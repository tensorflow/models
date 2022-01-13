# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input processing for TriviaQA."""
import os
from typing import Optional, Text, Union

import tensorflow as tf
import tensorflow_datasets as tfds

from official.modeling import tf_utils
from official.projects.triviaqa import dataset  # pylint: disable=unused-import


def _flatten_dims(tensor: tf.Tensor,
                  first_dim: Optional[int] = 0,
                  last_dim: Optional[int] = -1,
                  name: Optional[Text] = None) -> tf.Tensor:
  """Flattens the given span of dimensions in `tensor`.

  Args:
    tensor: [..., first_dim_size, ...middle_dims..., last_dim_size, ...] shaped
      Tensor.
    first_dim: The first dimension to flatten (inclusive). Must be a valid index
      for the rank of `tensor`. Default is 0.
    last_dim: The last dimension to flatten (inclusive). Must be a valid index
      for the rank of `tensor`. Default is -1.
    name: A name for the operation (optional).

  Returns:
    Tensor of shape [..., flattened_dim_size, ...] where
    flattened_dim_size = first_dim_size * ...middle_dims... * last_dim_size.
  """
  with tf.name_scope(name or 'flatten_dims'):
    tensor = tf.convert_to_tensor(tensor)

    rank = tensor.shape.rank
    if rank is None:
      raise ValueError('Static rank of `tensor` must be known.')
    if first_dim < 0:  # pytype: disable=unsupported-operands
      first_dim += rank
    if first_dim < 0 or first_dim >= rank:  # pytype: disable=unsupported-operands
      raise ValueError('`first_dim` out of bounds for `tensor` rank.')
    if last_dim < 0:  # pytype: disable=unsupported-operands
      last_dim += rank
    if last_dim < 0 or last_dim >= rank:  # pytype: disable=unsupported-operands
      raise ValueError('`last_dim` out of bounds for `tensor` rank.')
    if first_dim > last_dim:  # pytype: disable=unsupported-operands
      raise ValueError('`first_dim` must not be larger than `last_dim`.')

    # Try to calculate static flattened dim size if all input sizes to flatten
    # are statically known. Otherwise, just use -1.
    flat_dims_shape = tensor.shape[first_dim:(last_dim + 1)].as_list()
    flattened_dim_size = 1
    for size in flat_dims_shape:
      if size is None:
        flattened_dim_size = -1
        break
      flattened_dim_size *= size

    old_shape = tf.shape(tensor)
    output_shape = tf.concat([
        old_shape[:first_dim], [flattened_dim_size], old_shape[(last_dim + 1):]
    ], 0)
    return tf.reshape(tensor, output_shape)


def _pad_to_multiple(tensor: tf.Tensor,
                     factor: Union[int, tf.Tensor],
                     axis: int,
                     mode: Optional[Text] = 'CONSTANT',
                     constant_values=0,
                     name: Optional[Text] = None) -> tf.Tensor:
  """Pads `tensor` on a given `axis` to be a multiple of `factor`.

  Padding will be concatenated to the end of the axis only, not the beginning.
  If the length along `axis` is already a multiple of `factor`, this is
  effectively a no-op.

  Args:
    tensor: A Tensor with rank >= 1 to pad.
    factor: Positive integer factor to pad for. If a Tensor, must be a scalar
      int.
    axis: A valid axis in `tensor` to pad.
    mode: The padding mode to use according to `tf.pad`. Defaults to 'CONSTANT'.
    constant_values: For 'CONSTANT' mode, the scalar pad value to use within
      `tf.pad`. Defaults to 0. Must be same type as `tensor`.
    name: A name for the operation (optional).

  Returns:
    The padded Tensor result.
  """
  with tf.name_scope(name or 'pad_to_multiple'):
    tensor = tf.convert_to_tensor(tensor)

    if isinstance(factor, int) and factor < 1:
      raise ValueError('`factor` must be positive.')
    rank = tensor.shape.rank
    if rank is None:
      raise ValueError('Static rank of `tensor` must be known.')
    if axis < 0:
      axis += rank
    if axis < 0 or axis >= rank:
      raise ValueError('`axis` out of bounds for `tensor` rank.')

    axis_len = tf_utils.get_shape_list(tensor)[axis]
    pad_len = -axis_len % factor
    paddings = pad_len * tf.one_hot([-1, axis], rank, axis=0, dtype=tf.int32)
    return tf.pad(
        tensor=tensor,
        paddings=paddings,
        mode=mode,
        constant_values=constant_values)


def _skew_elements_right(tensor: tf.Tensor,
                         axis: int,
                         pad_value=0,
                         name: Optional[Text] = None) -> tf.Tensor:
  """Skews successive elements right along the given `axis`.

  This changes an input like
  [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
  ]
  into the following:
  [
    [1, 2, 3, 0, 0],
    [0, 4, 5, 6, 0],
    [0, 0, 7, 8, 9]
  ]

  Args:
    tensor: Tensor of shape [..., num_rows, axis_len, ...].
    axis: A valid axis in `tensor` to skew along. It must not be the first axis
      in `tensor`.
    pad_value: The scalar pad value to use. Defaults to 0. Must be the same type
      as `tensor`.
    name: A name for the operation (optional).

  Returns:
    Tensor of shape [..., num_rows, axis_len + num_rows - 1, ...].
  """
  with tf.name_scope(name or 'skew_elements_right'):
    tensor = tf.convert_to_tensor(tensor)

    rank = tensor.shape.rank
    num_rows = tf_utils.get_shape_list(tensor)[axis - 1]
    axis_len = tf_utils.get_shape_list(tensor)[axis]

    if rank is None:
      raise ValueError('Static rank of `tensor` must be known.')
    if axis < 0:
      axis += rank
    if axis <= 0 or axis >= rank:
      raise ValueError('`axis` out of bounds for `tensor` rank.')

    output_len = axis_len + num_rows - 1

    paddings = num_rows * tf.one_hot([-1, axis], rank, axis=0, dtype=tf.int32)

    # [..., num_rows, axis_len + num_rows, ...]
    padded_tensor = tf.pad(tensor, paddings, constant_values=pad_value)

    # [..., num_rows * (axis_len + num_rows), ...]
    flat_tensor = _flatten_dims(
        padded_tensor, first_dim=axis - 1, last_dim=axis)

    padded_tensor2 = _pad_to_multiple(
        flat_tensor,
        factor=output_len,
        axis=axis - 1,
        constant_values=pad_value)

    # [..., num_rows + 1, output_len, ...]
    new_shape = tf.concat([
        tf.shape(tensor)[:(axis - 1)], [num_rows + 1, output_len],
        tf.shape(tensor)[(axis + 1):]
    ], 0)
    reshaped_tensor = tf.reshape(padded_tensor2, new_shape)

    # [..., num_rows, output_len, ...]
    output_shape = new_shape - tf.one_hot(axis - 1, depth=rank, dtype=tf.int32)
    return tf.slice(
        reshaped_tensor, begin=tf.zeros_like(output_shape), size=output_shape)


class RelativePositionGenerator(object):
  """Generates `relative_att_ids` for purely distance-based relative positions.

  This implements the clipped relative position representations originally
  described in https://arxiv.org/abs/1803.02155 .

  Attributes:
    max_distance: Integer passed from `__init__`.
    ignore_direction: Bool passed from `__init__`.
    relative_vocab_size: Integer representing the maximum number of unique ids
      output from this generator.
    left_pad_value: Integer id for all positions at or beyond max_distance to
      the left.
    right_pad_value: Integer id for all positions at or beyond max_distance to
      the right.
  """

  def __init__(self, max_distance: int, ignore_direction: bool = False):
    """Init.

    Args:
      max_distance: The maximum distance to represent. Must not be negative. All
        larger distances will be clipped to this value.
      ignore_direction: If True, both left and right position representations
        will have the same ids based on absolute distance (resulting in
        symmetric ids around the center token).
    """
    if max_distance < 0:
      raise ValueError('`max_distance` must not be negative.')
    self.max_distance = max_distance
    self.ignore_direction = ignore_direction

    self.right_pad_value = max_distance
    self.left_pad_value = max_distance if ignore_direction else 2 * max_distance

    # 0 is the first id, so vocab size is 1 + the largest id (left pad value).
    self.relative_vocab_size = self.left_pad_value + 1

  def make_relative_att_ids(self,
                            seq_len: Union[int, tf.Tensor],
                            batch_size: Optional[Union[int, tf.Tensor]] = 1,
                            name: Optional[Text] = None) -> tf.Tensor:
    """Makes relative position ids for full self-attention.

    For example, if `max_distance` is 3, `ignore_direction` is False, `seq_len`
    is 6, and `batch_size` is 1, the result is the following:
      [[
          [0, 1, 2, 3, 3, 3],
          [4, 0, 1, 2, 3, 3],
          [5, 4, 0, 1, 2, 3],
          [6, 5, 4, 0, 1, 2],
          [6, 6, 5, 4, 0, 1],
          [6, 6, 6, 5, 4, 0],
      ]]

    Args:
      seq_len: The sequence length to create ids for. Must be positive. If a
        Tensor, must be a scalar int.
      batch_size: The batch size of the result (default 1). Must be positive. If
        a Tensor, must be a scalar int. All examples in the batch will have the
        same id pattern.
      name: A name for the operation (optional).

    Returns:
      <int32>[batch_size, seq_len, seq_len] Tensor of relative position ids.
    """
    with tf.name_scope(name or 'make_relative_att_ids'):
      if isinstance(seq_len, int) and seq_len < 1:
        raise ValueError('`seq_len` must be positive.')
      if isinstance(batch_size, int) and batch_size < 1:
        raise ValueError('`batch_size` must be positive.')

      # We need the id_pattern to cover all tokens to the left of the last token
      # and all tokens to the right of the first token at the same time.
      window_size = 2 * seq_len - 1

      # [window_size]
      id_pattern = self._make_relative_id_pattern(window_size)

      # [seq_len, window_size]
      id_tensor = tf.tile(id_pattern[tf.newaxis, :], [seq_len, 1])

      # [seq_len, window_size + seq_len - 1]
      id_tensor = _skew_elements_right(id_tensor, -1)

      # [seq_len, seq_len]
      id_tensor = tf.slice(id_tensor, [0, seq_len - 1], [seq_len, seq_len])

      return tf.tile(id_tensor[tf.newaxis, :, :], [batch_size, 1, 1])

  def make_local_relative_att_ids(self,
                                  seq_len: Union[int, tf.Tensor],
                                  local_radius: int,
                                  batch_size: Optional[Union[int,
                                                             tf.Tensor]] = 1,
                                  name: Optional[Text] = None) -> tf.Tensor:
    """Makes relative position ids for local self-attention.

    The result can be used as `relative_att_ids` in
    `layers.RelativeLocalSelfAttention`.

    For example, if `max_distance` is 3, `ignore_direction` is False, `seq_len`
    is 4, `local_radius` is 5, and `batch_size` is 1, the result is the
    following:
      [[
          [6, 6, 6, 5, 4, 0, 1, 2, 3, 3, 3],
          [6, 6, 6, 5, 4, 0, 1, 2, 3, 3, 3],
          [6, 6, 6, 5, 4, 0, 1, 2, 3, 3, 3],
          [6, 6, 6, 5, 4, 0, 1, 2, 3, 3, 3],
      ]]

    Args:
      seq_len: The sequence length to create ids for. Must be positive. If a
        Tensor, must be a scalar int.
      local_radius: The local radius as expected by
        `layers.RelativeLocalSelfAttention`. Must be positive.
      batch_size: The batch size of the result (default 1). Must be positive. If
        a Tensor, must be a scalar int. All examples in the batch will have the
        same id pattern.
      name: A name for the operation (optional).

    Returns:
      <int32>[batch_size, seq_len, 2*local_radius + 1] Tensor of relative
      position ids.
    """
    with tf.name_scope(name or 'make_local_relative_att_ids'):
      if isinstance(seq_len, int) and seq_len < 1:
        raise ValueError('`seq_len` must be positive.')
      if local_radius < 1:
        raise ValueError('`local_radius` must be positive.')
      if isinstance(batch_size, int) and batch_size < 1:
        raise ValueError('`batch_size` must be positive.')

      window_size = 2 * local_radius + 1

      # [window_size]
      id_pattern = self._make_relative_id_pattern(window_size)

      return tf.tile(id_pattern[tf.newaxis, tf.newaxis, :],
                     [batch_size, seq_len, 1])

  def _make_relative_id_pattern(
      self, window_size: Union[int, tf.Tensor]) -> tf.Tensor:
    """Helper for making the relative id pattern for a particular window size.

    For example, if `max_distance` is 3, `ignore_direction` is False, and
    `window_size` is 11, the result is the following:
    [6, 6, 6, 5, 4, 0, 1, 2, 3, 3, 3].

    Args:
      window_size: Window size to return relative ids for. Must be positive and
        odd since ids will be relative to the center of the window. If a Tensor,
        must be a scalar int.

    Returns:
      <int32>[window_size] Tensor of relative position ids.
    """
    if isinstance(window_size, int):
      if window_size < 1:
        raise ValueError('`window_size` must be positive.')
      if window_size % 2 != 1:
        raise ValueError('`window_size` must be odd.')

    x = tf.range(self.max_distance + 1, dtype=tf.int32)
    x = tf.pad(x, [[self.max_distance, 0]], mode='REFLECT')
    if not self.ignore_direction:
      direction_adder = tf.concat([
          tf.fill([self.max_distance], self.max_distance),
          tf.zeros([self.max_distance + 1], dtype=tf.int32)
      ], 0)
      x += direction_adder

    len_x = x.shape.as_list()[0]
    if len_x > window_size:
      trim_amount = (len_x - window_size) // 2
      return x[trim_amount:-trim_amount]

    pad_amount = (window_size - len_x) // 2
    result = tf.pad(x, [[pad_amount, 0]], constant_values=self.left_pad_value)
    result = tf.pad(
        result, [[0, pad_amount]], constant_values=self.right_pad_value)
    return result


def read_batches(data_dir,
                 split,
                 batch_size,
                 include_answers=True,
                 shuffle=False,
                 drop_final_batch=False,
                 compression_type=''):
  """Read TriviaQA batches."""
  features = {
      'id': tf.io.FixedLenFeature([], tf.string),
      'qid': tf.io.FixedLenFeature([], tf.string),
      'context': tf.io.FixedLenFeature([], tf.string),
      'question': tf.io.FixedLenFeature([], tf.string),
      'global_token_ids': tf.io.RaggedFeature(tf.int64),
      'token_ids': tf.io.RaggedFeature(tf.int64),
      'segment_ids': tf.io.RaggedFeature(tf.int64),
      'token_offsets': tf.io.RaggedFeature(tf.int64),
  }
  if include_answers:
    features['answers'] = tf.io.RaggedFeature(
        tf.int64, partitions=(tf.io.RaggedFeature.UniformRowLength(2),))  # pytype: disable=attribute-error

  dataset_builder = tfds.builder(
      'bigbird_trivia_qa/rc_wiki.preprocessed', data_dir=data_dir)
  split_info = dataset_builder.info.splits[split]
  return tf.data.experimental.make_batched_features_dataset(
      [
          os.path.join(dataset_builder.data_dir, filename)
          for filename in split_info.filenames
      ],
      batch_size=batch_size,
      features=features,
      reader=lambda path: tf.data.TFRecordDataset(path, compression_type),
      label_key='answers' if include_answers else None,
      num_epochs=1,
      shuffle=shuffle,
      shuffle_buffer_size=split_info.num_examples,
      prefetch_buffer_size=tf.data.experimental.AUTOTUNE,
      sloppy_ordering=True,
      drop_final_batch=drop_final_batch,
      reader_num_threads=8,
      parser_num_threads=16)


def scatter_labels(labels, batch_size, sequence_length):
  """Create one hot labels."""
  row_ids = labels.value_rowids()
  indices = tf.concat(
      (tf.stack((row_ids, tf.cast(labels.flat_values[:, 0],
                                  tf.int32), tf.zeros_like(row_ids)), -1),
       tf.stack((row_ids, tf.cast(labels.flat_values[:, 1],
                                  tf.int32), tf.ones_like(row_ids)), -1)), 0)
  one_hot_labels = tf.scatter_nd(indices,
                                 tf.ones(tf.shape(indices)[0], tf.float32),
                                 (batch_size, sequence_length, 2))
  return tf.minimum(one_hot_labels, 1.)


def features_map_fn(features, local_radius, relative_pos_max_distance,
                    use_hard_g2l_mask, padding_id, eos_id, null_id, cls_id,
                    sep_id, sequence_length, global_sequence_length):
  """Make features."""
  batch_size = tf.get_static_value(features['token_ids'].shape[0])
  # sequence_lengths = features['token_ids'].row_lengths()
  question_lengths = tf.argmax(
      tf.equal(features['token_ids'].to_tensor(
          shape=(batch_size, global_sequence_length)), sep_id), -1) + 1
  mapped_features = dict(
      token_ids=tf.cast(
          features['token_ids'].to_tensor(shape=(batch_size, sequence_length)),
          tf.int32),
      global_token_ids=tf.cast(
          features['global_token_ids'].to_tensor(
              shape=(batch_size, global_sequence_length)), tf.int32),
      segment_ids=tf.cast(
          features['segment_ids'].to_tensor(
              shape=(batch_size, sequence_length)), tf.int32),
  )
  relative_pos_generator = RelativePositionGenerator(
      max_distance=relative_pos_max_distance)
  # Only do long-to-long attention for non-null tokens.
  # Let the null token attend to itself.
  l2l_att_mask = tf.ones((batch_size, sequence_length, 2 * local_radius + 1),
                         tf.int32)
  l2l_att_mask *= 1 - tf.cast(
      tf.logical_or(
          tf.equal(mapped_features['token_ids'], padding_id),
          tf.equal(mapped_features['token_ids'], null_id)),
      tf.int32)[:, :, tf.newaxis]
  l2l_relative_att_ids = relative_pos_generator.make_local_relative_att_ids(
      seq_len=sequence_length, local_radius=local_radius, batch_size=batch_size)
  #
  l2g_att_mask = tf.ones((batch_size, sequence_length, global_sequence_length),
                         tf.int32)
  l2g_att_mask *= tf.cast(
      tf.not_equal(mapped_features['token_ids'], padding_id),
      tf.int32)[:, :, tf.newaxis]
  l2g_att_mask *= tf.cast(
      tf.not_equal(mapped_features['global_token_ids'], padding_id),
      tf.int32)[:, tf.newaxis, :]
  l2g_relative_att_ids = tf.fill(
      (batch_size, sequence_length, global_sequence_length),
      relative_pos_generator.relative_vocab_size + 1)
  #
  g2g_att_mask = tf.ones(
      (batch_size, global_sequence_length, global_sequence_length), tf.int32)
  g2g_att_mask *= tf.cast(
      tf.not_equal(mapped_features['global_token_ids'], padding_id),
      tf.int32)[:, :, tf.newaxis]
  g2g_relative_att_ids = relative_pos_generator.make_relative_att_ids(
      seq_len=global_sequence_length, batch_size=batch_size)
  global_sentence_mask = tf.equal(mapped_features['global_token_ids'], eos_id)
  global_question_mask = tf.logical_not(
      tf.logical_or(
          tf.logical_or(
              tf.equal(mapped_features['global_token_ids'], cls_id),
              tf.equal(mapped_features['global_token_ids'], eos_id)),
          tf.equal(mapped_features['global_token_ids'], padding_id)))
  g2g_question_mask = tf.logical_and(global_question_mask[:, tf.newaxis, :],
                                     global_question_mask[:, :, tf.newaxis])
  g2g_sentence_mask = tf.logical_and(global_sentence_mask[:, tf.newaxis, :],
                                     global_sentence_mask[:, :, tf.newaxis])
  g2g_local_mask = tf.cast(
      tf.logical_or(g2g_question_mask, g2g_sentence_mask), tf.int32)
  g2g_relative_att_ids *= g2g_local_mask
  g2g_relative_att_ids += (1 - g2g_local_mask) * (
      relative_pos_generator.relative_vocab_size + 2)
  #
  g2l_att_mask = tf.transpose(l2g_att_mask, [0, 2, 1])
  if use_hard_g2l_mask:
    global_range = tf.range(
        global_sequence_length, dtype=mapped_features['global_token_ids'].dtype)
    g2l_att_mask *= tf.cast(
        tf.logical_or(
            tf.equal(
                mapped_features['global_token_ids'], cls_id)[:, :, tf.newaxis],
            tf.equal(global_range[tf.newaxis, :, tf.newaxis],
                     mapped_features['segment_ids'][:, tf.newaxis, :])),
        tf.int32)
  g2l_relative_att_ids = tf.transpose(l2g_relative_att_ids, [0, 2, 1])
  mapped_features.update(
      dict(
          l2l_att_mask=l2l_att_mask,
          l2l_relative_att_ids=l2l_relative_att_ids,
          l2g_att_mask=l2g_att_mask,
          l2g_relative_att_ids=l2g_relative_att_ids,
          g2g_att_mask=g2g_att_mask,
          g2g_relative_att_ids=g2g_relative_att_ids,
          g2l_att_mask=g2l_att_mask,
          g2l_relative_att_ids=g2l_relative_att_ids,
          question_lengths=question_lengths,
      ))
  return mapped_features


def labels_map_fn(token_ids, labels, sequence_length):
  batch_size = tf.get_static_value(labels.shape[0])
  row_lengths = labels.row_lengths()
  empty_token_index = token_ids.row_lengths() - 1
  one_hot_labels = scatter_labels(labels, batch_size, sequence_length)
  one_hot_labels += (tf.cast(row_lengths == 0, tf.float32)[:, tf.newaxis] *
                     tf.one_hot(empty_token_index, sequence_length))[:, :,
                                                                     tf.newaxis]
  return one_hot_labels
