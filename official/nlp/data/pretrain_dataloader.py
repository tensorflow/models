# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Loads dataset for the BERT pretraining task."""
import dataclasses
from typing import Mapping, Optional

from absl import logging

import numpy as np
import tensorflow as tf, tf_keras
from official.common import dataset_fn
from official.core import config_definitions as cfg
from official.core import input_reader
from official.nlp.data import data_loader
from official.nlp.data import data_loader_factory


@dataclasses.dataclass
class BertPretrainDataConfig(cfg.DataConfig):
  """Data config for BERT pretraining task (tasks/masked_lm)."""
  input_path: str = ''
  global_batch_size: int = 512
  is_training: bool = True
  seq_length: int = 512
  max_predictions_per_seq: int = 76
  use_next_sentence_label: bool = True
  use_position_id: bool = False
  # Historically, BERT implementations take `input_ids` and `segment_ids` as
  # feature names. Inside the TF Model Garden implementation, the Keras model
  # inputs are set as `input_word_ids` and `input_type_ids`. When
  # v2_feature_names is True, the data loader assumes the tf.Examples use
  # `input_word_ids` and `input_type_ids` as keys.
  use_v2_feature_names: bool = False
  file_type: str = 'tfrecord'


@data_loader_factory.register_data_loader_cls(BertPretrainDataConfig)
class BertPretrainDataLoader(data_loader.DataLoader):
  """A class to load dataset for bert pretraining task."""

  def __init__(self, params):
    """Inits `BertPretrainDataLoader` class.

    Args:
      params: A `BertPretrainDataConfig` object.
    """
    self._params = params
    self._seq_length = params.seq_length
    self._max_predictions_per_seq = params.max_predictions_per_seq
    self._use_next_sentence_label = params.use_next_sentence_label
    self._use_position_id = params.use_position_id

  def _name_to_features(self):
    name_to_features = {
        'input_mask':
            tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'masked_lm_positions':
            tf.io.FixedLenFeature([self._max_predictions_per_seq], tf.int64),
        'masked_lm_ids':
            tf.io.FixedLenFeature([self._max_predictions_per_seq], tf.int64),
        'masked_lm_weights':
            tf.io.FixedLenFeature([self._max_predictions_per_seq], tf.float32),
    }
    if self._params.use_v2_feature_names:
      name_to_features.update({
          'input_word_ids': tf.io.FixedLenFeature([self._seq_length], tf.int64),
          'input_type_ids': tf.io.FixedLenFeature([self._seq_length], tf.int64),
      })
    else:
      name_to_features.update({
          'input_ids': tf.io.FixedLenFeature([self._seq_length], tf.int64),
          'segment_ids': tf.io.FixedLenFeature([self._seq_length], tf.int64),
      })
    if self._use_next_sentence_label:
      name_to_features['next_sentence_labels'] = tf.io.FixedLenFeature([1],
                                                                       tf.int64)
    if self._use_position_id:
      name_to_features['position_ids'] = tf.io.FixedLenFeature(
          [self._seq_length], tf.int64)
    return name_to_features

  def _decode(self, record: tf.Tensor):
    """Decodes a serialized tf.Example."""
    name_to_features = self._name_to_features()
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example

  def _parse(self, record: Mapping[str, tf.Tensor]):
    """Parses raw tensors into a dict of tensors to be consumed by the model."""
    x = {
        'input_mask': record['input_mask'],
        'masked_lm_positions': record['masked_lm_positions'],
        'masked_lm_ids': record['masked_lm_ids'],
        'masked_lm_weights': record['masked_lm_weights'],
    }
    if self._params.use_v2_feature_names:
      x['input_word_ids'] = record['input_word_ids']
      x['input_type_ids'] = record['input_type_ids']
    else:
      x['input_word_ids'] = record['input_ids']
      x['input_type_ids'] = record['segment_ids']
    if self._use_next_sentence_label:
      x['next_sentence_labels'] = record['next_sentence_labels']
    if self._use_position_id:
      x['position_ids'] = record['position_ids']

    return x

  def load(self, input_context: Optional[tf.distribute.InputContext] = None):
    """Returns a tf.dataset.Dataset."""
    reader = input_reader.InputReader(
        params=self._params,
        dataset_fn=dataset_fn.pick_dataset_fn(self._params.file_type),
        decoder_fn=self._decode,
        parser_fn=self._parse)
    return reader.read(input_context)


@dataclasses.dataclass
class XLNetPretrainDataConfig(cfg.DataConfig):
  """Data config for XLNet pretraining task.

  Attributes:
    input_path: See base class.
    global_batch_size: See base calss.
    is_training: See base class.
    seq_length: The length of each sequence.
    max_predictions_per_seq: The number of predictions per sequence.
    reuse_length: The number of tokens in a previous segment to reuse. This
      should be the same value used during pretrain data creation.
    sample_strategy: The strategy used to sample factorization permutations.
      Possible values: 'single_token', 'whole_word', 'token_span', 'word_span'.
    min_num_tokens: The minimum number of tokens to sample in a span. This is
      used when `sample_strategy` is 'token_span'.
    max_num_tokens: The maximum number of tokens to sample in a span. This is
      used when `sample_strategy` is 'token_span'.
    min_num_words: The minimum number of words to sample in a span. This is used
      when `sample_strategy` is 'word_span'.
    max_num_words: The maximum number of words to sample in a span. This is used
      when `sample_strategy` is 'word_span'.
    permutation_size: The length of the longest permutation. This can be set to
      `reuse_length`. This should NOT be greater than `reuse_length`, otherwise
      this may introduce data leaks.
    leak_ratio: The percentage of masked tokens that are leaked.
    segment_sep_id: The ID of the SEP token used when preprocessing the dataset.
    segment_cls_id: The ID of the CLS token used when preprocessing the dataset.
  """
  input_path: str = ''
  global_batch_size: int = 512
  is_training: bool = True
  seq_length: int = 512
  max_predictions_per_seq: int = 76
  reuse_length: int = 256
  sample_strategy: str = 'word_span'
  min_num_tokens: int = 1
  max_num_tokens: int = 5
  min_num_words: int = 1
  max_num_words: int = 5
  permutation_size: int = 256
  leak_ratio: float = 0.1
  segment_sep_id: int = 4
  segment_cls_id: int = 3


@data_loader_factory.register_data_loader_cls(XLNetPretrainDataConfig)
class XLNetPretrainDataLoader(data_loader.DataLoader):
  """A class to load dataset for xlnet pretraining task."""

  def __init__(self, params: XLNetPretrainDataConfig):
    """Inits `XLNetPretrainDataLoader` class.

    Args:
      params: A `XLNetPretrainDataConfig` object.
    """
    self._params = params
    self._seq_length = params.seq_length
    self._max_predictions_per_seq = params.max_predictions_per_seq
    self._reuse_length = params.reuse_length
    self._num_replicas_in_sync = None
    self._permutation_size = params.permutation_size
    self._sep_id = params.segment_sep_id
    self._cls_id = params.segment_cls_id
    self._sample_strategy = params.sample_strategy
    self._leak_ratio = params.leak_ratio

  def _decode(self, record: tf.Tensor):
    """Decodes a serialized tf.Example."""
    name_to_features = {
        'input_word_ids': tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'input_type_ids': tf.io.FixedLenFeature([self._seq_length], tf.int64),
        'boundary_indices': tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example

  def _parse(self, record: Mapping[str, tf.Tensor]):
    """Parses raw tensors into a dict of tensors to be consumed by the model."""
    x = {}

    inputs = record['input_word_ids']
    x['input_type_ids'] = record['input_type_ids']

    if self._sample_strategy in ['whole_word', 'word_span']:
      boundary = tf.sparse.to_dense(record['boundary_indices'])
    else:
      boundary = None

    input_mask = self._online_sample_mask(inputs=inputs, boundary=boundary)

    if self._reuse_length > 0:
      if self._permutation_size > self._reuse_length:
        logging.warning(
            '`permutation_size` is greater than `reuse_length` (%d > %d).'
            'This may introduce data leakage.', self._permutation_size,
            self._reuse_length)

      # Enable the memory mechanism.
      # Permute the reuse and non-reuse segments separately.
      non_reuse_len = self._seq_length - self._reuse_length
      if not (self._reuse_length % self._permutation_size == 0 and
              non_reuse_len % self._permutation_size == 0):
        raise ValueError('`reuse_length` and `seq_length` should both be '
                         'a multiple of `permutation_size`.')

      # Creates permutation mask and target mask for the first reuse_len tokens.
      # The tokens in this part are reused from the last sequence.
      perm_mask_0, target_mask_0, tokens_0, masked_0 = self._get_factorization(
          inputs=inputs[:self._reuse_length],
          input_mask=input_mask[:self._reuse_length])

      # Creates permutation mask and target mask for the rest of tokens in
      # current example, which are concatentation of two new segments.
      perm_mask_1, target_mask_1, tokens_1, masked_1 = self._get_factorization(
          inputs[self._reuse_length:], input_mask[self._reuse_length:])

      perm_mask_0 = tf.concat([
          perm_mask_0,
          tf.zeros([self._reuse_length, non_reuse_len], dtype=tf.int32)
      ],
                              axis=1)
      perm_mask_1 = tf.concat([
          tf.ones([non_reuse_len, self._reuse_length], dtype=tf.int32),
          perm_mask_1
      ],
                              axis=1)
      perm_mask = tf.concat([perm_mask_0, perm_mask_1], axis=0)
      target_mask = tf.concat([target_mask_0, target_mask_1], axis=0)
      tokens = tf.concat([tokens_0, tokens_1], axis=0)
      masked_tokens = tf.concat([masked_0, masked_1], axis=0)
    else:
      # Disable the memory mechanism.
      if self._seq_length % self._permutation_size != 0:
        raise ValueError('`seq_length` should be a multiple of '
                         '`permutation_size`.')
      # Permute the entire sequence together
      perm_mask, target_mask, tokens, masked_tokens = self._get_factorization(
          inputs=inputs, input_mask=input_mask)
    x['permutation_mask'] = tf.reshape(perm_mask,
                                       [self._seq_length, self._seq_length])
    x['input_word_ids'] = tokens
    x['masked_tokens'] = masked_tokens

    target = tokens
    if self._max_predictions_per_seq is not None:
      indices = tf.range(self._seq_length, dtype=tf.int32)
      bool_target_mask = tf.cast(target_mask, tf.bool)
      indices = tf.boolean_mask(indices, bool_target_mask)

      # account for extra padding due to CLS/SEP.
      actual_num_predict = tf.shape(indices)[0]
      pad_len = self._max_predictions_per_seq - actual_num_predict

      target_mapping = tf.one_hot(indices, self._seq_length, dtype=tf.int32)
      paddings = tf.zeros([pad_len, self._seq_length],
                          dtype=target_mapping.dtype)
      target_mapping = tf.concat([target_mapping, paddings], axis=0)
      x['target_mapping'] = tf.reshape(
          target_mapping, [self._max_predictions_per_seq, self._seq_length])

      target = tf.boolean_mask(target, bool_target_mask)
      paddings = tf.zeros([pad_len], dtype=target.dtype)
      target = tf.concat([target, paddings], axis=0)
      x['target'] = tf.reshape(target, [self._max_predictions_per_seq])

      target_mask = tf.concat([
          tf.ones([actual_num_predict], dtype=tf.int32),
          tf.zeros([pad_len], dtype=tf.int32)
      ],
                              axis=0)
      x['target_mask'] = tf.reshape(target_mask,
                                    [self._max_predictions_per_seq])
    else:
      x['target'] = tf.reshape(target, [self._seq_length])
      x['target_mask'] = tf.reshape(target_mask, [self._seq_length])
    return x

  def _index_pair_to_mask(self, begin_indices: tf.Tensor,
                          end_indices: tf.Tensor,
                          inputs: tf.Tensor) -> tf.Tensor:
    """Converts beginning and end indices into an actual mask."""
    non_func_mask = tf.logical_and(
        tf.not_equal(inputs, self._sep_id), tf.not_equal(inputs, self._cls_id))
    all_indices = tf.where(
        non_func_mask, tf.range(self._seq_length, dtype=tf.int32),
        tf.constant(-1, shape=[self._seq_length], dtype=tf.int32))
    candidate_matrix = tf.cast(
        tf.logical_and(all_indices[None, :] >= begin_indices[:, None],
                       all_indices[None, :] < end_indices[:, None]), tf.float32)
    cumsum_matrix = tf.reshape(
        tf.cumsum(tf.reshape(candidate_matrix, [-1])), [-1, self._seq_length])
    masked_matrix = tf.cast(cumsum_matrix <= self._max_predictions_per_seq,
                            tf.float32)
    target_mask = tf.reduce_sum(candidate_matrix * masked_matrix, axis=0)
    return tf.cast(target_mask, tf.bool)

  def _single_token_mask(self, inputs: tf.Tensor) -> tf.Tensor:
    """Samples individual tokens as prediction targets."""
    all_indices = tf.range(self._seq_length, dtype=tf.int32)
    non_func_mask = tf.logical_and(
        tf.not_equal(inputs, self._sep_id), tf.not_equal(inputs, self._cls_id))
    non_func_indices = tf.boolean_mask(all_indices, non_func_mask)

    masked_pos = tf.random.shuffle(non_func_indices)
    masked_pos = tf.sort(masked_pos[:self._max_predictions_per_seq])

    sparse_indices = tf.stack([tf.zeros_like(masked_pos), masked_pos], axis=-1)
    sparse_indices = tf.cast(sparse_indices, tf.int64)

    sparse_indices = tf.sparse.SparseTensor(
        sparse_indices,
        values=tf.ones_like(masked_pos),
        dense_shape=(1, self._seq_length))

    target_mask = tf.sparse.to_dense(sp_input=sparse_indices, default_value=0)

    return tf.squeeze(tf.cast(target_mask, tf.bool))

  def _whole_word_mask(self, inputs: tf.Tensor,
                       boundary: tf.Tensor) -> tf.Tensor:
    """Samples whole words as prediction targets."""
    pair_indices = tf.concat([boundary[:-1, None], boundary[1:, None]], axis=1)
    cand_pair_indices = tf.random.shuffle(
        pair_indices)[:self._max_predictions_per_seq]
    begin_indices = cand_pair_indices[:, 0]
    end_indices = cand_pair_indices[:, 1]

    return self._index_pair_to_mask(
        begin_indices=begin_indices, end_indices=end_indices, inputs=inputs)

  def _token_span_mask(self, inputs: tf.Tensor) -> tf.Tensor:
    """Samples token spans as prediction targets."""
    min_num_tokens = self._params.min_num_tokens
    max_num_tokens = self._params.max_num_tokens

    mask_alpha = self._seq_length / self._max_predictions_per_seq
    round_to_int = lambda x: tf.cast(tf.round(x), tf.int32)

    # Sample span lengths from a zipf distribution
    span_len_seq = np.arange(min_num_tokens, max_num_tokens + 1)
    probs = np.array([1.0 / (i + 1) for i in span_len_seq])

    probs /= np.sum(probs)
    logits = tf.constant(np.log(probs), dtype=tf.float32)
    span_lens = tf.random.categorical(
        logits=logits[None],
        num_samples=self._max_predictions_per_seq,
        dtype=tf.int32,
    )[0] + min_num_tokens

    # Sample the ratio [0.0, 1.0) of left context lengths
    span_lens_float = tf.cast(span_lens, tf.float32)
    left_ratio = tf.random.uniform(
        shape=[self._max_predictions_per_seq], minval=0.0, maxval=1.0)
    left_ctx_len = left_ratio * span_lens_float * (mask_alpha - 1)
    left_ctx_len = round_to_int(left_ctx_len)

    # Compute the offset from left start to the right end
    right_offset = round_to_int(span_lens_float * mask_alpha) - left_ctx_len

    # Get the actual begin and end indices
    begin_indices = (
        tf.cumsum(left_ctx_len) + tf.cumsum(right_offset, exclusive=True))
    end_indices = begin_indices + span_lens

    # Remove out of range indices
    valid_idx_mask = end_indices < self._seq_length
    begin_indices = tf.boolean_mask(begin_indices, valid_idx_mask)
    end_indices = tf.boolean_mask(end_indices, valid_idx_mask)

    # Shuffle valid indices
    num_valid = tf.cast(tf.shape(begin_indices)[0], tf.int32)
    order = tf.random.shuffle(tf.range(num_valid, dtype=tf.int32))
    begin_indices = tf.gather(begin_indices, order)
    end_indices = tf.gather(end_indices, order)

    return self._index_pair_to_mask(
        begin_indices=begin_indices, end_indices=end_indices, inputs=inputs)

  def _word_span_mask(self, inputs: tf.Tensor, boundary: tf.Tensor):
    """Sample whole word spans as prediction targets."""
    min_num_words = self._params.min_num_words
    max_num_words = self._params.max_num_words

    # Note: 1.2 is the token-to-word ratio
    mask_alpha = self._seq_length / self._max_predictions_per_seq / 1.2
    round_to_int = lambda x: tf.cast(tf.round(x), tf.int32)

    # Sample span lengths from a zipf distribution
    span_len_seq = np.arange(min_num_words, max_num_words + 1)
    probs = np.array([1.0 / (i + 1) for i in span_len_seq])
    probs /= np.sum(probs)
    logits = tf.constant(np.log(probs), dtype=tf.float32)

    # Sample `num_predict` words here: note that this is over sampling
    span_lens = tf.random.categorical(
        logits=logits[None],
        num_samples=self._max_predictions_per_seq,
        dtype=tf.int32,
    )[0] + min_num_words

    # Sample the ratio [0.0, 1.0) of left context lengths
    span_lens_float = tf.cast(span_lens, tf.float32)
    left_ratio = tf.random.uniform(
        shape=[self._max_predictions_per_seq], minval=0.0, maxval=1.0)
    left_ctx_len = left_ratio * span_lens_float * (mask_alpha - 1)

    left_ctx_len = round_to_int(left_ctx_len)
    right_offset = round_to_int(span_lens_float * mask_alpha) - left_ctx_len

    begin_indices = (
        tf.cumsum(left_ctx_len) + tf.cumsum(right_offset, exclusive=True))
    end_indices = begin_indices + span_lens

    # Remove out of range indices
    max_boundary_index = tf.cast(tf.shape(boundary)[0] - 1, tf.int32)
    valid_idx_mask = end_indices < max_boundary_index
    begin_indices = tf.boolean_mask(begin_indices, valid_idx_mask)
    end_indices = tf.boolean_mask(end_indices, valid_idx_mask)

    begin_indices = tf.gather(boundary, begin_indices)
    end_indices = tf.gather(boundary, end_indices)

    # Shuffle valid indices
    num_valid = tf.cast(tf.shape(begin_indices)[0], tf.int32)
    order = tf.random.shuffle(tf.range(num_valid, dtype=tf.int32))
    begin_indices = tf.gather(begin_indices, order)
    end_indices = tf.gather(end_indices, order)

    return self._index_pair_to_mask(
        begin_indices=begin_indices, end_indices=end_indices, inputs=inputs)

  def _online_sample_mask(self, inputs: tf.Tensor,
                          boundary: tf.Tensor) -> tf.Tensor:
    """Samples target positions for predictions.

    Descriptions of each strategy:
      - 'single_token': Samples individual tokens as prediction targets.
      - 'token_span': Samples spans of tokens as prediction targets.
      - 'whole_word': Samples individual words as prediction targets.
      - 'word_span': Samples spans of words as prediction targets.

    Args:
      inputs: The input tokens.
      boundary: The `int` Tensor of indices indicating whole word boundaries.
        This is used in 'whole_word' and 'word_span'

    Returns:
      The sampled `bool` input mask.

    Raises:
      `ValueError`: if `max_predictions_per_seq` is not set or if boundary is
        not provided for 'whole_word' and 'word_span' sample strategies.
    """
    if self._max_predictions_per_seq is None:
      raise ValueError('`max_predictions_per_seq` must be set.')

    if boundary is None and 'word' in self._sample_strategy:
      raise ValueError('`boundary` must be provided for {} strategy'.format(
          self._sample_strategy))

    if self._sample_strategy == 'single_token':
      return self._single_token_mask(inputs)
    elif self._sample_strategy == 'token_span':
      return self._token_span_mask(inputs)
    elif self._sample_strategy == 'whole_word':
      return self._whole_word_mask(inputs, boundary)
    elif self._sample_strategy == 'word_span':
      return self._word_span_mask(inputs, boundary)
    else:
      raise NotImplementedError('Invalid sample strategy.')

  def _get_factorization(self, inputs: tf.Tensor, input_mask: tf.Tensor):
    """Samples a permutation of the factorization order.

    Args:
      inputs: the input tokens.
      input_mask: the `bool` Tensor of the same shape as `inputs`. If `True`,
        then this means select for partial prediction.

    Returns:
      perm_mask: An `int32` Tensor of shape [seq_length, seq_length] consisting
        of 0s and 1s. If perm_mask[i][j] == 0, then this means that the i-th
        token (in original order) cannot attend to the jth attention token.
      target_mask: An `int32` Tensor of shape [seq_len] consisting of 0s and 1s.
        If target_mask[i] == 1, then the i-th token needs to be predicted and
        the mask will be used as input. This token will be included in the loss.
        If target_mask[i] == 0, then the token (or [SEP], [CLS]) will be used as
        input. This token will not be included in the loss.
      tokens: int32 Tensor of shape [seq_length].
      masked_tokens: int32 Tensor of shape [seq_length].
    """
    factorization_length = tf.shape(inputs)[0]
    # Generate permutation indices
    index = tf.range(factorization_length, dtype=tf.int32)
    index = tf.transpose(tf.reshape(index, [-1, self._permutation_size]))
    index = tf.random.shuffle(index)
    index = tf.reshape(tf.transpose(index), [-1])

    input_mask = tf.cast(input_mask, tf.bool)

    # non-functional tokens
    non_func_tokens = tf.logical_not(
        tf.logical_or(
            tf.equal(inputs, self._sep_id), tf.equal(inputs, self._cls_id)))
    masked_tokens = tf.logical_and(input_mask, non_func_tokens)
    non_masked_or_func_tokens = tf.logical_not(masked_tokens)

    smallest_index = -2 * tf.ones([factorization_length], dtype=tf.int32)

    # Similar to BERT, randomly leak some masked tokens
    if self._leak_ratio > 0:
      leak_tokens = tf.logical_and(
          masked_tokens,
          tf.random.uniform([factorization_length], maxval=1.0) <
          self._leak_ratio)
      can_attend_self = tf.logical_or(non_masked_or_func_tokens, leak_tokens)
    else:
      can_attend_self = non_masked_or_func_tokens
    to_index = tf.where(can_attend_self, smallest_index, index)
    from_index = tf.where(can_attend_self, to_index + 1, to_index)

    # For masked tokens, can attend if i > j
    # For context tokens, always can attend each other
    can_attend = from_index[:, None] > to_index[None, :]

    perm_mask = tf.cast(can_attend, tf.int32)

    # Only masked tokens are included in the loss
    target_mask = tf.cast(masked_tokens, tf.int32)

    return perm_mask, target_mask, inputs, masked_tokens

  def load(self, input_context: Optional[tf.distribute.InputContext] = None):
    """Returns a tf.dataset.Dataset."""
    if input_context:
      self._num_replicas_in_sync = input_context.num_replicas_in_sync
    reader = input_reader.InputReader(
        params=self._params, decoder_fn=self._decode, parser_fn=self._parse)
    return reader.read(input_context)
