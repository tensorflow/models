# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""An embedding network supporting packed sequences and position ids."""
# pylint: disable=g-classes-have-attributes
import collections
import tensorflow as tf

from official.modeling import tf_utils
from official.nlp.modeling import layers


@tf.keras.utils.register_keras_serializable(package='Text')
class PackedSequenceEmbedding(tf.keras.Model):
  """An embedding network supporting packed sequences and position ids.

  This network implements an embedding layer similar to the one described in
  "BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding" (https://arxiv.org/abs/1810.04805). On top of it, it supports
  to (1) pack multiple sequences into one sequence and (2) allow additional
  "position_ids" as input.

  Args:
    vocab_size: The size of the token vocabulary.
    type_vocab_size: The size of the type vocabulary.
    embedding_width: Width of token embeddings.
    hidden_size: The output size for this encoder.
    max_seq_length: The maximum sequence length for this encoder.
    initializer: The initializer for the embedding portion of this encoder.
    dropout_rate: The dropout rate to apply before the encoding layers.
    pack_multiple_sequences: If `True`, we can feed multiple sequences into one
      sequence for training and inference (they don't impact each other).
    use_position_id: Whether to expect `position_ids` as an input to the
      network. If False, the `position_ids` will be inferred: (1) when
        pack_multiple_sequences is False, we assume the position ids are `0, 1,
        2, ..., seq_length - 1`; (2) when `pack_multiple_sequences` is `True`,
        there may be multiple sub sequences, and for each sub sequence, its
        position ids start from 0, 1, 2, ...
  """

  def __init__(self,
               vocab_size,
               type_vocab_size,
               embedding_width,
               hidden_size,
               max_seq_length,
               initializer,
               dropout_rate,
               use_position_id=False,
               pack_multiple_sequences=False,
               **kwargs):
    initializer = tf.keras.initializers.get(initializer)
    if embedding_width is None:
      embedding_width = hidden_size
    config_dict = {
        'vocab_size': vocab_size,
        'type_vocab_size': type_vocab_size,
        'embedding_width': embedding_width,
        'hidden_size': hidden_size,
        'max_seq_length': max_seq_length,
        'initializer': tf.keras.initializers.serialize(initializer),
        'dropout_rate': dropout_rate,
        'use_position_id': use_position_id,
        'pack_multiple_sequences': pack_multiple_sequences,
    }

    word_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_word_ids')
    mask = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_mask')
    type_ids = tf.keras.layers.Input(
        shape=(None,), dtype=tf.int32, name='input_type_ids')
    inputs = [word_ids, mask, type_ids]
    if use_position_id:
      position_ids = tf.keras.layers.Input(
          shape=(None,), dtype=tf.int32, name='position_ids')
      inputs.append(position_ids)
    else:
      position_ids = None

    if pack_multiple_sequences:
      sub_seq_mask = PackedSequenceMask()(word_ids)
    else:
      sub_seq_mask = None

    embedding_layer = layers.OnDeviceEmbedding(
        vocab_size=vocab_size,
        embedding_width=embedding_width,
        initializer=initializer,
        name='word_embeddings')
    word_embeddings = embedding_layer(word_ids)

    # Always uses dynamic slicing for simplicity.
    position_embedding_layer = PositionEmbeddingWithSubSeqMask(
        initializer=initializer,
        use_dynamic_slicing=True,
        max_sequence_length=max_seq_length,
        name='position_embedding')
    position_embeddings = position_embedding_layer(
        word_embeddings, position_ids, sub_seq_mask)

    type_embeddings = (
        layers.OnDeviceEmbedding(
            vocab_size=type_vocab_size,
            embedding_width=embedding_width,
            initializer=initializer,
            use_one_hot=True,
            name='type_embeddings')(type_ids))

    embeddings = tf.keras.layers.Add()(
        [word_embeddings, position_embeddings, type_embeddings])
    embeddings = tf.keras.layers.LayerNormalization(
        name='embeddings/layer_norm', axis=-1, epsilon=1e-12, dtype=tf.float32)(
            embeddings)
    embeddings = tf.keras.layers.Dropout(
        rate=dropout_rate, dtype=tf.float32)(
            embeddings)

    if embedding_width != hidden_size:
      embeddings = tf.keras.layers.experimental.EinsumDense(
          '...x,xy->...y',
          output_shape=hidden_size,
          bias_axes=None,
          kernel_initializer=initializer,
          name='embedding_projection')(
              embeddings)

    attention_mask = layers.SelfAttentionMask()(embeddings, mask)
    if sub_seq_mask is not None:
      attention_mask = tf.keras.layers.Lambda(
          lambda x: x[0] * tf.cast(x[1], x[0].dtype))(
              [attention_mask, sub_seq_mask])

    outputs = [embeddings, attention_mask]
    super(PackedSequenceEmbedding, self).__init__(
        inputs=inputs, outputs=outputs, **kwargs)
    # TF does not track immutable attrs which do not contain Trackables,
    # so by creating a config namedtuple instead of a dict we avoid tracking it.
    config_cls = collections.namedtuple('Config', config_dict.keys())
    self._config = config_cls(**config_dict)
    self._embedding_layer = embedding_layer
    self._position_embedding_layer = position_embedding_layer

  def get_embedding_table(self):
    return self._embedding_layer.embeddings

  def get_config(self):
    return dict(self._config._asdict())

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


@tf.keras.utils.register_keras_serializable(package='Text')
class PackedSequenceMask(tf.keras.layers.Layer):
  """A layer to create a mask to indicate multiple sub sequences."""

  def call(self, input_ids):
    """Implements call() for the layer.

    Args:
      input_ids: int32 Tensor of shape [batch_size, seq_length].

    Returns:
      boolean Tensor of shape [batch_size, seq_length, seq_length]. [x, y, z]
      is True if for x'th instance in a batch, y'th token and z'th token are
      from the same sub sequence.
    """
    # Suppose
    # - the first token in the parent sequence is [CLS].
    # - every sequence starts from [CLS].
    # - every sequence only contains one [CLS].
    seq_start_token = input_ids[:, 0:1]
    seq_start_loc = tf.cast(tf.equal(input_ids, seq_start_token), tf.int32)
    # Set different ids for different sub sequences.
    seq_ids = tf.expand_dims(tf.cumsum(seq_start_loc, -1), -1)
    return tf.equal(seq_ids, tf.transpose(seq_ids, [0, 2, 1]))


@tf.keras.utils.register_keras_serializable(package='Text')
class PositionEmbeddingWithSubSeqMask(tf.keras.layers.Layer):
  """Creates a positional embedding with sub-sequence masking.

  This layer creates a positional embedding as described in "BERT: Pre-training
  of Deep Bidirectional Transformers for Language Understanding"
  (https://arxiv.org/abs/1810.04805). On top of it, it supports
  `position_ids` and `sub_sequence_mask` tensors.

  This layer can be set up to either create a statically shaped slice or a
  dynamically shaped slice. If `use_dynamic_slicing` is True, the input tensor
  can have a dynamic 1st dimension, while if `use_dynamic_slicing` is False the
  input size must be fixed.

  Args:
    initializer: The initializer to use for the embedding weights. Defaults to
      "glorot_uniform".
    use_dynamic_slicing: Whether to use the dynamic slicing path.
    max_sequence_length: The maximum size of the dynamic sequence. Only
      applicable if `use_dynamic_slicing` is True.
  """

  def __init__(self,
               initializer='glorot_uniform',
               use_dynamic_slicing=False,
               max_sequence_length=None,
               **kwargs):
    # We need to have a default dtype of float32, since the inputs (which Keras
    # usually uses to infer the dtype) will always be int32.
    if 'dtype' not in kwargs:
      kwargs['dtype'] = 'float32'

    super(PositionEmbeddingWithSubSeqMask, self).__init__(**kwargs)
    if use_dynamic_slicing and max_sequence_length is None:
      raise ValueError(
          'If `use_dynamic_slicing` is True, `max_sequence_length` must be set.'
      )
    self._max_sequence_length = max_sequence_length
    self._initializer = tf.keras.initializers.get(initializer)
    self._use_dynamic_slicing = use_dynamic_slicing

  def get_config(self):
    config = {
        'max_sequence_length': self._max_sequence_length,
        'initializer': tf.keras.initializers.serialize(self._initializer),
        'use_dynamic_slicing': self._use_dynamic_slicing,
    }
    base_config = super(PositionEmbeddingWithSubSeqMask, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    """Implements build() for the layer."""
    dimension_list = input_shape.as_list()

    if len(dimension_list) != 3:
      raise ValueError('PositionEmbedding expects a 3-dimensional input tensor '
                       'of shape [batch, sequence, width]')
    seq_length = dimension_list[1]
    width = dimension_list[2]

    # If we are not using dynamic slicing, we must assume that the sequence
    # length is fixed and max_sequence_length should not be specified.
    if not self._use_dynamic_slicing:
      if seq_length is None:
        raise ValueError(
            'PositionEmbedding must have `use_dynamic_slicing` set '
            'to True (and max_sequence_length set) when the '
            'sequence (1st) dimension of the input is None.')
      if self._max_sequence_length is not None:
        raise ValueError(
            'When `use_dynamic_slicing` is False, max_sequence_length should '
            'not be specified and we ought to use seq_length to get the '
            'variable shape.')

    if self._max_sequence_length is not None:
      weight_sequence_length = self._max_sequence_length
    else:
      weight_sequence_length = seq_length

    self._position_embeddings = self.add_weight(
        'embeddings',
        shape=[weight_sequence_length, width],
        initializer=self._initializer)

    super(PositionEmbeddingWithSubSeqMask, self).build(input_shape)

  def call(self, inputs, position_ids=None, sub_sequence_mask=None):
    """Implements call() for the layer.

    When `position_ids` is specified, it will return the position embeddings
    corresponding to this `position_ids`; otherwise, `position_ids` will be
    inferred in the following way:

    (1) When `sub_sequence_mask` is None, we assume the position ids are
        0, 1, 2, ..., seq_length - 1.
    (2) When `sub_sequence_mask` is specified, there may be multiple sub
        sequences, and for each sub sequence, its position ids start from
        0, 1, 2, ...

    Args:
      inputs: Word embeddings in shape [batch, seq_length, embedding_dim].
      position_ids: An optional int32 tensor in shape [batch, seq_length].
      sub_sequence_mask: An optional bool tensor in shape [batch, seq_length,
        seq_length]. [x, y, z] is True if for x'th instance in a batch, y'th
        token and z'th token are from the same sub sequence.

    Returns:
      The position embeddings in shape [batch, seq_length, embedding_dim].
    """
    input_shape = tf_utils.get_shape_list(inputs, expected_rank=3)
    if self._use_dynamic_slicing:
      position_embeddings = self._position_embeddings[:input_shape[1], :]
    else:
      position_embeddings = self._position_embeddings

    if position_ids is not None:
      return tf.gather(position_embeddings, position_ids)

    if sub_sequence_mask is None:
      return tf.broadcast_to(position_embeddings, input_shape)
    else:
      sub_sequence_mask = tf.cast(sub_sequence_mask, tf.int32)
      # For each sub sequence, its position ids start from 0, 1, 2, ...
      position_ids = tf.linalg.diag_part(tf.cumsum(sub_sequence_mask, -1)) - 1
      return tf.gather(position_embeddings, position_ids)
