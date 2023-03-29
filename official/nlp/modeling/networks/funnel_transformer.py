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

"""Funnel Transformer network."""
# pylint: disable=g-classes-have-attributes

import math
from typing import Any, Callable, Optional, Sequence, Union

from absl import logging
import numpy as np
import tensorflow as tf

from official.modeling import tf_utils
from official.nlp.modeling import layers

_Initializer = Union[str, tf.keras.initializers.Initializer]
_Activation = Union[str, Callable[..., Any]]

_MAX = 'max'
_AVG = 'avg'
_TRUNCATED_AVG = 'truncated_avg'

_transformer_cls2str = {
    layers.TransformerEncoderBlock: 'TransformerEncoderBlock',
    layers.ReZeroTransformer: 'ReZeroTransformer'
}

_str2transformer_cls = {
    'TransformerEncoderBlock': layers.TransformerEncoderBlock,
    'ReZeroTransformer': layers.ReZeroTransformer
}

_approx_gelu = lambda x: tf.keras.activations.gelu(x, approximate=True)


def _get_policy_dtype():
  try:
    return tf.keras.mixed_precision.global_policy().compute_dtype or tf.float32
  except AttributeError:  # tf1 has no attribute 'global_policy'
    return tf.float32


def _pool_and_concat(mask, unpool_length: int, strides: Union[Sequence[int],
                                                              int],
                     axes: Union[Sequence[int], int]):
  """Pools the mask along a given axis with stride.

  It also skips first unpool_length elements.

  Args:
    mask: Tensor to be pooled.
    unpool_length: Leading elements to be skipped.
    strides: Strides for the given axes.
    axes: Axes to pool the Tensor.

  Returns:
    Pooled and concatenated Tensor.
  """
  # Wraps the axes as a list.
  if isinstance(axes, int):
    axes = [axes]
  if isinstance(strides, int):
    strides = [strides] * len(axes)
  else:
    if len(strides) != len(axes):
      raise ValueError('The lengths of strides and axes need to match.')
  # Bypass no pooling cases.
  if np.all(np.array(strides) == 1):
    return mask

  for axis, stride in zip(axes, strides):
    # Skips first `unpool_length` tokens.
    unpool_tensor_shape = [slice(None)] * axis + [slice(None, unpool_length)]
    unpool_tensor = mask[unpool_tensor_shape]
    # Pools the second half.
    pool_tensor_shape = [slice(None)] * axis + [
        slice(unpool_length, None, stride)
    ]
    pool_tensor = mask[pool_tensor_shape]
    mask = tf.concat((unpool_tensor, pool_tensor), axis=axis)
  return mask


def _create_fractional_pool_transform(sl: int, pool_factor: float):
  """Create pooling transform for fractional pooling factor."""

  assert pool_factor > 1.0, '`pool_factor` should be > 1.0.'

  psl = int(sl / pool_factor)
  gcd_ = math.gcd(sl, psl)
  # It is expected chunk_sl and chunk_psl are small integers.
  # The transform is built by tiling a [chunk_sl, chunk_psl] submatrix
  # gcd_ times. The submatrix sums to chunk_psl.
  chunk_sl = sl // gcd_
  chunk_psl = psl // gcd_
  num_one_entries = chunk_psl - 1
  num_frac_entries = chunk_sl - (chunk_psl - 1)

  # The transform is of shape [sl, psl].
  transform = np.zeros((sl, psl))
  for i in range(sl // chunk_sl):
    row_start = chunk_sl * i
    col_start = chunk_psl * i
    for idx in range(num_one_entries):
      transform[row_start + idx][col_start + idx] = 1.0
    for idx in range(num_frac_entries):
      transform[row_start + num_one_entries + idx][
          col_start + num_one_entries
      ] = (1.0 / num_frac_entries)

  return tf.constant(transform, dtype=_get_policy_dtype())


def _create_truncated_avg_transforms(
    seq_length: int, pool_strides: Sequence[int]
):
  """Computes pooling transforms.

  The pooling_transform is of shape [seq_length,
  seq_length//pool_stride] and
  pooling_transform[i,j] = 1.0/pool_stride if i//pool_stride == j
                           0.0                otherwise.
  It's in essense average pooling but truncate the final window if it
  seq_length % pool_stride != 0.
  For seq_length==6 and pool_stride==2, it is
  [[ 0.5, 0.0, 0.0 ],
   [ 0.5, 0.0, 0.0 ],
   [ 0.0, 0.5, 0.0 ],
   [ 0.0, 0.5, 0.0 ],
   [ 0.0, 0.0, 0.5 ],
   [ 0.0, 0.0, 0.5 ]]

  Args:
    seq_length: int, sequence length.
    pool_strides: Sequence of pooling strides for each layer.

  Returns:
    pooling_transforms: Sequence of pooling transforms (Tensors) for each layer.
  """

  pooling_transforms = []
  for pool_stride in pool_strides:
    if pool_stride == 1:
      pooling_transforms.append(None)
    else:
      pooled_seq_length = int(seq_length / pool_stride)
      if (1.0 * pool_stride).is_integer():
        pfac, sl, psl = pool_stride, seq_length, pooled_seq_length

        transform = [
            [1.0 if (i // pfac) == j else 0.0 for j in range(psl)]
            for i in range(sl)
        ]
        transform = (
            tf.constant(transform, dtype=_get_policy_dtype()) / pool_stride
        )
      else:
        transform = _create_fractional_pool_transform(seq_length, pool_stride)
      pooling_transforms.append(transform)
      seq_length = pooled_seq_length

  return pooling_transforms


def _create_truncated_avg_masks(input_mask: tf.Tensor,
                                pool_strides: Sequence[int],
                                transforms: Sequence[tf.Tensor]):
  """Computes attention masks.

  For [1,1,1,0,0]

  Args:
    input_mask: Tensor of shape [batch_size, seq_length].
    pool_strides: Sequence of pooling strides for each layer.
    transforms: Sequence of off-diagonal matrices filling with 0.0 and
      1/pool_stride.

  Returns:
    attention_masks: Sequence of attention masks for each layer.
  """

  def create_2d_mask(from_length, mask):
    return tf.einsum('F,BT->BFT', tf.ones([from_length], dtype=mask.dtype),
                     mask)

  attention_masks = []
  seq_length = tf.shape(input_mask)[-1]
  layer_mask = tf.cast(input_mask, dtype=_get_policy_dtype())
  for pool_stride, transform in zip(pool_strides, transforms):
    if pool_stride == 1:
      attention_masks.append(create_2d_mask(seq_length, layer_mask))
    else:
      pooled_seq_length = tf.cast(
          tf.cast(seq_length, tf.float32) / tf.cast(pool_stride, tf.float32),
          tf.int32,
      )
      attention_masks.append(create_2d_mask(pooled_seq_length, layer_mask))

      layer_mask = tf.cast(
          tf.einsum('BF,FT->BT', layer_mask, transform) > 0.0,
          dtype=layer_mask.dtype,
      )
      seq_length = pooled_seq_length
  del seq_length

  return attention_masks


@tf.keras.utils.register_keras_serializable(package='Text')
class FunnelTransformerEncoder(tf.keras.layers.Layer):
  """Funnel Transformer-based encoder network.

  Funnel Transformer Implementation of https://arxiv.org/abs/2006.03236.
  This implementation utilizes the base framework with Bert
  (https://arxiv.org/abs/1810.04805).
  Its output is compatible with `BertEncoder`.

  Args:
    vocab_size: The size of the token vocabulary.
    hidden_size: The size of the transformer hidden layers.
    num_layers: The number of transformer layers.
    num_attention_heads: The number of attention heads for each transformer. The
      hidden size must be divisible by the number of attention heads.
    max_sequence_length: The maximum sequence length that this encoder can
      consume. If None, max_sequence_length uses the value from sequence length.
      This determines the variable shape for positional embeddings.
    type_vocab_size: The number of types that the 'type_ids' input can take.
    inner_dim: The output dimension of the first Dense layer in a two-layer
      feedforward network for each transformer.
    inner_activation: The activation for the first Dense layer in a two-layer
      feedforward network for each transformer.
    output_dropout: Dropout probability for the post-attention and output
      dropout.
    attention_dropout: The dropout rate to use for the attention layers within
      the transformer layers.
    pool_type: Pooling type. Choose from ['max', 'avg', 'truncated_avg'].
    pool_stride: An int or a list of ints. Pooling stride(s) to compress the
      sequence length. If set to int, each layer will have the same stride size.
      If set to list, the number of elements needs to match num_layers.
    unpool_length: Leading n tokens to be skipped from pooling.
    initializer: The initialzer to use for all weights in this encoder.
    output_range: The sequence output range, [0, output_range), by slicing the
      target sequence of the last transformer layer. `None` means the entire
      target sequence will attend to the source sequence, which yields the full
      output.
    embedding_width: The width of the word embeddings. If the embedding width is
      not equal to hidden size, embedding parameters will be factorized into two
      matrices in the shape of ['vocab_size', 'embedding_width'] and
      ['embedding_width', 'hidden_size'] ('embedding_width' is usually much
      smaller than 'hidden_size').
    embedding_layer: An optional Layer instance which will be called to generate
      embeddings for the input word IDs.
    norm_first: Whether to normalize inputs to attention and intermediate dense
      layers. If set False, output of attention and intermediate dense layers is
      normalized. This does not apply to ReZero.
    transformer_cls: str or a keras Layer. This is the base TransformerBlock the
      funnel encoder relies on.
    share_rezero: bool. Whether to share ReZero alpha between the attention
      layer and the ffn layer. This option is specific to ReZero.
    with_dense_inputs: Whether to accept dense embeddings as the input.
  """

  def __init__(
      self,
      vocab_size: int,
      hidden_size: int = 768,
      num_layers: int = 12,
      num_attention_heads: int = 12,
      max_sequence_length: int = 512,
      type_vocab_size: int = 16,
      inner_dim: int = 3072,
      inner_activation: _Activation = _approx_gelu,
      output_dropout: float = 0.1,
      attention_dropout: float = 0.1,
      pool_type: str = _MAX,
      pool_stride: Union[int, Sequence[Union[int, float]]] = 2,
      unpool_length: int = 0,
      initializer: _Initializer = tf.keras.initializers.TruncatedNormal(
          stddev=0.02
      ),
      output_range: Optional[int] = None,
      embedding_width: Optional[int] = None,
      embedding_layer: Optional[tf.keras.layers.Layer] = None,
      norm_first: bool = False,
      transformer_cls: Union[
          str, tf.keras.layers.Layer
      ] = layers.TransformerEncoderBlock,
      share_rezero: bool = False,
      **kwargs
  ):
    super().__init__(**kwargs)

    if output_range is not None:
      logging.warning('`output_range` is available as an argument for `call()`.'
                      'The `output_range` as __init__ argument is deprecated.')

    activation = tf.keras.activations.get(inner_activation)
    initializer = tf.keras.initializers.get(initializer)

    if embedding_width is None:
      embedding_width = hidden_size

    if embedding_layer is None:
      self._embedding_layer = layers.OnDeviceEmbedding(
          vocab_size=vocab_size,
          embedding_width=embedding_width,
          initializer=tf_utils.clone_initializer(initializer),
          name='word_embeddings')
    else:
      self._embedding_layer = embedding_layer

    self._position_embedding_layer = layers.PositionEmbedding(
        initializer=tf_utils.clone_initializer(initializer),
        max_length=max_sequence_length,
        name='position_embedding')

    self._type_embedding_layer = layers.OnDeviceEmbedding(
        vocab_size=type_vocab_size,
        embedding_width=embedding_width,
        initializer=tf_utils.clone_initializer(initializer),
        use_one_hot=True,
        name='type_embeddings')

    self._embedding_norm_layer = tf.keras.layers.LayerNormalization(
        name='embeddings/layer_norm', axis=-1, epsilon=1e-12, dtype=tf.float32)

    self._embedding_dropout = tf.keras.layers.Dropout(
        rate=output_dropout, name='embedding_dropout')

    # We project the 'embedding' output to 'hidden_size' if it is not already
    # 'hidden_size'.
    self._embedding_projection = None
    if embedding_width != hidden_size:
      self._embedding_projection = tf.keras.layers.EinsumDense(
          '...x,xy->...y',
          output_shape=hidden_size,
          bias_axes='y',
          kernel_initializer=tf_utils.clone_initializer(initializer),
          name='embedding_projection')

    self._transformer_layers = []
    self._attention_mask_layer = layers.SelfAttentionMask(
        name='self_attention_mask')
    # Will raise an error if the string is not supported.
    if isinstance(transformer_cls, str):
      transformer_cls = _str2transformer_cls[transformer_cls]
    self._num_layers = num_layers
    for i in range(num_layers):
      layer = transformer_cls(
          num_attention_heads=num_attention_heads,
          intermediate_size=inner_dim,
          inner_dim=inner_dim,
          intermediate_activation=inner_activation,
          inner_activation=inner_activation,
          output_dropout=output_dropout,
          attention_dropout=attention_dropout,
          norm_first=norm_first,
          kernel_initializer=tf_utils.clone_initializer(initializer),
          share_rezero=share_rezero,
          name='transformer/layer_%d' % i)
      self._transformer_layers.append(layer)

    self._pooler_layer = tf.keras.layers.Dense(
        units=hidden_size,
        activation='tanh',
        kernel_initializer=tf_utils.clone_initializer(initializer),
        name='pooler_transform')
    if isinstance(pool_stride, int):
      # TODO(b/197133196): Pooling layer can be shared.
      pool_strides = [pool_stride] * num_layers
    else:
      if len(pool_stride) != num_layers:
        raise ValueError('Lengths of pool_stride and num_layers are not equal.')
      pool_strides = pool_stride

    is_fractional_pooling = False in [
        (1.0 * pool_stride).is_integer() for pool_stride in pool_strides
    ]
    if is_fractional_pooling and pool_type in [_MAX, _AVG]:
      raise ValueError(
          'Fractional pooling is only supported for'
          ' `pool_type`=`truncated_average`'
      )

    # TODO(crickwu): explore tf.keras.layers.serialize method.
    if pool_type == _MAX:
      pool_cls = tf.keras.layers.MaxPooling1D
    elif pool_type == _AVG:
      pool_cls = tf.keras.layers.AveragePooling1D
    elif pool_type == _TRUNCATED_AVG:
      # TODO(b/203665205): unpool_length should be implemented.
      if unpool_length != 0:
        raise ValueError('unpool_length is not supported by truncated_avg now.')
    else:
      raise ValueError('pool_type not supported.')

    if pool_type in (_MAX, _AVG):
      self._att_input_pool_layers = []
      for layer_pool_stride in pool_strides:
        att_input_pool_layer = pool_cls(
            pool_size=layer_pool_stride,
            strides=layer_pool_stride,
            padding='same',
            name='att_input_pool_layer')
        self._att_input_pool_layers.append(att_input_pool_layer)

    self._max_sequence_length = max_sequence_length
    self._pool_strides = pool_strides  # This is a list here.
    self._unpool_length = unpool_length
    self._pool_type = pool_type

    self._config = {
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_attention_heads': num_attention_heads,
        'max_sequence_length': max_sequence_length,
        'type_vocab_size': type_vocab_size,
        'inner_dim': inner_dim,
        'inner_activation': tf.keras.activations.serialize(activation),
        'output_dropout': output_dropout,
        'attention_dropout': attention_dropout,
        'initializer': tf.keras.initializers.serialize(initializer),
        'output_range': output_range,
        'embedding_width': embedding_width,
        'embedding_layer': embedding_layer,
        'norm_first': norm_first,
        'pool_type': pool_type,
        'pool_stride': pool_stride,
        'unpool_length': unpool_length,
        'transformer_cls': _transformer_cls2str.get(
            transformer_cls, str(transformer_cls)
        ),
    }

    self.inputs = dict(
        input_word_ids=tf.keras.Input(shape=(None,), dtype=tf.int32),
        input_mask=tf.keras.Input(shape=(None,), dtype=tf.int32),
        input_type_ids=tf.keras.Input(shape=(None,), dtype=tf.int32))

  def call(self, inputs, output_range: Optional[tf.Tensor] = None):
    # inputs are [word_ids, mask, type_ids]
    if isinstance(inputs, (list, tuple)):
      logging.warning('List inputs to  %s are discouraged.', self.__class__)
      if len(inputs) == 3:
        word_ids, mask, type_ids = inputs
        dense_inputs = None
        dense_mask = None
        dense_type_ids = None
      elif len(inputs) == 6:
        word_ids, mask, type_ids, dense_inputs, dense_mask, dense_type_ids = (
            inputs
        )
      else:
        raise ValueError(
            'Unexpected inputs to %s with length at %d.'
            % (self.__class__, len(inputs))
        )
    elif isinstance(inputs, dict):
      word_ids = inputs.get('input_word_ids')
      mask = inputs.get('input_mask')
      type_ids = inputs.get('input_type_ids')

      dense_inputs = inputs.get('dense_inputs', None)
      dense_mask = inputs.get('dense_mask', None)
      dense_type_ids = inputs.get('dense_type_ids', None)
    else:
      raise ValueError('Unexpected inputs type to %s.' % self.__class__)

    word_embeddings = self._embedding_layer(word_ids)

    if dense_inputs is not None:
      # Concat the dense embeddings at sequence begin so unpool_len can control
      # embedding not being pooled.
      word_embeddings = tf.concat([dense_inputs, word_embeddings], axis=1)
      type_ids = tf.concat([dense_type_ids, type_ids], axis=1)
      mask = tf.concat([dense_mask, mask], axis=1)
    # absolute position embeddings
    position_embeddings = self._position_embedding_layer(word_embeddings)
    type_embeddings = self._type_embedding_layer(type_ids)

    embeddings = tf.keras.layers.add(
        [word_embeddings, position_embeddings, type_embeddings])
    embeddings = self._embedding_norm_layer(embeddings)
    embeddings = self._embedding_dropout(embeddings)

    if self._embedding_projection is not None:
      embeddings = self._embedding_projection(embeddings)

    attention_mask = self._attention_mask_layer(embeddings, mask)

    encoder_outputs = []
    x = embeddings
    # TODO(b/195972228): attention_mask can be co-generated with pooling.
    if self._pool_type in (_MAX, _AVG):
      attention_mask = _pool_and_concat(
          attention_mask,
          unpool_length=self._unpool_length,
          strides=self._pool_strides[0],
          axes=[1])

      for i, layer in enumerate(self._transformer_layers):
        # Bypass no pooling cases.
        if self._pool_strides[i] == 1:
          x = layer([x, x, attention_mask])
        else:
          # Pools layer for compressing the query length.
          pooled_inputs = self._att_input_pool_layers[i](
              x[:, self._unpool_length:, :])
          query_inputs = tf.concat(
              values=(tf.cast(
                  x[:, :self._unpool_length, :],
                  dtype=pooled_inputs.dtype), pooled_inputs),
              axis=1)
          x = layer([query_inputs, x, attention_mask],
                    output_range=output_range if i == self._num_layers -
                    1 else None)
        # Pools the corresponding attention_mask.
        if i < len(self._transformer_layers) - 1:
          attention_mask = _pool_and_concat(
              attention_mask,
              unpool_length=self._unpool_length,
              strides=[self._pool_strides[i + 1], self._pool_strides[i]],
              axes=[1, 2])
        encoder_outputs.append(x)
    elif self._pool_type == _TRUNCATED_AVG:
      # Compute the attention masks and pooling transforms.
      # Note we do not compute this in __init__ due to inference converter issue
      # b/215659399.
      pooling_transforms = _create_truncated_avg_transforms(
          self._max_sequence_length, self._pool_strides)
      attention_masks = _create_truncated_avg_masks(mask, self._pool_strides,
                                                    pooling_transforms)
      for i, layer in enumerate(self._transformer_layers):
        attention_mask = attention_masks[i]
        transformer_output_range = None
        if i == self._num_layers - 1:
          transformer_output_range = output_range
        # Bypass no pooling cases.
        if self._pool_strides[i] == 1:
          x = layer([x, x, attention_mask],
                    output_range=transformer_output_range)
        else:
          pooled_inputs = tf.einsum(
              'BFD,FT->BTD',
              tf.cast(x[:, self._unpool_length:, :], _get_policy_dtype()
                     ),  # extra casting for faster mixed computation.
              pooling_transforms[i])
          query_inputs = tf.concat(
              values=(tf.cast(
                  x[:, :self._unpool_length, :],
                  dtype=pooled_inputs.dtype), pooled_inputs),
              axis=1)
          x = layer([query_inputs, x, attention_mask],
                    output_range=transformer_output_range)
        encoder_outputs.append(x)

    last_encoder_output = encoder_outputs[-1]
    first_token_tensor = last_encoder_output[:, 0, :]
    pooled_output = self._pooler_layer(first_token_tensor)

    return dict(
        word_embeddings=word_embeddings,
        embedding_output=embeddings,
        sequence_output=encoder_outputs[-1],
        pooled_output=pooled_output,
        encoder_outputs=encoder_outputs)

  def get_embedding_table(self):
    return self._embedding_layer.embeddings

  def get_embedding_layer(self):
    return self._embedding_layer

  def get_config(self):
    return dict(self._config)

  @property
  def transformer_layers(self):
    """List of Transformer layers in the encoder."""
    return self._transformer_layers

  @property
  def pooler_layer(self):
    """The pooler dense layer after the transformer layers."""
    return self._pooler_layer

  @classmethod
  def from_config(cls, config, custom_objects=None):
    if 'embedding_layer' in config and config['embedding_layer'] is not None:
      warn_string = (
          'You are reloading a model that was saved with a '
          'potentially-shared embedding layer object. If you contine to '
          'train this model, the embedding layer will no longer be shared. '
          'To work around this, load the model outside of the Keras API.')
      print('WARNING: ' + warn_string)
      logging.warn(warn_string)

    return cls(**config)
