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

"""Layers and Model class for MaxViT."""

import functools
import string
from typing import Any, Callable, Optional, Tuple, Union

from absl import logging
import tensorflow as tf, tf_keras

from official.projects.maxvit.modeling import common_ops


class TrailDense(tf_keras.layers.Layer):
  """Dense module that projects multiple trailing dimensions."""

  def __init__(
      self,
      output_trailing_dims: Union[int, Tuple[int, ...]],
      begin_axis: int = -1,
      use_bias: bool = True,
      kernel_initializer: Optional[str] = 'glorot_uniform',
      bias_initializer: Optional[str] = 'zeros',
      name: str = 'dense',
  ):
    super().__init__(name=name)

    if isinstance(output_trailing_dims, int):
      self._output_trailing_dims = [output_trailing_dims]
    else:
      assert isinstance(output_trailing_dims, (list, tuple)) and all(
          isinstance(i, int) for i in output_trailing_dims
      ), f'Invalid output shape: {output_trailing_dims}.'
      self._output_trailing_dims = list(output_trailing_dims)
    self.begin_axis = begin_axis
    self.use_bias = use_bias

    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer

  def build(self, input_shape: tf.TensorShape) -> None:
    """Create variables and einsum expression based on input shape."""
    # Create variables
    weight_shape = input_shape[self.begin_axis :] + self._output_trailing_dims
    self.weight = self.add_weight(
        name='weight',
        shape=weight_shape,
        initializer=self.kernel_initializer,
        trainable=True,
    )
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=self._output_trailing_dims,
          initializer=self.bias_initializer,
          trainable=True,
      )

    # Create einsum expression
    input_rank = input_shape.rank
    shared_size = self.begin_axis % input_rank
    i_only_size = input_rank - shared_size
    o_only_size = len(self._output_trailing_dims)

    assert input_rank + o_only_size < len(
        string.ascii_uppercase
    ), 'Cannot use einsum as input rank + output rank > 26.'
    einsum_str = string.ascii_uppercase[: input_rank + o_only_size]

    offset = 0
    shared_str = einsum_str[offset : offset + shared_size]
    offset += shared_size
    i_only_str = einsum_str[offset : offset + i_only_size]
    offset += i_only_size
    o_only_str = einsum_str[offset : offset + o_only_size]

    input_str = f'{shared_str}{i_only_str}'
    output_str = f'{shared_str}{o_only_str}'
    weight_str = f'{i_only_str}{o_only_str}'
    # Examples
    # - For 4D tensors in conv, a common expr would be 'ABCD,DE->ABCE'.
    # - For `q/k/v` head projection in multi-head attention with two output
    #   trailing dims, the expr is 'ABC,CDE->ABDE'
    # - For `o` output projection in multi-head attention with begin_axis = -2,
    #   the expr is 'ABCD,CDE->ABE'
    self.einsum_expr = f'{input_str},{weight_str}->{output_str}'

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    output = tf.einsum(self.einsum_expr, inputs, self.weight)
    if self.use_bias:
      output += self.bias
    return output


class Attention(tf_keras.layers.Layer):
  """Multi-headed attention module."""

  def __init__(
      self,
      hidden_size: int,
      head_size: int,
      input_origin_height: int = 1,
      input_origin_width: int = 1,
      num_heads: Optional[int] = None,
      dropatt: float = 0.0,
      attn_axis: int = 0,
      rel_attn_type: Optional[str] = None,
      scale_ratio: Optional[float] = None,
      kernel_initializer: Optional[str] = 'glorot_uniform',
      bias_initializer: Optional[str] = 'zeros',
      name: str = 'attention',
  ):
    super().__init__(name=name)

    self.hidden_size = hidden_size
    self.head_size = head_size
    self.input_origin_height = input_origin_height
    self.input_origin_width = input_origin_width
    self.num_heads = num_heads or hidden_size // head_size
    self.dropatt = dropatt
    self.attn_axis = attn_axis
    self.rel_attn_type = rel_attn_type
    self.scale_ratio = scale_ratio

    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer

    self._q_proj = TrailDense(
        output_trailing_dims=(self.num_heads, self.head_size),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='q',
    )
    self._k_proj = TrailDense(
        output_trailing_dims=(self.num_heads, self.head_size),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='k',
    )
    self._v_proj = TrailDense(
        output_trailing_dims=(self.num_heads, self.head_size),
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='v',
    )
    self._o_proj = TrailDense(
        output_trailing_dims=self.hidden_size,
        begin_axis=-2,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='o',
    )

    self.q_scale = self.head_size**-0.5
    self.relative_bias = None

  def build(self, query_shape: Any) -> None:
    ##### Content attention
    # Einsum expression:
    #   B = batch_size
    #   N = num_heads
    #   K = head_size
    #   S = query_len (of the given attn_axis)
    #   T = key/value_len (of the given attn_axis)
    #   [U-Z] = length of other attension axes
    # Example for 5D query_heads, (e.g. images [B x H x W x N x K])
    # - when attn_axis = 0 (H axis):
    #     symbols = 'U'  => num_attn_dims = 2
    #     q_expr = 'BSUNK' => 'S' is inserted, prefix = 'B', suffix = 'NK'
    #     k_expr = 'BTUNK' => 'T' is inserted, prefix = 'B', suffix = 'NK'
    #     v_expr = 'BTUNK' => 'T' is inserted, prefix = 'B', suffix = 'NK'
    #     a_expr = 'BUNST' => 'N x S x T' attention map
    num_attn_dims = query_shape.rank - 2  # -2 to account for bsz, hidden size
    assert num_attn_dims < 6, 'Only support at most 6 attention dims.'
    symbols = ''.join([chr(ord('U') + i) for i in range(num_attn_dims - 1)])
    insert = lambda s, i, c: s[:i] + c + s[i:]
    create_expr = lambda s, prefix='B', suffix='NK': prefix + s + suffix
    self.q_expr = create_expr(insert(symbols, self.attn_axis, 'S'))
    self.k_expr = create_expr(insert(symbols, self.attn_axis, 'T'))
    self.v_expr = create_expr(insert(symbols, self.attn_axis, 'T'))
    self.a_expr = create_expr(symbols, suffix='NST')

    ##### Relative attention
    if self.rel_attn_type in ['2d_multi_head', '2d_single_head']:
      query_shape_list = query_shape.as_list()
      if query_shape.rank == 4:
        height, width = query_shape_list[1:3]
      elif query_shape.rank == 3:
        seq_len = query_shape_list[1]
        height, width = common_ops.get_shape_from_length(
            seq_len, self.input_origin_height, self.input_origin_width
        )
        if height * width != seq_len:
          raise ValueError(
              'Sequence length: %s violates input size: (%s, %s).'
              % (seq_len, height, width)
          )
      else:
        raise ValueError(
            'Does not support relative attention for query shape: %s.'
            % query_shape_list
        )

      if self.scale_ratio is not None:
        scale_ratio = eval(self.scale_ratio)  # pylint:disable=eval-used
        vocab_height = 2 * int(height / scale_ratio) - 1
        vocab_width = 2 * int(width / scale_ratio) - 1
      else:
        vocab_height = 2 * height - 1
        vocab_width = 2 * width - 1

      if self.rel_attn_type == '2d_multi_head':
        rel_bias_shape = [self.num_heads, vocab_height, vocab_width]
      elif self.rel_attn_type == '2d_single_head':
        rel_bias_shape = [vocab_height, vocab_width]
      else:
        raise NotImplementedError(
            f'rel_attn_type {self.rel_attn_type} not implemented yet.'
        )

      self._feat_height = height
      self._feat_width = width
      self.relative_bias = self.add_weight(
          'relative_bias',
          rel_bias_shape,
          initializer=self.kernel_initializer,
          trainable=True,
      )

  def call(
      self,
      query: tf.Tensor,
      training: bool,
      context: Optional[tf.Tensor] = None,
      attn_mask: Optional[tf.Tensor] = None,
  ) -> tf.Tensor:
    if context is None:
      context = query

    q_heads = self._q_proj(query)
    k_heads = self._k_proj(context)
    v_heads = self._v_proj(context)
    q_heads *= self.q_scale

    # attention
    attn_logits = tf.einsum(
        f'{self.q_expr},{self.k_expr}->{self.a_expr}', q_heads, k_heads
    )

    if self.relative_bias is not None:
      if self.rel_attn_type == '2d_multi_head':
        h_axis = 1
      else:
        h_axis = 0

      if self.scale_ratio is not None:
        src_shape = self.relative_bias.shape.as_list()
        relative_bias = tf.expand_dims(self.relative_bias, axis=-1)
        relative_bias = tf.image.resize(
            relative_bias, [2 * self._feat_height - 1, 2 * self._feat_width - 1]
        )
        relative_bias = tf.cast(
            tf.squeeze(relative_bias, axis=-1), self.compute_dtype
        )
        tgt_shape = relative_bias.shape.as_list()
        logging.info(
            'Bilinear resize relative position bias %s -> %s.',
            src_shape,
            tgt_shape,
        )
      else:
        relative_bias = tf.cast(self.relative_bias, self.compute_dtype)

      reindexed_bias = common_ops.reindex_2d_einsum_lookup(
          relative_position_tensor=relative_bias,
          height=self._feat_height,
          width=self._feat_width,
          max_relative_height=self._feat_height - 1,
          max_relative_width=self._feat_width - 1,
          h_axis=h_axis,
      )
      attn_logits += reindexed_bias

    if attn_mask is not None:
      # attn_mask: 1.0 means CAN attend, 0.0 means CANNOT attend
      attn_logits += (1.0 - attn_mask) * attn_logits.dtype.min

    attn_probs = common_ops.float32_softmax(attn_logits, axis=-1)
    if self.dropatt:
      attn_probs = tf_keras.layers.Dropout(self.dropatt, name='attn_prob_drop')(
          attn_probs, training=training
      )

    attn_out = tf.einsum(
        f'{self.a_expr},{self.v_expr}->{self.q_expr}', attn_probs, v_heads
    )
    output = self._o_proj(attn_out)

    return output


class FFN(tf_keras.layers.Layer):
  """Positionwise feed-forward network."""

  def __init__(
      self,
      hidden_size: int,
      dropout: float = 0.0,
      expansion_rate: int = 4,
      activation: str = 'gelu',
      kernel_initializer: Optional[str] = 'glorot_uniform',
      bias_initializer: Optional[str] = 'zeros',
      name: str = 'ffn',
  ):
    super().__init__(name=name)

    self.hidden_size = hidden_size
    self.expansion_rate = expansion_rate
    self.expanded_size = self.hidden_size * self.expansion_rate
    self.dropout = dropout
    self.activation = activation

    self._expand_dense = TrailDense(
        output_trailing_dims=self.expanded_size,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='expand_dense',
    )
    self._shrink_dense = TrailDense(
        output_trailing_dims=self.hidden_size,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='shrink_dense',
    )
    self._activation_fn = common_ops.get_act_fn(self.activation)

  def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
    output = inputs
    output = self._expand_dense(output)
    output = self._activation_fn(output)
    if self.dropout:
      output = tf_keras.layers.Dropout(self.dropout, name='nonlinearity_drop')(
          output, training=training
      )
    output = self._shrink_dense(output)

    return output


class TransformerBlock(tf_keras.layers.Layer):
  """Transformer block = Attention + FFN."""

  def __init__(
      self,
      hidden_size: int,
      head_size: int,
      input_origin_height: int = 1,
      input_origin_width: int = 1,
      num_heads: Optional[int] = None,
      expansion_rate: int = 4,
      activation: str = 'gelu',
      pool_type: str = '2d:avg',
      pool_stride: int = 1,
      pool_query_only: bool = False,
      dropatt: Optional[Union[float, tf.Tensor]] = None,
      dropout: Optional[Union[float, tf.Tensor]] = None,
      rel_attn_type: Optional[str] = None,
      scale_ratio: Optional[str] = None,
      survival_prob: Optional[Union[float, tf.Tensor]] = None,
      ln_epsilon: float = 1e-5,
      ln_dtype: Optional[tf.DType] = None,
      kernel_initializer: Optional[str] = 'glorot_uniform',
      bias_initializer: Optional[str] = 'zeros',
      name: str = 'transformer',
  ) -> None:
    super().__init__(name=name)

    self._hidden_size = hidden_size
    self._head_size = head_size
    self._input_origin_height = input_origin_height
    self._input_origin_width = input_origin_width
    self._num_heads = num_heads
    self._expansion_rate = expansion_rate
    self._activation = activation
    self._pool_type = pool_type
    self._pool_stride = pool_stride
    self._pool_query_only = pool_query_only
    self._dropatt = dropatt
    self._dropout = dropout
    self._rel_attn_type = rel_attn_type
    self._scale_ratio = scale_ratio
    self._survival_prob = survival_prob
    self._ln_epsilon = ln_epsilon
    self._ln_dtype = ln_dtype
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  def build(self, input_shape: tf.TensorShape) -> None:
    if len(input_shape.as_list()) == 4:
      _, height, width, _ = input_shape.as_list()
    elif len(input_shape.as_list()) == 3:
      _, seq_len, _ = input_shape.as_list()
      height, width = common_ops.get_shape_from_length(
          seq_len, self._input_origin_height, self._input_origin_width
      )
    else:
      raise ValueError(f'Unsupported input shape: {input_shape.as_list()}.')

    self.height, self.width = height, width
    input_size = input_shape.as_list()[-1]

    if input_size != self._hidden_size:
      self._shortcut_proj = TrailDense(
          self._hidden_size,
          kernel_initializer=self._kernel_initializer,
          bias_initializer=self._bias_initializer,
          name='shortcut_proj',
      )
    else:
      self._shortcut_proj = None

    self._attn_layer_norm = tf_keras.layers.LayerNormalization(
        axis=-1,
        epsilon=self._ln_epsilon,
        dtype=self._ln_dtype,
        name='attn_layer_norm',
    )

    self._attention = Attention(
        self._hidden_size,
        self._head_size,
        height // self._pool_stride,
        width // self._pool_stride,
        num_heads=self._num_heads,
        dropatt=self._dropatt,
        rel_attn_type=self._rel_attn_type,
        scale_ratio=self._scale_ratio,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
    )

    self._ffn_layer_norm = tf_keras.layers.LayerNormalization(
        axis=-1,
        epsilon=self._ln_epsilon,
        dtype=self._ln_dtype,
        name='ffn_layer_norm',
    )

    self._ffn = FFN(
        self._hidden_size,
        dropout=self._dropout,
        expansion_rate=self._expansion_rate,
        activation=self._activation,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
    )

  def downsample(self, inputs: tf.Tensor, name: str) -> tf.Tensor:
    output = inputs
    if self._pool_stride > 1:
      assert self._pool_type in [
          '2d:avg',
          '2d:max',
          '1d:avg',
          '1d:max',
      ], f'Invalid pool_type {self._pool_type}'
      if self._pool_type.startswith('2d'):
        output = common_ops.maybe_reshape_to_2d(output, height=self.height)
        output = common_ops.pooling_2d(
            output,
            self._pool_type.split(':')[-1],
            self._pool_stride,
            padding='same',
            data_format='channels_last',
            name=name,
        )
      else:
        output = common_ops.pooling_1d(
            output,
            self._pool_type.split(':')[-1],
            self._pool_stride,
            padding='same',
            data_format='channels_last',
            name=name,
        )
    return output

  def shortcut_branch(self, shortcut: tf.Tensor) -> tf.Tensor:
    shortcut = self.downsample(shortcut, 'shortcut_pool')
    shortcut = common_ops.maybe_reshape_to_1d(shortcut)
    if self._shortcut_proj:
      shortcut = self._shortcut_proj(shortcut)

    return shortcut

  def attn_branch(
      self,
      inputs: tf.Tensor,
      training: bool,
      attn_mask: Optional[tf.Tensor] = None,
  ) -> tf.Tensor:
    output = self._attn_layer_norm(inputs)
    if self._pool_query_only:
      query = self.downsample(output, 'query_pool')
      query = common_ops.maybe_reshape_to_1d(query)
      output = common_ops.maybe_reshape_to_1d(output)
      output = self._attention(
          query, training, context=output, attn_mask=attn_mask
      )
    else:
      output = self.downsample(output, 'residual_pool')
      output = common_ops.maybe_reshape_to_1d(output)
      output = self._attention(output, training, attn_mask=attn_mask)
    return output

  def ffn_branch(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
    output = self._ffn_layer_norm(inputs)
    output = self._ffn(output, training)
    return output

  def call(
      self,
      inputs: tf.Tensor,
      training: bool,
      attn_mask: Optional[tf.Tensor] = None,
  ) -> tf.Tensor:
    logging.info(
        'Block %s input shape: %s, (%s).', self.name, inputs.shape, inputs.dtype
    )

    shortcut = self.shortcut_branch(inputs)
    output = self.attn_branch(inputs, training, attn_mask)
    if self._dropout:
      output = tf_keras.layers.Dropout(self._dropout, name='after_attn_drop')(
          output, training=training
      )
    output = common_ops.residual_add(
        output, shortcut, self._survival_prob, training
    )

    shortcut = output
    output = self.ffn_branch(output, training)
    if self._dropout:
      output = tf_keras.layers.Dropout(self._dropout, name='after_ffn_drop')(
          output, training=training
      )
    output = common_ops.residual_add(
        output, shortcut, self._survival_prob, training
    )

    return output


class SqueezeAndExcitation(tf_keras.layers.Layer):
  """Squeeze-and-excitation layer."""

  def __init__(
      self,
      se_filters: int,
      output_filters: int,
      local_pooling: bool = False,
      data_format: str = 'channels_last',
      activation: str = 'swish',
      kernel_initializer: Optional[str] = 'glorot_uniform',
      bias_initializer: Optional[str] = 'zeros',
      name: str = 'se',
  ):
    super().__init__(name=name)

    self._local_pooling = local_pooling
    self._data_format = data_format
    self._activation_fn = common_ops.get_act_fn(activation)

    # Squeeze and Excitation layer.
    self._se_reduce = tf_keras.layers.Conv2D(
        se_filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        padding='same',
        data_format=self._data_format,
        use_bias=True,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='reduce_conv2d',
    )
    self._se_expand = tf_keras.layers.Conv2D(
        output_filters,
        kernel_size=[1, 1],
        strides=[1, 1],
        padding='same',
        data_format=self._data_format,
        use_bias=True,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        name='expand_conv2d',
    )

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    h_axis, w_axis = [2, 3] if self._data_format == 'channels_first' else [1, 2]
    if self._local_pooling:
      se_tensor = tf.nn.avg_pool(
          inputs,
          ksize=[1, inputs.shape[h_axis], inputs.shape[w_axis], 1],
          strides=[1, 1, 1, 1],
          padding='VALID',
      )
    else:
      se_tensor = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    se_tensor = self._se_expand(self._activation_fn(self._se_reduce(se_tensor)))
    return tf.sigmoid(se_tensor) * inputs


def _config_batch_norm(
    norm_type: str,
    ln_epsilon: float = 1e-6,
    bn_momentum: float = 0.99,
    bn_epsilon: float = 1e-6,
) -> Callable[..., Any]:
  """Defines the normalization class for MbConv based on `norm_type`."""

  if norm_type == 'layer_norm':
    return functools.partial(
        tf_keras.layers.LayerNormalization, epsilon=ln_epsilon
    )
  elif norm_type == 'batch_norm':
    return functools.partial(
        tf_keras.layers.BatchNormalization,
        momentum=bn_momentum,
        epsilon=bn_epsilon,
    )
  elif norm_type == 'sync_batch_norm':
    return functools.partial(
        tf_keras.layers.BatchNormalization,
        momentum=bn_momentum,
        epsilon=bn_epsilon,
        synchronized=True,
    )
  else:
    raise ValueError(f'Unsupported norm_type {norm_type}.')


def _build_downsample_layer(
    pool_type: str, pool_stride: int, data_format: str = 'channels_last'
) -> tf_keras.layers.Layer:
  """Builds a downsample layer for MbConv based on pool type."""
  if pool_type == 'max':
    return tf_keras.layers.MaxPooling2D(
        pool_size=(pool_stride, pool_stride),
        strides=(pool_stride, pool_stride),
        padding='same',
        data_format=data_format,
    )
  elif pool_type == 'avg':
    return tf_keras.layers.AveragePooling2D(
        pool_size=(pool_stride, pool_stride),
        strides=(pool_stride, pool_stride),
        padding='same',
        data_format=data_format,
    )
  else:
    raise ValueError(f'Unsurpported pool_type {pool_type}')


class MBConvBlock(tf_keras.layers.Layer):
  """Mobile Inverted Residual Bottleneck (https://arxiv.org/abs/1905.02244)."""

  def __init__(
      self,
      hidden_size: int,
      downsample_loc: str = 'depth_conv',
      data_format: str = 'channels_last',
      kernel_size: int = 3,
      expansion_rate: int = 4,
      se_ratio: float = 0.25,
      activation: str = 'gelu',
      pool_type: str = 'avg',
      pool_stride: int = 1,
      dropcnn: Optional[float] = None,
      survival_prob: Optional[float] = None,
      norm_type: str = 'sync_batch_norm',
      bn_epsilon: float = 1e-3,
      bn_momentum: float = 0.99,
      kernel_initializer: Optional[str] = 'glorot_uniform',
      bias_initializer: Optional[str] = 'zeros',
      name: str = 'mbconv',
  ):
    super().__init__(name=name)

    self._hidden_size = hidden_size
    self._downsample_loc = downsample_loc
    self._data_format = data_format
    self._kernel_size = kernel_size
    self._expansion_rate = expansion_rate
    self._se_ratio = se_ratio
    self._activation = activation
    self._pool_type = pool_type
    self._pool_stride = pool_stride
    self._dropcnn = dropcnn
    self._survival_prob = survival_prob
    self._norm_type = norm_type
    self._bn_epsilon = bn_epsilon
    self._bn_momentum = bn_momentum
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._pool_layer = _build_downsample_layer(
        pool_type, pool_stride, data_format)
    self._activation_fn = common_ops.get_act_fn(self._activation)

  def build(self, input_shape: tf.TensorShape) -> None:
    """Builds block according to the arguments."""

    channel_axis = 3 if self._data_format == 'channels_last' else 1
    input_size = input_shape[channel_axis]
    inner_size = self._hidden_size * self._expansion_rate

    norm_cls = _config_batch_norm(
        self._norm_type,
        bn_momentum=self._bn_momentum,
        bn_epsilon=self._bn_epsilon,
    )

    # Shortcut projection.
    if input_size != self._hidden_size:
      self._shortcut_conv = tf_keras.layers.Conv2D(
          filters=self._hidden_size,
          kernel_size=1,
          strides=1,
          padding='same',
          data_format=self._data_format,
          kernel_initializer=self._kernel_initializer,
          bias_initializer=self._bias_initializer,
          use_bias=True,
          name='shortcut_conv',
      )
    else:
      self._shortcut_conv = None

    # Pre-Activation norm
    self._pre_norm = norm_cls(name='pre_norm')

    # Expansion phase. Called if not using fused convolutions and expansion
    # phase is necessary.
    if self._expansion_rate != 1:
      self._expand_conv = tf_keras.layers.Conv2D(
          filters=inner_size,
          kernel_size=1,
          strides=(
              self._pool_stride if self._downsample_loc == 'expand_conv' else 1
          ),
          kernel_initializer=self._kernel_initializer,
          padding='same',
          data_format=self._data_format,
          use_bias=False,
          name='expand_conv',
      )
      self._expand_norm = norm_cls(name='expand_norm')

    # Depth-wise convolution phase. Called if not using fused convolutions.
    self._depthwise_conv = tf_keras.layers.DepthwiseConv2D(
        kernel_size=self._kernel_size,
        strides=(
            self._pool_stride if self._downsample_loc == 'depth_conv' else 1
        ),
        depthwise_initializer=self._kernel_initializer,
        padding='same',
        data_format=self._data_format,
        use_bias=False,
        name='depthwise_conv',
    )
    self._depthwise_norm = norm_cls(name='depthwise_norm')

    if self._se_ratio is not None and 0 < self._se_ratio <= 1:
      se_filters = max(1, int(self._hidden_size * self._se_ratio))
      self._se = SqueezeAndExcitation(
          se_filters=se_filters,
          output_filters=inner_size,
          data_format=self._data_format,
          kernel_initializer=self._kernel_initializer,
          bias_initializer=self._bias_initializer,
          name='se',
      )
    else:
      self._se = None

    # Output phase.
    self._shrink_conv = tf_keras.layers.Conv2D(
        filters=self._hidden_size,
        kernel_size=1,
        strides=1,
        padding='same',
        data_format=self._data_format,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        use_bias=True,
        name='shrink_conv',
    )

  def downsample(self, inputs: tf.Tensor, name: str) -> tf.Tensor:
    output = inputs
    if self._pool_stride > 1:
      output = self._pool_layer(output)
    return output

  def shortcut_branch(self, shortcut: tf.Tensor) -> tf.Tensor:
    shortcut = self.downsample(shortcut, name='shortcut_pool')
    if self._shortcut_conv:
      shortcut = self._shortcut_conv(shortcut)

    return shortcut

  def residual_branch(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
    output = self._pre_norm(inputs, training=training)
    if self._downsample_loc == 'inputs':
      output = self.downsample(output, name='residual_pool')
    if self._expansion_rate != 1:
      output = self._expand_conv(output)
      output = self._expand_norm(output, training=training)
      output = self._activation_fn(output)
      logging.debug('Expand shape: %s', output.shape)

    output = self._depthwise_conv(output)
    output = self._depthwise_norm(output, training=training)
    output = self._activation_fn(output)
    logging.debug('DConv shape: %s', output.shape)

    if self._dropcnn:
      output = tf_keras.layers.Dropout(self._dropcnn, 'after_dconv_drop')(
          output, training=training
      )

    if self._se:
      output = self._se(output)
    self.endpoints = {'expansion_output': output}

    output = self._shrink_conv(output)
    logging.debug('Shrink shape: %s', output.shape)

    return output

  def call(
      self,
      inputs: tf.Tensor,
      training: bool,
      survival_prob: Optional[Union[float, tf.Tensor]] = None,
  ) -> tf.Tensor:
    """Implementation of call().

    Args:
      inputs: the inputs tensor.
      training: boolean, whether the model is constructed for training.
      survival_prob: float, between 0 to 1, drop connect rate.

    Returns:
      A output tensor.
    """
    logging.debug(
        'Block %s input shape: %s (%s)', self.name, inputs.shape, inputs.dtype
    )

    residual = self.residual_branch(inputs, training)
    shortcut = self.shortcut_branch(inputs)
    survival_prob = survival_prob or self._survival_prob
    output = common_ops.residual_add(
        residual, shortcut, survival_prob, training
    )

    return output
