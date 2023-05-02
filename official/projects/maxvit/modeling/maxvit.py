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

# pylint: disable=logging-fstring-interpolation
r"""MaxViT layers and model class."""

import functools
from typing import Any, Mapping, Optional, Tuple, Union

from absl import logging
import tensorflow as tf

from official.projects.maxvit.modeling import common_ops as ops
from official.projects.maxvit.modeling import layers
from official.vision.modeling.backbones import factory


MAXVIT_SPECS = {
    'maxvit-tiny-for-test': dict(
        survival_prob=None,
        stem_hsize=(8, 8),
        block_type=('maxvit', 'maxvit', 'maxvit', 'maxvit'),
        num_blocks=(2, 3, 3, 2),
        hidden_size=(32, 32, 32, 768),
    ),
    'maxvit-tiny': dict(
        survival_prob=0.8,
        stem_hsize=(64, 64),
        block_type=('maxvit', 'maxvit', 'maxvit', 'maxvit'),
        num_blocks=(2, 2, 5, 2),
        hidden_size=(64, 128, 256, 512),
    ),
    'maxvit-small': dict(
        survival_prob=0.7,
        stem_hsize=(64, 64),
        block_type=('maxvit', 'maxvit', 'maxvit', 'maxvit'),
        num_blocks=(2, 2, 5, 2),
        hidden_size=(96, 192, 384, 768),
    ),
    'maxvit-base': dict(
        survival_prob=0.6,
        stem_hsize=(64, 64),
        block_type=('maxvit', 'maxvit', 'maxvit', 'maxvit'),
        num_blocks=(2, 6, 14, 2),
        hidden_size=(96, 192, 384, 768),
    ),
    'maxvit-large': dict(
        survival_prob=0.4,
        stem_hsize=(128, 128),
        block_type=('maxvit', 'maxvit', 'maxvit', 'maxvit'),
        num_blocks=(2, 6, 14, 2),
        hidden_size=(128, 256, 512, 1024),
    ),
    'maxvit-xlarge': dict(
        survival_prob=0.3,
        stem_hsize=(192, 192),
        block_type=('maxvit', 'maxvit', 'maxvit', 'maxvit'),
        num_blocks=(2, 6, 14, 2),
        hidden_size=(192, 384, 768, 1536),
    ),
}


class MaxViTBlock(tf.keras.layers.Layer):
  """MaxViT block = MBConv + Block-Attention + FFN + Grid-Attention + FFN."""

  def __init__(
      self,
      hidden_size: int,
      head_size: int,
      window_size: int,
      grid_size: int,
      num_heads: Optional[int] = None,
      downsample_loc: str = 'depth_conv',
      data_format: str = 'channels_last',
      kernel_size: int = 3,
      expansion_rate: int = 4,
      se_ratio: float = 0.25,
      activation: str = 'gelu',
      pool_type: str = '2d:avg',
      pool_stride: int = 1,
      dropcnn: Optional[float] = None,
      dropatt: Optional[Union[float, tf.Tensor]] = None,
      dropout: Optional[Union[float, tf.Tensor]] = None,
      rel_attn_type: Optional[str] = None,
      scale_ratio: Optional[str] = None,
      survival_prob: Optional[Union[float, tf.Tensor]] = None,
      ln_epsilon: float = 1e-5,
      ln_dtype: Optional[tf.DType] = None,
      norm_type: str = 'sync_batch_norm',
      bn_epsilon: float = 1e-3,
      bn_momentum: float = 0.99,
      kernel_initializer: Optional[str] = 'glorot_uniform',
      bias_initializer: Optional[str] = 'zeros',
      name: str = 'maxvit_block',
  ) -> None:
    super().__init__(name=name)

    self._hidden_size = hidden_size
    self._head_size = head_size
    self._window_size = window_size
    self._grid_size = grid_size
    self._num_heads = num_heads
    self._downsample_loc = downsample_loc
    self._data_format = data_format
    self._kernel_size = kernel_size
    self._expansion_rate = expansion_rate
    self._se_ratio = se_ratio
    self._dropcnn = dropcnn
    self._activation = activation
    self._norm_type = norm_type
    self._bn_epsilon = bn_epsilon
    self._bn_momentum = bn_momentum
    self._pool_type = pool_type
    self._pool_stride = pool_stride
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
    input_size = input_shape.as_list()[-1]

    if input_size != self._hidden_size:
      self._shortcut_proj = layers.TrailDense(
          self._hidden_size,
          kernel_initializer=self._kernel_initializer,
          bias_initializer=self._bias_initializer,
          name='shortcut_proj',
      )
    else:
      self._shortcut_proj = None

    self._block_attn_layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1,
        epsilon=self._ln_epsilon,
        dtype=self._ln_dtype,
        name='attn_layer_norm',
    )

    self._grid_attn_layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1,
        epsilon=self._ln_epsilon,
        dtype=self._ln_dtype,
        name='attn_layer_norm_1',
    )

    self._block_attention = layers.Attention(
        self._hidden_size,
        self._head_size,
        num_heads=self._num_heads,
        dropatt=self._dropatt,
        rel_attn_type=self._rel_attn_type,
        scale_ratio=self._scale_ratio,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        name='attention',
    )

    self._grid_attention = layers.Attention(
        self._hidden_size,
        self._head_size,
        num_heads=self._num_heads,
        dropatt=self._dropatt,
        rel_attn_type=self._rel_attn_type,
        scale_ratio=self._scale_ratio,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        name='attention_1',
    )

    self._block_ffn_layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1,
        epsilon=self._ln_epsilon,
        dtype=self._ln_dtype,
        name='ffn_layer_norm',
    )

    self._grid_ffn_layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1,
        epsilon=self._ln_epsilon,
        dtype=self._ln_dtype,
        name='ffn_layer_norm_1',
    )

    self._block_ffn = layers.FFN(
        self._hidden_size,
        dropout=self._dropout,
        expansion_rate=self._expansion_rate,
        activation=self._activation,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        name='ffn',
    )

    self._grid_ffn = layers.FFN(
        self._hidden_size,
        dropout=self._dropout,
        expansion_rate=self._expansion_rate,
        activation=self._activation,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        name='ffn_1',
    )

    self._mbconv = layers.MBConvBlock(
        self._hidden_size,
        downsample_loc=self._downsample_loc,
        data_format=self._data_format,
        kernel_size=self._kernel_size,
        expansion_rate=self._expansion_rate,
        se_ratio=self._se_ratio,
        activation=self._activation,
        pool_type='avg' if self._pool_type == '2d:avg' else 'max',
        pool_stride=self._pool_stride,
        dropcnn=self._dropcnn,
        survival_prob=self._survival_prob,
        norm_type=self._norm_type,
        bn_epsilon=self._bn_epsilon,
        bn_momentum=self._bn_momentum,
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        name='mbconv',
    )

  def downsample(self, inputs, name):
    output = inputs
    if self._pool_stride > 1:
      output = ops.maybe_reshape_to_2d(output)
      output = ops.pooling_2d(
          output,
          self._pool_type,
          self._pool_stride,
          padding='same',
          data_format='channels_last',
          name=name,
      )
    return output

  def window_partition(self, features: tf.Tensor) -> tf.Tensor:
    """Partition the input feature maps into non-overlapping windows.

    Note that unsuitable feature or window sizes may be costly on TPU due to
    padding sizes:
    https://docs.google.com/document/d/1GojE1Q7hR2qyi0mIfnTHgERfl7Dmsj6xPQ31MQo3xUk/edit#

    Args:
      features: [B, H, W, C] feature maps.

    Returns:
      Partitioned features: [B, nH, nW, wSize, wSize, c].

    Raises:
      ValueError: If the feature map sizes are not divisible by window sizes.
    """

    _, h, w, c = features.shape
    window_size = self._window_size

    if h % window_size != 0 or w % window_size != 0:
      raise ValueError(
          f'Feature map sizes {(h, w)} '
          f'not divisible by window size ({window_size}).'
      )

    features = tf.reshape(
        features,
        (-1, h // window_size, window_size, w // window_size, window_size, c),
    )
    features = tf.transpose(features, (0, 1, 3, 2, 4, 5))
    features = tf.reshape(features, (-1, window_size, window_size, c))
    return features

  def window_stitch_back(
      self, features: tf.Tensor, window_size: int, h: int, w: int
  ) -> tf.Tensor:
    """Reverse window_partition."""
    features = tf.reshape(
        features,
        [
            -1,
            h // window_size,
            w // window_size,
            window_size,
            window_size,
            features.shape[-1],
        ],
    )
    return tf.reshape(
        tf.transpose(features, (0, 1, 3, 2, 4, 5)),
        [-1, h, w, features.shape[-1]],
    )

  def grid_partition(self, features: tf.Tensor) -> tf.Tensor:
    """Partition the input feature maps into non-overlapping windows.

    Note that unsuitable feature or window sizes may be costly on TPU due to
    padding sizes:
    https://docs.google.com/document/d/1GojE1Q7hR2qyi0mIfnTHgERfl7Dmsj6xPQ31MQo3xUk/edit#

    Args:
      features: [B, H, W, C] feature maps.

    Returns:
      Partitioned features: [B, nH, nW, wSize, wSize, c].

    Raises:
      ValueError: If the feature map sizes are not divisible by window sizes.
    """
    _, h, w, c = features.shape
    grid_size = self._grid_size
    if h % grid_size != 0 or w % grid_size != 0:
      raise ValueError(
          f'Feature map sizes {(h, w)} '
          f'not divisible by window size ({grid_size}).'
      )
    features = tf.reshape(
        features, (-1, grid_size, h // grid_size, grid_size, w // grid_size, c)
    )
    features = tf.transpose(features, (0, 2, 4, 1, 3, 5))
    features = tf.reshape(features, (-1, grid_size, grid_size, c))
    return features

  def grid_stitch_back(
      self, features: tf.Tensor, grid_size: int, h: int, w: int
  ) -> tf.Tensor:
    """Reverse window_partition."""
    features = tf.reshape(
        features,
        [
            -1,
            h // grid_size,
            w // grid_size,
            grid_size,
            grid_size,
            features.shape[-1],
        ],
    )
    return tf.reshape(
        tf.transpose(features, (0, 3, 1, 4, 2, 5)),
        [-1, h, w, features.shape[-1]],
    )

  def block_attn_branch(
      self, inputs: tf.Tensor, training: bool, attn_mask: tf.Tensor
  ) -> tf.Tensor:
    output = self._block_attn_layer_norm(inputs)
    # If put grid-attention in front, we don't need to downsample.
    # Apply local block-attention
    _, h, w, _ = output.shape
    output = self.window_partition(output)
    output = ops.maybe_reshape_to_1d(output)
    output = self._block_attention(output, training, attn_mask=attn_mask)
    output = self.window_stitch_back(output, self._window_size, h, w)
    return output

  def grid_attn_branch(
      self, inputs: tf.Tensor, training: bool, attn_mask: tf.Tensor
  ) -> tf.Tensor:
    output = self._grid_attn_layer_norm(inputs)
    # output = self.downsample(output, 'residual_pool')
    # Apply global grid
    _, h, w, _ = output.shape
    output = self.grid_partition(output)
    output = ops.maybe_reshape_to_1d(output)
    output = self._grid_attention(output, training, attn_mask=attn_mask)
    output = self.grid_stitch_back(output, self._grid_size, h, w)
    return output

  def block_ffn_branch(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
    output = self._block_ffn_layer_norm(inputs)
    output = self._block_ffn(output, training)
    return output

  def grid_ffn_branch(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
    output = self._grid_ffn_layer_norm(inputs)
    output = self._grid_ffn(output, training)
    return output

  def mbconv_branch(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
    output = self._mbconv(inputs, training=training)
    return output

  def call(
      self,
      inputs: tf.Tensor,
      training: bool,
      attn_mask: Optional[tf.Tensor] = None,
  ) -> tf.Tensor:
    logging.debug(
        'Block %s input shape: %s (%s)', self.name, inputs.shape, inputs.dtype
    )

    # MBConv
    output = self.mbconv_branch(inputs, training)

    # block self-attention
    shortcut = output
    output = self.block_attn_branch(output, training, attn_mask)
    if self._dropout:
      output = tf.keras.layers.Dropout(
          self._dropout, name='after_block_attn_drop'
      )(output, training=training)
    output = ops.residual_add(output, shortcut, self._survival_prob, training)

    shortcut = output
    output = self.block_ffn_branch(output, training)
    if self._dropout:
      output = tf.keras.layers.Dropout(
          self._dropout, name='after_block_ffn_drop_1'
      )(output, training=training)
    output = ops.residual_add(output, shortcut, self._survival_prob, training)

    # grid self-attention
    shortcut = output
    output = self.grid_attn_branch(output, training, attn_mask)
    if self._dropout:
      output = tf.keras.layers.Dropout(
          self._dropout, name='after_grid_attn_drop'
      )(output, training=training)
    output = ops.residual_add(output, shortcut, self._survival_prob, training)

    shortcut = output
    output = self.grid_ffn_branch(output, training)
    if self._dropout:
      output = tf.keras.layers.Dropout(
          self._dropout, name='after_grid_ffn_drop'
      )(output, training=training)
    output = ops.residual_add(output, shortcut, self._survival_prob, training)

    return output


class MaxViT(tf.keras.Model):
  """MaxViT's backbone that outputs the pre-global-pooled features."""

  def __init__(
      self,
      block_type: Tuple[str, ...],
      num_blocks: Tuple[int, ...],
      hidden_size: Tuple[int, ...],
      stem_hsize: Tuple[int, ...],
      head_size: int = 32,
      num_heads: Optional[int] = None,
      dropatt: Optional[float] = None,
      dropout: Optional[float] = None,
      rel_attn_type: str = '2d_multi_head',
      window_size: int = 7,
      grid_size: int = 7,
      scale_ratio: Optional[str] = None,
      ln_epsilon: float = 1e-5,
      ln_dtype: Optional[tf.DType] = None,
      downsample_loc: str = 'depth_conv',
      kernel_size: int = 3,
      se_ratio: float = 0.25,
      dropcnn: Optional[float] = None,
      data_format: str = 'channels_last',
      norm_type: str = 'sync_batch_norm',
      bn_epsilon: float = 1e-3,
      bn_momentum: float = 0.99,
      add_pos_enc: bool = False,
      pool_type: str = '2d:avg',
      pool_stride: int = 2,
      expansion_rate: int = 4,
      activation: str = 'gelu',
      survival_prob: Optional[float] = None,
      survival_prob_anneal: bool = True,
      representation_size: Optional[int] = None,
      add_gap_layer_norm: bool = False,
      kernel_initializer: Optional[str] = 'glorot_uniform',
      bias_initializer: Optional[str] = 'zeros',
      name: str = 'maxvit',
      **kwargs,
  ):
    """Initializes MaxViT backbone.

    Args:
      block_type: a tuple of `str`, specify each block type.
      num_blocks: a tuple of `int`, specify the number of blocks in each stage.
      hidden_size: a tuple of `int`, specify hidden size of block in each stage.
      stem_hsize: a tuple of `int`, specify the hidden size of stem network.
      head_size: embedding size of each attention head.
      num_heads: number of attention head.
      dropatt: an optional float of attention dropout rate.
      dropout: an optional float of dropping rate for dropout regularization.
      rel_attn_type: =a `str` specify the type of relative attention head,
        possible values are ['2d_multi_head', '2d_single_head'].
      window_size: window size for conducting block attention module.
      grid_size: grid size for conducting sparse global grid attention.
      scale_ratio: a optional string for finetuning at different window size,
        e.g. '14/7'.
      ln_epsilon: layer normalization epsilon.
      ln_dtype: layer normalization data type.
      downsample_loc: location to conduct downsampleing to feature maps.
      kernel_size: stem convoluation kernal size.
      se_ratio: se ratio for `mbconv` block.
      dropcnn: an optional float of CNN dropout rate.
      data_format: image data format, usualy 'channels_last'.
      norm_type: normalization type, one of ['batch_norm', 'sync_batch_norm',
        'layer_norm'].
      bn_epsilon: batch normalization epsilon.
      bn_momentum: batch normalization momentum.
      add_pos_enc: if add position embedding.
      pool_type: pooling operation type, one of ['2d:avg', '2d:max', '1d:avg',
        '1d:max'].
      pool_stride: pooling stride size.
      expansion_rate: expansion rate value.
      activation: activate function.
      survival_prob: survival probability.
      survival_prob_anneal: if anneal survival probability.
      representation_size: an optional `int` of representation size.
      add_gap_layer_norm: if add layer norm to GAP of backbone final output.
      kernel_initializer: kernel initializer.
      bias_initializer: bias initializer.
      name: specify module name.
      **kwargs: extra keyword arguments to be passed.
    """

    super().__init__(name=name)
    self._block_type = block_type
    self._num_blocks = num_blocks
    self._hidden_size = hidden_size
    self._stem_hsize = stem_hsize
    self._head_size = head_size
    self._num_heads = num_heads
    self._dropatt = dropatt
    self._dropout = dropout
    self._rel_attn_type = rel_attn_type
    self._window_size = window_size
    self._grid_size = grid_size
    self._scale_ratio = scale_ratio
    self._ln_epsilon = ln_epsilon
    self._ln_dtype = ln_dtype
    self._downsample_loc = downsample_loc
    self._kernel_size = kernel_size
    self._se_ratio = se_ratio
    self._dropcnn = dropcnn
    self._data_format = data_format
    self._norm_type = norm_type
    self._bn_epsilon = bn_epsilon
    self._bn_momentum = bn_momentum
    self._add_pos_enc = add_pos_enc
    self._pool_type = pool_type
    self._pool_stride = pool_stride
    self._expansion_rate = expansion_rate
    self._activation = activation
    self._survival_prob = survival_prob
    self._survival_prob_anneal = survival_prob_anneal
    self._representation_size = representation_size
    self._add_gap_layer_norm = add_gap_layer_norm
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._output_specs = {}

  def build(self, input_shape: tf.TensorShape) -> None:
    if self._norm_type == 'layer_norm':
      bn_class = functools.partial(
          tf.keras.layers.LayerNormalization, epsilon=self._ln_epsilon
      )
    elif self._norm_type == 'batch_norm':
      bn_class = functools.partial(
          tf.keras.layers.BatchNormalization,
          momentum=self._bn_momentum,
          epsilon=self._bn_epsilon,
      )
    elif self._norm_type == 'sync_batch_norm':
      bn_class = functools.partial(
          tf.keras.layers.BatchNormalization,
          momentum=self._bn_momentum,
          epsilon=self._bn_epsilon,
          synchronized=True,
      )
    else:
      raise ValueError(f'Unsupported norm_type {self._norm_type}.')

    _, self.height, self.width, _ = input_shape.as_list()
    logging.info(
        f'Build backbone with input size: ({self.height}, {self.width}).'
    )

    # Stem
    stem_layers = []
    for i, _ in enumerate(self._stem_hsize):
      conv_layer = tf.keras.layers.Conv2D(
          filters=self._stem_hsize[i],
          kernel_size=self._kernel_size,
          strides=2 if i == 0 else 1,
          padding='same',
          data_format=self._data_format,
          kernel_initializer=self._kernel_initializer,
          bias_initializer=self._bias_initializer,
          use_bias=True,
          name='conv_{}'.format(i),
      )
      stem_layers.append(conv_layer)
      if i < len(self._stem_hsize) - 1:
        stem_layers.append(bn_class(name='norm_{}'.format(i)))
        stem_layers.append(
            tf.keras.layers.Activation(
                ops.get_act_fn(self._activation), name=f'act_{i}'
            )
        )
    self._stem = tf.keras.Sequential(layers=stem_layers, name='stem')

    # Backbone
    self._blocks = []
    total_num_blocks = sum(self._num_blocks)
    bid = 0
    for i, _ in enumerate(self._block_type):
      self._blocks.append([])
      for j in range(self._num_blocks[i]):
        # block name
        block_name = f'block_{i:0>2d}_{j:0>2d}'

        ##### Update per-block config
        # No pooling if not the first block in the stage
        if j == 0:
          pool_stride = self._pool_stride
        else:
          pool_stride = 1

        # anneal the survival prob
        survival_prob = self._survival_prob
        if survival_prob and self._survival_prob_anneal:
          drop_rate = 1.0 - survival_prob
          survival_prob = 1.0 - drop_rate * bid / total_num_blocks
          logging.info(
              '[%02d/%02d] %s survival_prob: %.4f',
              bid,
              total_num_blocks,
              block_name,
              survival_prob,
          )

        ##### Init block
        if self._block_type[i] == 'tfm':
          block = layers.TransformerBlock(
              hidden_size=self._hidden_size[i],
              head_size=self._head_size,
              input_origin_height=self.height,
              input_origin_width=self.width,
              num_heads=self._num_heads,
              expansion_rate=self._expansion_rate,
              activation=self._activation,
              pool_type=self._pool_type,
              pool_stride=pool_stride,
              dropatt=self._dropatt,
              dropout=self._dropout,
              rel_attn_type=self._rel_attn_type,
              scale_ratio=self._scale_ratio,
              survival_prob=survival_prob,
              ln_epsilon=self._ln_epsilon,
              ln_dtype=self._ln_dtype,
              kernel_initializer=self._kernel_initializer,
              bias_initializer=self._bias_initializer,
              name=block_name,
          )
        elif self._block_type[i] == 'mbconv':
          assert self._pool_type in ['2d:max', '2d:avg'], (
              'Invalid pool_type %s for MBConv block' % self._pool_type
          )
          pool_type = self._pool_type.split(':')[-1]
          block = layers.MBConvBlock(
              hidden_size=self._hidden_size[i],
              downsample_loc=self._downsample_loc,
              data_format=self._data_format,
              kernel_size=self._kernel_size,
              expansion_rate=self._expansion_rate,
              se_ratio=self._se_ratio,
              activation=self._activation,
              pool_type=pool_type,
              pool_stride=pool_stride,
              dropcnn=self._dropcnn,
              survival_prob=survival_prob,
              norm_type=self._norm_type,
              bn_epsilon=self._bn_epsilon,
              bn_momentum=self._bn_momentum,
              kernel_initializer=self._kernel_initializer,
              bias_initializer=self._bias_initializer,
              name=block_name,
          )
        elif self._block_type[i] == 'maxvit':
          block = MaxViTBlock(
              hidden_size=self._hidden_size[i],
              head_size=self._head_size,
              window_size=self._window_size,
              grid_size=self._grid_size,
              num_heads=self._num_heads,
              downsample_loc=self._downsample_loc,
              data_format=self._data_format,
              kernel_size=self._kernel_size,
              expansion_rate=self._expansion_rate,
              se_ratio=self._se_ratio,
              activation=self._activation,
              pool_type=self._pool_type,
              pool_stride=pool_stride,
              dropcnn=self._dropcnn,
              dropatt=self._dropatt,
              dropout=self._dropout,
              rel_attn_type=self._rel_attn_type,
              scale_ratio=self._scale_ratio,
              survival_prob=survival_prob,
              ln_epsilon=self._ln_epsilon,
              ln_dtype=self._ln_dtype,
              norm_type=self._norm_type,
              bn_epsilon=self._bn_epsilon,
              bn_momentum=self._bn_momentum,
              kernel_initializer=self._kernel_initializer,
              bias_initializer=self._bias_initializer,
              name=block_name,
          )
        else:
          raise ValueError(f'Unsupported block_type {self._block_type[i]}')
        self._blocks[-1].append(block)
        bid += 1

    if self._representation_size and self._representation_size > 0:
      self._dense = tf.keras.layers.Dense(
          self._representation_size, name='pre_logits')
      if self._add_gap_layer_norm:
        self._final_layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=self._ln_epsilon, name='final_layer_norm')

  def _add_absolute_position_encoding(self, inputs: tf.Tensor) -> tf.Tensor:
    """Add absolute sinusoid position encoding, which is computed on the fly."""
    output = ops.maybe_reshape_to_2d(inputs)
    h, w = tf.shape(output)[1], tf.shape(output)[2]
    enc_size = output.shape.as_list()[-1] // 2
    # sinusoid positional encoding that can be generated online
    h_seq = tf.range(-h / 2, h / 2)
    w_seq = tf.range(-w / 2, w / 2)
    pos_enc_h = ops.absolute_position_encoding(
        h_seq, enc_size, dtype=output.dtype
    )
    pos_enc_w = ops.absolute_position_encoding(
        w_seq, enc_size, dtype=output.dtype
    )
    abs_pos_enc = tf.concat(
        [
            tf.tile(pos_enc_h[:, None, :], [1, w, 1]),
            tf.tile(pos_enc_w[None, :, :], [h, 1, 1]),
        ],
        axis=-1,
    )
    output += abs_pos_enc
    if inputs.shape.rank == 3:
      output = ops.maybe_reshape_to_1d(output)
    return output

  def call(
      self, inputs: tf.Tensor, mask: Optional[Any] = None, training: bool = None
  ) -> Mapping[str, tf.Tensor]:
    logging.info(
        'MaxViT inputs: shape %s, dtype %s.', inputs.shape, inputs.dtype
    )
    output = self._stem(inputs, training=training)
    logging.info(
        'Stage 0 (stem) output: shape %s, dtype %s.', output.shape, output.dtype
    )

    endpoints = {}
    add_pos_enc = self._add_pos_enc
    for idx, stage_blocks in enumerate(self._blocks):
      # Add position encoding
      # Note: the position encoding is usually added to the input of the first
      # transformer block. For MaxViT, it is the first block of stage 3.
      if (isinstance(add_pos_enc, (tuple, list)) and add_pos_enc[idx]) or (
          isinstance(add_pos_enc, bool) and add_pos_enc
      ):
        logging.info('Add position encoding at stage %d.', idx + 1)
        output = self._add_absolute_position_encoding(output)

      # Blocks forward
      for block in stage_blocks:
        output = block(output, training=training)

      if self._block_type[idx] == 'tfm':
        height, width = ops.get_shape_from_length(
            output.shape[1], self.height, self.width
        )
        output = tf.reshape(output, [-1, height, width, output.shape[-1]])

      endpoints[str(idx + 2)] = output
      logging.info(
          'Stage %d output: feature level %s shape %s, dtype %s.',
          idx + 1,
          idx + 2,
          output.shape,
          output.dtype,
      )

    self._output_specs = {
        idx: endpoint.get_shape() for idx, endpoint in endpoints.items()
    }

    if self._representation_size and self._representation_size > 0:
      # Backbone's output is [batch_size, height, weight, channel_size].
      output = tf.keras.layers.GlobalAveragePooling2D()(output)
      # Maybe add a layer_norm after global average pooling.
      if self._add_gap_layer_norm:
        output = self._final_layer_norm(output)
      endpoints['pre_logits'] = tf.nn.tanh(self._dense(output))

    return endpoints

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


def override_predefined_spec_and_build_maxvit(
    predefined_maxvit_spec, backbone_cfg, norm_activation_config
):
  """Builds a MaxViT backbone.

  Args:
    predefined_maxvit_spec: a dict predefined maxvit specifications.
    backbone_cfg: the MaxViT backbone config.
    norm_activation_config: normalization and activation config.

  Returns:
    The built MaxViT backbone.
  """
  survival_prob = (
      predefined_maxvit_spec['survival_prob']
      if backbone_cfg.survival_prob is None
      else backbone_cfg.survival_prob
  )
  stem_hsize = (
      predefined_maxvit_spec['stem_hsize']
      if backbone_cfg.stem_hsize is None
      else backbone_cfg.stem_hsize
  )
  block_type = (
      predefined_maxvit_spec['block_type']
      if backbone_cfg.block_type is None
      else backbone_cfg.block_type
  )
  num_blocks = (
      predefined_maxvit_spec['num_blocks']
      if backbone_cfg.num_blocks is None
      else backbone_cfg.num_blocks
  )
  hidden_size = (
      predefined_maxvit_spec['hidden_size']
      if backbone_cfg.hidden_size is None
      else backbone_cfg.hidden_size
  )

  logging.info(
      (
          'Final MaxViT specs: survival_prob=%s, stem_hsize=%s, hidden_size=%s,'
          'block_type=%s, num_blocks=%s,.'
      ),
      survival_prob,
      stem_hsize,
      hidden_size,
      block_type,
      num_blocks,
  )

  return MaxViT(
      block_type=block_type,
      num_blocks=num_blocks,
      hidden_size=hidden_size,
      stem_hsize=stem_hsize,
      head_size=backbone_cfg.head_size,
      dropatt=backbone_cfg.dropatt,
      dropout=backbone_cfg.dropout,
      rel_attn_type=backbone_cfg.rel_attn_type,
      window_size=backbone_cfg.window_size,
      grid_size=backbone_cfg.grid_size,
      scale_ratio=backbone_cfg.scale_ratio,
      ln_epsilon=backbone_cfg.ln_epsilon,
      ln_dtype=backbone_cfg.ln_dtype,
      downsample_loc=backbone_cfg.downsample_loc,
      kernel_size=backbone_cfg.kernel_size,
      se_ratio=backbone_cfg.se_ratio,
      dropcnn=backbone_cfg.dropcnn,
      data_format=backbone_cfg.data_format,
      norm_type=backbone_cfg.norm_type,
      bn_epsilon=norm_activation_config.norm_epsilon,
      bn_momentum=norm_activation_config.norm_momentum,
      add_pos_enc=backbone_cfg.add_pos_enc,
      pool_type=backbone_cfg.pool_type,
      pool_stride=backbone_cfg.pool_stride,
      expansion_rate=backbone_cfg.expansion_rate,
      activation=norm_activation_config.activation,
      survival_prob=survival_prob,
      survival_prob_anneal=backbone_cfg.survival_prob_anneal,
      representation_size=backbone_cfg.representation_size,
      add_gap_layer_norm=backbone_cfg.add_gap_layer_norm,
      kernel_initializer=backbone_cfg.kernel_initializer,
      bias_initializer=backbone_cfg.bias_initializer,
  )


@factory.register_backbone_builder('maxvit')
def build_maxvit(
    input_specs,
    backbone_config,
    norm_activation_config,
    l2_regularizer=None,
):
  """Builds a MaxViT backbone."""
  del l2_regularizer
  backbone_cfg = backbone_config.get()
  maxvit = override_predefined_spec_and_build_maxvit(
      predefined_maxvit_spec=MAXVIT_SPECS[backbone_cfg.model_name],
      backbone_cfg=backbone_cfg,
      norm_activation_config=norm_activation_config,
  )
  # Build the backbone to get a proper `output_specs`.
  dummy_inputs = tf.keras.Input(input_specs.shape[1:])
  _ = maxvit(dummy_inputs, training=False)
  return maxvit
