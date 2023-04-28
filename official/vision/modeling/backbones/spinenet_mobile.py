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

# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Contains definitions of Mobile SpineNet Networks."""
import math
from typing import Any, List, Optional, Tuple

# Import libraries

from absl import logging
import tensorflow as tf

from official.modeling import hyperparams
from official.modeling import tf_utils
from official.vision.modeling.backbones import factory
from official.vision.modeling.layers import nn_blocks
from official.vision.modeling.layers import nn_layers
from official.vision.ops import spatial_transform_ops

layers = tf.keras.layers

FILTER_SIZE_MAP = {
    0: 8,
    1: 16,
    2: 24,
    3: 40,
    4: 80,
    5: 112,
    6: 112,
    7: 112,
}

# The fixed SpineNet architecture discovered by NAS.
# Each element represents a specification of a building block:
#   (block_level, block_fn, (input_offset0, input_offset1), is_output).
SPINENET_BLOCK_SPECS = [
    (2, 'mbconv', (0, 1), False),
    (2, 'mbconv', (1, 2), False),
    (4, 'mbconv', (1, 2), False),
    (3, 'mbconv', (3, 4), False),
    (4, 'mbconv', (3, 5), False),
    (6, 'mbconv', (4, 6), False),
    (4, 'mbconv', (4, 6), False),
    (5, 'mbconv', (7, 8), False),
    (7, 'mbconv', (7, 9), False),
    (5, 'mbconv', (9, 10), False),
    (5, 'mbconv', (9, 11), False),
    (4, 'mbconv', (6, 11), True),
    (3, 'mbconv', (5, 11), True),
    (5, 'mbconv', (8, 13), True),
    (7, 'mbconv', (6, 15), True),
    (6, 'mbconv', (13, 15), True),
]

SCALING_MAP = {
    '49': {
        'endpoints_num_filters': 48,
        'filter_size_scale': 1.0,
        'block_repeats': 1,
    },
    '49S': {
        'endpoints_num_filters': 40,
        'filter_size_scale': 0.65,
        'block_repeats': 1,
    },
    '49XS': {
        'endpoints_num_filters': 24,
        'filter_size_scale': 0.6,
        'block_repeats': 1,
    },
}


class BlockSpec(object):
  """A container class that specifies the block configuration for SpineNet."""

  def __init__(self, level: int, block_fn: str, input_offsets: Tuple[int, int],
               is_output: bool):
    self.level = level
    self.block_fn = block_fn
    self.input_offsets = input_offsets
    self.is_output = is_output


def build_block_specs(
    block_specs: Optional[List[Tuple[Any, ...]]] = None) -> List[BlockSpec]:
  """Builds the list of BlockSpec objects for SpineNet."""
  if not block_specs:
    block_specs = SPINENET_BLOCK_SPECS
  logging.info('Building SpineNet block specs: %s', block_specs)
  return [BlockSpec(*b) for b in block_specs]


@tf.keras.utils.register_keras_serializable(package='Vision')
class SpineNetMobile(tf.keras.Model):
  """Creates a Mobile SpineNet family model.

  This implements:
    [1] Xianzhi Du, Tsung-Yi Lin, Pengchong Jin, Golnaz Ghiasi, Mingxing Tan,
    Yin Cui, Quoc V. Le, Xiaodan Song.
    SpineNet: Learning Scale-Permuted Backbone for Recognition and Localization.
    (https://arxiv.org/abs/1912.05027).
    [2] Xianzhi Du, Tsung-Yi Lin, Pengchong Jin, Yin Cui, Mingxing Tan,
    Quoc Le, Xiaodan Song.
    Efficient Scale-Permuted Backbone with Learned Resource Distribution.
    (https://arxiv.org/abs/2010.11426).
  """

  def __init__(
      self,
      input_specs: tf.keras.layers.InputSpec = tf.keras.layers.InputSpec(
          shape=[None, None, None, 3]),
      min_level: int = 3,
      max_level: int = 7,
      block_specs: Optional[List[BlockSpec]] = None,
      endpoints_num_filters: int = 256,
      se_ratio: float = 0.2,
      block_repeats: int = 1,
      filter_size_scale: float = 1.0,
      expand_ratio: int = 6,
      init_stochastic_depth_rate=0.0,
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      use_keras_upsampling_2d: bool = False,
      **kwargs):
    """Initializes a Mobile SpineNet model.

    Args:
      input_specs: A `tf.keras.layers.InputSpec` of the input tensor.
      min_level: An `int` of min level for output mutiscale features.
      max_level: An `int` of max level for output mutiscale features.
      block_specs: The block specifications for the SpineNet model discovered by
        NAS.
      endpoints_num_filters: An `int` of feature dimension for the output
        endpoints.
      se_ratio: A `float` of Squeeze-and-Excitation ratio.
      block_repeats: An `int` of number of blocks contained in the layer.
      filter_size_scale: A `float` of multiplier for the filters (number of
        channels) for all convolution ops. The value must be greater than zero.
        Typical usage will be to set this value in (0, 1) to reduce the number
        of parameters or computation cost of the model.
      expand_ratio: An `integer` of expansion ratios for inverted bottleneck
        blocks.
      init_stochastic_depth_rate: A `float` of initial stochastic depth rate.
      kernel_initializer: A str for kernel initializer of convolutional layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
        Default to None.
      activation: A `str` name of the activation function.
      use_sync_bn: If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A small `float` added to variance to avoid dividing by zero.
      use_keras_upsampling_2d: If True, use keras UpSampling2D layer.
      **kwargs: Additional keyword arguments to be passed.
    """
    self._input_specs = input_specs
    self._min_level = min_level
    self._max_level = max_level
    self._block_specs = (
        build_block_specs() if block_specs is None else block_specs
    )
    self._endpoints_num_filters = endpoints_num_filters
    self._se_ratio = se_ratio
    self._block_repeats = block_repeats
    self._filter_size_scale = filter_size_scale
    self._expand_ratio = expand_ratio
    self._init_stochastic_depth_rate = init_stochastic_depth_rate
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._activation = activation
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._use_keras_upsampling_2d = use_keras_upsampling_2d
    self._num_init_blocks = 2
    self._norm = layers.BatchNormalization

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1

    # Build SpineNet.
    inputs = tf.keras.Input(shape=input_specs.shape[1:])

    net = self._build_stem(inputs=inputs)
    input_width = input_specs.shape[2]
    if input_width is None:
      max_stride = max(map(lambda b: b.level, self._block_specs))
      input_width = 2 ** max_stride
    net = self._build_scale_permuted_network(net=net, input_width=input_width)
    endpoints = self._build_endpoints(net=net)

    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}
    super().__init__(inputs=inputs, outputs=endpoints)

  def _block_group(self,
                   inputs: tf.Tensor,
                   in_filters: int,
                   out_filters: int,
                   strides: int,
                   expand_ratio: int = 6,
                   block_repeats: int = 1,
                   se_ratio: float = 0.2,
                   stochastic_depth_drop_rate: Optional[float] = None,
                   name: str = 'block_group'):
    """Creates one group of blocks for the SpineNet model."""
    x = nn_blocks.InvertedBottleneckBlock(
        in_filters=in_filters,
        out_filters=out_filters,
        strides=strides,
        se_gating_activation='hard_sigmoid',
        se_ratio=se_ratio,
        expand_ratio=expand_ratio,
        stochastic_depth_drop_rate=stochastic_depth_drop_rate,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=self._activation,
        use_sync_bn=self._use_sync_bn,
        norm_momentum=self._norm_momentum,
        norm_epsilon=self._norm_epsilon)(
            inputs)
    for _ in range(1, block_repeats):
      x = nn_blocks.InvertedBottleneckBlock(
          in_filters=in_filters,
          out_filters=out_filters,
          strides=1,
          se_ratio=se_ratio,
          expand_ratio=expand_ratio,
          stochastic_depth_drop_rate=stochastic_depth_drop_rate,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation=self._activation,
          use_sync_bn=self._use_sync_bn,
          norm_momentum=self._norm_momentum,
          norm_epsilon=self._norm_epsilon)(
              inputs)
    return tf.keras.layers.Activation('linear', name=name)(x)

  def _build_stem(self, inputs):
    """Builds SpineNet stem."""
    x = layers.Conv2D(
        filters=int(FILTER_SIZE_MAP[0] * self._filter_size_scale),
        kernel_size=3,
        strides=2,
        use_bias=False,
        padding='same',
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)(
            inputs)
    x = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        synchronized=self._use_sync_bn)(
            x)
    x = tf_utils.get_activation(self._activation, use_keras_layer=True)(x)

    net = []
    stem_strides = [1, 2]
    # Build the initial level 2 blocks.
    for i in range(self._num_init_blocks):
      x = self._block_group(
          inputs=x,
          in_filters=int(FILTER_SIZE_MAP[i] * self._filter_size_scale),
          out_filters=int(FILTER_SIZE_MAP[i + 1] * self._filter_size_scale),
          expand_ratio=self._expand_ratio,
          strides=stem_strides[i],
          se_ratio=self._se_ratio,
          block_repeats=self._block_repeats,
          name='stem_block_{}'.format(i + 1))
      net.append(x)
    return net

  def _build_scale_permuted_network(self,
                                    net,
                                    input_width,
                                    weighted_fusion=False):
    """Builds scale-permuted network."""
    net_sizes = [
        int(math.ceil(input_width / 2)),
        int(math.ceil(input_width / 2**2))
    ]
    num_outgoing_connections = [0] * len(net)

    endpoints = {}
    for i, block_spec in enumerate(self._block_specs):
      # Update block level if it is larger than max_level to avoid building
      # blocks smaller than requested.
      block_spec.level = min(block_spec.level, self._max_level)
      # Find out specs for the target block.
      target_width = int(math.ceil(input_width / 2**block_spec.level))
      target_num_filters = int(FILTER_SIZE_MAP[block_spec.level] *
                               self._filter_size_scale)

      # Resample then merge input0 and input1.
      parents = []
      input0 = block_spec.input_offsets[0]
      input1 = block_spec.input_offsets[1]

      x0 = self._resample_with_sepconv(
          inputs=net[input0],
          input_width=net_sizes[input0],
          target_width=target_width,
          target_num_filters=target_num_filters)
      parents.append(x0)
      num_outgoing_connections[input0] += 1

      x1 = self._resample_with_sepconv(
          inputs=net[input1],
          input_width=net_sizes[input1],
          target_width=target_width,
          target_num_filters=target_num_filters)
      parents.append(x1)
      num_outgoing_connections[input1] += 1

      # Merge 0 outdegree blocks to the output block.
      if block_spec.is_output:
        for j, (j_feat,
                j_connections) in enumerate(zip(net, num_outgoing_connections)):
          if j_connections == 0 and (j_feat.shape[2] == target_width and
                                     j_feat.shape[3] == x0.shape[3]):
            parents.append(j_feat)
            num_outgoing_connections[j] += 1

      # pylint: disable=g-direct-tensorflow-import
      if weighted_fusion:
        dtype = parents[0].dtype
        parent_weights = [
            tf.nn.relu(tf.cast(tf.Variable(1.0, name='block{}_fusion{}'.format(
                i, j)), dtype=dtype)) for j in range(len(parents))]
        weights_sum = parent_weights[0]
        for adder in parent_weights[1:]:
          weights_sum = layers.Add()([weights_sum, adder])

        parents = [
            parents[i] * parent_weights[i] / (weights_sum + 0.0001)
            for i in range(len(parents))
        ]

      # Fuse all parent nodes then build a new block.
      x = parents[0]
      for adder in parents[1:]:
        x = layers.Add()([x, adder])
      x = tf_utils.get_activation(
          self._activation, use_keras_layer=True)(x)
      x = self._block_group(
          inputs=x,
          in_filters=target_num_filters,
          out_filters=target_num_filters,
          strides=1,
          se_ratio=self._se_ratio,
          expand_ratio=self._expand_ratio,
          block_repeats=self._block_repeats,
          stochastic_depth_drop_rate=nn_layers.get_stochastic_depth_rate(
              self._init_stochastic_depth_rate, i + 1, len(self._block_specs)),
          name='scale_permuted_block_{}'.format(i + 1))

      net.append(x)
      net_sizes.append(target_width)
      num_outgoing_connections.append(0)

      # Save output feats.
      if block_spec.is_output:
        if block_spec.level in endpoints:
          raise ValueError('Duplicate feats found for output level {}.'.format(
              block_spec.level))
        if (block_spec.level < self._min_level or
            block_spec.level > self._max_level):
          logging.warning(
              'SpineNet output level out of range [min_level, max_levle] = [%s, %s] will not be used for further processing.',
              self._min_level, self._max_level)
        endpoints[str(block_spec.level)] = x

    return endpoints

  def _build_endpoints(self, net):
    """Matches filter size for endpoints before sharing conv layers."""
    endpoints = {}
    for level in range(self._min_level, self._max_level + 1):
      x = layers.Conv2D(
          filters=self._endpoints_num_filters,
          kernel_size=1,
          strides=1,
          use_bias=False,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)(
              net[str(level)])
      x = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon,
          synchronized=self._use_sync_bn)(
              x)
      x = tf_utils.get_activation(self._activation, use_keras_layer=True)(x)
      endpoints[str(level)] = x
    return endpoints

  def _resample_with_sepconv(self, inputs, input_width, target_width,
                             target_num_filters):
    """Matches resolution and feature dimension."""
    x = inputs
    # Spatial resampling.
    if input_width > target_width:
      while input_width > target_width:
        x = layers.DepthwiseConv2D(
            kernel_size=3,
            strides=2,
            padding='SAME',
            use_bias=False,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer)(
                x)
        x = self._norm(
            axis=self._bn_axis,
            momentum=self._norm_momentum,
            epsilon=self._norm_epsilon,
            synchronized=self._use_sync_bn)(
                x)
        x = tf_utils.get_activation(
            self._activation, use_keras_layer=True)(x)
        input_width /= 2
    elif input_width < target_width:
      scale = target_width // input_width
      x = spatial_transform_ops.nearest_upsampling(
          x, scale=scale, use_keras_layer=self._use_keras_upsampling_2d)

    # Last 1x1 conv to match filter size.
    x = layers.Conv2D(
        filters=target_num_filters,
        kernel_size=1,
        strides=1,
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)(
            x)
    x = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        synchronized=self._use_sync_bn)(
            x)
    return x

  def get_config(self):
    config_dict = {
        'min_level': self._min_level,
        'max_level': self._max_level,
        'endpoints_num_filters': self._endpoints_num_filters,
        'se_ratio': self._se_ratio,
        'expand_ratio': self._expand_ratio,
        'block_repeats': self._block_repeats,
        'filter_size_scale': self._filter_size_scale,
        'init_stochastic_depth_rate': self._init_stochastic_depth_rate,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'use_keras_upsampling_2d': self._use_keras_upsampling_2d,
    }
    return config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_backbone_builder('spinenet_mobile')
def build_spinenet_mobile(
    input_specs: tf.keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
  """Builds Mobile SpineNet backbone from a config."""
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'spinenet_mobile', (f'Inconsistent backbone type '
                                              f'{backbone_type}')

  model_id = backbone_cfg.model_id
  if model_id not in SCALING_MAP:
    raise ValueError(
        'Mobile SpineNet-{} is not a valid architecture.'.format(model_id))
  scaling_params = SCALING_MAP[model_id]

  return SpineNetMobile(
      input_specs=input_specs,
      min_level=backbone_cfg.min_level,
      max_level=backbone_cfg.max_level,
      endpoints_num_filters=scaling_params['endpoints_num_filters'],
      block_repeats=scaling_params['block_repeats'],
      filter_size_scale=scaling_params['filter_size_scale'],
      se_ratio=backbone_cfg.se_ratio,
      expand_ratio=backbone_cfg.expand_ratio,
      init_stochastic_depth_rate=backbone_cfg.stochastic_depth_drop_rate,
      kernel_regularizer=l2_regularizer,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      use_keras_upsampling_2d=backbone_cfg.use_keras_upsampling_2d)
