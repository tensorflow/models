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

# ==============================================================================
"""Implementation of SpineNet model.

X. Du, T-Y. Lin, P. Jin, G. Ghiasi, M. Tan, Y. Cui, Q. V. Le, X. Song
SpineNet: Learning Scale-Permuted Backbone for Recognition and Localization
https://arxiv.org/abs/1912.05027
"""
import math

from absl import logging
import tensorflow as tf, tf_keras
from official.legacy.detection.modeling.architecture import nn_blocks
from official.modeling import tf_utils

layers = tf_keras.layers

FILTER_SIZE_MAP = {
    1: 32,
    2: 64,
    3: 128,
    4: 256,
    5: 256,
    6: 256,
    7: 256,
}

# The fixed SpineNet architecture discovered by NAS.
# Each element represents a specification of a building block:
#   (block_level, block_fn, (input_offset0, input_offset1), is_output).
SPINENET_BLOCK_SPECS = [
    (2, 'bottleneck', (0, 1), False),
    (4, 'residual', (0, 1), False),
    (3, 'bottleneck', (2, 3), False),
    (4, 'bottleneck', (2, 4), False),
    (6, 'residual', (3, 5), False),
    (4, 'bottleneck', (3, 5), False),
    (5, 'residual', (6, 7), False),
    (7, 'residual', (6, 8), False),
    (5, 'bottleneck', (8, 9), False),
    (5, 'bottleneck', (8, 10), False),
    (4, 'bottleneck', (5, 10), True),
    (3, 'bottleneck', (4, 10), True),
    (5, 'bottleneck', (7, 12), True),
    (7, 'bottleneck', (5, 14), True),
    (6, 'bottleneck', (12, 14), True),
]

SCALING_MAP = {
    '49S': {
        'endpoints_num_filters': 128,
        'filter_size_scale': 0.65,
        'resample_alpha': 0.5,
        'block_repeats': 1,
    },
    '49': {
        'endpoints_num_filters': 256,
        'filter_size_scale': 1.0,
        'resample_alpha': 0.5,
        'block_repeats': 1,
    },
    '96': {
        'endpoints_num_filters': 256,
        'filter_size_scale': 1.0,
        'resample_alpha': 0.5,
        'block_repeats': 2,
    },
    '143': {
        'endpoints_num_filters': 256,
        'filter_size_scale': 1.0,
        'resample_alpha': 1.0,
        'block_repeats': 3,
    },
    '190': {
        'endpoints_num_filters': 512,
        'filter_size_scale': 1.3,
        'resample_alpha': 1.0,
        'block_repeats': 4,
    },
}


class BlockSpec(object):
  """A container class that specifies the block configuration for SpineNet."""

  def __init__(self, level, block_fn, input_offsets, is_output):
    self.level = level
    self.block_fn = block_fn
    self.input_offsets = input_offsets
    self.is_output = is_output


def build_block_specs(block_specs=None):
  """Builds the list of BlockSpec objects for SpineNet."""
  if not block_specs:
    block_specs = SPINENET_BLOCK_SPECS
  logging.info('Building SpineNet block specs: %s', block_specs)
  return [BlockSpec(*b) for b in block_specs]


class SpineNet(tf_keras.Model):
  """Class to build SpineNet models."""

  def __init__(self,
               input_specs=tf_keras.layers.InputSpec(shape=[None, 640, 640, 3]),
               min_level=3,
               max_level=7,
               block_specs=None,
               endpoints_num_filters=256,
               resample_alpha=0.5,
               block_repeats=1,
               filter_size_scale=1.0,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               **kwargs):
    """SpineNet model."""
    self._min_level = min_level
    self._max_level = max_level
    self._block_specs = (
        build_block_specs() if block_specs is None else block_specs
    )
    self._endpoints_num_filters = endpoints_num_filters
    self._resample_alpha = resample_alpha
    self._block_repeats = block_repeats
    self._filter_size_scale = filter_size_scale
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    if activation == 'relu':
      self._activation = tf.nn.relu
    elif activation == 'swish':
      self._activation = tf.nn.swish
    else:
      raise ValueError('Activation {} not implemented.'.format(activation))
    self._init_block_fn = 'bottleneck'
    self._num_init_blocks = 2

    if use_sync_bn:
      self._norm = layers.experimental.SyncBatchNormalization
    else:
      self._norm = layers.BatchNormalization

    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1

    # Build SpineNet.
    inputs = tf_keras.Input(shape=input_specs.shape[1:])

    net = self._build_stem(inputs=inputs)
    net = self._build_scale_permuted_network(
        net=net, input_width=input_specs.shape[1])
    net = self._build_endpoints(net=net)

    super(SpineNet, self).__init__(inputs=inputs, outputs=net)

  def _block_group(self,
                   inputs,
                   filters,
                   strides,
                   block_fn_cand,
                   block_repeats=1,
                   name='block_group'):
    """Creates one group of blocks for the SpineNet model."""
    block_fn_candidates = {
        'bottleneck': nn_blocks.BottleneckBlock,
        'residual': nn_blocks.ResidualBlock,
    }
    block_fn = block_fn_candidates[block_fn_cand]
    _, _, _, num_filters = inputs.get_shape().as_list()

    if block_fn_cand == 'bottleneck':
      use_projection = not (num_filters == (filters * 4) and strides == 1)
    else:
      use_projection = not (num_filters == filters and strides == 1)

    x = block_fn(
        filters=filters,
        strides=strides,
        use_projection=use_projection,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=self._activation,
        use_sync_bn=self._use_sync_bn,
        norm_momentum=self._norm_momentum,
        norm_epsilon=self._norm_epsilon)(
            inputs)
    for _ in range(1, block_repeats):
      x = block_fn(
          filters=filters,
          strides=1,
          use_projection=False,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation=self._activation,
          use_sync_bn=self._use_sync_bn,
          norm_momentum=self._norm_momentum,
          norm_epsilon=self._norm_epsilon)(
              x)
    return tf.identity(x, name=name)

  def _build_stem(self, inputs):
    """Build SpineNet stem."""
    x = layers.Conv2D(
        filters=64,
        kernel_size=7,
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
        epsilon=self._norm_epsilon)(
            x)
    x = tf_utils.get_activation(self._activation)(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    net = []
    # Build the initial level 2 blocks.
    for i in range(self._num_init_blocks):
      x = self._block_group(
          inputs=x,
          filters=int(FILTER_SIZE_MAP[2] * self._filter_size_scale),
          strides=1,
          block_fn_cand=self._init_block_fn,
          block_repeats=self._block_repeats,
          name='stem_block_{}'.format(i + 1))
      net.append(x)
    return net

  def _build_scale_permuted_network(self,
                                    net,
                                    input_width,
                                    weighted_fusion=False):
    """Build scale-permuted network."""
    net_sizes = [int(math.ceil(input_width / 2**2))] * len(net)
    net_block_fns = [self._init_block_fn] * len(net)
    num_outgoing_connections = [0] * len(net)

    endpoints = {}
    for i, block_spec in enumerate(self._block_specs):
      # Find out specs for the target block.
      target_width = int(math.ceil(input_width / 2**block_spec.level))
      target_num_filters = int(FILTER_SIZE_MAP[block_spec.level] *
                               self._filter_size_scale)
      target_block_fn = block_spec.block_fn

      # Resample then merge input0 and input1.
      parents = []
      input0 = block_spec.input_offsets[0]
      input1 = block_spec.input_offsets[1]

      x0 = self._resample_with_alpha(
          inputs=net[input0],
          input_width=net_sizes[input0],
          input_block_fn=net_block_fns[input0],
          target_width=target_width,
          target_num_filters=target_num_filters,
          target_block_fn=target_block_fn,
          alpha=self._resample_alpha)
      parents.append(x0)
      num_outgoing_connections[input0] += 1

      x1 = self._resample_with_alpha(
          inputs=net[input1],
          input_width=net_sizes[input1],
          input_block_fn=net_block_fns[input1],
          target_width=target_width,
          target_num_filters=target_num_filters,
          target_block_fn=target_block_fn,
          alpha=self._resample_alpha)
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
        weights_sum = tf.add_n(parent_weights)
        parents = [
            parents[i] * parent_weights[i] / (weights_sum + 0.0001)
            for i in range(len(parents))
        ]

      # Fuse all parent nodes then build a new block.
      x = tf_utils.get_activation(self._activation)(tf.add_n(parents))
      x = self._block_group(
          inputs=x,
          filters=target_num_filters,
          strides=1,
          block_fn_cand=target_block_fn,
          block_repeats=self._block_repeats,
          name='scale_permuted_block_{}'.format(i + 1))

      net.append(x)
      net_sizes.append(target_width)
      net_block_fns.append(target_block_fn)
      num_outgoing_connections.append(0)

      # Save output feats.
      if block_spec.is_output:
        if block_spec.level in endpoints:
          raise ValueError('Duplicate feats found for output level {}.'.format(
              block_spec.level))
        if (block_spec.level < self._min_level or
            block_spec.level > self._max_level):
          raise ValueError('Output level is out of range [{}, {}]'.format(
              self._min_level, self._max_level))
        endpoints[block_spec.level] = x

    return endpoints

  def _build_endpoints(self, net):
    """Match filter size for endpoints before sharing conv layers."""
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
              net[level])
      x = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon)(
              x)
      x = tf_utils.get_activation(self._activation)(x)
      endpoints[level] = x
    return endpoints

  def _resample_with_alpha(self,
                           inputs,
                           input_width,
                           input_block_fn,
                           target_width,
                           target_num_filters,
                           target_block_fn,
                           alpha=0.5):
    """Match resolution and feature dimension."""
    _, _, _, input_num_filters = inputs.get_shape().as_list()
    if input_block_fn == 'bottleneck':
      input_num_filters /= 4
    new_num_filters = int(input_num_filters * alpha)

    x = layers.Conv2D(
        filters=new_num_filters,
        kernel_size=1,
        strides=1,
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)(
            inputs)
    x = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon)(
            x)
    x = tf_utils.get_activation(self._activation)(x)

    # Spatial resampling.
    if input_width > target_width:
      x = layers.Conv2D(
          filters=new_num_filters,
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
          epsilon=self._norm_epsilon)(
              x)
      x = tf_utils.get_activation(self._activation)(x)
      input_width /= 2
      while input_width > target_width:
        x = layers.MaxPool2D(pool_size=3, strides=2, padding='SAME')(x)
        input_width /= 2
    elif input_width < target_width:
      scale = target_width // input_width
      x = layers.UpSampling2D(size=(scale, scale))(x)

    # Last 1x1 conv to match filter size.
    if target_block_fn == 'bottleneck':
      target_num_filters *= 4
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
        epsilon=self._norm_epsilon)(
            x)

    return x


class SpineNetBuilder(object):
  """SpineNet builder."""

  def __init__(self,
               model_id,
               input_specs=tf_keras.layers.InputSpec(shape=[None, 640, 640, 3]),
               min_level=3,
               max_level=7,
               block_specs=None,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001):
    if model_id not in SCALING_MAP:
      raise ValueError(
          'SpineNet {} is not a valid architecture.'.format(model_id))
    scaling_params = SCALING_MAP[model_id]
    self._input_specs = input_specs
    self._min_level = min_level
    self._max_level = max_level
    self._block_specs = block_specs or build_block_specs()
    self._endpoints_num_filters = scaling_params['endpoints_num_filters']
    self._resample_alpha = scaling_params['resample_alpha']
    self._block_repeats = scaling_params['block_repeats']
    self._filter_size_scale = scaling_params['filter_size_scale']
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._activation = activation
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon

  def __call__(self, inputs, is_training=None):
    model = SpineNet(
        input_specs=self._input_specs,
        min_level=self._min_level,
        max_level=self._max_level,
        block_specs=self._block_specs,
        endpoints_num_filters=self._endpoints_num_filters,
        resample_alpha=self._resample_alpha,
        block_repeats=self._block_repeats,
        filter_size_scale=self._filter_size_scale,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=self._activation,
        use_sync_bn=self._use_sync_bn,
        norm_momentum=self._norm_momentum,
        norm_epsilon=self._norm_epsilon)
    return model(inputs)
