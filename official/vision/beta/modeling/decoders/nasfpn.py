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
"""NAS-FPN.

Golnaz Ghiasi, Tsung-Yi Lin, Ruoming Pang, Quoc V. Le.
NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection.
https://arxiv.org/abs/1904.07392. CVPR 2019.
"""

# Import libraries
from absl import logging
import tensorflow as tf

from official.vision.beta.ops import spatial_transform_ops


# The fixed NAS-FPN architecture discovered by NAS.
# Each element represents a specification of a building block:
#   (block_level, combine_fn, (input_offset0, input_offset1), is_output).
NASFPN_BLOCK_SPECS = [
    (4, 'attention', (1, 3), False),
    (4, 'sum', (1, 5), False),
    (3, 'sum', (0, 6), True),
    (4, 'sum', (6, 7), True),
    (5, 'attention', (7, 8), True),
    (7, 'attention', (6, 9), True),
    (6, 'attention', (9, 10), True),
]


class BlockSpec(object):
  """A container class that specifies the block configuration for NAS-FPN."""

  def __init__(self, level, combine_fn, input_offsets, is_output):
    self.level = level
    self.combine_fn = combine_fn
    self.input_offsets = input_offsets
    self.is_output = is_output


def build_block_specs(block_specs=None):
  """Builds the list of BlockSpec objects for NAS-FPN."""
  if not block_specs:
    block_specs = NASFPN_BLOCK_SPECS
  logging.info('Building NAS-FPN block specs: %s', block_specs)
  return [BlockSpec(*b) for b in block_specs]


@tf.keras.utils.register_keras_serializable(package='Vision')
class NASFPN(tf.keras.Model):
  """NAS-FPN."""

  def __init__(self,
               input_specs,
               min_level=3,
               max_level=7,
               block_specs=build_block_specs(),
               num_filters=256,
               num_repeats=5,
               use_separable_conv=False,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               **kwargs):
    """FPN initialization function.

    Args:
      input_specs: `dict` input specifications. A dictionary consists of
        {level: TensorShape} from a backbone.
      min_level: `int` minimum level in FPN output feature maps.
      max_level: `int` maximum level in FPN output feature maps.
      block_specs: a list of BlockSpec objects that specifies the NAS-FPN
        network topology. By default, the previously discovered architecture is
        used.
      num_filters: `int` number of filters in FPN layers.
      num_repeats: number of repeats for feature pyramid network.
      use_separable_conv: `bool`, if True use separable convolution for
        convolution in FPN layers.
      activation: `str` name of the activation function.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
      **kwargs: keyword arguments to be passed.
    """
    self._config_dict = {
        'input_specs': input_specs,
        'min_level': min_level,
        'max_level': max_level,
        'num_filters': num_filters,
        'num_repeats': num_repeats,
        'use_separable_conv': use_separable_conv,
        'activation': activation,
        'use_sync_bn': use_sync_bn,
        'norm_momentum': norm_momentum,
        'norm_epsilon': norm_epsilon,
        'kernel_initializer': kernel_initializer,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
    }
    self._min_level = min_level
    self._max_level = max_level
    self._block_specs = block_specs
    self._num_repeats = num_repeats
    self._conv_op = (tf.keras.layers.SeparableConv2D
                     if self._config_dict['use_separable_conv']
                     else tf.keras.layers.Conv2D)
    if self._config_dict['use_separable_conv']:
      self._conv_kwargs = {
          'depthwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'pointwise_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'depthwise_regularizer': self._config_dict['kernel_regularizer'],
          'pointwise_regularizer': self._config_dict['kernel_regularizer'],
          'bias_regularizer': self._config_dict['bias_regularizer'],
      }
    else:
      self._conv_kwargs = {
          'kernel_initializer': tf.keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
          'bias_regularizer': self._config_dict['bias_regularizer'],
      }
    self._norm_op = (tf.keras.layers.experimental.SyncBatchNormalization
                     if self._config_dict['use_sync_bn']
                     else tf.keras.layers.BatchNormalization)
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._norm_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._config_dict['norm_momentum'],
        'epsilon': self._config_dict['norm_epsilon'],
    }
    if activation == 'relu':
      self._activation = tf.nn.relu
    elif activation == 'swish':
      self._activation = tf.nn.swish
    else:
      raise ValueError('Activation {} not implemented.'.format(activation))

    # Gets input feature pyramid from backbone.
    inputs = self._build_input_pyramid(input_specs, min_level)

    # Projects the input features.
    feats = []
    for level in range(self._min_level, self._max_level + 1):
      if str(level) in inputs.keys():
        feats.append(self._resample_feature_map(
            inputs[str(level)], level, level, self._config_dict['num_filters']))
      else:
        feats.append(self._resample_feature_map(
            feats[-1], level - 1, level, self._config_dict['num_filters']))

    # Repeatly builds the NAS-FPN modules.
    for _ in range(self._num_repeats):
      output_feats = self._build_feature_pyramid(feats)
      feats = [output_feats[level]
               for level in range(self._min_level, self._max_level + 1)]

    self._output_specs = {
        str(level): output_feats[level].get_shape()
        for level in range(min_level, max_level + 1)
    }
    output_feats = {str(level): output_feats[level]
                    for level in output_feats.keys()}
    super(NASFPN, self).__init__(inputs=inputs, outputs=output_feats, **kwargs)

  def _build_input_pyramid(self, input_specs, min_level):
    assert isinstance(input_specs, dict)
    if min(input_specs.keys()) > str(min_level):
      raise ValueError(
          'Backbone min level should be less or equal to FPN min level')

    inputs = {}
    for level, spec in input_specs.items():
      inputs[level] = tf.keras.Input(shape=spec[1:])
    return inputs

  def _resample_feature_map(self,
                            inputs,
                            input_level,
                            target_level,
                            target_num_filters=256):
    x = inputs
    _, _, _, input_num_filters = x.get_shape().as_list()
    if input_num_filters != target_num_filters:
      x = self._conv_op(
          filters=target_num_filters,
          kernel_size=1,
          padding='same',
          **self._conv_kwargs)(x)
      x = self._norm_op(**self._norm_kwargs)(x)

    if input_level < target_level:
      stride = int(2 ** (target_level - input_level))
      x = tf.keras.layers.MaxPool2D(
          pool_size=stride, strides=stride, padding='same')(x)
    elif input_level > target_level:
      scale = int(2 ** (input_level - target_level))
      x = spatial_transform_ops.nearest_upsampling(x, scale=scale)

    return x

  def _global_attention(self, feat0, feat1):
    m = tf.math.reduce_max(feat0, axis=[1, 2], keepdims=True)
    m = tf.math.sigmoid(m)
    return feat0 + feat1 * m

  def _build_feature_pyramid(self, feats):
    num_output_connections = [0] * len(feats)
    num_output_levels = self._max_level - self._min_level + 1
    feat_levels = list(range(self._min_level, self._max_level + 1))

    for i, block_spec in enumerate(self._block_specs):
      new_level = block_spec.level

      # Checks the range of input_offsets.
      for input_offset in block_spec.input_offsets:
        if input_offset >= len(feats):
          raise ValueError(
              'input_offset ({}) is larger than num feats({})'.format(
                  input_offset, len(feats)))
      input0 = block_spec.input_offsets[0]
      input1 = block_spec.input_offsets[1]

      # Update graph with inputs.
      node0 = feats[input0]
      node0_level = feat_levels[input0]
      num_output_connections[input0] += 1
      node0 = self._resample_feature_map(node0, node0_level, new_level)
      node1 = feats[input1]
      node1_level = feat_levels[input1]
      num_output_connections[input1] += 1
      node1 = self._resample_feature_map(node1, node1_level, new_level)

      # Combine node0 and node1 to create new feat.
      if block_spec.combine_fn == 'sum':
        new_node = node0 + node1
      elif block_spec.combine_fn == 'attention':
        if node0_level >= node1_level:
          new_node = self._global_attention(node0, node1)
        else:
          new_node = self._global_attention(node1, node0)
      else:
        raise ValueError('unknown combine_fn `{}`.'
                         .format(block_spec.combine_fn))

      # Add intermediate nodes that do not have any connections to output.
      if block_spec.is_output:
        for j, (feat, feat_level, num_output) in enumerate(
            zip(feats, feat_levels, num_output_connections)):
          if num_output == 0 and feat_level == new_level:
            num_output_connections[j] += 1

            feat_ = self._resample_feature_map(feat, feat_level, new_level)
            new_node += feat_

      new_node = self._activation(new_node)
      new_node = self._conv_op(
          filters=self._config_dict['num_filters'],
          kernel_size=(3, 3),
          padding='same',
          **self._conv_kwargs)(new_node)
      new_node = self._norm_op(**self._norm_kwargs)(new_node)

      feats.append(new_node)
      feat_levels.append(new_level)
      num_output_connections.append(0)

    output_feats = {}
    for i in range(len(feats) - num_output_levels, len(feats)):
      level = feat_levels[i]
      output_feats[level] = feats[i]
    logging.info('Output feature pyramid: %s', output_feats)
    return output_feats

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs
