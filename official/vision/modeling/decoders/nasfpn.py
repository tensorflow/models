# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Contains definitions of NAS-FPN."""

from typing import Any, List, Mapping, Optional, Tuple

from absl import logging
import tensorflow as tf, tf_keras

from official.modeling import hyperparams
from official.modeling import tf_utils
from official.vision.modeling.decoders import factory
from official.vision.ops import spatial_transform_ops


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


class BlockSpec():
  """A container class that specifies the block configuration for NAS-FPN."""

  def __init__(self, level: int, combine_fn: str,
               input_offsets: Tuple[int, int], is_output: bool):
    self.level = level
    self.combine_fn = combine_fn
    self.input_offsets = input_offsets
    self.is_output = is_output


def build_block_specs(
    block_specs: Optional[List[Tuple[Any, ...]]] = None) -> List[BlockSpec]:
  """Builds the list of BlockSpec objects for NAS-FPN."""
  if not block_specs:
    block_specs = NASFPN_BLOCK_SPECS
  logging.info('Building NAS-FPN block specs: %s', block_specs)
  return [BlockSpec(*b) for b in block_specs]


@tf_keras.utils.register_keras_serializable(package='Vision')
class NASFPN(tf_keras.Model):
  """Creates a NAS-FPN model.

  This implements the paper:
  Golnaz Ghiasi, Tsung-Yi Lin, Ruoming Pang, Quoc V. Le.
  NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection.
  (https://arxiv.org/abs/1904.07392)
  """

  def __init__(
      self,
      input_specs: Mapping[str, tf.TensorShape],
      min_level: int = 3,
      max_level: int = 7,
      block_specs: Optional[List[BlockSpec]] = None,
      num_filters: int = 256,
      num_repeats: int = 5,
      use_separable_conv: bool = False,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a NAS-FPN model.

    Args:
      input_specs: A `dict` of input specifications. A dictionary consists of
        {level: TensorShape} from a backbone.
      min_level: An `int` of minimum level in FPN output feature maps.
      max_level: An `int` of maximum level in FPN output feature maps.
      block_specs: a list of BlockSpec objects that specifies the NAS-FPN
        network topology. By default, the previously discovered architecture is
        used.
      num_filters: An `int` number of filters in FPN layers.
      num_repeats: number of repeats for feature pyramid network.
      use_separable_conv: A `bool`.  If True use separable convolution for
        convolution in FPN layers.
      activation: A `str` name of the activation function.
      use_sync_bn: A `bool`. If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_initializer: A `str` name of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf_keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf_keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
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
    self._block_specs = (
        build_block_specs() if block_specs is None else block_specs
    )
    self._num_repeats = num_repeats
    self._conv_op = (tf_keras.layers.SeparableConv2D
                     if self._config_dict['use_separable_conv']
                     else tf_keras.layers.Conv2D)
    self._norm_op = tf_keras.layers.BatchNormalization
    if tf_keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1
    self._norm_kwargs = {
        'axis': self._bn_axis,
        'momentum': self._config_dict['norm_momentum'],
        'epsilon': self._config_dict['norm_epsilon'],
        'synchronized': self._config_dict['use_sync_bn'],
    }
    self._activation = tf_utils.get_activation(activation)

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

  def _build_input_pyramid(self, input_specs: Mapping[str, tf.TensorShape],
                           min_level: int):
    assert isinstance(input_specs, dict)
    if min(input_specs.keys()) > str(min_level):
      raise ValueError(
          'Backbone min level should be less or equal to FPN min level')

    inputs = {}
    for level, spec in input_specs.items():
      inputs[level] = tf_keras.Input(shape=spec[1:])
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
      return tf_keras.layers.MaxPool2D(
          pool_size=stride, strides=stride, padding='same')(x)
    if input_level > target_level:
      scale = int(2 ** (input_level - target_level))
      return spatial_transform_ops.nearest_upsampling(x, scale=scale)

    # Force output x to be the same dtype as mixed precision policy. This avoids
    # dtype mismatch when one input (by default float32 dtype) does not meet all
    # the above conditions and is output unchanged, while other inputs are
    # processed to have different dtype, e.g., using bfloat16 on TPU.
    compute_dtype = tf_keras.layers.Layer().dtype_policy.compute_dtype
    if (compute_dtype is not None) and (x.dtype != compute_dtype):
      return tf.cast(x, dtype=compute_dtype)
    else:
      return x

  @property
  def _conv_kwargs(self):
    if self._config_dict['use_separable_conv']:
      return {
          'depthwise_initializer': tf_keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'pointwise_initializer': tf_keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'depthwise_regularizer': self._config_dict['kernel_regularizer'],
          'pointwise_regularizer': self._config_dict['kernel_regularizer'],
          'bias_regularizer': self._config_dict['bias_regularizer'],
      }
    else:
      return {
          'kernel_initializer': tf_keras.initializers.VarianceScaling(
              scale=2, mode='fan_out', distribution='untruncated_normal'),
          'bias_initializer': tf.zeros_initializer(),
          'kernel_regularizer': self._config_dict['kernel_regularizer'],
          'bias_regularizer': self._config_dict['bias_regularizer'],
      }

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

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self) -> Mapping[str, tf.TensorShape]:
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_decoder_builder('nasfpn')
def build_nasfpn_decoder(
    input_specs: Mapping[str, tf.TensorShape],
    model_config: hyperparams.Config,
    l2_regularizer: Optional[tf_keras.regularizers.Regularizer] = None
) -> tf_keras.Model:
  """Builds NASFPN decoder from a config.

  Args:
    input_specs: A `dict` of input specifications. A dictionary consists of
      {level: TensorShape} from a backbone.
    model_config: A OneOfConfig. Model config.
    l2_regularizer: A `tf_keras.regularizers.Regularizer` instance. Default to
      None.

  Returns:
    A `tf_keras.Model` instance of the NASFPN decoder.

  Raises:
    ValueError: If the model_config.decoder.type is not `nasfpn`.
  """
  decoder_type = model_config.decoder.type
  decoder_cfg = model_config.decoder.get()
  if decoder_type != 'nasfpn':
    raise ValueError(f'Inconsistent decoder type {decoder_type}. '
                     'Need to be `nasfpn`.')

  norm_activation_config = model_config.norm_activation
  return NASFPN(
      input_specs=input_specs,
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      num_filters=decoder_cfg.num_filters,
      num_repeats=decoder_cfg.num_repeats,
      use_separable_conv=decoder_cfg.use_separable_conv,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)
