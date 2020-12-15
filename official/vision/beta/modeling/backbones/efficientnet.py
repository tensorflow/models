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
"""Contains definitions of EfficientNet Networks."""

import math
# Import libraries
import tensorflow as tf
from official.modeling import tf_utils
from official.vision.beta.modeling.backbones import factory
from official.vision.beta.modeling.layers import nn_blocks
from official.vision.beta.modeling.layers import nn_layers

layers = tf.keras.layers

# The fixed EfficientNet-B0 architecture discovered by NAS.
# Each element represents a specification of a building block:
# (block_fn, block_repeats, kernel_size, strides, expand_ratio, in_filters,
# out_filters, is_output)
EN_B0_BLOCK_SPECS = [
    ('mbconv', 1, 3, 1, 1, 32, 16, False),
    ('mbconv', 2, 3, 2, 6, 16, 24, True),
    ('mbconv', 2, 5, 2, 6, 24, 40, True),
    ('mbconv', 3, 3, 2, 6, 40, 80, False),
    ('mbconv', 3, 5, 1, 6, 80, 112, True),
    ('mbconv', 4, 5, 2, 6, 112, 192, False),
    ('mbconv', 1, 3, 1, 6, 192, 320, True),
]

SCALING_MAP = {
    'b0': dict(width_scale=1.0, depth_scale=1.0),
    'b1': dict(width_scale=1.0, depth_scale=1.1),
    'b2': dict(width_scale=1.1, depth_scale=1.2),
    'b3': dict(width_scale=1.2, depth_scale=1.4),
    'b4': dict(width_scale=1.4, depth_scale=1.8),
    'b5': dict(width_scale=1.6, depth_scale=2.2),
    'b6': dict(width_scale=1.8, depth_scale=2.6),
    'b7': dict(width_scale=2.0, depth_scale=3.1),
}


def round_repeats(repeats, multiplier, skip=False):
  """Round number of filters based on depth multiplier."""
  if skip or not multiplier:
    return repeats
  return int(math.ceil(multiplier * repeats))


def block_spec_decoder(specs, width_scale, depth_scale):
  """Decode specs for a block."""
  decoded_specs = []
  for s in specs:
    s = s + (
        width_scale,
        depth_scale,
    )
    decoded_specs.append(BlockSpec(*s))
  return decoded_specs


class BlockSpec(object):
  """A container class that specifies the block configuration for MnasNet."""

  def __init__(self, block_fn, block_repeats, kernel_size, strides,
               expand_ratio, in_filters, out_filters, is_output, width_scale,
               depth_scale):
    self.block_fn = block_fn
    self.block_repeats = round_repeats(block_repeats, depth_scale)
    self.kernel_size = kernel_size
    self.strides = strides
    self.expand_ratio = expand_ratio
    self.in_filters = nn_layers.round_filters(in_filters, width_scale)
    self.out_filters = nn_layers.round_filters(out_filters, width_scale)
    self.is_output = is_output


@tf.keras.utils.register_keras_serializable(package='Vision')
class EfficientNet(tf.keras.Model):
  """Class to build EfficientNet family model."""

  def __init__(self,
               model_id,
               input_specs=layers.InputSpec(shape=[None, None, None, 3]),
               se_ratio=0.0,
               stochastic_depth_drop_rate=0.0,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               use_sync_bn=False,
               norm_momentum=0.99,
               norm_epsilon=0.001,
               **kwargs):
    """EfficientNet initialization function.

    Args:
      model_id: `str` model id of EfficientNet.
      input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
      se_ratio: `float` squeeze and excitation ratio for inverted bottleneck
        blocks.
      stochastic_depth_drop_rate: `float` drop rate for drop connect layer.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
        Default to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
        Default to None.
      activation: `str` name of the activation function.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: `float` normalization omentum for the moving average.
      norm_epsilon: `float` small float added to variance to avoid dividing by
        zero.
      **kwargs: keyword arguments to be passed.
    """
    self._model_id = model_id
    self._input_specs = input_specs
    self._se_ratio = se_ratio
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    if use_sync_bn:
      self._norm = layers.experimental.SyncBatchNormalization
    else:
      self._norm = layers.BatchNormalization

    if tf.keras.backend.image_data_format() == 'channels_last':
      bn_axis = -1
    else:
      bn_axis = 1

    # Build EfficientNet.
    inputs = tf.keras.Input(shape=input_specs.shape[1:])
    width_scale = SCALING_MAP[model_id]['width_scale']
    depth_scale = SCALING_MAP[model_id]['depth_scale']

    # Build stem.
    x = layers.Conv2D(
        filters=nn_layers.round_filters(32, width_scale),
        kernel_size=3,
        strides=2,
        use_bias=False,
        padding='same',
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)(
            inputs)
    x = self._norm(
        axis=bn_axis, momentum=norm_momentum, epsilon=norm_epsilon)(
            x)
    x = tf_utils.get_activation(activation)(x)

    # Build intermediate blocks.
    endpoints = {}
    endpoint_level = 2
    decoded_specs = block_spec_decoder(EN_B0_BLOCK_SPECS, width_scale,
                                       depth_scale)

    for i, specs in enumerate(decoded_specs):
      x = self._block_group(
          inputs=x, specs=specs, name='block_group_{}'.format(i))
      if specs.is_output:
        endpoints[str(endpoint_level)] = x
        endpoint_level += 1

    # Build output specs for downstream tasks.
    self._output_specs = {l: endpoints[l].get_shape for l in endpoints.keys()}

    # Build the final conv for classification.
    x = layers.Conv2D(
        filters=nn_layers.round_filters(1280, width_scale),
        kernel_size=1,
        strides=1,
        use_bias=False,
        padding='same',
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)(
            x)
    x = self._norm(
        axis=bn_axis, momentum=norm_momentum, epsilon=norm_epsilon)(
            x)
    endpoints[str(endpoint_level)] = tf_utils.get_activation(activation)(x)

    super(EfficientNet, self).__init__(
        inputs=inputs, outputs=endpoints, **kwargs)

  def _block_group(self, inputs, specs, name='block_group'):
    """Creates one group of blocks for the EfficientNet model.

    Args:
      inputs: `Tensor` of size `[batch, channels, height, width]`.
      specs: specifications for one inverted bottleneck block group.
      name: `str`name for the block.

    Returns:
      The output `Tensor` of the block layer.
    """
    if specs.block_fn == 'mbconv':
      block_fn = nn_blocks.InvertedBottleneckBlock
    else:
      raise ValueError('Block func {} not supported.'.format(specs.block_fn))

    x = block_fn(
        in_filters=specs.in_filters,
        out_filters=specs.out_filters,
        expand_ratio=specs.expand_ratio,
        strides=specs.strides,
        kernel_size=specs.kernel_size,
        se_ratio=self._se_ratio,
        stochastic_depth_drop_rate=self._stochastic_depth_drop_rate,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=self._activation,
        use_sync_bn=self._use_sync_bn,
        norm_momentum=self._norm_momentum,
        norm_epsilon=self._norm_epsilon)(
            inputs)

    for _ in range(1, specs.block_repeats):
      x = block_fn(
          in_filters=specs.out_filters,  # Set 'in_filters' to 'out_filters'.
          out_filters=specs.out_filters,
          expand_ratio=specs.expand_ratio,
          strides=1,  # Fix strides to 1.
          kernel_size=specs.kernel_size,
          se_ratio=self._se_ratio,
          stochastic_depth_drop_rate=self._stochastic_depth_drop_rate,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          activation=self._activation,
          use_sync_bn=self._use_sync_bn,
          norm_momentum=self._norm_momentum,
          norm_epsilon=self._norm_epsilon)(
              x)

    return tf.identity(x, name=name)

  def get_config(self):
    config_dict = {
        'model_id': self._model_id,
        'se_ratio': self._se_ratio,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon
    }
    return config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_backbone_builder('efficientnet')
def build_efficientnet(
    input_specs: tf.keras.layers.InputSpec,
    model_config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
  """Builds ResNet 3d backbone from a config."""
  backbone_type = model_config.backbone.type
  backbone_cfg = model_config.backbone.get()
  norm_activation_config = model_config.norm_activation
  assert backbone_type == 'efficientnet', (f'Inconsistent backbone type '
                                           f'{backbone_type}')

  return EfficientNet(
      model_id=backbone_cfg.model_id,
      input_specs=input_specs,
      stochastic_depth_drop_rate=backbone_cfg.stochastic_depth_drop_rate,
      se_ratio=backbone_cfg.se_ratio,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)
