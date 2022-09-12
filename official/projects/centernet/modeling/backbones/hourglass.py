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

"""Build Hourglass backbone."""

from typing import Optional

import tensorflow as tf

from official.modeling import hyperparams
from official.projects.centernet.modeling.layers import cn_nn_blocks
from official.vision.modeling.backbones import factory
from official.vision.modeling.backbones import mobilenet
from official.vision.modeling.layers import nn_blocks

HOURGLASS_SPECS = {
    10: {
        'blocks_per_stage': [1, 1],
        'channel_dims_per_stage': [2, 2]
    },
    20: {
        'blocks_per_stage': [1, 2, 2],
        'channel_dims_per_stage': [2, 2, 3]
    },
    32: {
        'blocks_per_stage': [2, 2, 2, 2],
        'channel_dims_per_stage': [2, 2, 3, 3]
    },
    52: {
        'blocks_per_stage': [2, 2, 2, 2, 2, 4],
        'channel_dims_per_stage': [2, 2, 3, 3, 3, 4]
    },
    100: {
        'blocks_per_stage': [4, 4, 4, 4, 4, 8],
        'channel_dims_per_stage': [2, 2, 3, 3, 3, 4]
    },
}


class Hourglass(tf.keras.Model):
  """CenterNet Hourglass backbone."""

  def __init__(
      self,
      model_id: int,
      input_channel_dims: int,
      input_specs=tf.keras.layers.InputSpec(shape=[None, None, None, 3]),
      num_hourglasses: int = 1,
      initial_downsample: bool = True,
      activation: str = 'relu',
      use_sync_bn: bool = True,
      norm_momentum=0.1,
      norm_epsilon=1e-5,
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initialize Hourglass backbone.

    Args:
      model_id: An `int` of the scale of Hourglass backbone model.
      input_channel_dims: `int`, number of filters used to downsample the
        input image.
      input_specs: A `tf.keras.layers.InputSpec` of specs of the input tensor.
      num_hourglasses: `int``, number of hourglass blocks in backbone. For
        example, hourglass-104 has two hourglass-52 modules.
      initial_downsample: `bool`, whether or not to downsample the input.
      activation: A `str` name of the activation function.
      use_sync_bn: If True, use synchronized batch normalization.
      norm_momentum: `float`, momentum for the batch normalization layers.
      norm_epsilon: `float`, epsilon for the batch normalization layers.
      kernel_initializer: A `str` for kernel initializer of conv layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
        Default to None.
      **kwargs: Additional keyword arguments to be passed.
    """
    self._input_channel_dims = input_channel_dims
    self._model_id = model_id
    self._num_hourglasses = num_hourglasses
    self._initial_downsample = initial_downsample
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon

    specs = HOURGLASS_SPECS[model_id]
    self._blocks_per_stage = specs['blocks_per_stage']
    self._channel_dims_per_stage = [item * self._input_channel_dims
                                    for item in specs['channel_dims_per_stage']]

    inputs = tf.keras.layers.Input(shape=input_specs.shape[1:])

    inp_filters = self._channel_dims_per_stage[0]

    # Downsample the input
    if initial_downsample:
      prelayer_kernel_size = 7
      prelayer_strides = 2
    else:
      prelayer_kernel_size = 3
      prelayer_strides = 1

    x_downsampled = mobilenet.Conv2DBNBlock(
        filters=self._input_channel_dims,
        kernel_size=prelayer_kernel_size,
        strides=prelayer_strides,
        use_explicit_padding=True,
        activation=self._activation,
        bias_regularizer=self._bias_regularizer,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        use_sync_bn=self._use_sync_bn,
        norm_momentum=self._norm_momentum,
        norm_epsilon=self._norm_epsilon)(inputs)

    x_downsampled = nn_blocks.ResidualBlock(
        filters=inp_filters,
        use_projection=True,
        use_explicit_padding=True,
        strides=prelayer_strides,
        bias_regularizer=self._bias_regularizer,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        use_sync_bn=self._use_sync_bn,
        norm_momentum=self._norm_momentum,
        norm_epsilon=self._norm_epsilon)(x_downsampled)

    all_heatmaps = {}
    for i in range(num_hourglasses):
      # Create an hourglass stack
      x_hg = cn_nn_blocks.HourglassBlock(
          channel_dims_per_stage=self._channel_dims_per_stage,
          blocks_per_stage=self._blocks_per_stage,
      )(x_downsampled)

      x_hg = mobilenet.Conv2DBNBlock(
          filters=inp_filters,
          kernel_size=3,
          strides=1,
          use_explicit_padding=True,
          activation=self._activation,
          bias_regularizer=self._bias_regularizer,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          use_sync_bn=self._use_sync_bn,
          norm_momentum=self._norm_momentum,
          norm_epsilon=self._norm_epsilon
      )(x_hg)

      # Given two down-sampling blocks above, the starting level is set to 2
      # To make it compatible with implementation of remaining backbones, the
      # output of hourglass backbones is organized as
      # '2' -> the last layer of output
      # '2_0' -> the first layer of output
      # ......
      # '2_{num_hourglasses-2}' -> the second to last layer of output
      if i < num_hourglasses - 1:
        all_heatmaps['2_{}'.format(i)] = x_hg
      else:
        all_heatmaps['2'] = x_hg

      # Intermediate conv and residual layers between hourglasses
      if i < num_hourglasses - 1:
        inter_hg_conv1 = mobilenet.Conv2DBNBlock(
            filters=inp_filters,
            kernel_size=1,
            strides=1,
            activation='identity',
            bias_regularizer=self._bias_regularizer,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            use_sync_bn=self._use_sync_bn,
            norm_momentum=self._norm_momentum,
            norm_epsilon=self._norm_epsilon
        )(x_downsampled)

        inter_hg_conv2 = mobilenet.Conv2DBNBlock(
            filters=inp_filters,
            kernel_size=1,
            strides=1,
            activation='identity',
            bias_regularizer=self._bias_regularizer,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            use_sync_bn=self._use_sync_bn,
            norm_momentum=self._norm_momentum,
            norm_epsilon=self._norm_epsilon
        )(x_hg)

        x_downsampled = tf.keras.layers.Add()([inter_hg_conv1, inter_hg_conv2])
        x_downsampled = tf.keras.layers.ReLU()(x_downsampled)

        x_downsampled = nn_blocks.ResidualBlock(
            filters=inp_filters,
            use_projection=False,
            use_explicit_padding=True,
            strides=1,
            bias_regularizer=self._bias_regularizer,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            use_sync_bn=self._use_sync_bn,
            norm_momentum=self._norm_momentum,
            norm_epsilon=self._norm_epsilon
        )(x_downsampled)

    self._output_specs = {l: all_heatmaps[l].get_shape() for l in all_heatmaps}

    super().__init__(inputs=inputs, outputs=all_heatmaps, **kwargs)

  def get_config(self):
    config = {
        'model_id': self._model_id,
        'input_channel_dims': self._input_channel_dims,
        'num_hourglasses': self._num_hourglasses,
        'initial_downsample': self._initial_downsample,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon
    }
    config.update(super(Hourglass, self).get_config())
    return config

  @property
  def num_hourglasses(self):
    return self._num_hourglasses

  @property
  def output_specs(self):
    return self._output_specs


@factory.register_backbone_builder('hourglass')
def build_hourglass(
    input_specs: tf.keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: Optional[tf.keras.regularizers.Regularizer] = None
    ) -> tf.keras.Model:
  """Builds Hourglass backbone from a configuration."""
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'hourglass', (f'Inconsistent backbone type '
                                        f'{backbone_type}')

  return Hourglass(
      model_id=backbone_cfg.model_id,
      input_channel_dims=backbone_cfg.input_channel_dims,
      num_hourglasses=backbone_cfg.num_hourglasses,
      input_specs=input_specs,
      initial_downsample=backbone_cfg.initial_downsample,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer,
  )
