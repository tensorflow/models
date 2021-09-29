# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Contains definitions of Residual Networks with Deeplab modifications."""

from typing import Callable, Optional, Tuple, List

import numpy as np
import tensorflow as tf
from official.modeling import hyperparams
from official.modeling import tf_utils
from official.vision.beta.modeling.backbones import factory
from official.vision.beta.modeling.layers import nn_blocks
from official.vision.beta.modeling.layers import nn_layers

layers = tf.keras.layers

# Specifications for different ResNet variants.
# Each entry specifies block configurations of the particular ResNet variant.
# Each element in the block configuration is in the following format:
# (block_fn, num_filters, block_repeats)
RESNET_SPECS = {
    50: [
        ('bottleneck', 64, 3),
        ('bottleneck', 128, 4),
        ('bottleneck', 256, 6),
        ('bottleneck', 512, 3),
    ],
    101: [
        ('bottleneck', 64, 3),
        ('bottleneck', 128, 4),
        ('bottleneck', 256, 23),
        ('bottleneck', 512, 3),
    ],
}


@tf.keras.utils.register_keras_serializable(package='Vision')
class DilatedResNet(tf.keras.Model):
  """Creates a ResNet model with Deeplabv3 modifications.

  This backbone is suitable for semantic segmentation. This implements
    Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam.
    Rethinking Atrous Convolution for Semantic Image Segmentation.
    (https://arxiv.org/pdf/1706.05587)
  """

  def __init__(
      self,
      model_id: int,
      output_stride: int,
      input_specs: tf.keras.layers.InputSpec = layers.InputSpec(
          shape=[None, None, None, 3]),
      stem_type: str = 'v0',
      se_ratio: Optional[float] = None,
      init_stochastic_depth_rate: float = 0.0,
      multigrid: Optional[Tuple[int]] = None,
      last_stage_repeats: int = 1,
      activation: str = 'relu',
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a ResNet model with DeepLab modification.

    Args:
      model_id: An `int` specifies depth of ResNet backbone model.
      output_stride: An `int` of output stride, ratio of input to output
        resolution.
      input_specs: A `tf.keras.layers.InputSpec` of the input tensor.
      stem_type: A `str` of stem type. Can be `v0` or `v1`. `v1` replaces 7x7
        conv by 3 3x3 convs.
      se_ratio: A `float` or None. Ratio of the Squeeze-and-Excitation layer.
      init_stochastic_depth_rate: A `float` of initial stochastic depth rate.
      multigrid: A tuple of the same length as the number of blocks in the last
        resnet stage.
      last_stage_repeats: An `int` that specifies how many times last stage is
        repeated.
      activation: A `str` name of the activation function.
      use_sync_bn: If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      kernel_initializer: A str for kernel initializer of convolutional layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
        Default to None.
      **kwargs: Additional keyword arguments to be passed.
    """
    self._model_id = model_id
    self._output_stride = output_stride
    self._input_specs = input_specs
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    if use_sync_bn:
      self._norm = layers.experimental.SyncBatchNormalization
    else:
      self._norm = layers.BatchNormalization
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._stem_type = stem_type
    self._se_ratio = se_ratio
    self._init_stochastic_depth_rate = init_stochastic_depth_rate

    if tf.keras.backend.image_data_format() == 'channels_last':
      bn_axis = -1
    else:
      bn_axis = 1

    # Build ResNet.
    inputs = tf.keras.Input(shape=input_specs.shape[1:])

    if stem_type == 'v0':
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
          axis=bn_axis, momentum=norm_momentum, epsilon=norm_epsilon)(
              x)
      x = tf_utils.get_activation(activation)(x)
    elif stem_type == 'v1':
      x = layers.Conv2D(
          filters=64,
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
      x = layers.Conv2D(
          filters=64,
          kernel_size=3,
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
      x = tf_utils.get_activation(activation)(x)
      x = layers.Conv2D(
          filters=128,
          kernel_size=3,
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
      x = tf_utils.get_activation(activation)(x)
    else:
      raise ValueError('Stem type {} not supported.'.format(stem_type))

    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    normal_resnet_stage = int(np.math.log2(self._output_stride)) - 2

    endpoints = {}
    for i in range(normal_resnet_stage + 1):
      spec = RESNET_SPECS[model_id][i]
      if spec[0] == 'bottleneck':
        block_fn = nn_blocks.BottleneckBlock
      else:
        raise ValueError('Block fn `{}` is not supported.'.format(spec[0]))
      x = self._block_group(
          inputs=x,
          filters=spec[1],
          strides=(1 if i == 0 else 2),
          dilation_rate=1,
          block_fn=block_fn,
          block_repeats=spec[2],
          stochastic_depth_drop_rate=nn_layers.get_stochastic_depth_rate(
              self._init_stochastic_depth_rate, i + 2, 4 + last_stage_repeats),
          name='block_group_l{}'.format(i + 2))
      endpoints[str(i + 2)] = x

    dilation_rate = 2
    for i in range(normal_resnet_stage + 1, 3 + last_stage_repeats):
      spec = RESNET_SPECS[model_id][i] if i < 3 else RESNET_SPECS[model_id][-1]
      if spec[0] == 'bottleneck':
        block_fn = nn_blocks.BottleneckBlock
      else:
        raise ValueError('Block fn `{}` is not supported.'.format(spec[0]))
      x = self._block_group(
          inputs=x,
          filters=spec[1],
          strides=1,
          dilation_rate=dilation_rate,
          block_fn=block_fn,
          block_repeats=spec[2],
          stochastic_depth_drop_rate=nn_layers.get_stochastic_depth_rate(
              self._init_stochastic_depth_rate, i + 2, 4 + last_stage_repeats),
          multigrid=multigrid if i >= 3 else None,
          name='block_group_l{}'.format(i + 2))
      dilation_rate *= 2

    endpoints[str(normal_resnet_stage + 2)] = x

    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}

    super(DilatedResNet, self).__init__(
        inputs=inputs, outputs=endpoints, **kwargs)

  def _block_group(self,
                   inputs: tf.Tensor,
                   filters: int,
                   strides: int,
                   dilation_rate: int,
                   block_fn: Callable[..., tf.keras.layers.Layer],
                   block_repeats: int = 1,
                   stochastic_depth_drop_rate: float = 0.0,
                   multigrid: Optional[List[int]] = None,
                   name: str = 'block_group'):
    """Creates one group of blocks for the ResNet model.

    Deeplab applies strides at the last block.

    Args:
      inputs: A `tf.Tensor` of size `[batch, channels, height, width]`.
      filters: An `int` off number of filters for the first convolution of the
        layer.
      strides: An `int` of stride to use for the first convolution of the layer.
        If greater than 1, this layer will downsample the input.
      dilation_rate: An `int` of diluted convolution rates.
      block_fn: Either `nn_blocks.ResidualBlock` or `nn_blocks.BottleneckBlock`.
      block_repeats: An `int` of number of blocks contained in the layer.
      stochastic_depth_drop_rate: A `float` of drop rate of the current block
        group.
      multigrid: A list of `int` or None. If specified, dilation rates for each
        block is scaled up by its corresponding factor in the multigrid.
      name: A `str` name for the block.

    Returns:
      The output `tf.Tensor` of the block layer.
    """
    if multigrid is not None and len(multigrid) != block_repeats:
      raise ValueError('multigrid has to match number of block_repeats')

    if multigrid is None:
      multigrid = [1] * block_repeats

    # TODO(arashwan): move striding at the of the block.
    x = block_fn(
        filters=filters,
        strides=strides,
        dilation_rate=dilation_rate * multigrid[0],
        use_projection=True,
        stochastic_depth_drop_rate=stochastic_depth_drop_rate,
        se_ratio=self._se_ratio,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activation=self._activation,
        use_sync_bn=self._use_sync_bn,
        norm_momentum=self._norm_momentum,
        norm_epsilon=self._norm_epsilon)(
            inputs)
    for i in range(1, block_repeats):
      x = block_fn(
          filters=filters,
          strides=1,
          dilation_rate=dilation_rate * multigrid[i],
          use_projection=False,
          stochastic_depth_drop_rate=stochastic_depth_drop_rate,
          se_ratio=self._se_ratio,
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
        'output_stride': self._output_stride,
        'stem_type': self._stem_type,
        'se_ratio': self._se_ratio,
        'init_stochastic_depth_rate': self._init_stochastic_depth_rate,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
    }
    return config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_backbone_builder('dilated_resnet')
def build_dilated_resnet(
    input_specs: tf.keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds ResNet backbone from a config."""
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'dilated_resnet', (f'Inconsistent backbone type '
                                             f'{backbone_type}')

  return DilatedResNet(
      model_id=backbone_cfg.model_id,
      output_stride=backbone_cfg.output_stride,
      input_specs=input_specs,
      stem_type=backbone_cfg.stem_type,
      se_ratio=backbone_cfg.se_ratio,
      init_stochastic_depth_rate=backbone_cfg.stochastic_depth_drop_rate,
      multigrid=backbone_cfg.multigrid,
      last_stage_repeats=backbone_cfg.last_stage_repeats,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)
