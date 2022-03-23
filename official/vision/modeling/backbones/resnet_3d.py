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

"""Contains definitions of 3D Residual Networks."""
from typing import Callable, List, Tuple, Optional

# Import libraries
import tensorflow as tf

from official.modeling import hyperparams
from official.modeling import tf_utils
from official.vision.modeling.backbones import factory
from official.vision.modeling.layers import nn_blocks_3d
from official.vision.modeling.layers import nn_layers

layers = tf.keras.layers

RESNET_SPECS = {
    50: [
        ('bottleneck3d', 64, 3),
        ('bottleneck3d', 128, 4),
        ('bottleneck3d', 256, 6),
        ('bottleneck3d', 512, 3),
    ],
    101: [
        ('bottleneck3d', 64, 3),
        ('bottleneck3d', 128, 4),
        ('bottleneck3d', 256, 23),
        ('bottleneck3d', 512, 3),
    ],
    152: [
        ('bottleneck3d', 64, 3),
        ('bottleneck3d', 128, 8),
        ('bottleneck3d', 256, 36),
        ('bottleneck3d', 512, 3),
    ],
    200: [
        ('bottleneck3d', 64, 3),
        ('bottleneck3d', 128, 24),
        ('bottleneck3d', 256, 36),
        ('bottleneck3d', 512, 3),
    ],
    270: [
        ('bottleneck3d', 64, 4),
        ('bottleneck3d', 128, 29),
        ('bottleneck3d', 256, 53),
        ('bottleneck3d', 512, 4),
    ],
    300: [
        ('bottleneck3d', 64, 4),
        ('bottleneck3d', 128, 36),
        ('bottleneck3d', 256, 54),
        ('bottleneck3d', 512, 4),
    ],
    350: [
        ('bottleneck3d', 64, 4),
        ('bottleneck3d', 128, 36),
        ('bottleneck3d', 256, 72),
        ('bottleneck3d', 512, 4),
    ],
}


@tf.keras.utils.register_keras_serializable(package='Vision')
class ResNet3D(tf.keras.Model):
  """Creates a 3D ResNet family model."""

  def __init__(
      self,
      model_id: int,
      temporal_strides: List[int],
      temporal_kernel_sizes: List[Tuple[int]],
      use_self_gating: Optional[List[int]] = None,
      input_specs: tf.keras.layers.InputSpec = layers.InputSpec(
          shape=[None, None, None, None, 3]),
      stem_type: str = 'v0',
      stem_conv_temporal_kernel_size: int = 5,
      stem_conv_temporal_stride: int = 2,
      stem_pool_temporal_stride: int = 2,
      init_stochastic_depth_rate: float = 0.0,
      activation: str = 'relu',
      se_ratio: Optional[float] = None,
      use_sync_bn: bool = False,
      norm_momentum: float = 0.99,
      norm_epsilon: float = 0.001,
      kernel_initializer: str = 'VarianceScaling',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a 3D ResNet model.

    Args:
      model_id: An `int` of depth of ResNet backbone model.
      temporal_strides: A list of integers that specifies the temporal strides
        for all 3d blocks.
      temporal_kernel_sizes: A list of tuples that specifies the temporal kernel
        sizes for all 3d blocks in different block groups.
      use_self_gating: A list of booleans to specify applying self-gating module
        or not in each block group. If None, self-gating is not applied.
      input_specs: A `tf.keras.layers.InputSpec` of the input tensor.
      stem_type: A `str` of stem type of ResNet. Default to `v0`. If set to
        `v1`, use ResNet-D type stem (https://arxiv.org/abs/1812.01187).
      stem_conv_temporal_kernel_size: An `int` of temporal kernel size for the
        first conv layer.
      stem_conv_temporal_stride: An `int` of temporal stride for the first conv
        layer.
      stem_pool_temporal_stride: An `int` of temporal stride for the first pool
        layer.
      init_stochastic_depth_rate: A `float` of initial stochastic depth rate.
      activation: A `str` of name of the activation function.
      se_ratio: A `float` or None. Ratio of the Squeeze-and-Excitation layer.
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
    self._temporal_strides = temporal_strides
    self._temporal_kernel_sizes = temporal_kernel_sizes
    self._input_specs = input_specs
    self._stem_type = stem_type
    self._stem_conv_temporal_kernel_size = stem_conv_temporal_kernel_size
    self._stem_conv_temporal_stride = stem_conv_temporal_stride
    self._stem_pool_temporal_stride = stem_pool_temporal_stride
    self._use_self_gating = use_self_gating
    self._se_ratio = se_ratio
    self._init_stochastic_depth_rate = init_stochastic_depth_rate
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
    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1

    # Build ResNet3D backbone.
    inputs = tf.keras.Input(shape=input_specs.shape[1:])
    endpoints = self._build_model(inputs)
    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}

    super(ResNet3D, self).__init__(inputs=inputs, outputs=endpoints, **kwargs)

  def _build_model(self, inputs):
    """Builds model architecture.

    Args:
      inputs: the keras input spec.

    Returns:
      endpoints: A dictionary of backbone endpoint features.
    """
    # Build stem.
    x = self._build_stem(inputs, stem_type=self._stem_type)

    temporal_kernel_size = 1 if self._stem_pool_temporal_stride == 1 else 3
    x = layers.MaxPool3D(
        pool_size=[temporal_kernel_size, 3, 3],
        strides=[self._stem_pool_temporal_stride, 2, 2],
        padding='same')(x)

    # Build intermediate blocks and endpoints.
    resnet_specs = RESNET_SPECS[self._model_id]
    if len(self._temporal_strides) != len(resnet_specs) or len(
        self._temporal_kernel_sizes) != len(resnet_specs):
      raise ValueError(
          'Number of blocks in temporal specs should equal to resnet_specs.')

    endpoints = {}
    for i, resnet_spec in enumerate(resnet_specs):
      if resnet_spec[0] == 'bottleneck3d':
        block_fn = nn_blocks_3d.BottleneckBlock3D
      else:
        raise ValueError('Block fn `{}` is not supported.'.format(
            resnet_spec[0]))

      use_self_gating = (
          self._use_self_gating[i] if self._use_self_gating else False)
      x = self._block_group(
          inputs=x,
          filters=resnet_spec[1],
          temporal_kernel_sizes=self._temporal_kernel_sizes[i],
          temporal_strides=self._temporal_strides[i],
          spatial_strides=(1 if i == 0 else 2),
          block_fn=block_fn,
          block_repeats=resnet_spec[2],
          stochastic_depth_drop_rate=nn_layers.get_stochastic_depth_rate(
              self._init_stochastic_depth_rate, i + 2, 5),
          use_self_gating=use_self_gating,
          name='block_group_l{}'.format(i + 2))
      endpoints[str(i + 2)] = x

    return endpoints

  def _build_stem(self, inputs, stem_type):
    """Builds stem layer."""
    # Build stem.
    if stem_type == 'v0':
      x = layers.Conv3D(
          filters=64,
          kernel_size=[self._stem_conv_temporal_kernel_size, 7, 7],
          strides=[self._stem_conv_temporal_stride, 2, 2],
          use_bias=False,
          padding='same',
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)(
              inputs)
      x = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon)(x)
      x = tf_utils.get_activation(self._activation)(x)
    elif stem_type == 'v1':
      x = layers.Conv3D(
          filters=32,
          kernel_size=[self._stem_conv_temporal_kernel_size, 3, 3],
          strides=[self._stem_conv_temporal_stride, 2, 2],
          use_bias=False,
          padding='same',
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)(
              inputs)
      x = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon)(x)
      x = tf_utils.get_activation(self._activation)(x)
      x = layers.Conv3D(
          filters=32,
          kernel_size=[1, 3, 3],
          strides=[1, 1, 1],
          use_bias=False,
          padding='same',
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)(
              x)
      x = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon)(x)
      x = tf_utils.get_activation(self._activation)(x)
      x = layers.Conv3D(
          filters=64,
          kernel_size=[1, 3, 3],
          strides=[1, 1, 1],
          use_bias=False,
          padding='same',
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer)(
              x)
      x = self._norm(
          axis=self._bn_axis,
          momentum=self._norm_momentum,
          epsilon=self._norm_epsilon)(x)
      x = tf_utils.get_activation(self._activation)(x)
    else:
      raise ValueError(f'Stem type {stem_type} not supported.')

    return x

  def _block_group(self,
                   inputs: tf.Tensor,
                   filters: int,
                   temporal_kernel_sizes: Tuple[int],
                   temporal_strides: int,
                   spatial_strides: int,
                   block_fn: Callable[
                       ...,
                       tf.keras.layers.Layer] = nn_blocks_3d.BottleneckBlock3D,
                   block_repeats: int = 1,
                   stochastic_depth_drop_rate: float = 0.0,
                   use_self_gating: bool = False,
                   name: str = 'block_group'):
    """Creates one group of blocks for the ResNet3D model.

    Args:
      inputs: A `tf.Tensor` of size `[batch, channels, height, width]`.
      filters: An `int` of number of filters for the first convolution of the
        layer.
      temporal_kernel_sizes: A tuple that specifies the temporal kernel sizes
        for each block in the current group.
      temporal_strides: An `int` of temporal strides for the first convolution
        in this group.
      spatial_strides: An `int` stride to use for the first convolution of the
        layer. If greater than 1, this layer will downsample the input.
      block_fn: Either `nn_blocks.ResidualBlock` or `nn_blocks.BottleneckBlock`.
      block_repeats: An `int` of number of blocks contained in the layer.
      stochastic_depth_drop_rate: A `float` of drop rate of the current block
        group.
      use_self_gating: A `bool` that specifies whether to apply self-gating
        module or not.
      name: A `str` name for the block.

    Returns:
      The output `tf.Tensor` of the block layer.
    """
    if len(temporal_kernel_sizes) != block_repeats:
      raise ValueError(
          'Number of elements in `temporal_kernel_sizes` must equal to `block_repeats`.'
      )

    # Only apply self-gating module in the last block.
    use_self_gating_list = [False] * (block_repeats - 1) + [use_self_gating]

    x = block_fn(
        filters=filters,
        temporal_kernel_size=temporal_kernel_sizes[0],
        temporal_strides=temporal_strides,
        spatial_strides=spatial_strides,
        stochastic_depth_drop_rate=stochastic_depth_drop_rate,
        use_self_gating=use_self_gating_list[0],
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
          temporal_kernel_size=temporal_kernel_sizes[i],
          temporal_strides=1,
          spatial_strides=1,
          stochastic_depth_drop_rate=stochastic_depth_drop_rate,
          use_self_gating=use_self_gating_list[i],
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
        'temporal_strides': self._temporal_strides,
        'temporal_kernel_sizes': self._temporal_kernel_sizes,
        'stem_type': self._stem_type,
        'stem_conv_temporal_kernel_size': self._stem_conv_temporal_kernel_size,
        'stem_conv_temporal_stride': self._stem_conv_temporal_stride,
        'stem_pool_temporal_stride': self._stem_pool_temporal_stride,
        'use_self_gating': self._use_self_gating,
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


@factory.register_backbone_builder('resnet_3d')
def build_resnet3d(
    input_specs: tf.keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: Optional[tf.keras.regularizers.Regularizer] = None
) -> tf.keras.Model:
  """Builds ResNet 3d backbone from a config."""
  backbone_cfg = backbone_config.get()

  # Flatten configs before passing to the backbone.
  temporal_strides = []
  temporal_kernel_sizes = []
  use_self_gating = []
  for block_spec in backbone_cfg.block_specs:
    temporal_strides.append(block_spec.temporal_strides)
    temporal_kernel_sizes.append(block_spec.temporal_kernel_sizes)
    use_self_gating.append(block_spec.use_self_gating)

  return ResNet3D(
      model_id=backbone_cfg.model_id,
      temporal_strides=temporal_strides,
      temporal_kernel_sizes=temporal_kernel_sizes,
      use_self_gating=use_self_gating,
      input_specs=input_specs,
      stem_type=backbone_cfg.stem_type,
      stem_conv_temporal_kernel_size=backbone_cfg
      .stem_conv_temporal_kernel_size,
      stem_conv_temporal_stride=backbone_cfg.stem_conv_temporal_stride,
      stem_pool_temporal_stride=backbone_cfg.stem_pool_temporal_stride,
      init_stochastic_depth_rate=backbone_cfg.stochastic_depth_drop_rate,
      se_ratio=backbone_cfg.se_ratio,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)


@factory.register_backbone_builder('resnet_3d_rs')
def build_resnet3d_rs(
    input_specs: tf.keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: Optional[tf.keras.regularizers.Regularizer] = None
) -> tf.keras.Model:
  """Builds ResNet-3D-RS backbone from a config."""
  backbone_cfg = backbone_config.get()

  # Flatten configs before passing to the backbone.
  temporal_strides = []
  temporal_kernel_sizes = []
  use_self_gating = []
  for i, block_spec in enumerate(backbone_cfg.block_specs):
    temporal_strides.append(block_spec.temporal_strides)
    use_self_gating.append(block_spec.use_self_gating)
    block_repeats_i = RESNET_SPECS[backbone_cfg.model_id][i][-1]
    temporal_kernel_sizes.append(list(block_spec.temporal_kernel_sizes) *
                                 block_repeats_i)
  return ResNet3D(
      model_id=backbone_cfg.model_id,
      temporal_strides=temporal_strides,
      temporal_kernel_sizes=temporal_kernel_sizes,
      use_self_gating=use_self_gating,
      input_specs=input_specs,
      stem_type=backbone_cfg.stem_type,
      stem_conv_temporal_kernel_size=backbone_cfg
      .stem_conv_temporal_kernel_size,
      stem_conv_temporal_stride=backbone_cfg.stem_conv_temporal_stride,
      stem_pool_temporal_stride=backbone_cfg.stem_pool_temporal_stride,
      init_stochastic_depth_rate=backbone_cfg.stochastic_depth_drop_rate,
      se_ratio=backbone_cfg.se_ratio,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)
