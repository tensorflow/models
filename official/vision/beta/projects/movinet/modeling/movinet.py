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

# Lint as: python3
"""Contains definitions of Mobile Video Networks.

Reference: https://arxiv.org/pdf/2103.11511.pdf
"""
from typing import Optional, Sequence, Tuple

import dataclasses
import tensorflow as tf

from official.modeling import hyperparams
from official.vision.beta.modeling.backbones import factory
from official.vision.beta.projects.movinet.modeling import movinet_layers

# Defines a set of kernel sizes and stride sizes to simplify and shorten
# architecture definitions for configs below.
KernelSize = Tuple[int, int, int]

# K(ab) represents a 3D kernel of size (a, b, b)
K13: KernelSize = (1, 3, 3)
K15: KernelSize = (1, 5, 5)
K33: KernelSize = (3, 3, 3)
K53: KernelSize = (5, 3, 3)

# S(ab) represents a 3D stride of size (a, b, b)
S11: KernelSize = (1, 1, 1)
S12: KernelSize = (1, 2, 2)
S22: KernelSize = (2, 2, 2)
S21: KernelSize = (2, 1, 1)


@dataclasses.dataclass
class BlockSpec:
  """Configuration of a block."""
  pass


@dataclasses.dataclass
class StemSpec(BlockSpec):
  """Configuration of a Movinet block."""
  filters: int = 0
  kernel_size: KernelSize = (0, 0, 0)
  strides: KernelSize = (0, 0, 0)


@dataclasses.dataclass
class MovinetBlockSpec(BlockSpec):
  """Configuration of a Movinet block."""
  base_filters: int = 0
  expand_filters: Sequence[int] = ()
  kernel_sizes: Sequence[KernelSize] = ()
  strides: Sequence[KernelSize] = ()


@dataclasses.dataclass
class HeadSpec(BlockSpec):
  """Configuration of a Movinet block."""
  project_filters: int = 0
  head_filters: int = 0
  output_per_frame: bool = False
  max_pool_predictions: bool = False


# Block specs specify the architecture of each model
BLOCK_SPECS = {
    'a0': (
        StemSpec(filters=8, kernel_size=K13, strides=S12),
        MovinetBlockSpec(
            base_filters=8,
            expand_filters=(24,),
            kernel_sizes=(K15,),
            strides=(S12,)),
        MovinetBlockSpec(
            base_filters=32,
            expand_filters=(80, 80, 80),
            kernel_sizes=(K33, K33, K33),
            strides=(S12, S11, S11)),
        MovinetBlockSpec(
            base_filters=56,
            expand_filters=(184, 112, 184),
            kernel_sizes=(K53, K33, K33),
            strides=(S12, S11, S11)),
        MovinetBlockSpec(
            base_filters=56,
            expand_filters=(184, 184, 184, 184),
            kernel_sizes=(K53, K33, K33, K33),
            strides=(S11, S11, S11, S11)),
        MovinetBlockSpec(
            base_filters=104,
            expand_filters=(384, 280, 280, 344),
            kernel_sizes=(K53, K15, K15, K15),
            strides=(S12, S11, S11, S11)),
        HeadSpec(project_filters=480, head_filters=2048),
    ),
    'a1': (
        StemSpec(filters=16, kernel_size=K13, strides=S12),
        MovinetBlockSpec(
            base_filters=16,
            expand_filters=(40, 40),
            kernel_sizes=(K15, K33),
            strides=(S12, S11)),
        MovinetBlockSpec(
            base_filters=40,
            expand_filters=(96, 120, 96, 96),
            kernel_sizes=(K33, K33, K33, K33),
            strides=(S12, S11, S11, S11)),
        MovinetBlockSpec(
            base_filters=64,
            expand_filters=(216, 128, 216, 168, 216),
            kernel_sizes=(K53, K33, K33, K33, K33),
            strides=(S12, S11, S11, S11, S11)),
        MovinetBlockSpec(
            base_filters=64,
            expand_filters=(216, 216, 216, 128, 128, 216),
            kernel_sizes=(K53, K33, K33, K33, K15, K33),
            strides=(S11, S11, S11, S11, S11, S11)),
        MovinetBlockSpec(
            base_filters=136,
            expand_filters=(456, 360, 360, 360, 456, 456, 544),
            kernel_sizes=(K53, K15, K15, K15, K15, K33, K13),
            strides=(S12, S11, S11, S11, S11, S11, S11)),
        HeadSpec(project_filters=600, head_filters=2048),
    ),
    'a2': (
        StemSpec(filters=16, kernel_size=K13, strides=S12),
        MovinetBlockSpec(
            base_filters=16,
            expand_filters=(40, 40, 64),
            kernel_sizes=(K15, K33, K33),
            strides=(S12, S11, S11)),
        MovinetBlockSpec(
            base_filters=40,
            expand_filters=(96, 120, 96, 96, 120),
            kernel_sizes=(K33, K33, K33, K33, K33),
            strides=(S12, S11, S11, S11, S11)),
        MovinetBlockSpec(
            base_filters=72,
            expand_filters=(240, 160, 240, 192, 240),
            kernel_sizes=(K53, K33, K33, K33, K33),
            strides=(S12, S11, S11, S11, S11)),
        MovinetBlockSpec(
            base_filters=72,
            expand_filters=(240, 240, 240, 240, 144, 240),
            kernel_sizes=(K53, K33, K33, K33, K15, K33),
            strides=(S11, S11, S11, S11, S11, S11)),
        MovinetBlockSpec(
            base_filters=144,
            expand_filters=(480, 384, 384, 480, 480, 480, 576),
            kernel_sizes=(K53, K15, K15, K15, K15, K33, K13),
            strides=(S12, S11, S11, S11, S11, S11, S11)),
        HeadSpec(project_filters=640, head_filters=2048),
    ),
    'a3': (
        StemSpec(filters=16, kernel_size=K13, strides=S12),
        MovinetBlockSpec(
            base_filters=16,
            expand_filters=(40, 40, 64, 40),
            kernel_sizes=(K15, K33, K33, K33),
            strides=(S12, S11, S11, S11)),
        MovinetBlockSpec(
            base_filters=48,
            expand_filters=(112, 144, 112, 112, 144, 144),
            kernel_sizes=(K33, K33, K33, K15, K33, K33),
            strides=(S12, S11, S11, S11, S11, S11)),
        MovinetBlockSpec(
            base_filters=80,
            expand_filters=(240, 152, 240, 192, 240),
            kernel_sizes=(K53, K33, K33, K33, K33),
            strides=(S12, S11, S11, S11, S11)),
        MovinetBlockSpec(
            base_filters=88,
            expand_filters=(264, 264, 264, 264, 160, 264, 264, 264),
            kernel_sizes=(K53, K33, K33, K33, K15, K33, K33, K33),
            strides=(S11, S11, S11, S11, S11, S11, S11, S11)),
        MovinetBlockSpec(
            base_filters=168,
            expand_filters=(560, 448, 448, 560, 560, 560, 448, 448, 560, 672),
            kernel_sizes=(K53, K15, K15, K15, K15, K33, K15, K15, K33, K13),
            strides=(S12, S11, S11, S11, S11, S11, S11, S11, S11, S11)),
        HeadSpec(project_filters=744, head_filters=2048),
    ),
    'a4': (
        StemSpec(filters=24, kernel_size=K13, strides=S12),
        MovinetBlockSpec(
            base_filters=24,
            expand_filters=(64, 64, 96, 64, 96, 64),
            kernel_sizes=(K15, K33, K33, K33, K33, K33),
            strides=(S12, S11, S11, S11, S11, S11)),
        MovinetBlockSpec(
            base_filters=56,
            expand_filters=(168, 168, 136, 136, 168, 168, 168, 136, 136),
            kernel_sizes=(K33, K33, K33, K33, K33, K33, K33, K15, K33),
            strides=(S12, S11, S11, S11, S11, S11, S11, S11, S11)),
        MovinetBlockSpec(
            base_filters=96,
            expand_filters=(320, 160, 320, 192, 320, 160, 320, 256, 320),
            kernel_sizes=(K53, K33, K33, K33, K33, K33, K33, K33, K33),
            strides=(S12, S11, S11, S11, S11, S11, S11, S11, S11)),
        MovinetBlockSpec(
            base_filters=96,
            expand_filters=(320, 320, 320, 320, 192, 320, 320, 192, 320, 320),
            kernel_sizes=(K53, K33, K33, K33, K15, K33, K33, K33, K33, K33),
            strides=(S11, S11, S11, S11, S11, S11, S11, S11, S11, S11)),
        MovinetBlockSpec(
            base_filters=192,
            expand_filters=(640, 512, 512, 640, 640, 640, 512, 512, 640, 768,
                            640, 640, 768),
            kernel_sizes=(K53, K15, K15, K15, K15, K33, K15, K15, K15, K15, K15,
                          K33, K33),
            strides=(S12, S11, S11, S11, S11, S11, S11, S11, S11, S11, S11, S11,
                     S11)),
        HeadSpec(project_filters=856, head_filters=2048),
    ),
    'a5': (
        StemSpec(filters=24, kernel_size=K13, strides=S12),
        MovinetBlockSpec(
            base_filters=24,
            expand_filters=(64, 64, 96, 64, 96, 64),
            kernel_sizes=(K15, K15, K33, K33, K33, K33),
            strides=(S12, S11, S11, S11, S11, S11)),
        MovinetBlockSpec(
            base_filters=64,
            expand_filters=(192, 152, 152, 152, 192, 192, 192, 152, 152, 192,
                            192),
            kernel_sizes=(K53, K33, K33, K33, K33, K33, K33, K33, K33, K33,
                          K33),
            strides=(S12, S11, S11, S11, S11, S11, S11, S11, S11, S11, S11)),
        MovinetBlockSpec(
            base_filters=112,
            expand_filters=(376, 224, 376, 376, 296, 376, 224, 376, 376, 296,
                            376, 376, 376),
            kernel_sizes=(K53, K33, K33, K33, K33, K33, K33, K33, K33, K33, K33,
                          K33, K33),
            strides=(S12, S11, S11, S11, S11, S11, S11, S11, S11, S11, S11, S11,
                     S11)),
        MovinetBlockSpec(
            base_filters=120,
            expand_filters=(376, 376, 376, 376, 224, 376, 376, 224, 376, 376,
                            376),
            kernel_sizes=(K53, K33, K33, K33, K15, K33, K33, K33, K33, K33,
                          K33),
            strides=(S11, S11, S11, S11, S11, S11, S11, S11, S11, S11, S11)),
        MovinetBlockSpec(
            base_filters=224,
            expand_filters=(744, 744, 600, 600, 744, 744, 744, 896, 600, 600,
                            896, 744, 744, 896, 600, 600, 744, 744),
            kernel_sizes=(K53, K33, K15, K15, K15, K15, K33, K15, K15, K15, K15,
                          K15, K33, K15, K15, K15, K15, K33),
            strides=(S12, S11, S11, S11, S11, S11, S11, S11, S11, S11, S11, S11,
                     S11, S11, S11, S11, S11, S11)),
        HeadSpec(project_filters=992, head_filters=2048),
    ),
    't0': (
        StemSpec(filters=8, kernel_size=K13, strides=S12),
        MovinetBlockSpec(
            base_filters=8,
            expand_filters=(16,),
            kernel_sizes=(K15,),
            strides=(S12,)),
        MovinetBlockSpec(
            base_filters=32,
            expand_filters=(72, 72),
            kernel_sizes=(K33, K15),
            strides=(S12, S11)),
        MovinetBlockSpec(
            base_filters=56,
            expand_filters=(112, 112, 112),
            kernel_sizes=(K53, K15, K33),
            strides=(S12, S11, S11)),
        MovinetBlockSpec(
            base_filters=56,
            expand_filters=(184, 184, 184, 184),
            kernel_sizes=(K53, K15, K33, K33),
            strides=(S11, S11, S11, S11)),
        MovinetBlockSpec(
            base_filters=104,
            expand_filters=(344, 344, 344, 344),
            kernel_sizes=(K53, K15, K15, K33),
            strides=(S12, S11, S11, S11)),
        HeadSpec(project_filters=240, head_filters=1024),
    ),
}


@tf.keras.utils.register_keras_serializable(package='Vision')
class Movinet(tf.keras.Model):
  """Class to build Movinet family model.

  Reference: https://arxiv.org/pdf/2103.11511.pdf
  """

  def __init__(self,
               model_id: str = 'a0',
               causal: bool = False,
               use_positional_encoding: bool = False,
               conv_type: str = '3d',
               input_specs: Optional[tf.keras.layers.InputSpec] = None,
               activation: str = 'swish',
               use_sync_bn: bool = True,
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               kernel_initializer: str = 'HeNormal',
               kernel_regularizer: Optional[str] = None,
               bias_regularizer: Optional[str] = None,
               stochastic_depth_drop_rate: float = 0.,
               **kwargs):
    """MoViNet initialization function.

    Args:
      model_id: name of MoViNet backbone model.
      causal: use causal mode, with CausalConv and CausalSE operations.
      use_positional_encoding:  if True, adds a positional encoding before
          temporal convolutions and the cumulative global average pooling
          layers.
      conv_type: '3d', '2plus1d', or '3d_2plus1d'. '3d' configures the network
        to use the default 3D convolution. '2plus1d' uses (2+1)D convolution
        with Conv2D operations and 2D reshaping (e.g., a 5x3x3 kernel becomes
        3x3 followed by 5x1 conv). '3d_2plus1d' uses (2+1)D convolution with
        Conv3D and no 2D reshaping (e.g., a 5x3x3 kernel becomes 1x3x3 followed
        by 5x1x1 conv).
      input_specs: the model input spec to use.
      activation: name of the activation function.
      use_sync_bn: if True, use synchronized batch normalization.
      norm_momentum: normalization momentum for the moving average.
      norm_epsilon: small float added to variance to avoid dividing by
        zero.
      kernel_initializer: kernel_initializer for convolutional layers.
      kernel_regularizer: tf.keras.regularizers.Regularizer object for Conv2D.
        Defaults to None.
      bias_regularizer: tf.keras.regularizers.Regularizer object for Conv2d.
        Defaults to None.
      stochastic_depth_drop_rate: the base rate for stochastic depth.
      **kwargs: keyword arguments to be passed.
    """
    block_specs = BLOCK_SPECS[model_id]
    if input_specs is None:
      input_specs = tf.keras.layers.InputSpec(shape=[None, None, None, None, 3])

    if conv_type not in ('3d', '2plus1d', '3d_2plus1d'):
      raise ValueError('Unknown conv type: {}'.format(conv_type))

    self._model_id = model_id
    self._block_specs = block_specs
    self._causal = causal
    self._use_positional_encoding = use_positional_encoding
    self._conv_type = conv_type
    self._input_specs = input_specs
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    if use_sync_bn:
      self._norm = tf.keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf.keras.layers.BatchNormalization
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate

    if not isinstance(block_specs[0], StemSpec):
      raise ValueError(
          'Expected first spec to be StemSpec, got {}'.format(block_specs[0]))
    if not isinstance(block_specs[-1], HeadSpec):
      raise ValueError(
          'Expected final spec to be HeadSpec, got {}'.format(block_specs[-1]))
    self._head_filters = block_specs[-1].head_filters

    if tf.keras.backend.image_data_format() == 'channels_last':
      bn_axis = -1
    else:
      bn_axis = 1

    # Build MoViNet backbone.
    inputs = tf.keras.Input(shape=input_specs.shape[1:], name='inputs')

    x = inputs
    states = {}
    endpoints = {}

    num_layers = sum(len(block.expand_filters) for block in block_specs
                     if isinstance(block, MovinetBlockSpec))
    stochastic_depth_idx = 1
    for block_idx, block in enumerate(block_specs):
      if isinstance(block, StemSpec):
        x, states = movinet_layers.Stem(
            block.filters,
            block.kernel_size,
            block.strides,
            conv_type=self._conv_type,
            causal=self._causal,
            activation=self._activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            batch_norm_layer=self._norm,
            batch_norm_momentum=self._norm_momentum,
            batch_norm_epsilon=self._norm_epsilon,
            name='stem')(x, states=states)
        endpoints['stem'] = x
      elif isinstance(block, MovinetBlockSpec):
        if not (len(block.expand_filters) == len(block.kernel_sizes) ==
                len(block.strides)):
          raise ValueError(
              'Lenths of block parameters differ: {}, {}, {}'.format(
                  len(block.expand_filters),
                  len(block.kernel_sizes),
                  len(block.strides)))
        params = list(zip(block.expand_filters,
                          block.kernel_sizes,
                          block.strides))
        for layer_idx, layer in enumerate(params):
          stochastic_depth_drop_rate = (
              self._stochastic_depth_drop_rate * stochastic_depth_idx /
              num_layers)
          expand_filters, kernel_size, strides = layer
          name = f'b{block_idx-1}/l{layer_idx}'
          x, states = movinet_layers.MovinetBlock(
              block.base_filters,
              expand_filters,
              kernel_size=kernel_size,
              strides=strides,
              causal=self._causal,
              activation=self._activation,
              stochastic_depth_drop_rate=stochastic_depth_drop_rate,
              conv_type=self._conv_type,
              use_positional_encoding=
              self._use_positional_encoding and self._causal,
              kernel_initializer=kernel_initializer,
              kernel_regularizer=kernel_regularizer,
              batch_norm_layer=self._norm,
              batch_norm_momentum=self._norm_momentum,
              batch_norm_epsilon=self._norm_epsilon,
              name=name)(x, states=states)
          endpoints[name] = x
          stochastic_depth_idx += 1
      elif isinstance(block, HeadSpec):
        x, states = movinet_layers.Head(
            project_filters=block.project_filters,
            conv_type=self._conv_type,
            activation=self._activation,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            batch_norm_layer=self._norm,
            batch_norm_momentum=self._norm_momentum,
            batch_norm_epsilon=self._norm_epsilon)(x, states=states)
        endpoints['head'] = x
      else:
        raise ValueError('Unknown block type {}'.format(block))

    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints}

    inputs = {
        'image': inputs,
        'states': {
            name: tf.keras.Input(shape=state.shape[1:], name=f'states/{name}')
            for name, state in states.items()
        },
    }
    outputs = (endpoints, states)

    super(Movinet, self).__init__(inputs=inputs, outputs=outputs, **kwargs)

  def get_config(self):
    config_dict = {
        'model_id': self._model_id,
        'causal': self._causal,
        'use_positional_encoding': self._use_positional_encoding,
        'conv_type': self._conv_type,
        'activation': self._activation,
        'use_sync_bn': self._use_sync_bn,
        'norm_momentum': self._norm_momentum,
        'norm_epsilon': self._norm_epsilon,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
    }
    return config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self):
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@factory.register_backbone_builder('movinet')
def build_movinet(
    input_specs: tf.keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
  """Builds MoViNet backbone from a config."""
  l2_regularizer = l2_regularizer or tf.keras.regularizers.L2(1.5e-5)

  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'movinet', ('Inconsistent backbone type '
                                      f'{backbone_type}')

  return Movinet(
      model_id=backbone_cfg.model_id,
      causal=backbone_cfg.causal,
      use_positional_encoding=backbone_cfg.use_positional_encoding,
      conv_type=backbone_cfg.conv_type,
      input_specs=input_specs,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer,
      stochastic_depth_drop_rate=backbone_cfg.stochastic_depth_drop_rate)
