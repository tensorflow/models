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
import dataclasses
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

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

# Type for a state container (map)
TensorMap = Mapping[str, tf.Tensor]


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
               se_type: str = '3d',
               input_specs: Optional[tf.keras.layers.InputSpec] = None,
               activation: str = 'swish',
               gating_activation: str = 'sigmoid',
               use_sync_bn: bool = True,
               norm_momentum: float = 0.99,
               norm_epsilon: float = 0.001,
               kernel_initializer: str = 'HeNormal',
               kernel_regularizer: Optional[str] = None,
               bias_regularizer: Optional[str] = None,
               stochastic_depth_drop_rate: float = 0.,
               use_external_states: bool = False,
               output_states: bool = True,
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
      se_type: '3d', '2d', or '2plus3d'. '3d' uses the default 3D
          spatiotemporal global average pooling for squeeze excitation. '2d'
          uses 2D spatial global average pooling  on each frame. '2plus3d'
          concatenates both 3D and 2D global average pooling.
      input_specs: the model input spec to use.
      activation: name of the main activation function.
      gating_activation: gating activation to use in squeeze excitation layers.
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
      use_external_states: if True, expects states to be passed as additional
        input.
      output_states: if True, output intermediate states that can be used to run
          the model in streaming mode. Inputting the output states of the
          previous input clip with the current input clip will utilize a stream
          buffer for streaming video.
      **kwargs: keyword arguments to be passed.
    """
    block_specs = BLOCK_SPECS[model_id]
    if input_specs is None:
      input_specs = tf.keras.layers.InputSpec(shape=[None, None, None, None, 3])

    if conv_type not in ('3d', '2plus1d', '3d_2plus1d'):
      raise ValueError('Unknown conv type: {}'.format(conv_type))
    if se_type not in ('3d', '2d', '2plus3d'):
      raise ValueError('Unknown squeeze excitation type: {}'.format(se_type))

    self._model_id = model_id
    self._block_specs = block_specs
    self._causal = causal
    self._use_positional_encoding = use_positional_encoding
    self._conv_type = conv_type
    self._se_type = se_type
    self._input_specs = input_specs
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._gating_activation = gating_activation
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
    self._use_external_states = use_external_states
    self._output_states = output_states

    if self._use_external_states and not self._causal:
      raise ValueError('External states should be used with causal mode.')
    if not isinstance(block_specs[0], StemSpec):
      raise ValueError(
          'Expected first spec to be StemSpec, got {}'.format(block_specs[0]))
    if not isinstance(block_specs[-1], HeadSpec):
      raise ValueError(
          'Expected final spec to be HeadSpec, got {}'.format(block_specs[-1]))
    self._head_filters = block_specs[-1].head_filters

    state_specs = None
    if use_external_states:
      self._set_dtype_policy(input_specs.dtype)
      state_specs = self.initial_state_specs(input_specs.shape)

    inputs, outputs = self._build_network(input_specs, state_specs=state_specs)

    super(Movinet, self).__init__(inputs=inputs, outputs=outputs, **kwargs)

    self._state_specs = state_specs

  def _build_network(
      self,
      input_specs: tf.keras.layers.InputSpec,
      state_specs: Optional[Mapping[str, tf.keras.layers.InputSpec]] = None,
  ) -> Tuple[TensorMap, Union[TensorMap, Tuple[TensorMap, TensorMap]]]:
    """Builds the model network.

    Args:
      input_specs: the model input spec to use.
      state_specs: a dict mapping a state name to the corresponding state spec.
        State names should match with the `state` input/output dict.

    Returns:
      Inputs and outputs as a tuple. Inputs are expected to be a dict with
      base input and states. Outputs are expected to be a dict of endpoints
      and (optional) output states.
    """
    state_specs = state_specs if state_specs is not None else {}

    image_input = tf.keras.Input(shape=input_specs.shape[1:], name='inputs')

    states = {
        name: tf.keras.Input(shape=spec.shape[1:], dtype=spec.dtype, name=name)
        for name, spec in state_specs.items()
    }

    inputs = {**states, 'image': image_input}
    endpoints = {}

    x = image_input

    num_layers = sum(
        len(block.expand_filters)
        for block in self._block_specs
        if isinstance(block, MovinetBlockSpec))
    stochastic_depth_idx = 1
    for block_idx, block in enumerate(self._block_specs):
      if isinstance(block, StemSpec):
        layer_obj = movinet_layers.Stem(
            block.filters,
            block.kernel_size,
            block.strides,
            conv_type=self._conv_type,
            causal=self._causal,
            activation=self._activation,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            batch_norm_layer=self._norm,
            batch_norm_momentum=self._norm_momentum,
            batch_norm_epsilon=self._norm_epsilon,
            state_prefix='state_stem',
            name='stem')
        x, states = layer_obj(x, states=states)
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
          name = f'block{block_idx-1}_layer{layer_idx}'
          layer_obj = movinet_layers.MovinetBlock(
              block.base_filters,
              expand_filters,
              kernel_size=kernel_size,
              strides=strides,
              causal=self._causal,
              activation=self._activation,
              gating_activation=self._gating_activation,
              stochastic_depth_drop_rate=stochastic_depth_drop_rate,
              conv_type=self._conv_type,
              se_type=self._se_type,
              use_positional_encoding=
              self._use_positional_encoding and self._causal,
              kernel_initializer=self._kernel_initializer,
              kernel_regularizer=self._kernel_regularizer,
              batch_norm_layer=self._norm,
              batch_norm_momentum=self._norm_momentum,
              batch_norm_epsilon=self._norm_epsilon,
              state_prefix=f'state_{name}',
              name=name)
          x, states = layer_obj(x, states=states)

          endpoints[name] = x
          stochastic_depth_idx += 1
      elif isinstance(block, HeadSpec):
        layer_obj = movinet_layers.Head(
            project_filters=block.project_filters,
            conv_type=self._conv_type,
            activation=self._activation,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            batch_norm_layer=self._norm,
            batch_norm_momentum=self._norm_momentum,
            batch_norm_epsilon=self._norm_epsilon,
            state_prefix='state_head',
            name='head')
        x, states = layer_obj(x, states=states)
        endpoints['head'] = x
      else:
        raise ValueError('Unknown block type {}'.format(block))

    outputs = (endpoints, states) if self._output_states else endpoints

    return inputs, outputs

  def _get_initial_state_shapes(
      self,
      block_specs: Sequence[BlockSpec],
      input_shape: Union[Sequence[int], tf.Tensor],
      use_positional_encoding: bool = False) -> Dict[str, Sequence[int]]:
    """Generates names and shapes for all input states.

    Args:
      block_specs: sequence of specs used for creating a model.
      input_shape: the expected 5D shape of the image input.
      use_positional_encoding: whether the model will use positional encoding.

    Returns:
      A dict mapping state names to state shapes.
    """
    def divide_resolution(shape, num_downsamples):
      """Downsamples the dimension to calculate strided convolution shape."""
      if shape is None:
        return None
      if isinstance(shape, tf.Tensor):
        # Avoid using div and ceil to support tf lite
        shape = tf.cast(shape, tf.float32)
        resolution_divisor = 2 ** num_downsamples
        resolution_multiplier = 0.5 ** num_downsamples
        shape = ((shape + resolution_divisor - 1) * resolution_multiplier)
        return tf.cast(shape, tf.int32)
      else:
        resolution_divisor = 2 ** num_downsamples
        return math.ceil(shape / resolution_divisor)

    states = {}
    num_downsamples = 0

    for block_idx, block in enumerate(block_specs):
      if isinstance(block, StemSpec):
        if block.kernel_size[0] > 1:
          states['state_stem_stream_buffer'] = (
              input_shape[0],
              input_shape[1],
              divide_resolution(input_shape[2], num_downsamples),
              divide_resolution(input_shape[3], num_downsamples),
              block.filters,
          )
        num_downsamples += 1
      elif isinstance(block, MovinetBlockSpec):
        block_idx -= 1
        params = list(zip(
            block.expand_filters,
            block.kernel_sizes,
            block.strides))
        for layer_idx, layer in enumerate(params):
          expand_filters, kernel_size, strides = layer

          # If we use a 2D kernel, we apply spatial downsampling
          # before the buffer.
          if (tuple(strides[1:3]) != (1, 1) and
              self._conv_type in ['2plus1d', '3d_2plus1d']):
            num_downsamples += 1

          prefix = f'state_block{block_idx}_layer{layer_idx}'

          if kernel_size[0] > 1:
            states[f'{prefix}_stream_buffer'] = (
                input_shape[0],
                kernel_size[0] - 1,
                divide_resolution(input_shape[2], num_downsamples),
                divide_resolution(input_shape[3], num_downsamples),
                expand_filters,
            )

          states[f'{prefix}_pool_buffer'] = (
              input_shape[0], 1, 1, 1, expand_filters,
          )
          states[f'{prefix}_pool_frame_count'] = (1,)

          if use_positional_encoding:
            name = f'{prefix}_pos_enc_frame_count'
            states[name] = (1,)

          if strides[1] != strides[2]:
            raise ValueError('Strides must match in the spatial dimensions, '
                             'got {}'.format(strides))

          # If we use a 3D kernel, we apply spatial downsampling
          # after the buffer.
          if (tuple(strides[1:3]) != (1, 1) and
              self._conv_type not in ['2plus1d', '3d_2plus1d']):
            num_downsamples += 1
      elif isinstance(block, HeadSpec):
        states['state_head_pool_buffer'] = (
            input_shape[0], 1, 1, 1, block.project_filters,
        )
        states['state_head_pool_frame_count'] = (1,)

    return states

  def _get_state_dtype(self, name: str) -> str:
    """Returns the dtype associated with a state."""
    if 'frame_count' in name:
      return 'int32'
    return self.dtype

  def initial_state_specs(
      self, input_shape: Sequence[int]) -> Dict[str, tf.keras.layers.InputSpec]:
    """Creates a mapping of state name to InputSpec from the input shape."""
    state_shapes = self._get_initial_state_shapes(
        self._block_specs,
        input_shape,
        use_positional_encoding=self._use_positional_encoding)

    return {
        name: tf.keras.layers.InputSpec(
            shape=shape, dtype=self._get_state_dtype(name))
        for name, shape in state_shapes.items()
    }

  def init_states(self, input_shape: Sequence[int]) -> Dict[str, tf.Tensor]:
    """Returns initial states for the first call in steaming mode."""
    state_shapes = self._get_initial_state_shapes(
        self._block_specs,
        input_shape,
        use_positional_encoding=self._use_positional_encoding)

    states = {
        name: tf.zeros(shape, dtype=self._get_state_dtype(name))
        for name, shape in state_shapes.items()
    }
    return states

  @property
  def use_external_states(self) -> bool:
    """Whether this model is expecting input states as additional input."""
    return self._use_external_states

  @property
  def head_filters(self):
    """The number of filters expected to be in the head classifer layer."""
    return self._head_filters

  @property
  def conv_type(self):
    """The expected convolution type (see __init__ for more details)."""
    return self._conv_type

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
        'use_external_states': self._use_external_states,
        'output_states': self._output_states,
    }
    return config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


@factory.register_backbone_builder('movinet')
def build_movinet(
    input_specs: tf.keras.layers.InputSpec,
    backbone_config: hyperparams.Config,
    norm_activation_config: hyperparams.Config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds MoViNet backbone from a config."""
  backbone_type = backbone_config.type
  backbone_cfg = backbone_config.get()
  assert backbone_type == 'movinet', ('Inconsistent backbone type '
                                      f'{backbone_type}')

  return Movinet(
      model_id=backbone_cfg.model_id,
      causal=backbone_cfg.causal,
      use_positional_encoding=backbone_cfg.use_positional_encoding,
      conv_type=backbone_cfg.conv_type,
      se_type=backbone_cfg.se_type,
      input_specs=input_specs,
      activation=backbone_cfg.activation,
      gating_activation=backbone_cfg.gating_activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer,
      stochastic_depth_drop_rate=backbone_cfg.stochastic_depth_drop_rate,
      use_external_states=backbone_cfg.use_external_states)
