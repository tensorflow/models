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

"""Contains common building blocks for neural networks."""

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from absl import logging
import tensorflow as tf

from official.modeling import tf_utils
from official.vision.ops import spatial_transform_ops


# Type annotations.
States = Dict[str, tf.Tensor]
Activation = Union[str, Callable]


def make_divisible(value: float,
                   divisor: int,
                   min_value: Optional[float] = None,
                   round_down_protect: bool = True,
                   ) -> int:
  """This is to ensure that all layers have channels that are divisible by 8.

  Args:
    value: A `float` of original value.
    divisor: An `int` of the divisor that need to be checked upon.
    min_value: A `float` of  minimum value threshold.
    round_down_protect: A `bool` indicating whether round down more than 10%
      will be allowed.

  Returns:
    The adjusted value in `int` that is divisible against divisor.
  """
  if min_value is None:
    min_value = divisor
  new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if round_down_protect and new_value < 0.9 * value:
    new_value += divisor
  return int(new_value)


def round_filters(filters: int,
                  multiplier: float,
                  divisor: int = 8,
                  min_depth: Optional[int] = None,
                  round_down_protect: bool = True,
                  skip: bool = False) -> int:
  """Rounds number of filters based on width multiplier."""
  orig_f = filters
  if skip or not multiplier:
    return filters

  new_filters = make_divisible(value=filters * multiplier,
                               divisor=divisor,
                               min_value=min_depth,
                               round_down_protect=round_down_protect)

  logging.info('round_filter input=%s output=%s', orig_f, new_filters)
  return int(new_filters)


def get_padding_for_kernel_size(kernel_size):
  """Compute padding size given kernel size."""
  if kernel_size == 7:
    return (3, 3)
  elif kernel_size == 3:
    return (1, 1)
  else:
    raise ValueError('Padding for kernel size {} not known.'.format(
        kernel_size))


@tf.keras.utils.register_keras_serializable(package='Vision')
class SqueezeExcitation(tf.keras.layers.Layer):
  """Creates a squeeze and excitation layer."""

  def __init__(self,
               in_filters,
               out_filters,
               se_ratio,
               divisible_by=1,
               use_3d_input=False,
               kernel_initializer='VarianceScaling',
               kernel_regularizer=None,
               bias_regularizer=None,
               activation='relu',
               gating_activation='sigmoid',
               round_down_protect=True,
               **kwargs):
    """Initializes a squeeze and excitation layer.

    Args:
      in_filters: An `int` number of filters of the input tensor.
      out_filters: An `int` number of filters of the output tensor.
      se_ratio: A `float` or None. If not None, se ratio for the squeeze and
        excitation layer.
      divisible_by: An `int` that ensures all inner dimensions are divisible by
        this number.
      use_3d_input: A `bool` of whether input is 2D or 3D image.
      kernel_initializer: A `str` of kernel_initializer for convolutional
        layers.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default to None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2d.
        Default to None.
      activation: A `str` name of the activation function.
      gating_activation: A `str` name of the activation function for final
        gating function.
      round_down_protect: A `bool` of whether round down more than 10% will be
        allowed.
      **kwargs: Additional keyword arguments to be passed.
    """
    super(SqueezeExcitation, self).__init__(**kwargs)

    self._in_filters = in_filters
    self._out_filters = out_filters
    self._se_ratio = se_ratio
    self._divisible_by = divisible_by
    self._round_down_protect = round_down_protect
    self._use_3d_input = use_3d_input
    self._activation = activation
    self._gating_activation = gating_activation
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    if tf.keras.backend.image_data_format() == 'channels_last':
      if not use_3d_input:
        self._spatial_axis = [1, 2]
      else:
        self._spatial_axis = [1, 2, 3]
    else:
      if not use_3d_input:
        self._spatial_axis = [2, 3]
      else:
        self._spatial_axis = [2, 3, 4]
    self._activation_fn = tf_utils.get_activation(activation)
    self._gating_activation_fn = tf_utils.get_activation(gating_activation)

  def build(self, input_shape):
    num_reduced_filters = make_divisible(
        max(1, int(self._in_filters * self._se_ratio)),
        divisor=self._divisible_by,
        round_down_protect=self._round_down_protect)

    self._se_reduce = tf.keras.layers.Conv2D(
        filters=num_reduced_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)

    self._se_expand = tf.keras.layers.Conv2D(
        filters=self._out_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        use_bias=True,
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer)

    super(SqueezeExcitation, self).build(input_shape)

  def get_config(self):
    config = {
        'in_filters': self._in_filters,
        'out_filters': self._out_filters,
        'se_ratio': self._se_ratio,
        'divisible_by': self._divisible_by,
        'use_3d_input': self._use_3d_input,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activation': self._activation,
        'gating_activation': self._gating_activation,
        'round_down_protect': self._round_down_protect,
    }
    base_config = super(SqueezeExcitation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    x = tf.reduce_mean(inputs, self._spatial_axis, keepdims=True)
    x = self._activation_fn(self._se_reduce(x))
    x = self._gating_activation_fn(self._se_expand(x))
    return x * inputs


def get_stochastic_depth_rate(init_rate, i, n):
  """Get drop connect rate for the ith block.

  Args:
    init_rate: A `float` of initial drop rate.
    i: An `int` of order of the current block.
    n: An `int` total number of blocks.

  Returns:
    Drop rate of the ith block.
  """
  if init_rate is not None:
    if init_rate < 0 or init_rate > 1:
      raise ValueError('Initial drop rate must be within 0 and 1.')
    rate = init_rate * float(i) / n
  else:
    rate = None
  return rate


@tf.keras.utils.register_keras_serializable(package='Vision')
class StochasticDepth(tf.keras.layers.Layer):
  """Creates a stochastic depth layer."""

  def __init__(self, stochastic_depth_drop_rate, **kwargs):
    """Initializes a stochastic depth layer.

    Args:
      stochastic_depth_drop_rate: A `float` of drop rate.
      **kwargs: Additional keyword arguments to be passed.

    Returns:
      A output `tf.Tensor` of which should have the same shape as input.
    """
    super(StochasticDepth, self).__init__(**kwargs)
    self._drop_rate = stochastic_depth_drop_rate

  def get_config(self):
    config = {'stochastic_depth_drop_rate': self._drop_rate}
    base_config = super(StochasticDepth, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs, training=None):
    if training is None:
      training = tf.keras.backend.learning_phase()
    if not training or self._drop_rate is None or self._drop_rate == 0:
      return inputs

    keep_prob = 1.0 - self._drop_rate
    batch_size = tf.shape(inputs)[0]
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(
        [batch_size] + [1] * (inputs.shape.rank - 1), dtype=inputs.dtype)
    binary_tensor = tf.floor(random_tensor)
    output = tf.math.divide(inputs, keep_prob) * binary_tensor
    return output


@tf.keras.utils.register_keras_serializable(package='Vision')
def pyramid_feature_fusion(inputs, target_level):
  """Fuses all feature maps in the feature pyramid at the target level.

  Args:
    inputs: A dictionary containing the feature pyramid. The size of the input
      tensor needs to be fixed.
    target_level: An `int` of the target feature level for feature fusion.

  Returns:
    A `float` `tf.Tensor` of shape [batch_size, feature_height, feature_width,
      feature_channel].
  """
  # Convert keys to int.
  pyramid_feats = {int(k): v for k, v in inputs.items()}
  min_level = min(pyramid_feats.keys())
  max_level = max(pyramid_feats.keys())
  resampled_feats = []

  for l in range(min_level, max_level + 1):
    if l == target_level:
      resampled_feats.append(pyramid_feats[l])
    else:
      feat = pyramid_feats[l]
      target_size = list(feat.shape[1:3])
      target_size[0] *= 2**(l - target_level)
      target_size[1] *= 2**(l - target_level)
      # Casts feat to float32 so the resize op can be run on TPU.
      feat = tf.cast(feat, tf.float32)
      feat = tf.image.resize(
          feat, size=target_size, method=tf.image.ResizeMethod.BILINEAR)
      # Casts it back to be compatible with the rest opetations.
      feat = tf.cast(feat, pyramid_feats[l].dtype)
      resampled_feats.append(feat)

  return tf.math.add_n(resampled_feats)


class PanopticFPNFusion(tf.keras.Model):
  """Creates a Panoptic FPN feature Fusion layer.

  This implements feature fusion for semantic segmentation head from the paper:
  Alexander Kirillov, Ross Girshick, Kaiming He and Piotr Dollar.
  Panoptic Feature Pyramid Networks.
  (https://arxiv.org/pdf/1901.02446.pdf)
  """

  def __init__(
      self,
      min_level: int = 2,
      max_level: int = 5,
      target_level: int = 2,
      num_filters: int = 128,
      num_fpn_filters: int = 256,
      activation: str = 'relu',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs):

    """Initializes panoptic FPN feature fusion layer.

    Args:
      min_level: An `int` of minimum level to use in feature fusion.
      max_level: An `int` of maximum level to use in feature fusion.
      target_level: An `int` of the target feature level for feature fusion.
      num_filters: An `int` number of filters in conv2d layers.
      num_fpn_filters: An `int` number of filters in the FPN outputs
      activation: A `str` name of the activation function.
      kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
        Conv2D. Default is None.
      bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
      **kwargs: Additional keyword arguments to be passed.
    Returns:
      A `float` `tf.Tensor` of shape [batch_size, feature_height, feature_width,
        feature_channel].
    """
    if target_level > max_level:
      raise ValueError('target_level should be less than max_level')

    self._config_dict = {
        'min_level': min_level,
        'max_level': max_level,
        'target_level': target_level,
        'num_filters': num_filters,
        'num_fpn_filters': num_fpn_filters,
        'activation': activation,
        'kernel_regularizer': kernel_regularizer,
        'bias_regularizer': bias_regularizer,
    }
    norm = tf.keras.layers.GroupNormalization
    conv2d = tf.keras.layers.Conv2D
    activation_fn = tf_utils.get_activation(activation)
    if tf.keras.backend.image_data_format() == 'channels_last':
      norm_axis = -1
    else:
      norm_axis = 1
    inputs = self._build_inputs(num_fpn_filters, min_level, max_level)

    upscaled_features = []
    for level in range(min_level, max_level + 1):
      num_conv_layers = max(1, level - target_level)
      x = inputs[str(level)]
      for i in range(num_conv_layers):
        x = conv2d(
            filters=num_filters,
            kernel_size=3,
            padding='same',
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer)(x)
        x = norm(groups=32, axis=norm_axis)(x)
        x = activation_fn(x)
        if level != target_level:
          x = spatial_transform_ops.nearest_upsampling(x, scale=2)
      upscaled_features.append(x)

    fused_features = tf.math.add_n(upscaled_features)
    self._output_specs = {str(target_level): fused_features.get_shape()}

    super(PanopticFPNFusion, self).__init__(
        inputs=inputs, outputs=fused_features, **kwargs)

  def _build_inputs(self, num_filters: int,
                    min_level: int, max_level: int):
    inputs = {}
    for level in range(min_level, max_level + 1):
      inputs[str(level)] = tf.keras.Input(shape=[None, None, num_filters])
    return inputs

  def get_config(self) -> Mapping[str, Any]:
    return self._config_dict

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  @property
  def output_specs(self) -> Mapping[str, tf.TensorShape]:
    """A dict of {level: TensorShape} pairs for the model output."""
    return self._output_specs


@tf.keras.utils.register_keras_serializable(package='Vision')
class Scale(tf.keras.layers.Layer):
  """Scales the input by a trainable scalar weight.

  This is useful for applying ReZero to layers, which improves convergence
  speed. This implements the paper:
  ReZero is All You Need: Fast Convergence at Large Depth.
  (https://arxiv.org/pdf/2003.04887.pdf).
  """

  def __init__(
      self,
      initializer: tf.keras.initializers.Initializer = 'ones',
      regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      **kwargs):
    """Initializes a scale layer.

    Args:
      initializer: A `str` of initializer for the scalar weight.
      regularizer: A `tf.keras.regularizers.Regularizer` for the scalar weight.
      **kwargs: Additional keyword arguments to be passed to this layer.

    Returns:
      An `tf.Tensor` of which should have the same shape as input.
    """
    super(Scale, self).__init__(**kwargs)

    self._initializer = initializer
    self._regularizer = regularizer

    self._scale = self.add_weight(
        name='scale',
        shape=[],
        dtype=self.dtype,
        initializer=self._initializer,
        regularizer=self._regularizer,
        trainable=True)

  def get_config(self):
    """Returns a dictionary containing the config used for initialization."""
    config = {
        'initializer': self._initializer,
        'regularizer': self._regularizer,
    }
    base_config = super(Scale, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    """Calls the layer with the given inputs."""
    scale = tf.cast(self._scale, inputs.dtype)
    return scale * inputs


@tf.keras.utils.register_keras_serializable(package='Vision')
class TemporalSoftmaxPool(tf.keras.layers.Layer):
  """Creates a network layer corresponding to temporal softmax pooling.

  This is useful for multi-class logits (used in e.g., Charades). Modified from
  AssembleNet Charades evaluation from:

  Michael S. Ryoo, AJ Piergiovanni, Mingxing Tan, Anelia Angelova.
  AssembleNet: Searching for Multi-Stream Neural Connectivity in Video
  Architectures.
  (https://arxiv.org/pdf/1905.13209.pdf).
  """

  def call(self, inputs):
    """Calls the layer with the given inputs."""
    assert inputs.shape.rank in (3, 4, 5)
    frames = tf.shape(inputs)[1]
    pre_logits = inputs / tf.sqrt(tf.cast(frames, inputs.dtype))
    activations = tf.nn.softmax(pre_logits, axis=1)
    outputs = inputs * activations
    return outputs


@tf.keras.utils.register_keras_serializable(package='Vision')
class PositionalEncoding(tf.keras.layers.Layer):
  """Creates a network layer that adds a sinusoidal positional encoding.

  Positional encoding is incremented across frames, and is added to the input.
  The positional encoding is first weighted at 0 so that the network can choose
  to ignore it. This implements:

  Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
  Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.
  Attention Is All You Need.
  (https://arxiv.org/pdf/1706.03762.pdf).
  """

  def __init__(self,
               initializer: tf.keras.initializers.Initializer = 'zeros',
               cache_encoding: bool = False,
               state_prefix: Optional[str] = None,
               **kwargs):
    """Initializes positional encoding.

    Args:
      initializer: A `str` of initializer for weighting the positional encoding.
      cache_encoding: A `bool`. If True, cache the positional encoding tensor
        after calling build. Otherwise, rebuild the tensor for every call.
        Setting this to False can be useful when we want to input a variable
        number of frames, so the positional encoding tensor can change shape.
      state_prefix: a prefix string to identify states.
      **kwargs: Additional keyword arguments to be passed to this layer.

    Returns:
      A `tf.Tensor` of which should have the same shape as input.
    """
    super(PositionalEncoding, self).__init__(**kwargs)
    self._initializer = initializer
    self._cache_encoding = cache_encoding
    self._pos_encoding = None
    self._rezero = Scale(initializer=initializer, name='rezero')
    state_prefix = state_prefix if state_prefix is not None else ''
    self._state_prefix = state_prefix
    self._frame_count_name = f'{state_prefix}_pos_enc_frame_count'

  def get_config(self):
    """Returns a dictionary containing the config used for initialization."""
    config = {
        'initializer': self._initializer,
        'cache_encoding': self._cache_encoding,
        'state_prefix': self._state_prefix,
    }
    base_config = super(PositionalEncoding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _positional_encoding(self,
                           num_positions: Union[int, tf.Tensor],
                           hidden_size: Union[int, tf.Tensor],
                           start_position: Union[int, tf.Tensor] = 0,
                           dtype: str = 'float32') -> tf.Tensor:
    """Creates a sequence of sinusoidal positional encoding vectors.

    Args:
      num_positions: the total number of positions (frames).
      hidden_size: the number of channels used for the hidden vectors.
      start_position: the start position.
      dtype: the dtype of the output tensor.

    Returns:
      The positional encoding tensor with shape [num_positions, hidden_size].
    """
    if isinstance(start_position, tf.Tensor) and start_position.shape.rank == 1:
      start_position = start_position[0]

    # Calling `tf.range` with `dtype=tf.bfloat16` results in an error,
    # so we cast afterward.
    positions = tf.range(start_position, start_position + num_positions)
    positions = tf.cast(positions, dtype)[:, tf.newaxis]
    idx = tf.range(hidden_size)[tf.newaxis, :]

    power = tf.cast(2 * (idx // 2), dtype)
    power /= tf.cast(hidden_size, dtype)
    angles = 1. / tf.math.pow(10_000., power)
    radians = positions * angles

    sin = tf.math.sin(radians[:, 0::2])
    cos = tf.math.cos(radians[:, 1::2])
    pos_encoding = tf.concat([sin, cos], axis=-1)

    return pos_encoding

  def _get_pos_encoding(self,
                        input_shape: tf.Tensor,
                        frame_count: int = 0) -> tf.Tensor:
    """Calculates the positional encoding from the input shape.

    Args:
      input_shape: the shape of the input.
      frame_count: a count of frames that indicates the index of the first
        frame.

    Returns:
      The positional encoding tensor with shape [num_positions, hidden_size].

    """
    frames = input_shape[1]
    channels = input_shape[-1]
    pos_encoding = self._positional_encoding(
        frames, channels, start_position=frame_count, dtype=self.dtype)
    pos_encoding = tf.reshape(pos_encoding, [1, frames, 1, 1, channels])
    return pos_encoding

  def build(self, input_shape):
    """Builds the layer with the given input shape.

    Args:
      input_shape: The input shape.

    Raises:
      ValueError: If using 'channels_first' data format.
    """
    if tf.keras.backend.image_data_format() == 'channels_first':
      raise ValueError('"channels_first" mode is unsupported.')

    if self._cache_encoding:
      self._pos_encoding = self._get_pos_encoding(input_shape)

    super(PositionalEncoding, self).build(input_shape)

  def call(
      self,
      inputs: tf.Tensor,
      states: Optional[States] = None,
      output_states: bool = True,
  ) -> Union[tf.Tensor, Tuple[tf.Tensor, States]]:
    """Calls the layer with the given inputs.

    Args:
      inputs: An input `tf.Tensor`.
      states: A `dict` of states such that, if any of the keys match for this
        layer, will overwrite the contents of the buffer(s). Expected keys
        include `state_prefix + '_pos_enc_frame_count'`.
      output_states: A `bool`. If True, returns the output tensor and output
        states. Returns just the output tensor otherwise.

    Returns:
      An output `tf.Tensor` (and optionally the states if `output_states=True`).

    Raises:
      ValueError: If using 'channels_first' data format.
    """
    states = dict(states) if states is not None else {}

    # Keep a count of frames encountered across input iterations in
    # num_frames to be able to accurately update the positional encoding.
    num_frames = tf.shape(inputs)[1]
    frame_count = tf.cast(states.get(self._frame_count_name, [0]), tf.int32)
    states[self._frame_count_name] = frame_count + num_frames

    if self._cache_encoding:
      pos_encoding = self._pos_encoding
    else:
      pos_encoding = self._get_pos_encoding(
          tf.shape(inputs), frame_count=frame_count)
    pos_encoding = tf.cast(pos_encoding, inputs.dtype)
    pos_encoding = self._rezero(pos_encoding)
    outputs = inputs + pos_encoding

    return (outputs, states) if output_states else outputs


@tf.keras.utils.register_keras_serializable(package='Vision')
class GlobalAveragePool3D(tf.keras.layers.Layer):
  """Creates a global average pooling layer with causal mode.

  Implements causal mode, which runs a cumulative sum (with `tf.cumsum`) across
  frames in the time dimension, allowing the use of a stream buffer. Sums any
  valid input state with the current input to allow state to accumulate over
  several iterations.
  """

  def __init__(self,
               keepdims: bool = False,
               causal: bool = False,
               state_prefix: Optional[str] = None,
               **kwargs):
    """Initializes a global average pool layer.

    Args:
      keepdims: A `bool`. If True, keep the averaged dimensions.
      causal: A `bool` of whether to run in causal mode with a cumulative sum
        across frames.
      state_prefix: a prefix string to identify states.
      **kwargs: Additional keyword arguments to be passed to this layer.

    Returns:
      An output `tf.Tensor`.
    """
    super(GlobalAveragePool3D, self).__init__(**kwargs)

    self._keepdims = keepdims
    self._causal = causal
    state_prefix = state_prefix if state_prefix is not None else ''
    self._state_prefix = state_prefix

    self._state_name = f'{state_prefix}_pool_buffer'
    self._frame_count_name = f'{state_prefix}_pool_frame_count'

  def get_config(self):
    """Returns a dictionary containing the config used for initialization."""
    config = {
        'keepdims': self._keepdims,
        'causal': self._causal,
        'state_prefix': self._state_prefix,
    }
    base_config = super(GlobalAveragePool3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self,
           inputs: tf.Tensor,
           states: Optional[States] = None,
           output_states: bool = False
           ) -> Union[tf.Tensor, Tuple[tf.Tensor, States]]:
    """Calls the layer with the given inputs.

    Args:
      inputs: An input `tf.Tensor`.
      states: A `dict` of states such that, if any of the keys match for this
        layer, will overwrite the contents of the buffer(s).
        Expected keys include `state_prefix + '__pool_buffer'` and
        `state_prefix + '__pool_frame_count'`.
      output_states: A `bool`. If True, returns the output tensor and output
        states. Returns just the output tensor otherwise.

    Returns:
      An output `tf.Tensor` (and optionally the states if `output_states=True`).
      If `causal=True`, the output tensor will have shape
      `[batch_size, num_frames, 1, 1, channels]` if `keepdims=True`. We keep
      the frame dimension in this case to simulate a cumulative global average
      as if we are inputting one frame at a time. If `causal=False`, the output
      is equivalent to `tf.keras.layers.GlobalAveragePooling3D` with shape
      `[batch_size, 1, 1, 1, channels]` if `keepdims=True` (plus the optional
      buffer stored in `states`).

    Raises:
      ValueError: If using 'channels_first' data format.
    """
    states = dict(states) if states is not None else {}

    if tf.keras.backend.image_data_format() == 'channels_first':
      raise ValueError('"channels_first" mode is unsupported.')

    # Shape: [batch_size, 1, 1, 1, channels]
    buffer = states.get(self._state_name, None)
    if buffer is None:
      buffer = tf.zeros_like(inputs[:, :1, :1, :1], dtype=inputs.dtype)
      states[self._state_name] = buffer

    # Keep a count of frames encountered across input iterations in
    # num_frames to be able to accurately take a cumulative average across
    # all frames when running in streaming mode
    num_frames = tf.shape(inputs)[1]
    frame_count = states.get(self._frame_count_name, tf.constant([0]))
    frame_count = tf.cast(frame_count, tf.int32)
    states[self._frame_count_name] = frame_count + num_frames

    if self._causal:
      # Take a mean of spatial dimensions to make computation more efficient.
      x = tf.reduce_mean(inputs, axis=[2, 3], keepdims=True)
      x = tf.cumsum(x, axis=1)
      x = x + buffer

      # The last frame will be the value of the next state
      # Shape: [batch_size, 1, 1, 1, channels]
      states[self._state_name] = x[:, -1:]

      # In causal mode, the divisor increments by 1 for every frame to
      # calculate cumulative averages instead of one global average
      mean_divisors = tf.range(num_frames) + frame_count + 1
      mean_divisors = tf.reshape(mean_divisors, [1, num_frames, 1, 1, 1])
      mean_divisors = tf.cast(mean_divisors, x.dtype)

      # Shape: [batch_size, num_frames, 1, 1, channels]
      x = x / mean_divisors
    else:
      # In non-causal mode, we (optionally) sum across frames to take a
      # cumulative average across input iterations rather than individual
      # frames. If no buffer state is passed, this essentially becomes
      # regular global average pooling.
      # Shape: [batch_size, 1, 1, 1, channels]
      x = tf.reduce_sum(inputs, axis=(1, 2, 3), keepdims=True)
      x = x / tf.cast(tf.shape(inputs)[2] * tf.shape(inputs)[3], x.dtype)
      x = x + buffer

      # Shape: [batch_size, 1, 1, 1, channels]
      states[self._state_name] = x

      x = x / tf.cast(frame_count + num_frames, x.dtype)

    if not self._keepdims:
      x = tf.squeeze(x, axis=(1, 2, 3))

    return (x, states) if output_states else x


@tf.keras.utils.register_keras_serializable(package='Vision')
class SpatialAveragePool3D(tf.keras.layers.Layer):
  """Creates a global average pooling layer pooling across spatial dimentions."""

  def __init__(self, keepdims: bool = False, **kwargs):
    """Initializes a global average pool layer.

    Args:
      keepdims: A `bool`. If True, keep the averaged dimensions.
      **kwargs: Additional keyword arguments to be passed to this layer.

    Returns:
      An output `tf.Tensor`.
    """
    super(SpatialAveragePool3D, self).__init__(**kwargs)
    self._keepdims = keepdims

  def get_config(self):
    """Returns a dictionary containing the config used for initialization."""
    config = {
        'keepdims': self._keepdims,
    }
    base_config = super(SpatialAveragePool3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    """Builds the layer with the given input shape."""
    if tf.keras.backend.image_data_format() == 'channels_first':
      raise ValueError('"channels_first" mode is unsupported.')

    super(SpatialAveragePool3D, self).build(input_shape)

  def call(self, inputs, states=None, output_states: bool = False):
    """Calls the layer with the given inputs."""
    if inputs.shape.rank != 5:
      raise ValueError(
          'Input should have rank {}, got {}'.format(5, inputs.shape.rank))

    output = tf.reduce_mean(inputs, axis=(2, 3), keepdims=self._keepdims)
    return (output, states) if output_states else output


class CausalConvMixin:
  """Mixin class to implement CausalConv for `tf.keras.layers.Conv` layers."""

  @property
  def use_buffered_input(self) -> bool:
    return self._use_buffered_input

  @use_buffered_input.setter
  def use_buffered_input(self, variable: bool):
    self._use_buffered_input = variable

  def _compute_buffered_causal_padding(self,
                                       inputs: tf.Tensor,
                                       use_buffered_input: bool = False,
                                       time_axis: int = 1,
                                       ) -> List[List[int]]:
    """Calculates padding for 'causal' option for conv layers.

    Args:
      inputs: An optional input `tf.Tensor` to be padded.
      use_buffered_input: A `bool`. If True, use 'valid' padding along the time
        dimension. This should be set when applying the stream buffer.
      time_axis: An `int` of the axis of the time dimension.

    Returns:
      A list of paddings for `tf.pad`.
    """
    input_shape = tf.shape(inputs)[1:-1]

    if tf.keras.backend.image_data_format() == 'channels_first':
      raise ValueError('"channels_first" mode is unsupported.')

    kernel_size_effective = [
        (self.kernel_size[i] +
         (self.kernel_size[i] - 1) * (self.dilation_rate[i] - 1))
        for i in range(self.rank)
    ]
    pad_total = [kernel_size_effective[0] - 1]
    for i in range(1, self.rank):
      overlap = (input_shape[i] - 1) % self.strides[i] + 1
      pad_total.append(tf.maximum(kernel_size_effective[i] - overlap, 0))
    pad_beg = [pad_total[i] // 2 for i in range(self.rank)]
    pad_end = [pad_total[i] - pad_beg[i] for i in range(self.rank)]
    padding = [[pad_beg[i], pad_end[i]] for i in range(self.rank)]
    padding = [[0, 0]] + padding + [[0, 0]]

    if use_buffered_input:
      padding[time_axis] = [0, 0]
    else:
      padding[time_axis] = [padding[time_axis][0] + padding[time_axis][1], 0]
    return padding

  def _causal_validate_init(self):
    """Validates the Conv layer initial configuration."""
    # Overriding this method is meant to circumvent unnecessary errors when
    # using causal padding.
    if (self.filters is not None
        and self.filters % self.groups != 0):
      raise ValueError(
          'The number of filters must be evenly divisible by the number of '
          'groups. Received: groups={}, filters={}'.format(
              self.groups, self.filters))

    if not all(self.kernel_size):
      raise ValueError('The argument `kernel_size` cannot contain 0(s). '
                       'Received: %s' % (self.kernel_size,))

  def _buffered_spatial_output_shape(self, spatial_output_shape: List[int]):
    """Computes the spatial output shape from the input shape."""
    # When buffer padding, use 'valid' padding across time. The output shape
    # across time should be the input shape minus any padding, assuming
    # the stride across time is 1.
    if self._use_buffered_input and spatial_output_shape[0] is not None:
      padding = self._compute_buffered_causal_padding(
          tf.zeros([1] + spatial_output_shape + [1]), use_buffered_input=False)
      spatial_output_shape[0] -= sum(padding[1])
    return spatial_output_shape


@tf.keras.utils.register_keras_serializable(package='Vision')
class Conv2D(tf.keras.layers.Conv2D, CausalConvMixin):
  """Conv2D layer supporting CausalConv.

  Supports `padding='causal'` option (like in `tf.keras.layers.Conv1D`),
  which applies causal padding to the temporal dimension, and same padding in
  the spatial dimensions.
  """

  def __init__(self, *args, use_buffered_input=False, **kwargs):
    """Initializes conv2d.

    Args:
      *args: Arguments to be passed.
      use_buffered_input: A `bool`. If True, the input is expected to be padded
        beforehand. In effect, calling this layer will use 'valid' padding on
        the temporal dimension to simulate 'causal' padding.
      **kwargs: Additional keyword arguments to be passed.

    Returns:
      An output `tf.Tensor` of the Conv2D operation.
    """
    super(Conv2D, self).__init__(*args, **kwargs)
    self._use_buffered_input = use_buffered_input

  def get_config(self):
    """Returns a dictionary containing the config used for initialization."""
    config = {
        'use_buffered_input': self._use_buffered_input,
    }
    base_config = super(Conv2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def _compute_causal_padding(self, inputs):
    """Computes causal padding dimensions for the given inputs."""
    return self._compute_buffered_causal_padding(
        inputs, use_buffered_input=self._use_buffered_input)

  def _validate_init(self):
    """Validates the Conv layer initial configuration."""
    self._causal_validate_init()

  def _spatial_output_shape(self, spatial_input_shape: List[int]):
    """Computes the spatial output shape from the input shape."""
    shape = super(Conv2D, self)._spatial_output_shape(spatial_input_shape)
    return self._buffered_spatial_output_shape(shape)


@tf.keras.utils.register_keras_serializable(package='Vision')
class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D, CausalConvMixin):
  """DepthwiseConv2D layer supporting CausalConv.

  Supports `padding='causal'` option (like in `tf.keras.layers.Conv1D`),
  which applies causal padding to the temporal dimension, and same padding in
  the spatial dimensions.
  """

  def __init__(self, *args, use_buffered_input=False, **kwargs):
    """Initializes depthwise conv2d.

    Args:
      *args: Arguments to be passed.
      use_buffered_input: A `bool`. If True, the input is expected to be padded
        beforehand. In effect, calling this layer will use 'valid' padding on
        the temporal dimension to simulate 'causal' padding.
      **kwargs: Additional keyword arguments to be passed.

    Returns:
      An output `tf.Tensor` of the DepthwiseConv2D operation.
    """
    super(DepthwiseConv2D, self).__init__(*args, **kwargs)
    self._use_buffered_input = use_buffered_input

    # Causal padding is unsupported by default for DepthwiseConv2D,
    # so we resort to valid padding internally. However, we handle
    # causal padding as a special case with `self._is_causal`, which is
    # defined by the super class.
    if self.padding == 'causal':
      self.padding = 'valid'

  def get_config(self):
    """Returns a dictionary containing the config used for initialization."""
    config = {
        'use_buffered_input': self._use_buffered_input,
    }
    base_config = super(DepthwiseConv2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    """Calls the layer with the given inputs."""
    if self._is_causal:
      inputs = tf.pad(inputs, self._compute_causal_padding(inputs))
    return super(DepthwiseConv2D, self).call(inputs)

  def _compute_causal_padding(self, inputs):
    """Computes causal padding dimensions for the given inputs."""
    return self._compute_buffered_causal_padding(
        inputs, use_buffered_input=self._use_buffered_input)

  def _validate_init(self):
    """Validates the Conv layer initial configuration."""
    self._causal_validate_init()

  def _spatial_output_shape(self, spatial_input_shape: List[int]):
    """Computes the spatial output shape from the input shape."""
    shape = super(DepthwiseConv2D, self)._spatial_output_shape(
        spatial_input_shape)
    return self._buffered_spatial_output_shape(shape)


@tf.keras.utils.register_keras_serializable(package='Vision')
class Conv3D(tf.keras.layers.Conv3D, CausalConvMixin):
  """Conv3D layer supporting CausalConv.

  Supports `padding='causal'` option (like in `tf.keras.layers.Conv1D`),
  which applies causal padding to the temporal dimension, and same padding in
  the spatial dimensions.
  """

  def __init__(self, *args, use_buffered_input=False, **kwargs):
    """Initializes conv3d.

    Args:
      *args: Arguments to be passed.
      use_buffered_input: A `bool`. If True, the input is expected to be padded
        beforehand. In effect, calling this layer will use 'valid' padding on
        the temporal dimension to simulate 'causal' padding.
      **kwargs: Additional keyword arguments to be passed.

    Returns:
      An output `tf.Tensor` of the Conv3D operation.
    """
    super(Conv3D, self).__init__(*args, **kwargs)
    self._use_buffered_input = use_buffered_input

  def get_config(self):
    """Returns a dictionary containing the config used for initialization."""
    config = {
        'use_buffered_input': self._use_buffered_input,
    }
    base_config = super(Conv3D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    """Call the layer with the given inputs."""
    # Note: tf.nn.conv3d with depthwise kernels on CPU is currently only
    # supported when compiling with TF graph (XLA) using tf.function, so it
    # is compiled by default here (b/186463870).
    conv_fn = tf.function(super(Conv3D, self).call, jit_compile=True)
    return conv_fn(inputs)

  def _compute_causal_padding(self, inputs):
    """Computes causal padding dimensions for the given inputs."""
    return self._compute_buffered_causal_padding(
        inputs, use_buffered_input=self._use_buffered_input)

  def _validate_init(self):
    """Validates the Conv layer initial configuration."""
    self._causal_validate_init()

  def _spatial_output_shape(self, spatial_input_shape: List[int]):
    """Computes the spatial output shape from the input shape."""
    shape = super(Conv3D, self)._spatial_output_shape(spatial_input_shape)
    return self._buffered_spatial_output_shape(shape)


@tf.keras.utils.register_keras_serializable(package='Vision')
class SpatialPyramidPooling(tf.keras.layers.Layer):
  """Implements the Atrous Spatial Pyramid Pooling.

  References:
    [Rethinking Atrous Convolution for Semantic Image Segmentation](
      https://arxiv.org/pdf/1706.05587.pdf)
    [Encoder-Decoder with Atrous Separable Convolution for Semantic Image
    Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
  """

  def __init__(
      self,
      output_channels: int,
      dilation_rates: List[int],
      pool_kernel_size: Optional[List[int]] = None,
      use_sync_bn: bool = False,
      batchnorm_momentum: float = 0.99,
      batchnorm_epsilon: float = 0.001,
      activation: str = 'relu',
      dropout: float = 0.5,
      kernel_initializer: str = 'GlorotUniform',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      interpolation: str = 'bilinear',
      use_depthwise_convolution: bool = False,
      **kwargs):
    """Initializes `SpatialPyramidPooling`.

    Args:
      output_channels: Number of channels produced by SpatialPyramidPooling.
      dilation_rates: A list of integers for parallel dilated conv.
      pool_kernel_size: A list of integers or None. If None, global average
        pooling is applied, otherwise an average pooling of pool_kernel_size is
        applied.
      use_sync_bn: A bool, whether or not to use sync batch normalization.
      batchnorm_momentum: A float for the momentum in BatchNorm. Defaults to
        0.99.
      batchnorm_epsilon: A float for the epsilon value in BatchNorm. Defaults to
        0.001.
      activation: A `str` for type of activation to be used. Defaults to 'relu'.
      dropout: A float for the dropout rate before output. Defaults to 0.5.
      kernel_initializer: Kernel initializer for conv layers. Defaults to
        `glorot_uniform`.
      kernel_regularizer: Kernel regularizer for conv layers. Defaults to None.
      interpolation: The interpolation method for upsampling. Defaults to
        `bilinear`.
      use_depthwise_convolution: Allows spatial pooling to be separable
        depthwise convolusions. [Encoder-Decoder with Atrous Separable
        Convolution for Semantic Image Segmentation](
         https://arxiv.org/pdf/1802.02611.pdf)
      **kwargs: Other keyword arguments for the layer.
    """
    super().__init__(**kwargs)

    self._output_channels = output_channels
    self._dilation_rates = dilation_rates
    self._use_sync_bn = use_sync_bn
    self._batchnorm_momentum = batchnorm_momentum
    self._batchnorm_epsilon = batchnorm_epsilon
    self._activation = activation
    self._dropout = dropout
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._interpolation = interpolation
    self._pool_kernel_size = pool_kernel_size
    self._use_depthwise_convolution = use_depthwise_convolution
    self._activation_fn = tf_utils.get_activation(activation)
    self._bn_op = tf.keras.layers.BatchNormalization

    if tf.keras.backend.image_data_format() == 'channels_last':
      self._bn_axis = -1
    else:
      self._bn_axis = 1

  def build(self, input_shape):
    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3]

    self.aspp_layers = []

    conv1 = tf.keras.layers.Conv2D(
        filters=self._output_channels,
        kernel_size=(1, 1),
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        use_bias=False)
    norm1 = self._bn_op(
        axis=self._bn_axis,
        momentum=self._batchnorm_momentum,
        epsilon=self._batchnorm_epsilon,
        synchronized=self._use_sync_bn)

    self.aspp_layers.append([conv1, norm1])

    for dilation_rate in self._dilation_rates:
      leading_layers = []
      kernel_size = (3, 3)
      if self._use_depthwise_convolution:
        leading_layers += [
            tf.keras.layers.DepthwiseConv2D(
                depth_multiplier=1,
                kernel_size=kernel_size,
                padding='same',
                depthwise_regularizer=self._kernel_regularizer,
                depthwise_initializer=tf_utils.clone_initializer(
                    self._kernel_initializer),
                dilation_rate=dilation_rate,
                use_bias=False)
        ]
        kernel_size = (1, 1)
      conv_dilation = leading_layers + [
          tf.keras.layers.Conv2D(
              filters=self._output_channels,
              kernel_size=kernel_size,
              padding='same',
              kernel_regularizer=self._kernel_regularizer,
              kernel_initializer=tf_utils.clone_initializer(
                  self._kernel_initializer),
              dilation_rate=dilation_rate,
              use_bias=False)
      ]
      norm_dilation = self._bn_op(
          axis=self._bn_axis,
          momentum=self._batchnorm_momentum,
          epsilon=self._batchnorm_epsilon,
          synchronized=self._use_sync_bn)

      self.aspp_layers.append(conv_dilation + [norm_dilation])

    if self._pool_kernel_size is None:
      pooling = [
          tf.keras.layers.GlobalAveragePooling2D(),
          tf.keras.layers.Reshape((1, 1, channels))
      ]
    else:
      pooling = [tf.keras.layers.AveragePooling2D(self._pool_kernel_size)]

    conv2 = tf.keras.layers.Conv2D(
        filters=self._output_channels,
        kernel_size=(1, 1),
        kernel_initializer=tf_utils.clone_initializer(self._kernel_initializer),
        kernel_regularizer=self._kernel_regularizer,
        use_bias=False)
    norm2 = self._bn_op(
        axis=self._bn_axis,
        momentum=self._batchnorm_momentum,
        epsilon=self._batchnorm_epsilon,
        synchronized=self._use_sync_bn)

    self.aspp_layers.append(pooling + [conv2, norm2])

    self._resizing_layer = tf.keras.layers.Resizing(
        height, width, interpolation=self._interpolation, dtype=tf.float32)

    self._projection = [
        tf.keras.layers.Conv2D(
            filters=self._output_channels,
            kernel_size=(1, 1),
            kernel_initializer=tf_utils.clone_initializer(
                self._kernel_initializer),
            kernel_regularizer=self._kernel_regularizer,
            use_bias=False),
        self._bn_op(
            axis=self._bn_axis,
            momentum=self._batchnorm_momentum,
            epsilon=self._batchnorm_epsilon,
            synchronized=self._use_sync_bn)
    ]
    self._dropout_layer = tf.keras.layers.Dropout(rate=self._dropout)
    self._concat_layer = tf.keras.layers.Concatenate(axis=-1)

  def call(self,
           inputs: tf.Tensor,
           training: Optional[bool] = None) -> tf.Tensor:
    if training is None:
      training = tf.keras.backend.learning_phase()
    result = []
    for i, layers in enumerate(self.aspp_layers):
      x = inputs
      for layer in layers:
        # Apply layers sequentially.
        x = layer(x, training=training)
      x = self._activation_fn(x)

      # Apply resize layer to the end of the last set of layers.
      if i == len(self.aspp_layers) - 1:
        x = self._resizing_layer(x)

      result.append(tf.cast(x, inputs.dtype))
    x = self._concat_layer(result)
    for layer in self._projection:
      x = layer(x, training=training)
    x = self._activation_fn(x)
    return self._dropout_layer(x)

  def get_config(self):
    config = {
        'output_channels': self._output_channels,
        'dilation_rates': self._dilation_rates,
        'pool_kernel_size': self._pool_kernel_size,
        'use_sync_bn': self._use_sync_bn,
        'batchnorm_momentum': self._batchnorm_momentum,
        'batchnorm_epsilon': self._batchnorm_epsilon,
        'activation': self._activation,
        'dropout': self._dropout,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'interpolation': self._interpolation,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package='Vision')
class MultiHeadAttention(tf.keras.layers.MultiHeadAttention):
  """MultiHeadAttention layer.

  This is an implementation of multi-headed attention as described in the paper
  "Attention is all you Need" (Vaswani et al., 2017).
  """

  def __init__(
      self,
      *args,
      partition_dims: Optional[Tuple[int, int, int, int]] = None,
      max_inference_parallelism: Optional[int] = None,
      **kwargs,
  ):
    """Initializes MultiHeadAttention.

    Args:
      *args: Positional arguments passed to super().__init__.
      partition_dims: Spatial partition dimensions.
      max_inference_parallelism: The number of examples to run in parallel
        during inference. Set this limit to reduce the peak memory usage. If
        None, use vectorized operations to run the whole batch in parallel.
      **kwargs: Keyword arguments passed to super().__init__.
    """
    super().__init__(*args, **kwargs)
    self._partition_dims = partition_dims
    self._max_inference_parallelism = max_inference_parallelism

  def get_config(self):
    config = super().get_config()
    config.update({
        'partition_dims': self._partition_dims,
        'max_inference_parallelism': self._max_inference_parallelism,
    })
    return config

  def _compute_attention(
      self,
      query: tf.Tensor,
      key: tf.Tensor,
      value: tf.Tensor,
      attention_mask: Optional[tf.Tensor] = None,
      training: Optional[bool] = None,
  ):
    """Applies dot-product attention with query, key, value tensors.

    Args:
      query: Projected query `Tensor` of shape `(B, T, N, key_dim)`.
      key: Projected key `Tensor` of shape `(B, S, N, key_dim)`.
      value: Projected value `Tensor` of shape `(B, S, N, value_dim)`.
      attention_mask: a boolean mask of shape `(B, T, S)`, that prevents
        attention to certain positions. It is generally not needed if the
        `query` and `value` (and/or `key`) are masked.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).

    Returns:
      attention_output: Multi-headed outputs of attention computation.
      attention_scores: Multi-headed attention weights.
    """
    if self._partition_dims is not None:
      strategy = tf.distribute.get_strategy()
      # `query` = [B, T, N ,H]
      query = strategy.experimental_split_to_logical_devices(
          query, self._partition_dims)
      key = strategy.experimental_split_to_logical_devices(
          key, self._partition_dims)
      value = strategy.experimental_split_to_logical_devices(
          value, self._partition_dims)

    batch_size = query.get_shape().as_list()[0]  # None if dynamic.

    if (
        training
        or self._max_inference_parallelism is None
        or self._max_inference_parallelism <= 0
        or (
            # If the whole batch is allowed to be run in parallel, use fully
            # vectorized computation instead of tf.map_fn to make things more
            # efficient.
            batch_size is not None
            and batch_size <= self._max_inference_parallelism
        )
    ):
      return self._compute_attention_delegate(
          query, key, value, attention_mask, training
      )
    else:
      # Sequentialize the inference execution with limited parallelism.
      def _compute_fn(x):
        attention_output, attention_scores = self._compute_attention_delegate(
            query=x[0][tf.newaxis, ...],
            key=x[1][tf.newaxis, ...],
            value=x[2][tf.newaxis, ...],
            attention_mask=x[3][tf.newaxis, ...] if len(x) >= 4 else None,
            training=training,
        )
        attention_output = tf.squeeze(attention_output, axis=0)
        attention_scores = tf.squeeze(attention_scores, axis=0)
        return attention_output, attention_scores

      if attention_mask is not None:
        elems = [query, key, value, attention_mask]
      else:
        elems = [query, key, value]

      return tf.map_fn(
          fn=_compute_fn,
          elems=elems,
          fn_output_signature=(value.dtype, value.dtype),
          parallel_iterations=self._max_inference_parallelism,
      )

  def _compute_attention_delegate(
      self,
      query: tf.Tensor,
      key: tf.Tensor,
      value: tf.Tensor,
      attention_mask: Optional[tf.Tensor] = None,
      training: Optional[bool] = None,
  ):
    """Implements dot-product attention with query, key, value tensors."""
    # Simply calls the implementation of the super class here, while the users
    # can override this function for customizing attention computation.
    return super()._compute_attention(
        query, key, value, attention_mask, training
    )
