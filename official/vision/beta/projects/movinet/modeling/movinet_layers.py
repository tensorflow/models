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
"""Contains common building blocks for MoViNets.

Reference: https://arxiv.org/pdf/2103.11511.pdf
"""

from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import tensorflow as tf

from official.vision.beta.modeling.layers import nn_layers

# Default kernel weight decay that may be overridden
KERNEL_WEIGHT_DECAY = 1.5e-5


def normalize_tuple(value: Union[int, Tuple[int, ...]], size: int, name: str):
  """Transforms a single integer or iterable of integers into an integer tuple.

  Arguments:
    value: The value to validate and convert. Could an int, or any iterable of
      ints.
    size: The size of the tuple to be returned.
    name: The name of the argument being validated, e.g. "strides" or
      "kernel_size". This is only used to format error messages.
  Returns:
    A tuple of `size` integers.
  Raises:
    ValueError: If something else than an int/long or iterable thereof was
      passed.
  """
  if isinstance(value, int):
    return (value,) * size
  else:
    try:
      value_tuple = tuple(value)
    except TypeError:
      raise ValueError('The `' + name + '` argument must be a tuple of ' +
                       str(size) + ' integers. Received: ' + str(value))
    if len(value_tuple) != size:
      raise ValueError('The `' + name + '` argument must be a tuple of ' +
                       str(size) + ' integers. Received: ' + str(value))
    for single_value in value_tuple:
      try:
        int(single_value)
      except (ValueError, TypeError):
        raise ValueError('The `' + name + '` argument must be a tuple of ' +
                         str(size) + ' integers. Received: ' + str(value) + ' '
                         'including element ' + str(single_value) + ' of type' +
                         ' ' + str(type(single_value)))
    return value_tuple


@tf.keras.utils.register_keras_serializable(package='Vision')
class Squeeze3D(tf.keras.layers.Layer):
  """Squeeze3D layer to remove singular dimensions."""

  def call(self, inputs):
    """Calls the layer with the given inputs."""
    return tf.squeeze(inputs, axis=(1, 2, 3))


@tf.keras.utils.register_keras_serializable(package='Vision')
class MobileConv2D(tf.keras.layers.Layer):
  """Conv2D layer with extra options to support mobile devices.

  Reshapes 5D video tensor inputs to 4D, allowing Conv2D to run across
  dimensions (2, 3) or (3, 4). Reshapes tensors back to 5D when returning the
  output.
  """

  def __init__(
      self,
      filters: int,
      kernel_size: Union[int, Sequence[int]],
      strides: Union[int, Sequence[int]] = (1, 1),
      padding: str = 'valid',
      data_format: Optional[str] = None,
      dilation_rate: Union[int, Sequence[int]] = (1, 1),
      groups: int = 1,
      activation: Optional[nn_layers.Activation] = None,
      use_bias: bool = True,
      kernel_initializer: tf.keras.initializers.Initializer = 'glorot_uniform',
      bias_initializer: tf.keras.initializers.Initializer = 'zeros',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      activity_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
      kernel_constraint: Optional[tf.keras.constraints.Constraint] = None,
      bias_constraint: Optional[tf.keras.constraints.Constraint] = None,
      use_depthwise: bool = False,
      use_temporal: bool = False,
      use_buffered_input: bool = False,
      **kwargs):  # pylint: disable=g-doc-args
    """Initializes mobile conv2d.

    For the majority of arguments, see tf.keras.layers.Conv2D.

    Args:
      use_depthwise: if True, use DepthwiseConv2D instead of Conv2D
      use_temporal: if True, apply Conv2D starting from the temporal dimension
          instead of the spatial dimensions.
      use_buffered_input: if True, the input is expected to be padded
          beforehand. In effect, calling this layer will use 'valid' padding on
          the temporal dimension to simulate 'causal' padding.
      **kwargs: keyword arguments to be passed to this layer.

    Returns:
      A output tensor of the MobileConv2D operation.
    """
    super(MobileConv2D, self).__init__(**kwargs)
    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._padding = padding
    self._data_format = data_format
    self._dilation_rate = dilation_rate
    self._groups = groups
    self._activation = activation
    self._use_bias = use_bias
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._kernel_regularizer = kernel_regularizer
    self._bias_regularizer = bias_regularizer
    self._activity_regularizer = activity_regularizer
    self._kernel_constraint = kernel_constraint
    self._bias_constraint = bias_constraint
    self._use_depthwise = use_depthwise
    self._use_temporal = use_temporal
    self._use_buffered_input = use_buffered_input

    kernel_size = normalize_tuple(kernel_size, 2, 'kernel_size')

    if self._use_temporal and kernel_size[1] > 1:
      raise ValueError('Temporal conv with spatial kernel is not supported.')

    if use_depthwise:
      self._conv = nn_layers.DepthwiseConv2D(
          kernel_size=kernel_size,
          strides=strides,
          padding=padding,
          depth_multiplier=1,
          data_format=data_format,
          dilation_rate=dilation_rate,
          activation=activation,
          use_bias=use_bias,
          depthwise_initializer=kernel_initializer,
          bias_initializer=bias_initializer,
          depthwise_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer,
          activity_regularizer=activity_regularizer,
          depthwise_constraint=kernel_constraint,
          bias_constraint=bias_constraint,
          use_buffered_input=use_buffered_input)
    else:
      self._conv = nn_layers.Conv2D(
          filters=filters,
          kernel_size=kernel_size,
          strides=strides,
          padding=padding,
          data_format=data_format,
          dilation_rate=dilation_rate,
          groups=groups,
          activation=activation,
          use_bias=use_bias,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer,
          kernel_regularizer=kernel_regularizer,
          bias_regularizer=bias_regularizer,
          activity_regularizer=activity_regularizer,
          kernel_constraint=kernel_constraint,
          bias_constraint=bias_constraint,
          use_buffered_input=use_buffered_input)

  def get_config(self):
    """Returns a dictionary containing the config used for initialization."""
    config = {
        'filters': self._filters,
        'kernel_size': self._kernel_size,
        'strides': self._strides,
        'padding': self._padding,
        'data_format': self._data_format,
        'dilation_rate': self._dilation_rate,
        'groups': self._groups,
        'activation': self._activation,
        'use_bias': self._use_bias,
        'kernel_initializer': self._kernel_initializer,
        'bias_initializer': self._bias_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'bias_regularizer': self._bias_regularizer,
        'activity_regularizer': self._activity_regularizer,
        'kernel_constraint': self._kernel_constraint,
        'bias_constraint': self._bias_constraint,
        'use_depthwise': self._use_depthwise,
        'use_temporal': self._use_temporal,
        'use_buffered_input': self._use_buffered_input,
    }
    base_config = super(MobileConv2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    """Calls the layer with the given inputs."""
    if self._use_temporal:
      input_shape = [
          tf.shape(inputs)[0],
          tf.shape(inputs)[1],
          tf.shape(inputs)[2] * tf.shape(inputs)[3],
          inputs.shape[4]]
    else:
      input_shape = [
          tf.shape(inputs)[0] * tf.shape(inputs)[1],
          tf.shape(inputs)[2],
          tf.shape(inputs)[3],
          inputs.shape[4]]
    x = tf.reshape(inputs, input_shape)

    x = self._conv(x)

    if self._use_temporal:
      output_shape = [
          tf.shape(x)[0],
          tf.shape(x)[1],
          tf.shape(inputs)[2],
          tf.shape(inputs)[3],
          x.shape[3]]
    else:
      output_shape = [
          tf.shape(inputs)[0],
          tf.shape(inputs)[1],
          tf.shape(x)[1],
          tf.shape(x)[2],
          x.shape[3]]
    x = tf.reshape(x, output_shape)

    return x


@tf.keras.utils.register_keras_serializable(package='Vision')
class ConvBlock(tf.keras.layers.Layer):
  """A Conv followed by optional BatchNorm and Activation."""

  def __init__(
      self,
      filters: int,
      kernel_size: Union[int, Sequence[int]],
      strides: Union[int, Sequence[int]] = 1,
      depthwise: bool = False,
      causal: bool = False,
      use_bias: bool = False,
      kernel_initializer: tf.keras.initializers.Initializer = 'HeNormal',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] =
      tf.keras.regularizers.L2(KERNEL_WEIGHT_DECAY),
      use_batch_norm: bool = True,
      batch_norm_layer: tf.keras.layers.Layer =
      tf.keras.layers.experimental.SyncBatchNormalization,
      batch_norm_momentum: float = 0.99,
      batch_norm_epsilon: float = 1e-3,
      activation: Optional[Any] = None,
      conv_type: str = '3d',
      use_buffered_input: bool = False,
      **kwargs):
    """Initializes a conv block.

    Args:
      filters: filters for the conv operation.
      kernel_size: kernel size for the conv operation.
      strides: strides for the conv operation.
      depthwise: if True, use DepthwiseConv2D instead of Conv2D
      causal: if True, use causal mode for the conv operation.
      use_bias: use bias for the conv operation.
      kernel_initializer: kernel initializer for the conv operation.
      kernel_regularizer: kernel regularizer for the conv operation.
      use_batch_norm: if True, apply batch norm after the conv operation.
      batch_norm_layer: class to use for batch norm, if applied.
      batch_norm_momentum: momentum of the batch norm operation, if applied.
      batch_norm_epsilon: epsilon of the batch norm operation, if applied.
      activation: activation after the conv and batch norm operations.
      conv_type: '3d', '2plus1d', or '3d_2plus1d'. '3d' uses the default 3D
          ops. '2plus1d' split any 3D ops into two sequential 2D ops with their
          own batch norm and activation. '3d_2plus1d' is like '2plus1d', but
          uses two sequential 3D ops instead.
      use_buffered_input: if True, the input is expected to be padded
          beforehand. In effect, calling this layer will use 'valid' padding on
          the temporal dimension to simulate 'causal' padding.
      **kwargs: keyword arguments to be passed to this layer.

    Returns:
      A output tensor of the ConvBlock operation.
    """

    super(ConvBlock, self).__init__(**kwargs)

    kernel_size = normalize_tuple(kernel_size, 3, 'kernel_size')
    strides = normalize_tuple(strides, 3, 'strides')

    self._filters = filters
    self._kernel_size = kernel_size
    self._strides = strides
    self._depthwise = depthwise
    self._causal = causal
    self._use_bias = use_bias
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._use_batch_norm = use_batch_norm
    self._batch_norm_layer = batch_norm_layer
    self._batch_norm_momentum = batch_norm_momentum
    self._batch_norm_epsilon = batch_norm_epsilon
    self._activation = activation
    self._conv_type = conv_type
    self._use_buffered_input = use_buffered_input

    if activation is not None:
      self._activation_layer = tf.keras.layers.Activation(activation)
    else:
      self._activation_layer = None

    self._groups = None

  def get_config(self):
    """Returns a dictionary containing the config used for initialization."""
    config = {
        'filters': self._filters,
        'kernel_size': self._kernel_size,
        'strides': self._strides,
        'depthwise': self._depthwise,
        'causal': self._causal,
        'use_bias': self._use_bias,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'use_batch_norm': self._use_batch_norm,
        'batch_norm_momentum': self._batch_norm_momentum,
        'batch_norm_epsilon': self._batch_norm_epsilon,
        'activation': self._activation,
        'conv_type': self._conv_type,
        'use_buffered_input': self._use_buffered_input,
    }
    base_config = super(ConvBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    """Builds the layer with the given input shape."""
    padding = 'causal' if self._causal else 'same'
    self._groups = input_shape[-1] if self._depthwise else 1

    self._conv_temporal = None

    if self._conv_type == '3d_2plus1d' and self._kernel_size[0] > 1:
      self._conv = nn_layers.Conv3D(
          self._filters,
          (1, self._kernel_size[1], self._kernel_size[2]),
          strides=(1, self._strides[1], self._strides[2]),
          padding='same',
          groups=self._groups,
          use_bias=self._use_bias,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          use_buffered_input=False,
          name='conv3d')
      self._conv_temporal = nn_layers.Conv3D(
          self._filters,
          (self._kernel_size[0], 1, 1),
          strides=(self._strides[0], 1, 1),
          padding=padding,
          groups=self._groups,
          use_bias=self._use_bias,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          use_buffered_input=self._use_buffered_input,
          name='conv3d_temporal')
    elif self._conv_type == '2plus1d':
      self._conv = MobileConv2D(
          self._filters,
          (self._kernel_size[1], self._kernel_size[2]),
          strides=(self._strides[1], self._strides[2]),
          padding='same',
          use_depthwise=self._depthwise,
          groups=self._groups,
          use_bias=self._use_bias,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          use_buffered_input=False,
          name='conv2d')
      if self._kernel_size[0] > 1:
        self._conv_temporal = MobileConv2D(
            self._filters,
            (self._kernel_size[0], 1),
            strides=(self._strides[0], 1),
            padding=padding,
            use_temporal=True,
            use_depthwise=self._depthwise,
            groups=self._groups,
            use_bias=self._use_bias,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            use_buffered_input=self._use_buffered_input,
            name='conv2d_temporal')
    else:
      self._conv = nn_layers.Conv3D(
          self._filters,
          self._kernel_size,
          strides=self._strides,
          padding=padding,
          groups=self._groups,
          use_bias=self._use_bias,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          use_buffered_input=self._use_buffered_input,
          name='conv3d')

    self._batch_norm = None
    self._batch_norm_temporal = None

    if self._use_batch_norm:
      self._batch_norm = self._batch_norm_layer(
          momentum=self._batch_norm_momentum,
          epsilon=self._batch_norm_epsilon,
          name='bn')
      if self._conv_type != '3d' and self._conv_temporal is not None:
        self._batch_norm_temporal = self._batch_norm_layer(
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
            name='bn_temporal')

    super(ConvBlock, self).build(input_shape)

  def call(self, inputs):
    """Calls the layer with the given inputs."""
    x = inputs

    x = self._conv(x)
    if self._batch_norm is not None:
      x = self._batch_norm(x)
    if self._activation_layer is not None:
      x = self._activation_layer(x)

    if self._conv_temporal is not None:
      x = self._conv_temporal(x)
      if self._batch_norm_temporal is not None:
        x = self._batch_norm_temporal(x)
      if self._activation_layer is not None:
        x = self._activation_layer(x)

    return x


@tf.keras.utils.register_keras_serializable(package='Vision')
class StreamBuffer(tf.keras.layers.Layer):
  """Stream buffer wrapper which caches activations of previous frames."""

  def __init__(self,
               buffer_size: int,
               state_prefix: Optional[str] = None,
               **kwargs):
    """Initializes a stream buffer.

    Args:
      buffer_size: the number of input frames to cache.
      state_prefix: a prefix string to identify states.
      **kwargs: keyword arguments to be passed to this layer.

    Returns:
      A output tensor of the StreamBuffer operation.
    """
    super(StreamBuffer, self).__init__(**kwargs)

    state_prefix = state_prefix if state_prefix is not None else ''
    self._state_prefix = state_prefix
    self._state_name = f'{state_prefix}/stream_buffer'
    self._buffer_size = buffer_size

  def get_config(self):
    """Returns a dictionary containing the config used for initialization."""
    config = {
        'buffer_size': self._buffer_size,
        'state_prefix': self._state_prefix,
    }
    base_config = super(StreamBuffer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(
      self,
      inputs: tf.Tensor,
      states: Optional[nn_layers.States] = None,
  ) -> Tuple[Any, nn_layers.States]:
    """Calls the layer with the given inputs.

    Args:
      inputs: the input tensor.
      states: a dict of states such that, if any of the keys match for this
          layer, will overwrite the contents of the buffer(s).
          Expected keys include `state_prefix + '/stream_buffer'`.

    Returns:
      the output tensor and states
    """
    states = dict(states) if states is not None else {}
    buffer = states.get(self._state_name, None)

    # Create the buffer if it does not exist in the states.
    # Output buffer shape:
    # [batch_size, buffer_size, input_height, input_width, num_channels]
    if buffer is None:
      shape = tf.shape(inputs)
      buffer = tf.zeros(
          [shape[0], self._buffer_size, shape[2], shape[3], shape[4]],
          dtype=inputs.dtype)

    # tf.pad has limited support for tf lite, so use tf.concat instead.
    full_inputs = tf.concat([buffer, inputs], axis=1)

    # Cache the last b frames of the input where b is the buffer size and f
    # is the number of input frames. If b > f, then we will cache the last b - f
    # frames from the previous buffer concatenated with the current f input
    # frames.
    new_buffer = full_inputs[:, -self._buffer_size:]
    states[self._state_name] = new_buffer

    return full_inputs, states


@tf.keras.utils.register_keras_serializable(package='Vision')
class StreamConvBlock(ConvBlock):
  """ConvBlock with StreamBuffer."""

  def __init__(
      self,
      filters: int,
      kernel_size: Union[int, Sequence[int]],
      strides: Union[int, Sequence[int]] = 1,
      depthwise: bool = False,
      causal: bool = False,
      use_bias: bool = False,
      kernel_initializer: tf.keras.initializers.Initializer = 'HeNormal',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = tf.keras
      .regularizers.L2(KERNEL_WEIGHT_DECAY),
      use_batch_norm: bool = True,
      batch_norm_layer: tf.keras.layers.Layer = tf.keras.layers.experimental
      .SyncBatchNormalization,
      batch_norm_momentum: float = 0.99,
      batch_norm_epsilon: float = 1e-3,
      activation: Optional[Any] = None,
      conv_type: str = '3d',
      state_prefix: Optional[str] = None,
      **kwargs):
    """Initializes a stream conv block.

    Args:
      filters: filters for the conv operation.
      kernel_size: kernel size for the conv operation.
      strides: strides for the conv operation.
      depthwise: if True, use DepthwiseConv2D instead of Conv2D
      causal: if True, use causal mode for the conv operation.
      use_bias: use bias for the conv operation.
      kernel_initializer: kernel initializer for the conv operation.
      kernel_regularizer: kernel regularizer for the conv operation.
      use_batch_norm: if True, apply batch norm after the conv operation.
      batch_norm_layer: class to use for batch norm, if applied.
      batch_norm_momentum: momentum of the batch norm operation, if applied.
      batch_norm_epsilon: epsilon of the batch norm operation, if applied.
      activation: activation after the conv and batch norm operations.
      conv_type: '3d', '2plus1d', or '3d_2plus1d'. '3d' uses the default 3D
          ops. '2plus1d' split any 3D ops into two sequential 2D ops with their
          own batch norm and activation. '3d_2plus1d' is like '2plus1d', but
          uses two sequential 3D ops instead.
      state_prefix: a prefix string to identify states.
      **kwargs: keyword arguments to be passed to this layer.

    Returns:
      A output tensor of the StreamConvBlock operation.
    """
    kernel_size = normalize_tuple(kernel_size, 3, 'kernel_size')
    buffer_size = kernel_size[0] - 1
    use_buffer = buffer_size > 0 and causal

    self._state_prefix = state_prefix

    super(StreamConvBlock, self).__init__(
        filters,
        kernel_size,
        strides=strides,
        depthwise=depthwise,
        causal=causal,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_batch_norm=use_batch_norm,
        batch_norm_layer=batch_norm_layer,
        batch_norm_momentum=batch_norm_momentum,
        batch_norm_epsilon=batch_norm_epsilon,
        activation=activation,
        conv_type=conv_type,
        use_buffered_input=use_buffer,
        **kwargs)

    self._stream_buffer = None
    if use_buffer:
      self._stream_buffer = StreamBuffer(
          buffer_size=buffer_size, state_prefix=state_prefix)

  def get_config(self):
    """Returns a dictionary containing the config used for initialization."""
    config = {'state_prefix': self._state_prefix}
    base_config = super(StreamConvBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self,
           inputs: tf.Tensor,
           states: Optional[nn_layers.States] = None
           ) -> Tuple[tf.Tensor, nn_layers.States]:
    """Calls the layer with the given inputs.

    Args:
      inputs: the input tensor.
      states: a dict of states such that, if any of the keys match for this
          layer, will overwrite the contents of the buffer(s).

    Returns:
      the output tensor and states
    """
    states = dict(states) if states is not None else {}

    x = inputs
    if self._stream_buffer is not None:
      x, states = self._stream_buffer(x, states=states)
    x = super(StreamConvBlock, self).call(x)

    return x, states


@tf.keras.utils.register_keras_serializable(package='Vision')
class StreamSqueezeExcitation(tf.keras.layers.Layer):
  """Squeeze and excitation layer with causal mode.

  Reference: https://arxiv.org/pdf/1709.01507.pdf
  """

  def __init__(
      self,
      hidden_filters: int,
      activation: nn_layers.Activation = 'swish',
      gating_activation: nn_layers.Activation = 'sigmoid',
      causal: bool = False,
      conv_type: str = '3d',
      kernel_initializer: tf.keras.initializers.Initializer = 'HeNormal',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = tf.keras
      .regularizers.L2(KERNEL_WEIGHT_DECAY),
      use_positional_encoding: bool = False,
      state_prefix: Optional[str] = None,
      **kwargs):
    """Implementation for squeeze and excitation.

    Args:
      hidden_filters: The hidden filters of squeeze excite.
      activation: name of the activation function.
      gating_activation: name of the activation function for gating.
      causal: if True, use causal mode in the global average pool.
      conv_type: '3d', '2plus1d', or '3d_2plus1d'. '3d' uses the default 3D
          ops. '2plus1d' split any 3D ops into two sequential 2D ops with their
          own batch norm and activation. '3d_2plus1d' is like '2plus1d', but
          uses two sequential 3D ops instead.
      kernel_initializer: kernel initializer for the conv operations.
      kernel_regularizer: kernel regularizer for the conv operation.
      use_positional_encoding: add a positional encoding after the (cumulative)
          global average pooling layer.
      state_prefix: a prefix string to identify states.
      **kwargs: keyword arguments to be passed to this layer.
    """
    super(StreamSqueezeExcitation, self).__init__(**kwargs)

    self._hidden_filters = hidden_filters
    self._activation = activation
    self._gating_activation = gating_activation
    self._causal = causal
    self._conv_type = conv_type
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._use_positional_encoding = use_positional_encoding
    self._state_prefix = state_prefix

    self._pool = nn_layers.GlobalAveragePool3D(
        keepdims=True, causal=causal, state_prefix=state_prefix)

    self._pos_encoding = None
    if use_positional_encoding:
      self._pos_encoding = nn_layers.PositionalEncoding(
          initializer='zeros', state_prefix=state_prefix)

  def get_config(self):
    """Returns a dictionary containing the config used for initialization."""
    config = {
        'hidden_filters': self._hidden_filters,
        'activation': self._activation,
        'gating_activation': self._gating_activation,
        'causal': self._causal,
        'conv_type': self._conv_type,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'use_positional_encoding': self._use_positional_encoding,
        'state_prefix': self._state_prefix,
    }
    base_config = super(StreamSqueezeExcitation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    """Builds the layer with the given input shape."""
    self._se_reduce = ConvBlock(
        filters=self._hidden_filters,
        kernel_size=1,
        causal=self._causal,
        use_bias=True,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        use_batch_norm=False,
        activation=self._activation,
        conv_type=self._conv_type,
        name='se_reduce')

    self._se_expand = ConvBlock(
        filters=input_shape[-1],
        kernel_size=1,
        causal=self._causal,
        use_bias=True,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        use_batch_norm=False,
        activation=self._gating_activation,
        conv_type=self._conv_type,
        name='se_expand')

    super(StreamSqueezeExcitation, self).build(input_shape)

  def call(self,
           inputs: tf.Tensor,
           states: Optional[nn_layers.States] = None
           ) -> Tuple[tf.Tensor, nn_layers.States]:
    """Calls the layer with the given inputs.

    Args:
      inputs: the input tensor.
      states: a dict of states such that, if any of the keys match for this
          layer, will overwrite the contents of the buffer(s).

    Returns:
      the output tensor and states
    """
    states = dict(states) if states is not None else {}

    x, states = self._pool(inputs, states=states)

    if self._pos_encoding is not None:
      x, states = self._pos_encoding(x, states=states)

    x = self._se_reduce(x)
    x = self._se_expand(x)
    return x * inputs, states


@tf.keras.utils.register_keras_serializable(package='Vision')
class MobileBottleneck(tf.keras.layers.Layer):
  """A depthwise inverted bottleneck block.

  Uses dependency injection to allow flexible definition of different layers
  within this block.
  """

  def __init__(self,
               expansion_layer: tf.keras.layers.Layer,
               feature_layer: tf.keras.layers.Layer,
               projection_layer: tf.keras.layers.Layer,
               attention_layer: Optional[tf.keras.layers.Layer] = None,
               skip_layer: Optional[tf.keras.layers.Layer] = None,
               stochastic_depth_drop_rate: Optional[float] = None,
               **kwargs):
    """Implementation for mobile bottleneck.

    Args:
      expansion_layer: initial layer used for pointwise expansion.
      feature_layer: main layer used for computing 3D features.
      projection_layer: layer used for pointwise projection.
      attention_layer: optional layer used for attention-like operations (e.g.,
          squeeze excite).
      skip_layer: optional skip layer used to project the input before summing
          with the output for the residual connection.
      stochastic_depth_drop_rate: optional drop rate for stochastic depth.
      **kwargs: keyword arguments to be passed to this layer.
    """
    super(MobileBottleneck, self).__init__(**kwargs)

    self._projection_layer = projection_layer
    self._attention_layer = attention_layer
    self._skip_layer = skip_layer
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._identity = tf.keras.layers.Activation(tf.identity)
    self._rezero = nn_layers.Scale(initializer='zeros', name='rezero')

    if stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          stochastic_depth_drop_rate, name='stochastic_depth')
    else:
      self._stochastic_depth = None

    self._feature_layer = feature_layer
    self._expansion_layer = expansion_layer

  def get_config(self):
    """Returns a dictionary containing the config used for initialization."""
    config = {
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
    }
    base_config = super(MobileBottleneck, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self,
           inputs: tf.Tensor,
           states: Optional[nn_layers.States] = None
           ) -> Tuple[tf.Tensor, nn_layers.States]:
    """Calls the layer with the given inputs.

    Args:
      inputs: the input tensor.
      states: a dict of states such that, if any of the keys match for this
          layer, will overwrite the contents of the buffer(s).

    Returns:
      the output tensor and states
    """
    states = dict(states) if states is not None else {}

    x = self._expansion_layer(inputs)
    x, states = self._feature_layer(x, states=states)
    x, states = self._attention_layer(x, states=states)
    x = self._projection_layer(x)

    # Add identity so that the ops are ordered as written. This is useful for,
    # e.g., quantization.
    x = self._identity(x)
    x = self._rezero(x)

    if self._stochastic_depth is not None:
      x = self._stochastic_depth(x)

    if self._skip_layer is not None:
      skip = self._skip_layer(inputs)
    else:
      skip = inputs

    return x + skip, states


@tf.keras.utils.register_keras_serializable(package='Vision')
class SkipBlock(tf.keras.layers.Layer):
  """Skip block for bottleneck blocks."""

  def __init__(
      self,
      out_filters: int,
      downsample: bool = False,
      conv_type: str = '3d',
      kernel_initializer: tf.keras.initializers.Initializer = 'HeNormal',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] =
      tf.keras.regularizers.L2(KERNEL_WEIGHT_DECAY),
      batch_norm_layer: tf.keras.layers.Layer =
      tf.keras.layers.experimental.SyncBatchNormalization,
      batch_norm_momentum: float = 0.99,
      batch_norm_epsilon: float = 1e-3,
      **kwargs):
    """Implementation for skip block.

    Args:
      out_filters: the number of projected output filters.
      downsample: if True, downsamples the input by a factor of 2 by applying
          average pooling with a 3x3 kernel size on the spatial dimensions.
      conv_type: '3d', '2plus1d', or '3d_2plus1d'. '3d' uses the default 3D
          ops. '2plus1d' split any 3D ops into two sequential 2D ops with their
          own batch norm and activation. '3d_2plus1d' is like '2plus1d', but
          uses two sequential 3D ops instead.
      kernel_initializer: kernel initializer for the conv operations.
      kernel_regularizer: kernel regularizer for the conv projection.
      batch_norm_layer: class to use for batch norm.
      batch_norm_momentum: momentum of the batch norm operation.
      batch_norm_epsilon: epsilon of the batch norm operation.
      **kwargs: keyword arguments to be passed to this layer.
    """
    super(SkipBlock, self).__init__(**kwargs)

    self._out_filters = out_filters
    self._downsample = downsample
    self._conv_type = conv_type
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._batch_norm_layer = batch_norm_layer
    self._batch_norm_momentum = batch_norm_momentum
    self._batch_norm_epsilon = batch_norm_epsilon

    self._projection = ConvBlock(
        filters=self._out_filters,
        kernel_size=1,
        conv_type=conv_type,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_batch_norm=True,
        batch_norm_layer=self._batch_norm_layer,
        batch_norm_momentum=self._batch_norm_momentum,
        batch_norm_epsilon=self._batch_norm_epsilon,
        name='skip_project')

    if downsample:
      if self._conv_type == '2plus1d':
        self._pool = tf.keras.layers.AveragePooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='same',
            name='skip_pool')
      else:
        self._pool = tf.keras.layers.AveragePooling3D(
            pool_size=(1, 3, 3),
            strides=(1, 2, 2),
            padding='same',
            name='skip_pool')
    else:
      self._pool = None

  def get_config(self):
    """Returns a dictionary containing the config used for initialization."""
    config = {
        'out_filters': self._out_filters,
        'downsample': self._downsample,
        'conv_type': self._conv_type,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'batch_norm_momentum': self._batch_norm_momentum,
        'batch_norm_epsilon': self._batch_norm_epsilon,
    }
    base_config = super(SkipBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs):
    """Calls the layer with the given inputs."""
    x = inputs
    if self._pool is not None:
      if self._conv_type == '2plus1d':
        x = tf.reshape(x, [-1, tf.shape(x)[2], tf.shape(x)[3], x.shape[4]])

      x = self._pool(x)

      if self._conv_type == '2plus1d':
        x = tf.reshape(
            x,
            [tf.shape(inputs)[0], -1, tf.shape(x)[1],
             tf.shape(x)[2], x.shape[3]])
    return self._projection(x)


@tf.keras.utils.register_keras_serializable(package='Vision')
class MovinetBlock(tf.keras.layers.Layer):
  """A basic block for MoViNets.

  Applies a mobile inverted bottleneck with pointwise expansion, 3D depthwise
  convolution, 3D squeeze excite, pointwise projection, and residual connection.
  """

  def __init__(
      self,
      out_filters: int,
      expand_filters: int,
      kernel_size: Union[int, Sequence[int]] = (3, 3, 3),
      strides: Union[int, Sequence[int]] = (1, 1, 1),
      causal: bool = False,
      activation: nn_layers.Activation = 'swish',
      se_ratio: float = 0.25,
      stochastic_depth_drop_rate: float = 0.,
      conv_type: str = '3d',
      use_positional_encoding: bool = False,
      kernel_initializer: tf.keras.initializers.Initializer = 'HeNormal',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = tf.keras
      .regularizers.L2(KERNEL_WEIGHT_DECAY),
      batch_norm_layer: tf.keras.layers.Layer = tf.keras.layers.experimental
      .SyncBatchNormalization,
      batch_norm_momentum: float = 0.99,
      batch_norm_epsilon: float = 1e-3,
      state_prefix: Optional[str] = None,
      **kwargs):
    """Implementation for MoViNet block.

    Args:
      out_filters: number of output filters for the final projection.
      expand_filters: number of expansion filters after the input.
      kernel_size: kernel size of the main depthwise convolution.
      strides: strides of the main depthwise convolution.
      causal: if True, run the temporal convolutions in causal mode.
      activation: activation to use across all conv operations.
      se_ratio: squeeze excite filters ratio.
      stochastic_depth_drop_rate: optional drop rate for stochastic depth.
      conv_type: '3d', '2plus1d', or '3d_2plus1d'. '3d' uses the default 3D
          ops. '2plus1d' split any 3D ops into two sequential 2D ops with their
          own batch norm and activation. '3d_2plus1d' is like '2plus1d', but
          uses two sequential 3D ops instead.
      use_positional_encoding: add a positional encoding after the (cumulative)
          global average pooling layer in the squeeze excite layer.
      kernel_initializer: kernel initializer for the conv operations.
      kernel_regularizer: kernel regularizer for the conv operations.
      batch_norm_layer: class to use for batch norm.
      batch_norm_momentum: momentum of the batch norm operation.
      batch_norm_epsilon: epsilon of the batch norm operation.
      state_prefix: a prefix string to identify states.
      **kwargs: keyword arguments to be passed to this layer.
    """
    super(MovinetBlock, self).__init__(**kwargs)

    self._kernel_size = normalize_tuple(kernel_size, 3, 'kernel_size')
    self._strides = normalize_tuple(strides, 3, 'strides')

    se_hidden_filters = nn_layers.make_divisible(
        se_ratio * expand_filters, divisor=8)
    self._out_filters = out_filters
    self._expand_filters = expand_filters
    self._kernel_size = kernel_size
    self._causal = causal
    self._activation = activation
    self._se_ratio = se_ratio
    self._downsample = any(s > 1 for s in self._strides)
    self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
    self._conv_type = conv_type
    self._use_positional_encoding = use_positional_encoding
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._batch_norm_layer = batch_norm_layer
    self._batch_norm_momentum = batch_norm_momentum
    self._batch_norm_epsilon = batch_norm_epsilon
    self._state_prefix = state_prefix

    self._expansion = ConvBlock(
        expand_filters,
        (1, 1, 1),
        activation=activation,
        conv_type=conv_type,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_batch_norm=True,
        batch_norm_layer=self._batch_norm_layer,
        batch_norm_momentum=self._batch_norm_momentum,
        batch_norm_epsilon=self._batch_norm_epsilon,
        name='expansion')
    self._feature = StreamConvBlock(
        expand_filters,
        self._kernel_size,
        strides=self._strides,
        depthwise=True,
        causal=self._causal,
        activation=activation,
        conv_type=conv_type,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_batch_norm=True,
        batch_norm_layer=self._batch_norm_layer,
        batch_norm_momentum=self._batch_norm_momentum,
        batch_norm_epsilon=self._batch_norm_epsilon,
        state_prefix=state_prefix,
        name='feature')
    self._projection = ConvBlock(
        out_filters,
        (1, 1, 1),
        activation=None,
        conv_type=conv_type,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        use_batch_norm=True,
        batch_norm_layer=self._batch_norm_layer,
        batch_norm_momentum=self._batch_norm_momentum,
        batch_norm_epsilon=self._batch_norm_epsilon,
        name='projection')
    self._attention = StreamSqueezeExcitation(
        se_hidden_filters,
        activation=activation,
        causal=self._causal,
        conv_type=conv_type,
        use_positional_encoding=use_positional_encoding,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        state_prefix=state_prefix,
        name='se')

  def get_config(self):
    """Returns a dictionary containing the config used for initialization."""
    config = {
        'out_filters': self._out_filters,
        'expand_filters': self._expand_filters,
        'kernel_size': self._kernel_size,
        'strides': self._strides,
        'causal': self._causal,
        'activation': self._activation,
        'se_ratio': self._se_ratio,
        'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
        'conv_type': self._conv_type,
        'use_positional_encoding': self._use_positional_encoding,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'batch_norm_momentum': self._batch_norm_momentum,
        'batch_norm_epsilon': self._batch_norm_epsilon,
        'state_prefix': self._state_prefix,
    }
    base_config = super(MovinetBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def build(self, input_shape):
    """Builds the layer with the given input shape."""
    if input_shape[-1] == self._out_filters and not self._downsample:
      self._skip = None
    else:
      self._skip = SkipBlock(
          self._out_filters,
          downsample=self._downsample,
          conv_type=self._conv_type,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          name='skip')

    self._mobile_bottleneck = MobileBottleneck(
        self._expansion,
        self._feature,
        self._projection,
        attention_layer=self._attention,
        skip_layer=self._skip,
        stochastic_depth_drop_rate=self._stochastic_depth_drop_rate,
        name='bneck')

    super(MovinetBlock, self).build(input_shape)

  def call(self,
           inputs: tf.Tensor,
           states: Optional[nn_layers.States] = None
           ) -> Tuple[tf.Tensor, nn_layers.States]:
    """Calls the layer with the given inputs.

    Args:
      inputs: the input tensor.
      states: a dict of states such that, if any of the keys match for this
          layer, will overwrite the contents of the buffer(s).

    Returns:
      the output tensor and states
    """
    states = dict(states) if states is not None else {}
    return self._mobile_bottleneck(inputs, states=states)


@tf.keras.utils.register_keras_serializable(package='Vision')
class Stem(tf.keras.layers.Layer):
  """Stem layer for video networks.

  Applies an initial convolution block operation.
  """

  def __init__(
      self,
      out_filters: int,
      kernel_size: Union[int, Sequence[int]],
      strides: Union[int, Sequence[int]] = (1, 1, 1),
      causal: bool = False,
      conv_type: str = '3d',
      activation: nn_layers.Activation = 'swish',
      kernel_initializer: tf.keras.initializers.Initializer = 'HeNormal',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = tf.keras
      .regularizers.L2(KERNEL_WEIGHT_DECAY),
      batch_norm_layer: tf.keras.layers.Layer = tf.keras.layers.experimental
      .SyncBatchNormalization,
      batch_norm_momentum: float = 0.99,
      batch_norm_epsilon: float = 1e-3,
      state_prefix: Optional[str] = None,
      **kwargs):
    """Implementation for video model stem.

    Args:
      out_filters: number of output filters.
      kernel_size: kernel size of the convolution.
      strides: strides of the convolution.
      causal: if True, run the temporal convolutions in causal mode.
      conv_type: '3d', '2plus1d', or '3d_2plus1d'. '3d' uses the default 3D
          ops. '2plus1d' split any 3D ops into two sequential 2D ops with their
          own batch norm and activation. '3d_2plus1d' is like '2plus1d', but
          uses two sequential 3D ops instead.
      activation: the input activation name.
      kernel_initializer: kernel initializer for the conv operations.
      kernel_regularizer: kernel regularizer for the conv operations.
      batch_norm_layer: class to use for batch norm.
      batch_norm_momentum: momentum of the batch norm operation.
      batch_norm_epsilon: epsilon of the batch norm operation.
      state_prefix: a prefix string to identify states.
      **kwargs: keyword arguments to be passed to this layer.
    """
    super(Stem, self).__init__(**kwargs)

    self._out_filters = out_filters
    self._kernel_size = normalize_tuple(kernel_size, 3, 'kernel_size')
    self._strides = normalize_tuple(strides, 3, 'strides')
    self._causal = causal
    self._conv_type = conv_type
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._batch_norm_layer = batch_norm_layer
    self._batch_norm_momentum = batch_norm_momentum
    self._batch_norm_epsilon = batch_norm_epsilon
    self._state_prefix = state_prefix

    self._stem = StreamConvBlock(
        filters=self._out_filters,
        kernel_size=self._kernel_size,
        strides=self._strides,
        causal=self._causal,
        activation=self._activation,
        conv_type=self._conv_type,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        use_batch_norm=True,
        batch_norm_layer=self._batch_norm_layer,
        batch_norm_momentum=self._batch_norm_momentum,
        batch_norm_epsilon=self._batch_norm_epsilon,
        state_prefix=self._state_prefix,
        name='stem')

  def get_config(self):
    """Returns a dictionary containing the config used for initialization."""
    config = {
        'out_filters': self._out_filters,
        'kernel_size': self._kernel_size,
        'strides': self._strides,
        'causal': self._causal,
        'activation': self._activation,
        'conv_type': self._conv_type,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'batch_norm_momentum': self._batch_norm_momentum,
        'batch_norm_epsilon': self._batch_norm_epsilon,
        'state_prefix': self._state_prefix,
    }
    base_config = super(Stem, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self,
           inputs: tf.Tensor,
           states: Optional[nn_layers.States] = None
           ) -> Tuple[tf.Tensor, nn_layers.States]:
    """Calls the layer with the given inputs.

    Args:
      inputs: the input tensor.
      states: a dict of states such that, if any of the keys match for this
          layer, will overwrite the contents of the buffer(s).

    Returns:
      the output tensor and states
    """
    states = dict(states) if states is not None else {}
    return self._stem(inputs, states=states)


@tf.keras.utils.register_keras_serializable(package='Vision')
class Head(tf.keras.layers.Layer):
  """Head layer for video networks.

  Applies pointwise projection and global pooling.
  """

  def __init__(
      self,
      project_filters: int,
      conv_type: str = '3d',
      activation: nn_layers.Activation = 'swish',
      kernel_initializer: tf.keras.initializers.Initializer = 'HeNormal',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = tf.keras
      .regularizers.L2(KERNEL_WEIGHT_DECAY),
      batch_norm_layer: tf.keras.layers.Layer = tf.keras.layers.experimental
      .SyncBatchNormalization,
      batch_norm_momentum: float = 0.99,
      batch_norm_epsilon: float = 1e-3,
      state_prefix: Optional[str] = None,
      **kwargs):
    """Implementation for video model head.

    Args:
      project_filters: number of pointwise projection filters.
      conv_type: '3d', '2plus1d', or '3d_2plus1d'. '3d' uses the default 3D
          ops. '2plus1d' split any 3D ops into two sequential 2D ops with their
          own batch norm and activation. '3d_2plus1d' is like '2plus1d', but
          uses two sequential 3D ops instead.
      activation: the input activation name.
      kernel_initializer: kernel initializer for the conv operations.
      kernel_regularizer: kernel regularizer for the conv operations.
      batch_norm_layer: class to use for batch norm.
      batch_norm_momentum: momentum of the batch norm operation.
      batch_norm_epsilon: epsilon of the batch norm operation.
      state_prefix: a prefix string to identify states.
      **kwargs: keyword arguments to be passed to this layer.
    """
    super(Head, self).__init__(**kwargs)

    self._project_filters = project_filters
    self._conv_type = conv_type
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._batch_norm_layer = batch_norm_layer
    self._batch_norm_momentum = batch_norm_momentum
    self._batch_norm_epsilon = batch_norm_epsilon
    self._state_prefix = state_prefix

    self._project = ConvBlock(
        filters=project_filters,
        kernel_size=1,
        activation=activation,
        conv_type=conv_type,
        kernel_regularizer=kernel_regularizer,
        use_batch_norm=True,
        batch_norm_layer=self._batch_norm_layer,
        batch_norm_momentum=self._batch_norm_momentum,
        batch_norm_epsilon=self._batch_norm_epsilon,
        name='project')
    self._pool = nn_layers.GlobalAveragePool3D(
        keepdims=True, causal=False, state_prefix=state_prefix)

  def get_config(self):
    """Returns a dictionary containing the config used for initialization."""
    config = {
        'project_filters': self._project_filters,
        'conv_type': self._conv_type,
        'activation': self._activation,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
        'batch_norm_momentum': self._batch_norm_momentum,
        'batch_norm_epsilon': self._batch_norm_epsilon,
        'state_prefix': self._state_prefix,
    }
    base_config = super(Head, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(
      self,
      inputs: Union[tf.Tensor, Mapping[str, tf.Tensor]],
      states: Optional[nn_layers.States] = None,
  ) -> Tuple[tf.Tensor, nn_layers.States]:
    """Calls the layer with the given inputs.

    Args:
      inputs: the input tensor or dict of endpoints.
      states: a dict of states such that, if any of the keys match for this
          layer, will overwrite the contents of the buffer(s).

    Returns:
      the output tensor and states
    """
    states = dict(states) if states is not None else {}
    x = self._project(inputs)
    return self._pool(x, states=states)


@tf.keras.utils.register_keras_serializable(package='Vision')
class ClassifierHead(tf.keras.layers.Layer):
  """Head layer for video networks.

  Applies dense projection, dropout, and classifier projection. Expects input
  to be pooled vector with shape [batch_size, 1, 1, 1, num_channels]
  """

  def __init__(
      self,
      head_filters: int,
      num_classes: int,
      dropout_rate: float = 0.,
      conv_type: str = '3d',
      activation: nn_layers.Activation = 'swish',
      output_activation: Optional[nn_layers.Activation] = None,
      max_pool_predictions: bool = False,
      kernel_initializer: tf.keras.initializers.Initializer = 'HeNormal',
      kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] =
      tf.keras.regularizers.L2(KERNEL_WEIGHT_DECAY),
      **kwargs):
    """Implementation for video model classifier head.

    Args:
      head_filters: number of dense head projection filters.
      num_classes: number of output classes for the final logits.
      dropout_rate: the dropout rate applied to the head projection.
      conv_type: '3d', '2plus1d', or '3d_2plus1d'. '3d' uses the default 3D
          ops. '2plus1d' split any 3D ops into two sequential 2D ops with their
          own batch norm and activation. '3d_2plus1d' is like '2plus1d', but
          uses two sequential 3D ops instead.
      activation: the input activation name.
      output_activation: optional final activation (e.g., 'softmax').
      max_pool_predictions: apply temporal softmax pooling to predictions.
          Intended for multi-label prediction, where multiple labels are
          distributed across the video. Currently only supports single clips.
      kernel_initializer: kernel initializer for the conv operations.
      kernel_regularizer: kernel regularizer for the conv operations.
      **kwargs: keyword arguments to be passed to this layer.
    """
    super(ClassifierHead, self).__init__(**kwargs)

    self._head_filters = head_filters
    self._num_classes = num_classes
    self._dropout_rate = dropout_rate
    self._conv_type = conv_type
    self._output_activation = output_activation
    self._max_pool_predictions = max_pool_predictions
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer

    self._dropout = tf.keras.layers.Dropout(dropout_rate)
    self._head = ConvBlock(
        filters=head_filters,
        kernel_size=1,
        activation=activation,
        use_bias=True,
        use_batch_norm=False,
        conv_type=conv_type,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        name='head')
    self._classifier = ConvBlock(
        filters=num_classes,
        kernel_size=1,
        kernel_initializer=tf.keras.initializers.random_normal(stddev=0.01),
        kernel_regularizer=None,
        use_bias=True,
        use_batch_norm=False,
        conv_type=conv_type,
        name='classifier')
    self._max_pool = nn_layers.TemporalSoftmaxPool()
    self._squeeze = Squeeze3D()

    output_activation = output_activation if output_activation else 'linear'
    self._cast = tf.keras.layers.Activation(
        output_activation, dtype='float32', name='cast')

  def get_config(self):
    """Returns a dictionary containing the config used for initialization."""
    config = {
        'head_filters': self._head_filters,
        'num_classes': self._num_classes,
        'dropout_rate': self._dropout_rate,
        'conv_type': self._conv_type,
        'output_activation': self._output_activation,
        'max_pool_predictions': self._max_pool_predictions,
        'kernel_initializer': self._kernel_initializer,
        'kernel_regularizer': self._kernel_regularizer,
    }
    base_config = super(ClassifierHead, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    """Calls the layer with the given inputs."""
    # Input Shape: [batch_size, 1, 1, 1, input_channels]
    x = inputs

    x = self._head(x)

    if self._dropout_rate and self._dropout_rate > 0:
      x = self._dropout(x)

    x = self._classifier(x)

    if self._max_pool_predictions:
      x = self._max_pool(x)

    x = self._squeeze(x)
    x = self._cast(x)

    return x
