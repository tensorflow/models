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

"""Customized keras layers used in the EdgeTPU models."""

from collections.abc import MutableMapping
import inspect
from typing import Any, Optional, Union
import tensorflow as tf

from official.modeling import tf_utils


class GroupConv2D(tf.keras.layers.Conv2D):
  """2D group convolution as a Keras Layer."""

  def __init__(self,
               filters: int,
               kernel_size: Union[int, tuple[int, int]],
               groups: int,
               strides: tuple[int, int] = (1, 1),
               padding: str = 'valid',
               data_format: str = 'channels_last',
               dilation_rate: tuple[int, int] = (1, 1),
               activation: Any = None,
               use_bias: bool = True,
               kernel_initializer: Any = 'glorot_uniform',
               bias_initializer: Any = 'zeros',
               kernel_regularizer: Any = None,
               bias_regularizer: Any = None,
               activity_regularizer: Any = None,
               kernel_constraint: Any = None,
               bias_constraint: Any = None,
               batch_norm_layer: Optional[tf.keras.layers.Layer] = None,
               bn_epsilon: float = 1e-3,
               bn_momentum: float = 0.99,
               **kwargs: Any) -> tf.keras.layers.Layer:
    """Creates a 2D group convolution keras layer.

    Args:
      filters: Integer, the dimensionality of the output space (i.e. the number
        of output filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the height
        and width of the 2D convolution window. Can be a single integer to
        specify the same value for all spatial dimensions.
      groups: The number of input/output channel groups.
      strides: An integer or tuple/list of n integers, specifying the stride
        length of the convolution. Specifying any stride value != 1 is
        incompatible with specifying any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: The ordering of the dimensions in the inputs. `channels_last`
        corresponds to inputs with shape `(batch_size, height, width, channels)`
      dilation_rate: an integer or tuple/list of 2 integers, specifying the
        dilation rate to use for dilated convolution. Can be a single integer to
        specify the same value for all spatial dimensions. Currently, specifying
        any `dilation_rate` value != 1 is incompatible with specifying any
        stride value != 1.
      activation: Activation function to use. If you don't specify anything, no
        activation is applied ( see `keras.activations`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix ( see
        `keras.initializers`).
      bias_initializer: Initializer for the bias vector ( see
        `keras.initializers`).
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix (see `keras.regularizers`).
      bias_regularizer: Regularizer function applied to the bias vector ( see
        `keras.regularizers`).
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation") ( see `keras.regularizers`).
      kernel_constraint: Constraint function applied to the kernel matrix ( see
        `keras.constraints`).
      bias_constraint: Constraint function applied to the bias vector ( see
        `keras.constraints`).
      batch_norm_layer: The batch normalization layer to use. This is typically
        tf.keras.layer.BatchNormalization or a derived class.
      bn_epsilon: Batch normalization epsilon.
      bn_momentum: Momentum used for moving average in batch normalization.
      **kwargs: Additional keyword arguments.
    Input shape:
      4D tensor with shape: `(batch_size, rows, cols, channels)`
    Output shape:
      4D tensor with shape: `(batch_size, new_rows, new_cols, filters)` `rows`
        and `cols` values might have changed due to padding.

    Returns:
      A tensor of rank 4 representing
      `activation(GroupConv2D(inputs, kernel) + bias)`.

    Raises:
      ValueError: if groups < 1 or groups > filters
      ValueError: if data_format is not "channels_last".
      ValueError: if `padding` is not `same` or `valid`.
      ValueError: if `batch_norm_layer` is not a callable when provided.
      ValueError: when both `strides` > 1 and `dilation_rate` > 1.
    """
    if groups <= 1 or groups > filters:
      raise ValueError(f'Number of groups {groups} should be greater than 1 and'
                       f' less or equal than the output filters {filters}.')
    self._groups = groups
    if data_format != 'channels_last':
      raise ValueError(
          'GroupConv2D expects input to be in channels_last format.')

    if padding.lower() not in ('same', 'valid'):
      raise ValueError('Valid padding options are : same, or valid.')

    self.use_batch_norm = False
    if batch_norm_layer is not None:
      if not inspect.isclass(batch_norm_layer):
        raise ValueError('batch_norm_layer is not a class.')
      self.use_batch_norm = True
    self.bn_epsilon = bn_epsilon
    self.bn_momentum = bn_momentum
    self.batch_norm_layer = []
    if self.use_batch_norm:
      self.batch_norm_layer = [
          batch_norm_layer(
              axis=-1, momentum=self.bn_momentum, epsilon=self.bn_epsilon)
          for i in range(self._groups)
      ]

    super().__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        groups=1,
        **kwargs)  # pytype: disable=bad-return-type  # typed-keras

  def build(self, input_shape: tuple[int, ...]) -> None:
    """Builds GroupConv2D layer as a collection of smaller Conv2D layers."""
    input_shape = tf.TensorShape(input_shape)
    input_channel = self._get_input_channel(input_shape)
    if input_channel % self._groups != 0:
      raise ValueError(
          f'Number of input channels: {input_channel} are not divisible '
          f'by number of groups: {self._groups}.')

    self.group_input_channel = int(input_channel / self._groups)
    self.group_output_channel = int(self.filters / self._groups)
    self.group_kernel_shape = self.kernel_size + (self.group_input_channel,
                                                  self.group_output_channel)

    self.kernel = []
    self.bias = []
    for g in range(self._groups):
      self.kernel.append(
          self.add_weight(
              name='kernel_{}'.format(g),
              shape=self.group_kernel_shape,
              initializer=tf_utils.clone_initializer(self.kernel_initializer),
              regularizer=self.kernel_regularizer,
              constraint=self.kernel_constraint,
              trainable=True,
              dtype=self.dtype))
      if self.use_bias:
        self.bias.append(
            self.add_weight(
                name='bias_{}'.format(g),
                shape=(self.group_output_channel,),
                initializer=tf_utils.clone_initializer(self.bias_initializer),
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype))
    channel_axis = self._get_channel_axis()
    self.input_spec = tf.keras.layers.InputSpec(
        ndim=self.rank + 2, axes={channel_axis: input_channel})

    self._build_conv_op_data_shape = input_shape[-(self.rank + 1):]
    self._build_input_channel = input_channel
    self._padding_op = self._get_padding_op()
    # channels_last corresponds to 'NHWC' data format.
    self._conv_op_data_format = 'NHWC'

    self.bn_layers = []
    if self.use_batch_norm:
      for group_index in range(self._groups):
        self.bn_layers.append(self.batch_norm_layer[group_index])

    self.built = True

  def call(self, inputs: Any, training: Optional[bool] = None) -> Any:
    """Performs the GroupConv2D operation on the inputs."""
    input_slices = tf.split(inputs, num_or_size_splits=self._groups, axis=-1)
    output_slices = []
    for i in range(self._groups):
      # Apply conv2d to each slice
      output_slice = tf.nn.conv2d(
          input_slices[i],
          self.kernel[i],
          strides=self.strides,
          padding=self._padding_op,
          data_format=self._conv_op_data_format,
          dilations=self.dilation_rate)

      if self.use_bias:
        output_slice = tf.nn.bias_add(
            output_slice, self.bias[i], data_format='NHWC')

      # Apply batch norm after bias addition.
      if self.use_batch_norm:
        output_slice = self.bn_layers[i](output_slice, training=training)

      if self.activation is not None:
        output_slice = self.activation(output_slice)

      output_slices.append(output_slice)

    # Concat the outputs back along the channel dimension
    outputs = tf.concat(output_slices, axis=-1)
    return outputs

  def get_config(self) -> MutableMapping[str, Any]:
    """Enables serialization for the group convolution layer."""
    config = super().get_config()
    config['groups'] = self._groups
    config['batch_norm_layer'] = self.batch_norm_layer
    config['bn_epsilon'] = self.bn_epsilon
    config['bn_momentum'] = self.bn_momentum
    return config

  @classmethod
  def from_config(cls, config):
    """Creates a layer from its config.

    This method is the reverse of `get_config`, capable of instantiating the
    same layer from the config dictionary. It does not handle layer connectivity
    (handled by Network), nor weights (handled by `set_weights`).

    Also, the get_config returns a config with a list type of `batch_norm_layer`
    we need to convert it either to None or the batch_norm class.

    Arguments:
        config: A Python dictionary, typically the output of get_config.

    Returns:
        A layer instance.
    """
    if not config['batch_norm_layer']:
      config['batch_norm_layer'] = None
    else:
      config['batch_norm_layer'] = type(config['batch_norm_layer'][0])
    return cls(**config)


class GroupConv2DKerasModel(tf.keras.Model):
  """2D group convolution as a keras model."""

  def __init__(self,
               filters: int,
               kernel_size: tuple[int, int],
               groups: int,
               batch_norm_layer: Optional[tf.keras.layers.Layer] = None,
               bn_epsilon: float = 1e-3,
               bn_momentum: float = 0.99,
               data_format: str = 'channels_last',
               padding: str = 'valid',
               **kwargs: Any) -> tf.keras.Model:
    """Creates a 2D group convolution layer as a keras model.

    Args:
      filters: Integer, the dimensionality of the output space (i.e. the number
        of output filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the height
        and width of the 2D convolution window. Can be a single integer to
        specify the same value for all spatial dimensions.
      groups: The number of input/output channel groups.
      batch_norm_layer: The batch normalization layer to use. This is typically
        tf.keras.layer.BatchNormalization or a derived class.
      bn_epsilon: Batch normalization epsilon.
      bn_momentum: Momentum used for moving average in batch normalization.
      data_format: The ordering of the dimensions in the inputs. `channels_last`
        corresponds to inputs with shape `(batch_size, height, width, channels)`
      padding: one of `"valid"` or `"same"` (case-insensitive).
      **kwargs: Additional keyword arguments passed to the underlying conv
        layers.

    Raises:
      ValueError: if groups < 1 or groups > filters
      ValueError: if `batch_norm_layer` is not a callable when provided.
      ValueError: if `data_format` is not channels_last
      ValueError: if `padding` is not `same` or `valid`.
    """
    super().__init__()
    self.conv_layers = []
    self.bn_layers = []
    per_conv_filter_size = filters / groups

    if groups <= 1 or groups >= filters:
      raise ValueError('Number of groups should be greater than 1 and less '
                       'than the output filters.')

    self.batch_norm_layer = batch_norm_layer
    self.use_batch_norm = False
    if self.batch_norm_layer is not None:
      if not inspect.isclass(self.batch_norm_layer):  # pytype: disable=not-supported-yet
        raise ValueError('batch_norm_layer is not a class.')
      self.use_batch_norm = True

    if 'activation' in kwargs.keys():
      self.activation = tf.keras.activations.get(kwargs['activation'])
      kwargs.pop('activation')
    else:
      self.activation = None

    if data_format != 'channels_last':
      raise ValueError(
          'GroupConv2D expects input to be in channels_last format.')

    if padding.lower() not in ('same', 'valid'):
      raise ValueError('Valid padding options are : same, or valid.')

    self._groups = groups
    for _ in range(self._groups):
      # Override the activation so that batchnorm can be applied after the conv.
      self.conv_layers.append(
          tf.keras.layers.Conv2D(per_conv_filter_size, kernel_size, **kwargs))

    if self.use_batch_norm:
      for _ in range(self._groups):
        self.bn_layers.append(
            self.batch_norm_layer(
                axis=-1, momentum=bn_momentum, epsilon=bn_epsilon))  # pytype: disable=bad-return-type  # typed-keras

  def call(self, inputs: Any) -> Any:
    """Applies 2d group convolution on the inputs."""
    input_shape = inputs.get_shape().as_list()
    if input_shape[-1] % self._groups != 0:
      raise ValueError(
          f'Number of input channels: {input_shape[-1]} are not divisible '
          f'by number of groups: {self._groups}.')
    input_slices = tf.split(inputs, num_or_size_splits=self._groups, axis=-1)
    output_slices = []
    for g in range(self._groups):
      output_slice = self.conv_layers[g](input_slices[g])
      if self.use_batch_norm:
        output_slice = self.bn_layers[g](output_slice)
      output_slice = self.activation(output_slice)
      output_slices.append(output_slice)

    outputs = tf.concat(output_slices, axis=-1)
    return outputs


def _nnapi_scalar(value, dtype):
  # Resolves "Scalar operand should be constant" at cost of broadcasting
  return tf.constant(value, dtype=dtype, shape=(1,))


def _fqop(x, min_val=-128, max_val=127):
  """Wraps an op x with fake quant op and given min/max."""
  return tf.quantization.fake_quant_with_min_max_args(
      x, min=min_val, max=max_val)


def argmax(input_tensor,
           axis=-1,
           output_type: tf.DType = tf.dtypes.float32,
           name: Optional[str] = None,
           keepdims: bool = False,
           epsilon: Optional[float] = None):
  """Returns the index with the largest value across axes of a tensor.

  Approximately tf.compat.v1.argmax, but not equivalent. If arithmetic allows
  value to be anomalously close to the maximum, but not equal to it, the
  behavior is undefined.

  Args:
    input_tensor: A Tensor.
    axis: A Value. Must be in the range [-rank(input), rank(input)). Describes
      which axis of the input Tensor to reduce across. For vectors, use axis =
      0.
    output_type: An optional tf.DType. Note that default is different from
      tflite (int64) to make default behavior compatible with darwinn.
    name: Optional name for operations.
    keepdims: If true, retains reduced dimensions with length 1.
    epsilon: Optional small number which is intended to be always below
      quantization threshold, used to distinguish equal and not equal numbers.

  Returns:
    A Tensor of type output_type.
  """
  fqop = _fqop if output_type.is_floating else tf.identity
  safe_axis = axis
  if safe_axis < 0:
    safe_axis = len(input_tensor.shape) + safe_axis
  reduction_size = input_tensor.shape[axis]
  axis_max = tf.math.reduce_max(input_tensor, axis=axis, keepdims=True)
  zero_if_max = tf.subtract(axis_max, input_tensor)
  eps = epsilon if epsilon else 1e-6
  if input_tensor.dtype.is_floating:
    zero_if_max_else_eps = tf.math.minimum(
        _nnapi_scalar(eps, input_tensor.dtype), zero_if_max)
    zero_if_max_else_one = zero_if_max_else_eps * _nnapi_scalar(
        1 / eps, input_tensor.dtype)
  elif input_tensor.dtype.is_integer:
    zero_if_max_else_one = tf.math.minimum(
        _nnapi_scalar(1, input_tensor.dtype), zero_if_max)
  else:
    raise ValueError('Please specify epsilon for unknown input data type')

  # Input type ends here, output type starts here
  zero_if_max_else_one = tf.cast(zero_if_max_else_one, dtype=output_type)
  zero_if_max_else_one = fqop(zero_if_max_else_one)
  one_if_max_else_zero = fqop(
      tf.math.subtract(
          fqop(_nnapi_scalar(1, output_type)), zero_if_max_else_one))
  rev_index = tf.range(reduction_size, 0, -1, dtype=output_type)
  for index in range(safe_axis + 1, len(input_tensor.shape)):
    rev_index = tf.expand_dims(rev_index, axis=index - safe_axis)
  rev_index = fqop(rev_index)
  rev_index_if_max_else_zero = fqop(
      tf.math.multiply(one_if_max_else_zero, rev_index))
  reverse_argmax = fqop(
      tf.math.reduce_max(
          rev_index_if_max_else_zero, axis=axis, keepdims=keepdims, name=name))
  # Final operation obtains name if argmax layer if provided
  return fqop(
      tf.math.subtract(
          fqop(_nnapi_scalar(reduction_size, output_type)),
          reverse_argmax,
          name=name))


class ArgmaxKerasLayer(tf.keras.layers.Layer):
  """Implements argmax as a keras model."""

  def __init__(self,
               axis=-1,
               name=None,
               output_type=tf.dtypes.int32,
               **kwargs: Any) -> tf.keras.Model:
    """Implements argmax as a keras model.

    Args:
      axis: A Value. Must be in the range [-rank(input), rank(input)). Describes
        which axis of the input Tensor to reduce across. For vectors, use axis =
        0.
      name: Optional name for operations.
      output_type: An optional tf.DType.
      **kwargs: Other arguments passed to model constructor.

    Returns:
      A Tensor of type output_type.
    """
    super().__init__(name=name, **kwargs)
    self.axis = axis
    self.output_type = output_type  # pytype: disable=bad-return-type  # typed-keras

  def call(self, inputs: Any) -> Any:
    """Applies argmax on the inputs."""
    return argmax(
        input_tensor=inputs,
        axis=self.axis,
        output_type=self.output_type,
        name=self.name)
