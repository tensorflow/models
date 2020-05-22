# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Common util modules for MobileNet."""
from typing import Tuple, Text, List, Dict, Optional, Union

import tensorflow as tf

from research.mobilenet.configs import archs

layers = tf.keras.layers


class FixedPadding(layers.Layer):
  """Pads the input along the spatial dimensions independently of input size.

  Pads the input such that if it was used in a convolution with 'VALID' padding,
  the output would have the same dimensions as if the unpadded input was used
  in a convolution with 'SAME' padding.
  """

  def __init__(self,
               kernel_size: Tuple[int, int],
               rate: int = 1,
               name: Text = None):
    """
    Args:
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
      rate: An integer, rate for atrous convolution.
      name: Name of the operation.
    """
    super(FixedPadding, self).__init__(name=name)
    self.kernel_size = kernel_size
    self.rate = rate

  def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
    """

    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].

    Returns:
      A tensor of size [batch, height_out, width_out, channels] with the
      input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    kernel_size = self.kernel_size
    rate = self.rate
    kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                             kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
    pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
    pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
    pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
    padded_inputs = tf.pad(
      tensor=inputs,
      paddings=[[0, 0], [pad_beg[0], pad_end[0]], [pad_beg[1], pad_end[1]],
                [0, 0]])
    return padded_inputs


def make_divisible(value: float,
                   divisor: int,
                   min_value: Optional[float] = None
                   ) -> int:
  """This utility function is to ensure that all layers have a channel number
  that is divisible by 8.
  Args:
    value: original value.
    divisor: the divisor that need to be checked upon.
    min_value: minimum value threshold.

  Returns:
    The adjusted value that divisible again divisor.
  """
  if min_value is None:
    min_value = divisor
  new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_value < 0.9 * value:
    new_value += divisor
  return new_value


def split_divisible(num: int,
                    num_ways: int,
                    divisible_by: int = 8
                    ) -> List[int]:
  """Evenly splits num, num_ways so each piece is a multiple of divisible_by."""
  assert num % divisible_by == 0
  assert num / num_ways >= divisible_by
  # Note: want to round down, we adjust each split to match the total.
  base = num // num_ways // divisible_by * divisible_by
  result = []
  accumulated = 0
  for i in range(num_ways):
    r = base
    while accumulated + r < num * (i + 1) / num_ways:
      r += divisible_by
    result.append(r)
    accumulated += r
  assert accumulated == num
  return result


def width_multiplier_op(filters: int,
                        width_multiplier: float,
                        min_depth: int = 8) -> int:
  """Determine the number of channels given width multiplier"""
  return max(int(filters * width_multiplier), min_depth)


def width_multiplier_op_divisible(filters: int,
                                  width_multiplier: float,
                                  divisible_by: int = 8,
                                  min_depth: int = 8) -> int:
  """Determine the divisible number of channels given width multiplier"""
  return make_divisible(value=filters * width_multiplier,
                        divisor=divisible_by,
                        min_value=min_depth)


def expand_input_by_factor(num_inputs: int,
                           expansion_size: int,
                           divisible_by: int = 8):
  return make_divisible(num_inputs * expansion_size, divisible_by)


def conv2d_block(inputs: tf.Tensor,
                 filters: int,
                 width_multiplier: float,
                 min_depth: int,
                 weight_decay: float,
                 stddev: float,
                 activation_name: Text = 'relu6',
                 normalization_name: Text = 'batch_norm',
                 normalization_params: Dict = {},
                 use_explicit_padding: bool = False,
                 kernel: Union[int, Tuple[int, int]] = (3, 3),
                 strides: Union[int, Tuple[int, int]] = (1, 1),
                 block_id: int = 0
                 ) -> tf.Tensor:
  """Adds an initial convolution layer (with batch normalization).

  Args:
    inputs: Input tensor of shape [batch_size, height, width, channels]
    filters: the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    width_multiplier: controls the width of the network.
      - If `width_multiplier` < 1.0, proportionally decreases the number
            of filters in each layer.
      - If `width_multiplier` > 1.0, proportionally increases the number
            of filters in each layer.
      - If `width_multiplier` = 1, default number of filters from the paper
            are used at each layer.
      This is called `width multiplier (\alpha)` in the original paper.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when width_multiplier < 1, and not an active constraint when
      width_multiplier >= 1.
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.
    activation_name: Name of the activation function
    normalization_name: Name of the normalization layer
    normalization_params: Parameters passed to normalization layer
    use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
      inputs so that the output dimensions are the same as if 'SAME' padding
      were used.
    kernel: An integer or tuple/list of 2 integers, specifying the
      width and height of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution
        along the width and height.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
    block_id: a unique identification designating the block number.

  Returns
    Output tensor of block of shape [batch_size, height, width, filters].
  """

  filters = width_multiplier_op(filters=filters,
                                width_multiplier=width_multiplier,
                                min_depth=min_depth)

  activation_fn = archs.get_activation_function()[activation_name]
  normalization_layer = archs.get_normalization_layer()[
    normalization_name]

  padding = 'SAME'
  if use_explicit_padding:
    padding = 'VALID'
    inputs = FixedPadding(
      kernel_size=kernel,
      name='Conv2d_{}_FP'.format(block_id))(inputs)

  weights_init = tf.keras.initializers.TruncatedNormal(stddev=stddev)
  regularizer = tf.keras.regularizers.L1L2(l2=weight_decay)
  x = layers.Conv2D(filters=filters,
                    kernel_size=kernel,
                    strides=strides,
                    padding=padding,
                    kernel_initializer=weights_init,
                    kernel_regularizer=regularizer,
                    use_bias=False,
                    name='Conv2d_{}'.format(block_id))(inputs)

  x = normalization_layer(axis=-1,
                          name='Conv2d_{}_{}'.format(
                            block_id, normalization_name),
                          **normalization_params)(x)

  outputs = layers.Activation(activation=activation_fn,
                              name='Conv2d_{}_{}'.format(
                                block_id, activation_name))(x)

  return outputs


def se_block(inputs: tf.Tensor,
             squeeze_factor: int = 4,
             divisible_by: int = 8,
             inner_activation_name: Text = 'relu',
             gating_activation_name: Text = 'sigmoid',
             squeeze_input_tensor: Optional[tf.Tensor] = None,
             block_id: int = 0) -> tf.Tensor:
  """Squeeze excite block for Mobilenet V3.

  If the squeeze_input_tensor - or the input_tensor if squeeze_input_tensor is
  None - contains variable dimensions (Nonetype in tensor shape), perform
  average pooling (as the first step in the squeeze operation) by calling
  reduce_mean across the H/W of the input tensor.

  Args:
    inputs: input tensor to apply SE block to.
    divisible_by: ensures all inner dimensions are divisible by this number.
    squeeze_factor: the factor of squeezing in the inner fully connected layer
    inner_activation_name: non-linearity to be used in inner layer.
    gating_activation_name: non-linearity to be used for final gating function.
    squeeze_input_tensor: custom tensor to use for computing gating activation.
     If provided the result will be input_tensor * SE(squeeze_input_tensor)
     instead of input_tensor * SE(input_tensor).
    block_id: a unique identification designating the block number.

  Returns:
    Gated input_tensor. (e.g. X * SE(X))
  """
  prefix = 'block_{}_'.format(block_id)
  if squeeze_input_tensor is None:
    squeeze_input_tensor = inputs

  input_channels = squeeze_input_tensor.shape.as_list()[3]
  output_channels = inputs.shape.as_list()[3]
  squeeze_channels = make_divisible(
    input_channels / squeeze_factor, divisor=divisible_by)

  inner_activation_fn = archs.get_activation_function()[inner_activation_name]
  gating_activation_fn = archs.get_normalization_layer()[
    gating_activation_name]

  x = layers.GlobalAveragePooling2D(name=prefix + 'se_GlobalPool')(inputs)
  x = layers.Reshape((1, 1, input_channels))(x)

  x = layers.Conv2D(squeeze_channels,
                    kernel_size=1,
                    padding='SAME',
                    name=prefix + 'squeeze')(x)
  x = layers.Activation(activation=inner_activation_fn,
                        name=prefix + 'squeeze_{}'.format(
                          inner_activation_name))(x)
  x = layers.Conv2D(output_channels,
                    kernel_size=1,
                    padding='SAME',
                    name=prefix + 'excite')(x)
  x = layers.Activation(activation=gating_activation_fn,
                        name=prefix + 'excite_{}'.format(
                          gating_activation_name))(x)

  x = layers.Multiply(name=prefix + 'se_Mul')([inputs, x])
  return x


def inverted_res_block(inputs: tf.Tensor,
                       filters: int,
                       width_multiplier: float,
                       min_depth: int,
                       weight_decay: float,
                       stddev: float,
                       activation_name: Text = 'relu6',
                       normalization_name: Text = 'batch_norm',
                       normalization_params: Dict = {},
                       dilation_rate: int = 1,
                       expansion_size: int = 6,
                       regularize_depthwise: bool = False,
                       use_explicit_padding: bool = False,
                       residual=True,
                       kernel: Union[int, Tuple[int, int]] = (3, 3),
                       strides: Union[int, Tuple[int, int]] = 1,
                       block_id: int = 1
                       ) -> tf.Tensor:
  """Depthwise Convolution Block with expansion.

  Builds a composite convolution that has the following structure
  expansion (1x1) -> depthwise (kernel_size) -> projection (1x1)

  Args:
    inputs: Input tensor of shape [batch_size, height, width, channels]
    filters: the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    width_multiplier: controls the width of the network.
      - If `width_multiplier` < 1.0, proportionally decreases the number
            of filters in each layer.
      - If `width_multiplier` > 1.0, proportionally increases the number
            of filters in each layer.
      - If `width_multiplier` = 1, default number of filters from the paper
            are used at each layer.
      This is called `width multiplier (\alpha)` in the original paper.
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when width_multiplier < 1, and not an active constraint when
      width_multiplier >= 1.
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.
    activation_name: Name of the activation function
    normalization_name: Name of the normalization layer
    normalization_params: Parameters passed to normalization layer
    dilation_rate: an integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    expansion_size: the size of expansion, could be a constant or a callable.
      If latter it will be provided 'num_inputs' as an input. For forward
      compatibility it should accept arbitrary keyword arguments.
      Default will expand the input by factor of 6.
    regularize_depthwise: Whether or not apply regularization on depthwise.
    use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
      inputs so that the output dimensions are the same as if 'SAME' padding
      were used.
    residual: whether to include residual connection between input
      and output.
    kernel: An integer or tuple/list of 2 integers, specifying the
      width and height of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution
        along the width and height.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
    block_id: a unique identification designating the block number.

  Returns:
    Tensor of depth num_outputs
  """

  prefix = 'block_{}_'.format(block_id)
  filters = width_multiplier_op_divisible(
    filters=filters,
    width_multiplier=width_multiplier,
    min_depth=min_depth)

  activation_fn = archs.get_activation_function()[activation_name]
  normalization_layer = archs.get_normalization_layer()[
    normalization_name]

  weights_init = tf.keras.initializers.TruncatedNormal(stddev=stddev)
  regularizer = tf.keras.regularizers.L1L2(l2=weight_decay)
  depth_regularizer = regularizer if regularize_depthwise else None

  # Expand
  in_channels = inputs.shape.as_list()[-1]
  expended_size = expand_input_by_factor(
    num_inputs=in_channels,
    expansion_size=expansion_size)
  x = layers.Conv2D(filters=expended_size,
                    kernel_size=kernel,
                    strides=(1, 1),
                    padding='SAME',
                    kernel_initializer=weights_init,
                    kernel_regularizer=regularizer,
                    use_bias=False,
                    name=prefix + 'expand')(inputs)

  x = normalization_layer(axis=-1,
                          name=prefix + 'expend_{}'.format(normalization_name),
                          **normalization_params)(x)
  x = layers.Activation(activation=activation_fn,
                        name=prefix + 'expand_{}'.format(activation_name))(x)

  # Depthwise
  padding = 'SAME'
  if use_explicit_padding:
    padding = 'VALID'
    x = FixedPadding(
      kernel_size=kernel,
      name=prefix + 'pad')(x)

  x = layers.DepthwiseConv2D(kernel_size=kernel,
                             padding=padding,
                             depth_multiplier=1,
                             strides=strides,
                             kernel_initializer=weights_init,
                             kernel_regularizer=depth_regularizer,
                             dilation_rate=dilation_rate,
                             use_bias=False,
                             name=prefix + 'depthwise')(x)
  x = normalization_layer(axis=-1,
                          name=prefix + 'depthwise_{}'.format(
                            normalization_name),
                          **normalization_params)(x)
  x = layers.Activation(activation=activation_fn,
                        name=prefix + 'depthwise_{}'.format(activation_name))(x)

  # Project
  x = layers.Conv2D(filters=filters,
                    kernel_size=(1, 1),
                    padding='SAME',
                    strides=(1, 1),
                    kernel_initializer=weights_init,
                    kernel_regularizer=regularizer,
                    use_bias=False,
                    name=prefix + 'project')(x)
  x = normalization_layer(axis=-1,
                          name=prefix + 'project_{}'.format(normalization_name),
                          **normalization_params)(x)

  if (residual and
      # stride check enforces that we don't add residuals when spatial
      # dimensions are None
      strides == 1 and
      # Depth matches
      in_channels == filters):
    x = layers.Add(name=prefix + 'add')([inputs, x])
  return x
