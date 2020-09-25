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
from typing import Tuple, Text, Dict, Optional, Union

import tensorflow as tf

from research.mobilenet.configs import archs

layers = tf.keras.layers

MobileNetConfig = Union[archs.MobileNetV1Config,
                        archs.MobileNetV2Config,
                        archs.MobileNetV3SmallConfig,
                        archs.MobileNetV3LargeConfig,
                        archs.MobileNetV3EdgeTPUConfig]


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


def reduced_kernel_size_for_small_input(input_tensor: tf.Tensor,
                                        kernel_size: Tuple[int, int]
                                        ) -> Tuple[int, int]:
  """Define kernel size which is automatically reduced for small input.

  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are large enough.

  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

  Returns:
    a tensor with the kernel size.
  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = (min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1]))
  return kernel_size_out


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
                           expansion_size: float,
                           divisible_by: int = 8):
  return make_divisible(num_inputs * expansion_size, divisible_by)


def get_initializer(stddev: float) -> tf.keras.initializers.Initializer:
  if stddev < 0:
    weight_intitializer = tf.keras.initializers.GlorotUniform()
  else:
    weight_intitializer = tf.keras.initializers.TruncatedNormal(stddev=stddev)

  return weight_intitializer


def global_pooling_block(inputs: tf.Tensor,
                         block_id: int = 0):
  """Apply global pooling to reduce shape.

  Args:
    inputs: Input tensor of shape [batch_size, height, width, channels]
    block_id: A unique identification designating the block number

  Returns:
    Output tensor of block of shape [batch_size, 1, 1, filters]
  """
  x = layers.GlobalAveragePooling2D(
    data_format='channels_last',
    name='GlobalPool_{}'.format(block_id))(inputs)
  outputs = layers.Reshape((1, 1, x.shape[1]),
                           name='Reshape_{}'.format(block_id))(x)
  return outputs


def conv2d_block(inputs: tf.Tensor,
                 filters: int,
                 width_multiplier: float,
                 min_depth: int,
                 weight_decay: float,
                 stddev: float,
                 activation_name: Text = 'relu6',
                 use_biase: bool = False,
                 normalization: bool = True,
                 normalization_name: Text = 'batch_norm',
                 normalization_params: Dict = {},
                 use_explicit_padding: bool = False,
                 kernel: Union[int, Tuple[int, int]] = (3, 3),
                 strides: Union[int, Tuple[int, int]] = (1, 1),
                 divisable: bool = False,
                 block_id: int = 0,
                 conv_block_id: int = 0,
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
    use_biase: whether use biase
    normalization: whether apply normalization at end.
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
    divisable: Ensure the number of filters is divisable.
    block_id: A unique identification designating the block number.
    conv_block_id: A unique identification designating the conv block number.

  Returns
    Output tensor of block of shape [batch_size, height, width, filters].
  """

  if not divisable:
    filters = width_multiplier_op(filters=filters,
                                  width_multiplier=width_multiplier,
                                  min_depth=min_depth)
  else:
    filters = width_multiplier_op_divisible(
      filters=filters,
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
      name='Conv2d_{}_{}/FP'.format(block_id, conv_block_id))(inputs)

  weights_init = get_initializer(stddev)
  regularizer = tf.keras.regularizers.L1L2(l2=weight_decay)
  x = layers.Conv2D(filters=filters,
                    kernel_size=kernel,
                    strides=strides,
                    padding=padding,
                    kernel_initializer=weights_init,
                    kernel_regularizer=regularizer,
                    use_bias=use_biase,
                    name='Conv2d_{}_{}'.format(block_id, conv_block_id))(inputs)

  if normalization:
    x = normalization_layer(axis=-1,
                            name='Conv2d_{}_{}/{}'.format(
                              block_id, conv_block_id, normalization_name),
                            **normalization_params)(x)

  outputs = layers.Activation(activation=activation_fn,
                              name='Conv2d_{}_{}/{}'.format(
                                block_id, conv_block_id, activation_name))(x)

  return outputs


def depthwise_conv2d_block(inputs: tf.Tensor,
                           filters: int,
                           width_multiplier: float,
                           min_depth: int,
                           weight_decay: float,
                           stddev: float,
                           activation_name: Text = 'relu6',
                           normalization_name: Text = 'batch_norm',
                           normalization_params: Dict = {},
                           dilation_rate: int = 1,
                           regularize_depthwise: bool = False,
                           use_explicit_padding: bool = False,
                           kernel: Union[int, Tuple[int, int]] = (3, 3),
                           strides: Union[int, Tuple[int, int]] = (1, 1),
                           block_id: int = 1
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
    dilation_rate: an integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    regularize_depthwise: Whether or not apply regularization on depthwise.
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

  filters = width_multiplier_op(
    filters=filters,
    width_multiplier=width_multiplier,
    min_depth=min_depth)

  activation_fn = archs.get_activation_function()[activation_name]
  normalization_layer = archs.get_normalization_layer()[
    normalization_name]

  weights_init = get_initializer(stddev)
  regularizer = tf.keras.regularizers.L1L2(l2=weight_decay)
  depth_regularizer = regularizer if regularize_depthwise else None

  padding = 'SAME'
  if use_explicit_padding:
    padding = 'VALID'
    inputs = FixedPadding(
      kernel_size=kernel,
      name='Conv2d_{}/FP'.format(block_id))(inputs)

  # depth-wise convolution
  x = layers.DepthwiseConv2D(kernel_size=kernel,
                             padding=padding,
                             depth_multiplier=1,
                             strides=strides,
                             kernel_initializer=weights_init,
                             kernel_regularizer=depth_regularizer,
                             dilation_rate=dilation_rate,
                             use_bias=False,
                             name='Conv2d_{}/depthwise'.format(
                               block_id))(inputs)
  x = normalization_layer(axis=-1,
                          name='Conv2d_{}/depthwise/{}'.format(
                            block_id, normalization_name),
                          **normalization_params)(x)
  x = layers.Activation(activation=activation_fn,
                        name='Conv2d_{}/depthwise/{}'.format(
                          block_id, activation_name))(x)

  # point-wise convolution
  x = layers.Conv2D(filters=filters,
                    kernel_size=(1, 1),
                    padding='SAME',
                    strides=(1, 1),
                    kernel_initializer=weights_init,
                    kernel_regularizer=regularizer,
                    use_bias=False,
                    name='Conv2d_{}/pointwise'.format(block_id))(x)
  x = normalization_layer(axis=-1,
                          name='Conv2d_{}/pointwise/{}'.format(
                            block_id, normalization_name),
                          **normalization_params)(x)
  outputs = layers.Activation(activation=activation_fn,
                              name='Conv2d_{}/pointwise/{}'.format(
                                block_id, activation_name))(x)
  return outputs


def se_block(inputs: tf.Tensor,
             weight_decay: float,
             stddev: float,
             prefix: Text,
             squeeze_factor: int = 4,
             divisible_by: int = 8,
             inner_activation_name: Text = 'relu',
             gating_activation_name: Text = 'hard_sigmoid',
             squeeze_input_tensor: Optional[tf.Tensor] = None,
             ) -> tf.Tensor:
  """Squeeze excite block for Mobilenet V3.

  If the squeeze_input_tensor - or the input_tensor if squeeze_input_tensor is
  None - contains variable dimensions (Nonetype in tensor shape), perform
  average pooling (as the first step in the squeeze operation) by calling
  reduce_mean across the H/W of the input tensor.

  Args:
    inputs: input tensor to apply SE block to.
    weight_decay: The weight decay to use for regularizing the model.
    stddev: The standard deviation of the trunctated normal weight initializer.
    divisible_by: ensures all inner dimensions are divisible by this number.
    squeeze_factor: the factor of squeezing in the inner fully connected layer.
    inner_activation_name: non-linearity to be used in inner layer.
    gating_activation_name: non-linearity to be used for final gating function.
    squeeze_input_tensor: custom tensor to use for computing gating activation.
     If provided the result will be input_tensor * SE(squeeze_input_tensor)
     instead of input_tensor * SE(input_tensor).
    block_id: a unique identification designating the block number.

  Returns:
    Gated input_tensor. (e.g. X * SE(X))
  """

  prefix = prefix + 'squeeze_excite/'

  if squeeze_input_tensor is None:
    squeeze_input_tensor = inputs

  input_channels = squeeze_input_tensor.shape.as_list()[3]
  output_channels = inputs.shape.as_list()[3]
  squeeze_channels = make_divisible(
    input_channels / squeeze_factor, divisor=divisible_by)

  inner_activation_fn = archs.get_activation_function()[inner_activation_name]
  gating_activation_fn = archs.get_activation_function()[
    gating_activation_name]

  weights_init = get_initializer(stddev)
  regularizer = tf.keras.regularizers.L1L2(l2=weight_decay)

  x = layers.GlobalAveragePooling2D(name=prefix + 'GlobalPool')(inputs)
  x = layers.Reshape((1, 1, input_channels),
                     name=prefix + 'Reshape')(x)

  x = layers.Conv2D(squeeze_channels,
                    kernel_size=1,
                    kernel_initializer=weights_init,
                    kernel_regularizer=regularizer,
                    padding='SAME',
                    name=prefix + 'squeeze')(x)
  x = layers.Activation(activation=inner_activation_fn,
                        name=prefix + 'squeeze/{}'.format(
                          inner_activation_name))(x)
  x = layers.Conv2D(output_channels,
                    kernel_size=1,
                    kernel_initializer=weights_init,
                    kernel_regularizer=regularizer,
                    padding='SAME',
                    name=prefix + 'excite')(x)
  x = layers.Activation(activation=gating_activation_fn,
                        name=prefix + 'excite/{}'.format(
                          gating_activation_name))(x)

  x = layers.Multiply(name=prefix + 'Mul')([inputs, x])
  return x


def inverted_res_block(inputs: tf.Tensor,
                       filters: int,
                       width_multiplier: float,
                       min_depth: int,
                       weight_decay: float,
                       stddev: float,
                       activation_name: Text = 'relu6',
                       depthwise_activation_name: Text = None,
                       normalization_name: Text = 'batch_norm',
                       normalization_params: Dict = {},
                       dilation_rate: int = 1,
                       expansion_size: float = 6.,
                       regularize_depthwise: bool = False,
                       use_explicit_padding: bool = False,
                       depthwise: bool = True,
                       residual=True,
                       squeeze_factor: Optional[int] = None,
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
    activation_name: Name of the activation function for inner Conv.
    depthwise_activation_name: Name of the activation function for deptwhise only.
    normalization_name: Name of the normalization layer.
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
    depthwise: whether to uses fused convolutions instead of depthwise
    residual: whether to include residual connection between input
      and output.
    squeeze_factor: the factor of squeezing in the inner fully connected layer
      for Squeeze excite block.
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

  prefix = 'expanded_conv_{}/'.format(block_id)
  filters = width_multiplier_op_divisible(
    filters=filters,
    width_multiplier=width_multiplier,
    min_depth=min_depth)

  if not depthwise_activation_name:
    depthwise_activation_name = activation_name

  activation_fn = archs.get_activation_function()[activation_name]
  depth_activation_fn = archs.get_activation_function()[
    depthwise_activation_name]
  normalization_layer = archs.get_normalization_layer()[
    normalization_name]

  weights_init = get_initializer(stddev)
  regularizer = tf.keras.regularizers.L1L2(l2=weight_decay)
  depth_regularizer = regularizer if regularize_depthwise else None

  x = inputs
  in_channels = inputs.shape.as_list()[-1]
  # Expand
  if expansion_size > 1:
    expended_size = expand_input_by_factor(
      num_inputs=in_channels,
      expansion_size=expansion_size)
    x = layers.Conv2D(filters=expended_size,
                      kernel_size=(1, 1) if depthwise else kernel,
                      strides=(1, 1) if depthwise else strides,
                      padding='SAME',
                      kernel_initializer=weights_init,
                      kernel_regularizer=regularizer,
                      use_bias=False,
                      name=prefix + 'expand')(x)

    x = normalization_layer(axis=-1,
                            name=prefix + 'expand/{}'.format(
                              normalization_name),
                            **normalization_params)(x)
    x = layers.Activation(activation=activation_fn,
                          name=prefix + 'expand/{}'.format(activation_name))(x)

  # Depthwise
  if depthwise:
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
                            name=prefix + 'depthwise/{}'.format(
                              normalization_name),
                            **normalization_params)(x)
    x = layers.Activation(activation=depth_activation_fn,
                          name=prefix + 'depthwise/{}'.format(
                            depthwise_activation_name))(x)

  if squeeze_factor:
    x = se_block(inputs=x,
                 squeeze_factor=squeeze_factor,
                 stddev=stddev,
                 weight_decay=weight_decay,
                 prefix=prefix)

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
                          name=prefix + 'project/{}'.format(normalization_name),
                          **normalization_params)(x)

  if (residual and
      # stride check enforces that we don't add residuals when spatial
      # dimensions are None
      strides == 1 and
      # Depth matches
      in_channels == filters):
    x = layers.Add(name=prefix + 'add')([inputs, x])
  return x


def mobilenet_base(inputs: tf.Tensor,
                   config: MobileNetConfig
                   ) -> tf.Tensor:
  """Build the base MobileNet architecture."""

  input_shape = inputs.get_shape().as_list()
  if len(input_shape) != 4:
    raise ValueError('Expected rank 4 input, was: %d' % len(input_shape))

  width_multiplier = config.width_multiplier
  min_depth = config.min_depth
  output_stride = config.output_stride
  use_explicit_padding = config.use_explicit_padding
  # regularization
  weight_decay = config.weight_decay
  stddev = config.stddev
  regularize_depthwise = config.regularize_depthwise
  # normalization
  normalization_name = config.normalization_name
  # need change accordingly if different normalization fn is used
  normalization_params = {
    'center': True,
    'scale': True,
    'momentum': config.batch_norm_decay,
    'epsilon': config.batch_norm_epsilon
  }
  # used only for MobileNetV2 in this base function
  finegrain_classification_mode = config.finegrain_classification_mode
  # base blocks definition
  blocks = config.blocks

  if width_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

  if output_stride is not None:
    if isinstance(config, archs.MobileNetV1Config):
      if output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')
    else:
      if output_stride == 0 or (output_stride > 1 and output_stride % 2):
        raise ValueError('Output stride must be None, 1 or a multiple of 2.')

  # This adjustment applies to V2 and V3
  if (not isinstance(config, archs.MobileNetV1Config)
      and finegrain_classification_mode
      and width_multiplier < 1.0):
    blocks[-1].filters /= width_multiplier

  # The current_stride variable keeps track of the output stride of the
  # activations, i.e., the running product of convolution strides up to the
  # current network layer. This allows us to invoke atrous convolution
  # whenever applying the next convolution would result in the activations
  # having output stride larger than the target output_stride.
  current_stride = 1

  # The atrous convolution rate parameter.
  rate = 1

  net = inputs
  con_block_tracker = 0
  for i, block_def in enumerate(blocks):
    if output_stride is not None and current_stride == output_stride:
      # If we have reached the target output_stride, then we need to employ
      # atrous convolution with stride=1 and multiply the atrous rate by the
      # current unit's stride for use in subsequent layers.
      layer_stride = 1
      layer_rate = rate
      rate *= block_def.stride
    else:
      layer_stride = block_def.stride
      layer_rate = 1
      current_stride *= block_def.stride

    if block_def.block_type == archs.BlockType.Conv.value:

      # For V2 and V3, divisable should be explicitly ensured.
      divisable = False
      if not isinstance(config, archs.MobileNetV1Config):
        divisable = True

      net = conv2d_block(
        inputs=net,
        filters=block_def.filters,
        kernel=block_def.kernel,
        strides=block_def.stride,
        activation_name=block_def.activation_name,
        use_biase=block_def.use_biase,
        normalization=block_def.normalization,
        width_multiplier=width_multiplier,
        min_depth=min_depth,
        weight_decay=weight_decay,
        stddev=stddev,
        use_explicit_padding=use_explicit_padding,
        normalization_name=normalization_name,
        normalization_params=normalization_params,
        divisable=divisable,
        block_id=i,
        conv_block_id=con_block_tracker
      )
      con_block_tracker += 1
    elif block_def.block_type == archs.BlockType.DepSepConv.value:
      net = depthwise_conv2d_block(
        inputs=net,
        filters=block_def.filters,
        kernel=block_def.kernel,
        strides=layer_stride,
        activation_name=block_def.activation_name,
        dilation_rate=layer_rate,
        width_multiplier=width_multiplier,
        min_depth=min_depth,
        weight_decay=weight_decay,
        stddev=stddev,
        normalization_name=normalization_name,
        normalization_params=normalization_params,
        regularize_depthwise=regularize_depthwise,
        use_explicit_padding=use_explicit_padding,
        block_id=i
      )
    elif block_def.block_type == archs.BlockType.InvertedResConv.value:
      use_rate = rate
      if layer_rate > 1 and block_def.kernel != (1, 1):
        # We will apply atrous rate in the following cases:
        # 1) When kernel_size is not in params, the operation then uses
        #   default kernel size 3x3.
        # 2) When kernel_size is in params, and if the kernel_size is not
        #   equal to (1, 1) (there is no need to apply atrous convolution to
        #   any 1x1 convolution).
        use_rate = layer_rate
      net = inverted_res_block(
        inputs=net,
        filters=block_def.filters,
        kernel=block_def.kernel,
        strides=layer_stride,
        expansion_size=block_def.expansion_size,
        squeeze_factor=block_def.squeeze_factor,
        activation_name=block_def.activation_name,
        depthwise=block_def.depthwise,
        residual=block_def.residual,
        dilation_rate=use_rate,
        width_multiplier=width_multiplier,
        min_depth=min_depth,
        weight_decay=weight_decay,
        stddev=stddev,
        regularize_depthwise=regularize_depthwise,
        use_explicit_padding=use_explicit_padding,
        normalization_name=normalization_name,
        normalization_params=normalization_params,
        block_id=i
      )
    elif block_def.block_type == archs.BlockType.GlobalPooling.value:
      net = global_pooling_block(
        inputs=net,
        block_id=i
      )
    else:
      raise ValueError('Unknown block type {} for layer {}'.format(
        block_def.block_type, i))
  return net


def mobilenet_head(inputs: tf.Tensor,
                   config: MobileNetConfig
                   ) -> tf.Tensor:
  """Build the head of MobileNet architecture."""
  dropout_keep_prob = config.dropout_keep_prob
  num_classes = config.num_classes
  spatial_squeeze = config.spatial_squeeze
  global_pool = config.global_pool

  # build top
  if global_pool:
    # global average pooling.
    x = layers.GlobalAveragePooling2D(data_format='channels_last',
                                      name='top/GlobalPool')(inputs)
    x = layers.Reshape((1, 1, x.shape[1]), name='top/Reshape')(x)
  else:
    # pooling with a fixed kernel size
    kernel_size = reduced_kernel_size_for_small_input(inputs, (7, 7))
    x = layers.AvgPool2D(pool_size=kernel_size,
                         padding='VALID',
                         data_format='channels_last',
                         name='top/AvgPool')(inputs)

  x = layers.Dropout(rate=1 - dropout_keep_prob,
                     name='top/Dropout')(x)
  # 1 x 1 x num_classes
  x = layers.Conv2D(filters=num_classes,
                    kernel_size=(1, 1),
                    padding='SAME',
                    bias_initializer=tf.keras.initializers.Zeros(),
                    name='top/Conv2d_1x1_output')(x)
  if spatial_squeeze:
    x = layers.Reshape(target_shape=(num_classes,),
                       name='top/SpatialSqueeze')(x)

  x = layers.Activation(activation='softmax',
                        name='top/Predictions')(x)

  return x
