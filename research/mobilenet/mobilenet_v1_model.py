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
"""MobileNet v1.

Adapted from tf.keras.applications.mobilenet.MobileNet().

MobileNet is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and different
head (for example: embeddings, localization and classification).

As described in https://arxiv.org/abs/1704.04861.

  MobileNets: Efficient Convolutional Neural Networks for
    Mobile Vision Applications
  Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang,
    Tobias Weyand, Marco Andreetto, Hartwig Adam

"""

import logging
from typing import Tuple, Union

import tensorflow as tf

from research.mobilenet import common_modules
from research.mobilenet.configs.mobilenet_config import MobileNetV1Config
from research.mobilenet.configs.mobilenet_config import Conv, DepSepConv

layers = tf.keras.layers


def _reduced_kernel_size_for_small_input(input_tensor: tf.Tensor,
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


def _depthwise_conv2d_block(inputs: tf.Tensor,
                            filters: int,
                            width_multiplier: float,
                            min_depth: int,
                            weight_decay: float,
                            stddev: float,
                            batch_norm_decay: float,
                            batch_norm_epsilon: float,
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
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
        in batch norm.
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

  filters = common_modules.width_multiplier_op(
    filters=filters,
    width_multiplier=width_multiplier,
    min_depth=min_depth)

  weights_init = tf.keras.initializers.TruncatedNormal(stddev=stddev)
  regularizer = tf.keras.regularizers.L1L2(l2=weight_decay)
  depth_regularizer = regularizer if regularize_depthwise else None

  padding = 'SAME'
  if use_explicit_padding:
    padding = 'VALID'
    inputs = common_modules.FixedPadding(
      kernel_size=kernel,
      name='Conv2d_{}_FP'.format(block_id))(inputs)

  # depth-wise convolution
  x = layers.DepthwiseConv2D(kernel_size=kernel,
                             padding=padding,
                             depth_multiplier=1,
                             strides=strides,
                             kernel_initializer=weights_init,
                             kernel_regularizer=depth_regularizer,
                             dilation_rate=dilation_rate,
                             use_bias=False,
                             name='Conv2d_{}_dw'.format(block_id))(inputs)
  x = layers.BatchNormalization(epsilon=batch_norm_epsilon,
                                momentum=batch_norm_decay,
                                axis=-1,
                                name='Conv2d_{}_dw_BN'.format(block_id))(x)
  x = layers.ReLU(max_value=6.,
                  name='Conv2d_{}_dw_ReLU'.format(block_id))(x)

  # point-wise convolution
  x = layers.Conv2D(filters=filters,
                    kernel_size=(1, 1),
                    padding='SAME',
                    strides=(1, 1),
                    kernel_initializer=weights_init,
                    kernel_regularizer=regularizer,
                    use_bias=False,
                    name='Conv2d_{}_pw'.format(block_id))(x)
  x = layers.BatchNormalization(epsilon=batch_norm_epsilon,
                                momentum=batch_norm_decay,
                                axis=-1,
                                name='Conv2d_{}_pw_BN'.format(block_id))(x)
  outputs = layers.ReLU(max_value=6.,
                        name='Conv2d_{}_pw_ReLU'.format(block_id))(x)

  return outputs


def mobilenet_v1_base(inputs: tf.Tensor,
                      config: MobileNetV1Config
                      ) -> tf.Tensor:
  """Build the base MobileNet architecture."""

  min_depth = config.min_depth
  width_multiplier = config.width_multiplier
  weight_decay = config.weight_decay
  stddev = config.stddev
  regularize_depthwise = config.regularize_depthwise
  batch_norm_decay = config.batch_norm_decay
  batch_norm_epsilon = config.batch_norm_epsilon
  output_stride = config.output_stride
  use_explicit_padding = config.use_explicit_padding

  if width_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

  if output_stride is not None and output_stride not in [8, 16, 32]:
    raise ValueError('Only allowed output_stride values are 8, 16, 32.')

  # The current_stride variable keeps track of the output stride of the
  # activations, i.e., the running product of convolution strides up to the
  # current network layer. This allows us to invoke atrous convolution
  # whenever applying the next convolution would result in the activations
  # having output stride larger than the target output_stride.
  current_stride = 1

  # The atrous convolution rate parameter.
  rate = 1

  net = inputs
  for i, block_def in enumerate(config.blocks):
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
    if block_def.block_type == Conv:
      net = common_modules.conv2d_block(
        inputs=net,
        filters=block_def.filters,
        kernel=block_def.kernel,
        strides=block_def.stride,
        width_multiplier=width_multiplier,
        min_depth=min_depth,
        weight_decay=weight_decay,
        stddev=stddev,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon,
        use_explicit_padding=use_explicit_padding,
        block_id=i
      )
    elif block_def.block_type == DepSepConv:
      net = _depthwise_conv2d_block(
        inputs=net,
        filters=block_def.filters,
        kernel=block_def.kernel,
        strides=layer_stride,
        dilation_rate=layer_rate,
        width_multiplier=width_multiplier,
        min_depth=min_depth,
        weight_decay=weight_decay,
        stddev=stddev,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon,
        regularize_depthwise=regularize_depthwise,
        use_explicit_padding=use_explicit_padding,
        block_id=i
      )
    else:
      raise ValueError('Unknown block type {} for layer {}'.format(
        block_def.block_type, i))
  return net


def mobilenet_v1(input_shape: Tuple[int, int, int] = (224, 224, 3),
                 config: MobileNetV1Config = MobileNetV1Config()
                 ) -> tf.keras.models.Model:
  """Instantiates the MobileNet Model."""

  dropout_keep_prob = config.dropout_keep_prob
  global_pool = config.global_pool
  num_classes = config.num_classes
  spatial_squeeze = config.spatial_squeeze
  model_name = config.name

  img_input = layers.Input(shape=input_shape, name='Input')
  x = mobilenet_v1_base(img_input, config)

  # Build top
  if global_pool:
    # Global average pooling.
    x = layers.GlobalAveragePooling2D(data_format='channels_last',
                                      name='top_GlobalPool')(x)
    x = layers.Reshape((1, 1, x.shape[1]))(x)
  else:
    # Pooling with a fixed kernel size.
    kernel_size = _reduced_kernel_size_for_small_input(x, (7, 7))
    x = layers.AvgPool2D(pool_size=kernel_size,
                         padding='VALID',
                         data_format='channels_last',
                         name='top_AvgPool')(x)

  # 1 x 1 x 1024
  x = layers.Dropout(rate=1 - dropout_keep_prob,
                     name='top_Dropout')(x)

  x = layers.Conv2D(filters=num_classes,
                    kernel_size=(1, 1),
                    padding='SAME',
                    name='top_Conv2d_1x1')(x)
  if spatial_squeeze:
    x = layers.Reshape(target_shape=(num_classes,),
                       name='top_SpatialSqueeze')(x)

  x = layers.Activation(activation='softmax',
                        name='top_Predictions')(x)

  return tf.keras.models.Model(inputs=img_input,
                               outputs=x,
                               name=model_name)


if __name__ == '__main__':
  logging.basicConfig(
    format='%(asctime)-15s:%(levelname)s:%(module)s:%(message)s',
    level=logging.INFO)
  model = mobilenet_v1()
  model.compile(
    optimizer='adam',
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=[tf.keras.metrics.categorical_crossentropy])
  logging.info(model.summary())
