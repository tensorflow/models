# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""MobileNet v2 models for Keras.

MobileNetV2 is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and
different width factors. This allows different width models to reduce
the number of multiply-adds and thereby
reduce inference cost on mobile devices.

MobileNetV2 is very similar to the original MobileNet,
except that it uses inverted residual blocks with
bottlenecking features. It has a drastically lower
parameter count than the original MobileNet.
MobileNets support any input size greater
than 32 x 32, with larger image sizes
offering better performance.

The number of parameters and number of multiply-adds
can be modified by using the `alpha` parameter,
which increases/decreases the number of filters in each layer.
By altering the image size and `alpha` parameter,
all 22 models from the paper can be built, with ImageNet weights provided.

The paper demonstrates the performance of MobileNets using `alpha` values of
1.0 (also called 100 % MobileNet), 0.35, 0.5, 0.75, 1.0, 1.3, and 1.4

For each of these `alpha` values, weights for 5 different input image sizes
are provided (224, 192, 160, 128, and 96).


The following table describes the performance of
MobileNet on various input sizes:
------------------------------------------------------------------------
MACs stands for Multiply Adds

 Classification Checkpoint| MACs (M)   | Parameters (M)| Top 1 Acc| Top 5 Acc
--------------------------|------------|---------------|---------|----|-------
| [mobilenet_v2_1.4_224]  | 582 | 6.06 |          75.0 | 92.5 |
| [mobilenet_v2_1.3_224]  | 509 | 5.34 |          74.4 | 92.1 |
| [mobilenet_v2_1.0_224]  | 300 | 3.47 |          71.8 | 91.0 |
| [mobilenet_v2_1.0_192]  | 221 | 3.47 |          70.7 | 90.1 |
| [mobilenet_v2_1.0_160]  | 154 | 3.47 |          68.8 | 89.0 |
| [mobilenet_v2_1.0_128]  | 99  | 3.47 |          65.3 | 86.9 |
| [mobilenet_v2_1.0_96]   | 56  | 3.47 |          60.3 | 83.2 |
| [mobilenet_v2_0.75_224] | 209 | 2.61 |          69.8 | 89.6 |
| [mobilenet_v2_0.75_192] | 153 | 2.61 |          68.7 | 88.9 |
| [mobilenet_v2_0.75_160] | 107 | 2.61 |          66.4 | 87.3 |
| [mobilenet_v2_0.75_128] | 69  | 2.61 |          63.2 | 85.3 |
| [mobilenet_v2_0.75_96]  | 39  | 2.61 |          58.8 | 81.6 |
| [mobilenet_v2_0.5_224]  | 97  | 1.95 |          65.4 | 86.4 |
| [mobilenet_v2_0.5_192]  | 71  | 1.95 |          63.9 | 85.4 |
| [mobilenet_v2_0.5_160]  | 50  | 1.95 |          61.0 | 83.2 |
| [mobilenet_v2_0.5_128]  | 32  | 1.95 |          57.7 | 80.8 |
| [mobilenet_v2_0.5_96]   | 18  | 1.95 |          51.2 | 75.8 |
| [mobilenet_v2_0.35_224] | 59  | 1.66 |          60.3 | 82.9 |
| [mobilenet_v2_0.35_192] | 43  | 1.66 |          58.2 | 81.2 |
| [mobilenet_v2_0.35_160] | 30  | 1.66 |          55.7 | 79.1 |
| [mobilenet_v2_0.35_128] | 20  | 1.66 |          50.8 | 75.0 |
| [mobilenet_v2_0.35_96]  | 11  | 1.66 |          45.5 | 70.4 |

The weights for all 16 models are obtained and translated from the Tensorflow
checkpoints from TensorFlow checkpoints found at
https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md

# Reference
This file contains building code for MobileNetV2, based on
[MobileNetV2: Inverted Residuals and Linear Bottlenecks]
(https://arxiv.org/abs/1801.04381)

Tests comparing this model to the existing Tensorflow model can be
found at
[mobilenet_v2_keras](https://github.com/JonathanCMitchell/mobilenet_v2_keras)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import numpy as np
import tensorflow.compat.v1 as tf

Model = tf.keras.Model
Input = tf.keras.layers.Input
Activation = tf.keras.layers.Activation
BatchNormalization = tf.keras.layers.BatchNormalization
Conv2D = tf.keras.layers.Conv2D
DepthwiseConv2D = tf.keras.layers.DepthwiseConv2D
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
Add = tf.keras.layers.Add
Dense = tf.keras.layers.Dense
K = tf.keras.Backend


def relu6(x):
  return K.relu(x, max_value=6)


def _obtain_input_shape(
    input_shape,
    default_size,
    min_size,
    data_format,
    require_flatten):
  """Internal utility to compute/validate an ImageNet model's input shape.

  Arguments:
      input_shape: either None (will return the default network input shape),
          or a user-provided shape to be validated.
      default_size: default input width/height for the model.
      min_size: minimum input width/height accepted by the model.
      data_format: image data format to use.
      require_flatten: whether the model is expected to
          be linked to a classifier via a Flatten layer.

  Returns:
      An integer shape tuple (may include None entries).

  Raises:
      ValueError: in case of invalid argument values.
  """
  if input_shape and len(input_shape) == 3:
    if data_format == 'channels_first':
      if input_shape[0] not in {1, 3}:
        warnings.warn(
            'This model usually expects 1 or 3 input channels. '
            'However, it was passed an input_shape with ' +
            str(input_shape[0]) + ' input channels.')
      default_shape = (input_shape[0], default_size, default_size)
    else:
      if input_shape[-1] not in {1, 3}:
        warnings.warn(
            'This model usually expects 1 or 3 input channels. '
            'However, it was passed an input_shape with ' +
            str(input_shape[-1]) + ' input channels.')
      default_shape = (default_size, default_size, input_shape[-1])
  else:
    if data_format == 'channels_first':
      default_shape = (3, default_size, default_size)
    else:
      default_shape = (default_size, default_size, 3)
  if input_shape:
    if data_format == 'channels_first':
      if input_shape is not None:
        if len(input_shape) != 3:
          raise ValueError(
              '`input_shape` must be a tuple of three integers.')
        if ((input_shape[1] is not None and input_shape[1] < min_size) or
            (input_shape[2] is not None and input_shape[2] < min_size)):
          raise ValueError('Input size must be at least ' +
                           str(min_size) + 'x' + str(min_size) +
                           '; got `input_shape=' +
                           str(input_shape) + '`')
    else:
      if input_shape is not None:
        if len(input_shape) != 3:
          raise ValueError(
              '`input_shape` must be a tuple of three integers.')
        if ((input_shape[0] is not None and input_shape[0] < min_size) or
            (input_shape[1] is not None and input_shape[1] < min_size)):
          raise ValueError('Input size must be at least ' +
                           str(min_size) + 'x' + str(min_size) +
                           '; got `input_shape=' +
                           str(input_shape) + '`')
  else:
    if require_flatten:
      input_shape = default_shape
    else:
      if data_format == 'channels_first':
        input_shape = (3, None, None)
      else:
        input_shape = (None, None, 3)
  if require_flatten:
    if None in input_shape:
      raise ValueError('If `include_top` is True, '
                       'you should specify a static `input_shape`. '
                       'Got `input_shape=' + str(input_shape) + '`')
  return input_shape


def preprocess_input(x):
  """Preprocesses a numpy array encoding a batch of images.

  This function applies the "Inception" preprocessing which converts
  the RGB values from [0, 255] to [-1, 1]. Note that this preprocessing
  function is different from `imagenet_utils.preprocess_input()`.

  Arguments:
    x: a 4D numpy array consists of RGB values within [0, 255].

  Returns:
    Preprocessed array.
  """
  x /= 128.
  x -= 1.
  return x.astype(np.float32)


# This function is taken from the original tf repo.
# It ensures that all layers have a channel number that is divisible by 8
# It can be seen here:
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py


def _make_divisible(v, divisor, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v


def mobilenet_v2(input_shape=None,
                 alpha=1.0,
                 include_top=True,
                 classes=1000):
  """Instantiates the MobileNetV2 architecture.

  To load a MobileNetV2 model via `load_model`, import the custom
  objects `relu6` and pass them to the `custom_objects` parameter.
  E.g.
  model = load_model('mobilenet.h5', custom_objects={
                     'relu6': mobilenet.relu6})

  Arguments:
    input_shape: optional shape tuple, to be specified if you would
      like to use a model with an input img resolution that is not
      (224, 224, 3).
      It should have exactly 3 inputs channels (224, 224, 3).
      You can also omit this option if you would like
      to infer input_shape from an input_tensor.
      If you choose to include both input_tensor and input_shape then
      input_shape will be used if they match, if the shapes
      do not match then we will throw an error.
      E.g. `(160, 160, 3)` would be one valid value.
    alpha: controls the width of the network. This is known as the
    width multiplier in the MobileNetV2 paper.
      - If `alpha` < 1.0, proportionally decreases the number
          of filters in each layer.
      - If `alpha` > 1.0, proportionally increases the number
          of filters in each layer.
      - If `alpha` = 1, default number of filters from the paper
           are used at each layer.
    include_top: whether to include the fully-connected
      layer at the top of the network.
    classes: optional number of classes to classify images
      into, only to be specified if `include_top` is True, and
      if no `weights` argument is specified.

  Returns:
    A Keras model instance.

  Raises:
    ValueError: in case of invalid argument for `weights`,
        or invalid input shape or invalid depth_multiplier, alpha,
        rows when weights='imagenet'
  """

  # Determine proper input shape and default size.
  # If input_shape is None and no input_tensor
  if input_shape is None:
    default_size = 224

  # If input_shape is not None, assume default size
  else:
    if K.image_data_format() == 'channels_first':
      rows = input_shape[1]
      cols = input_shape[2]
    else:
      rows = input_shape[0]
      cols = input_shape[1]

    if rows == cols and rows in [96, 128, 160, 192, 224]:
      default_size = rows
    else:
      default_size = 224

  input_shape = _obtain_input_shape(input_shape,
                                    default_size=default_size,
                                    min_size=32,
                                    data_format=K.image_data_format(),
                                    require_flatten=include_top)

  if K.image_data_format() == 'channels_last':
    row_axis, col_axis = (0, 1)
  else:
    row_axis, col_axis = (1, 2)
  rows = input_shape[row_axis]
  cols = input_shape[col_axis]

  if K.image_data_format() != 'channels_last':
    warnings.warn('The MobileNet family of models is only available '
                  'for the input data format "channels_last" '
                  '(width, height, channels). '
                  'However your settings specify the default '
                  'data format "channels_first" (channels, width, height).'
                  ' You should set `image_data_format="channels_last"` '
                  'in your Keras config located at ~/.keras/keras.json. '
                  'The model being returned right now will expect inputs '
                  'to follow the "channels_last" data format.')
    K.set_image_data_format('channels_last')
    old_data_format = 'channels_first'
  else:
    old_data_format = None

  img_input = Input(shape=input_shape)

  first_block_filters = _make_divisible(32 * alpha, 8)
  x = Conv2D(first_block_filters,
             kernel_size=3,
             strides=(2, 2), padding='same',
             use_bias=False, name='Conv1')(img_input)
  x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
  x = Activation(relu6, name='Conv1_relu')(x)

  x = _first_inverted_res_block(x,
                                filters=16,
                                alpha=alpha,
                                stride=1,
                                block_id=0)

  x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                          expansion=6, block_id=1)
  x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                          expansion=6, block_id=2)

  x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                          expansion=6, block_id=3)
  x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                          expansion=6, block_id=4)
  x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                          expansion=6, block_id=5)

  x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                          expansion=6, block_id=6)
  x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                          expansion=6, block_id=7)
  x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                          expansion=6, block_id=8)
  x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                          expansion=6, block_id=9)

  x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                          expansion=6, block_id=10)
  x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                          expansion=6, block_id=11)
  x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                          expansion=6, block_id=12)

  x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                          expansion=6, block_id=13)
  x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                          expansion=6, block_id=14)
  x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                          expansion=6, block_id=15)

  x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                          expansion=6, block_id=16)

  # no alpha applied to last conv as stated in the paper:
  # if the width multiplier is greater than 1 we
  # increase the number of output channels
  if alpha > 1.0:
    last_block_filters = _make_divisible(1280 * alpha, 8)
  else:
    last_block_filters = 1280

  x = Conv2D(last_block_filters,
             kernel_size=1,
             use_bias=False,
             name='Conv_1')(x)
  x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
  x = Activation(relu6, name='out_relu')(x)

  if include_top:
    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax',
              use_bias=True, name='Logits')(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  inputs = img_input

  # Create model.
  model = Model(inputs, x, name='mobilenetv2_%0.2f_%s' % (alpha, rows))

  if old_data_format:
    K.set_image_data_format(old_data_format)
  return model


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
  """Build an inverted res block."""
  in_channels = int(inputs.shape[-1])
  pointwise_conv_filters = int(filters * alpha)
  pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
  # Expand

  x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
             use_bias=False, activation=None,
             name='mobl%d_conv_expand' % block_id)(inputs)
  x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                         name='bn%d_conv_bn_expand' %
                         block_id)(x)
  x = Activation(relu6, name='conv_%d_relu' % block_id)(x)

  # Depthwise
  x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                      use_bias=False, padding='same',
                      name='mobl%d_conv_depthwise' % block_id)(x)
  x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                         name='bn%d_conv_depthwise' % block_id)(x)

  x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

  # Project
  x = Conv2D(pointwise_filters,
             kernel_size=1, padding='same', use_bias=False, activation=None,
             name='mobl%d_conv_project' % block_id)(x)
  x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                         name='bn%d_conv_bn_project' % block_id)(x)

  if in_channels == pointwise_filters and stride == 1:
    return Add(name='res_connect_' + str(block_id))([inputs, x])

  return x


def _first_inverted_res_block(inputs,
                              stride,
                              alpha, filters, block_id):
  """Build the first inverted res block."""
  in_channels = int(inputs.shape[-1])
  pointwise_conv_filters = int(filters * alpha)
  pointwise_filters = _make_divisible(pointwise_conv_filters, 8)

  # Depthwise
  x = DepthwiseConv2D(kernel_size=3,
                      strides=stride, activation=None,
                      use_bias=False, padding='same',
                      name='mobl%d_conv_depthwise' %
                      block_id)(inputs)
  x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                         name='bn%d_conv_depthwise' %
                         block_id)(x)
  x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)

  # Project
  x = Conv2D(pointwise_filters,
             kernel_size=1,
             padding='same',
             use_bias=False,
             activation=None,
             name='mobl%d_conv_project' %
             block_id)(x)
  x = BatchNormalization(epsilon=1e-3, momentum=0.999,
                         name='bn%d_conv_project' %
                         block_id)(x)

  if in_channels == pointwise_filters and stride == 1:
    return Add(name='res_connect_' + str(block_id))([inputs, x])

  return x
