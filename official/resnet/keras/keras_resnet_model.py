# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""ResNet50 model for Keras adapted from tf.keras.applications.ResNet50.

# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

import tensorflow as tf

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
L2_WEIGHT_DECAY = 1e-4


def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        require_flatten,
                        weights=None):
  """Internal utility to compute/validate a model's input shape.

  Arguments:
    input_shape: Either None (will return the default network input shape),
        or a user-provided shape to be validated.
    default_size: Default input width/height for the model.
    min_size: Minimum input width/height accepted by the model.
    data_format: Image data format to use.
    require_flatten: Whether the model is expected to
        be linked to a classifier via a Flatten layer.
    weights: One of `None` (random initialization)
        or 'imagenet' (pre-training on ImageNet).
        If weights='imagenet' input channels must be equal to 3.

  Returns:
    An integer shape tuple (may include None entries).

  Raises:
    ValueError: In case of invalid argument values.
  """
  if weights != 'imagenet' and input_shape and len(input_shape) == 3:
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
  if weights == 'imagenet' and require_flatten:
    if input_shape is not None:
      if input_shape != default_shape:
        raise ValueError('When setting`include_top=True` '
                         'and loading `imagenet` weights, '
                         '`input_shape` should be ' +
                         str(default_shape) + '.')
    return default_shape
  if input_shape:
    if data_format == 'channels_first':
      if input_shape is not None:
        if len(input_shape) != 3:
          raise ValueError(
              '`input_shape` must be a tuple of three integers.')
        if input_shape[0] != 3 and weights == 'imagenet':
          raise ValueError('The input must have 3 channels; got '
                           '`input_shape=' + str(input_shape) + '`')
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
        if input_shape[-1] != 3 and weights == 'imagenet':
          raise ValueError('The input must have 3 channels; got '
                           '`input_shape=' + str(input_shape) + '`')
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


def identity_block(input_tensor, kernel_size, filters, stage, block, training):
  """The identity block is the block that has no conv layer at shortcut.

  Arguments:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of
        middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names

  Returns:
    Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if tf.keras.backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = tf.keras.layers.Conv2D(filters1, (1, 1),
                             kernel_regularizer=
                             tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                             bias_regularizer=
                             tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                             name=conv_name_base + '2a')(input_tensor)
  x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                         name=bn_name_base + '2a',
                                         momentum=BATCH_NORM_DECAY,
                                         epsilon=BATCH_NORM_EPSILON)(
                                             x, training=training)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(filters2, kernel_size,
                             padding='same',
                             kernel_regularizer=
                             tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                             bias_regularizer=
                             tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                             name=conv_name_base + '2b')(x)
  x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                         name=bn_name_base + '2b',
                                         momentum=BATCH_NORM_DECAY,
                                         epsilon=BATCH_NORM_EPSILON)(
                                             x, training=training)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(filters3, (1, 1),
                             kernel_regularizer=
                             tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                             bias_regularizer=
                             tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                             name=conv_name_base + '2c')(x)
  x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                         name=bn_name_base + '2c',
                                         momentum=BATCH_NORM_DECAY,
                                         epsilon=BATCH_NORM_EPSILON)(
                                             x, training=training)

  x = tf.keras.layers.add([x, input_tensor])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               training=True):
  """A block that has a conv layer at shortcut.

  Arguments:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of
        middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    strides: Strides for the first conv layer in the block.
    training: Boolean to indicate if we are in the training loop.

  Returns:
    Output tensor for the block.

  Note that from stage 3,
  the first conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well
  """
  filters1, filters2, filters3 = filters
  if tf.keras.backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  bn_name_base = 'bn' + str(stage) + block + '_branch'

  x = tf.keras.layers.Conv2D(filters1, (1, 1),
                             kernel_regularizer=
                             tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                             bias_regularizer=
                             tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                             name=conv_name_base + '2a')(input_tensor)
  x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                         name=bn_name_base + '2a',
                                         momentum=BATCH_NORM_DECAY,
                                         epsilon=BATCH_NORM_EPSILON)(
                                             x, training=training)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same',
                             kernel_regularizer=
                             tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                             bias_regularizer=
                             tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                             name=conv_name_base + '2b', strides=strides)(x)
  x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                         name=bn_name_base + '2b',
                                         momentum=BATCH_NORM_DECAY,
                                         epsilon=BATCH_NORM_EPSILON)(
                                             x, training=training)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Conv2D(filters3, (1, 1),
                             kernel_regularizer=
                             tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                             bias_regularizer=
                             tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                             name=conv_name_base + '2c')(x)
  x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                         name=bn_name_base + '2c',
                                         momentum=BATCH_NORM_DECAY,
                                         epsilon=BATCH_NORM_EPSILON)(
                                             x, training=training)

  shortcut = tf.keras.layers.Conv2D(filters3, (1, 1), strides=strides,
                                    kernel_regularizer=
                                    tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                    bias_regularizer=
                                    tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                    name=conv_name_base + '1')(input_tensor)
  shortcut = tf.keras.layers.BatchNormalization(
      axis=bn_axis, name=bn_name_base + '1',
      momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(
          shortcut, training=training)

  x = tf.keras.layers.add([x, shortcut])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def ResNet50(include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             training=True):
  """Instantiates the ResNet50 architecture.

  Optionally loads weights pre-trained on ImageNet.
  Note that the data format convention used by the model is
  the one specified in your Keras config at `~/.keras/keras.json`.

  Arguments:
      include_top: whether to include the fully-connected
          layer at the top of the network.
      weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
          to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
          if `include_top` is False (otherwise the input shape
          has to be `(224, 224, 3)` (with `channels_last` data format)
          or `(3, 224, 224)` (with `channels_first` data format).
          It should have exactly 3 inputs channels,
          and width and height should be no smaller than 197.
          E.g. `(200, 200, 3)` would be one valid value.
      pooling: Optional pooling mode for feature extraction
          when `include_top` is `False`.
          - `None` means that the output of the model will be
              the 4D tensor output of the
              last convolutional layer.
          - `avg` means that global average pooling
              will be applied to the output of the
              last convolutional layer, and thus
              the output of the model will be a 2D tensor.
          - `max` means that global max pooling will
              be applied.
      classes: optional number of classes to classify images
          into, only to be specified if `include_top` is True, and
          if no `weights` argument is specified.
      training: optional boolean indicating if this model will be
          used for training or evaluation. This boolean is then
          passed to the BatchNorm layer.

  Returns:
      A Keras model instance.

  Raises:
      ValueError: in case of invalid argument for `weights`,
          or invalid input shape.
  """
  if not (weights in {'imagenet', None} or os.path.exists(weights)):
    raise ValueError('The `weights` argument should be either '
                     '`None` (random initialization), `imagenet` '
                     '(pre-training on ImageNet), '
                     'or the path to the weights file to be loaded.')

  if weights == 'imagenet' and include_top and classes != 1000:
    raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                     ' as true, `classes` should be 1000')

  # Determine proper input shape
  input_shape = _obtain_input_shape(
      input_shape,
      default_size=224,
      min_size=197,
      data_format=tf.keras.backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = tf.keras.layers.Input(shape=input_shape)
  else:
    if not tf.keras.backend.is_keras_tensor(input_tensor):
      img_input = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor
  if tf.keras.backend.image_data_format() == 'channels_last':
    bn_axis = 3
  else:
    bn_axis = 1

  x = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
  x = tf.keras.layers.Conv2D(64, (7, 7),
                             strides=(2, 2),
                             padding='valid',
                             name='conv1')(x)
  x = tf.keras.layers.BatchNormalization(axis=bn_axis, name='bn_conv1',
                                         momentum=BATCH_NORM_DECAY,
                                         epsilon=BATCH_NORM_EPSILON)(
                                             x, training=training)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

  x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1),
                 training=training)
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='b',
                     training=training)
  x = identity_block(x, 3, [64, 64, 256], stage=2, block='c',
                     training=training)

  x = conv_block(x, 3, [128, 128, 512], stage=3, block='a',
                 training=training)
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='b',
                     training=training)
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='c',
                     training=training)
  x = identity_block(x, 3, [128, 128, 512], stage=3, block='d',
                     training=training)

  x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a',
                 training=training)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b',
                     training=training)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c',
                     training=training)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d',
                     training=training)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e',
                     training=training)
  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f',
                     training=training)

  x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a',
                 training=training)
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b',
                     training=training)
  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c',
                     training=training)

  if include_top:
    x = tf.keras.layers.AveragePooling2D((7, 7), name='avg_pool')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(classes, activation='softmax', name='fc1000')(x)
  else:
    if pooling == 'avg':
      x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
      x = tf.keras.layers.GlobalMaxPooling2D()(x)
    else:
      warnings.warn('The output shape of `ResNet50(include_top=False)` '
                    'has been changed since Keras 2.2.0.')

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = tf.keras.engine.get_source_inputs(input_tensor)
  else:
    inputs = img_input
  # Create model.
  model = tf.keras.models.Model(inputs, x, name='resnet50')

  # Load weights.
  if weights == 'imagenet':
    if include_top:
      weights_path = tf.keras.utils.get_file(
          'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
          WEIGHTS_PATH,
          cache_subdir='models',
          md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
    else:
      weights_path = tf.keras.utils.get_file(
          'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
          WEIGHTS_PATH_NO_TOP,
          cache_subdir='models',
          md5_hash='a268eb855778b3df3c7506639542a6af')
    model.load_weights(weights_path)
  elif weights is not None:
    model.load_weights(weights)

  return model
