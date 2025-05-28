# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""VGG16 model for Keras.

Adapted from tf_keras.applications.vgg16.VGG16().

Related papers/blogs:
- https://arxiv.org/abs/1409.1556
"""

import tensorflow as tf, tf_keras

layers = tf_keras.layers


def _gen_l2_regularizer(use_l2_regularizer=True, l2_weight_decay=1e-4):
  return tf_keras.regularizers.L2(
      l2_weight_decay) if use_l2_regularizer else None


def vgg16(num_classes,
          batch_size=None,
          use_l2_regularizer=True,
          batch_norm_decay=0.9,
          batch_norm_epsilon=1e-5):
  """Instantiates the VGG16 architecture.

  Args:
    num_classes: `int` number of classes for image classification.
    batch_size: Size of the batches for each step.
    use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.
    batch_norm_decay: Moment of batch norm layers.
    batch_norm_epsilon: Epsilon of batch borm layers.

  Returns:
    A Keras model instance.

  """
  input_shape = (224, 224, 3)
  img_input = layers.Input(shape=input_shape, batch_size=batch_size)

  x = img_input

  if tf_keras.backend.image_data_format() == 'channels_first':
    x = layers.Permute((3, 1, 2))(x)
    bn_axis = 1
  else:  # channels_last
    bn_axis = 3
  # Block 1
  x = layers.Conv2D(
      64, (3, 3),
      padding='same',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block1_conv1')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name='bn_conv1')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.Conv2D(
      64, (3, 3),
      padding='same',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block1_conv2')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name='bn_conv2')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

  # Block 2
  x = layers.Conv2D(
      128, (3, 3),
      padding='same',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block2_conv1')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name='bn_conv3')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.Conv2D(
      128, (3, 3),
      padding='same',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block2_conv2')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name='bn_conv4')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

  # Block 3
  x = layers.Conv2D(
      256, (3, 3),
      padding='same',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block3_conv1')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name='bn_conv5')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.Conv2D(
      256, (3, 3),
      padding='same',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block3_conv2')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name='bn_conv6')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.Conv2D(
      256, (3, 3),
      padding='same',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block3_conv3')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name='bn_conv7')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

  # Block 4
  x = layers.Conv2D(
      512, (3, 3),
      padding='same',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block4_conv1')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name='bn_conv8')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.Conv2D(
      512, (3, 3),
      padding='same',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block4_conv2')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name='bn_conv9')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.Conv2D(
      512, (3, 3),
      padding='same',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block4_conv3')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name='bn_conv10')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

  # Block 5
  x = layers.Conv2D(
      512, (3, 3),
      padding='same',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block5_conv1')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name='bn_conv11')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.Conv2D(
      512, (3, 3),
      padding='same',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block5_conv2')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name='bn_conv12')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.Conv2D(
      512, (3, 3),
      padding='same',
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='block5_conv3')(
          x)
  x = layers.BatchNormalization(
      axis=bn_axis,
      momentum=batch_norm_decay,
      epsilon=batch_norm_epsilon,
      name='bn_conv13')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

  x = layers.Flatten(name='flatten')(x)
  x = layers.Dense(
      4096,
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='fc1')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(
      4096,
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='fc2')(
          x)
  x = layers.Activation('relu')(x)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(
      num_classes,
      kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
      name='fc1000')(
          x)

  x = layers.Activation('softmax', dtype='float32')(x)

  # Create model.
  return tf_keras.Model(img_input, x, name='vgg16')
