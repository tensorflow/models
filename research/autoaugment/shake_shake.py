# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Builds the Shake-Shake Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import models

layers = tf.keras.layers

def _shake_shake_skip_connection(x, 
                                 output_filters, 
                                 stride):
  """Adds a residual connection to the filter x for the shake-shake model."""
  curr_filters = int(x.shape[3])
  if curr_filters == output_filters:
    return x

  #Skip path 1
  path1 = layers.AveragePooling2D(
          pool_size=(1, 1), 
          strides=(stride,stride), 
          padding='valid')(
          x)
  path1 = layers.Conv2D(
          filters=int(output_filters / 2), 
          kernel_size=1, 
          strides=(1, 1), 
          padding='same', 
          kernel_initializer='he_normal')(
          path1)

  # Skip path 2
  # First pad with 0's then crop
  path2 = tf.keras.layers.ZeroPadding2D(padding=(1, 1))(x)
  path2 = tf.keras.layers.Cropping2D(cropping=((1, 0), (1, 0)))(x)
  path2 = layers.AveragePooling2D(
          pool_size=(1, 1), 
          strides=(stride,stride), 
          padding='valid')(
          x)  
  path2 = layers.Conv2D(
          filters=int(output_filters / 2),
          kernel_size=1, 
          strides=(1, 1), 
          padding='same', 
          kernel_initializer = 'he_normal')(
          path2)
  
  #Concat and apply BN
  final_path = layers.Concatenate(axis=3)([path1, path2])
  final_path = layers.BatchNormalization(
               momentum=0.999, 
               epsilon=0.001,
               fused=True)(
               final_path)
  return final_path

def _shake_shake_branch(x,
                        output_filters,
                        stride,
                        rand_forward,
                        rand_backward,
                        is_training):
  """Building a 2 branching convnet."""
  x=layers.Activation('relu')(x)
    
  x = layers.Conv2D(
      filters=output_filters, 
      kernel_size = 3, 
      strides=stride, 
      padding='same',
      kernel_initializer='he_normal')(
      x)
  x = layers.BatchNormalization(
      momentum=0.999,
      epsilon=0.001,
      fused=True)(
      x)
  x = layers.Activation('relu')(x)
  x = layers.Conv2D(
      filters=output_filters, 
      kernel_size = 3, 
      padding='same', 
      kernel_initializer='he_normal')(
      x)
  x = layers.BatchNormalization(
      momentum=0.999,
      epsilon=0.001,
      fused=True)(
      x)

  if is_training:
    x = x * rand_backward + tf.stop_gradient(x * rand_forward -
                                             x * rand_backward)
  else:
    x *= 1.0 / 2
  return x


def _shake_shake_block(x, 
                       output_filters, 
                       stride,
                       is_training):
  """Builds a full shake-shake sub layer."""
  batch_size = tf.shape(x)[0]

  # Generate random numbers for scaling the branches
  rand_forward = [
      tf.random.uniform(
          shape=[batch_size, 1, 1, 1],minval=-0, maxval=1, dtype=tf.dtypes.float32)
      for _ in range(2)
  ]
  rand_backward = [
      tf.random.uniform(
          shape=[batch_size, 1, 1, 1], minval=0, maxval=1, dtype=tf.dtypes.float32)
      for _ in range(2)
  ]

  # Normalize so that all sum to 1
  total_forward = tf.math.add_n(rand_forward)
  total_backward = tf.math.add_n(rand_backward)
  rand_forward = [samp / total_forward for samp in rand_forward]
  rand_backward = [samp / total_backward for samp in rand_backward]
  zipped_rand = zip(rand_forward, rand_backward)

  branches = []
  for branch, (r_forward, r_backward) in enumerate(zipped_rand):
    b = _shake_shake_branch(x, output_filters, stride, r_forward, r_backward,is_training)
    branches.append(b)
  res = _shake_shake_skip_connection(x, output_filters, stride)
  res = res + tf.math.add_n(branches)
  return res


def _shake_shake_layer(x, 
                       output_filters, 
                       num_blocks, 
                       stride,
                       is_training):
  """Builds many sub layers into one full layer."""
  for block_num in range(num_blocks):
    curr_stride = stride if (block_num == 0) else 1
    x = _shake_shake_block(x, output_filters, curr_stride,is_training)
  return x


def build_shake_shake_model(images,
                            num_classes,
                            hparams,
                            is_training):
  """Builds the Shake-Shake model.

  Build the Shake-Shake model from https://arxiv.org/abs/1705.07485.

  Args:
    images: Tensor of images that will be fed into the Wide ResNet Model.
    num_classes: Number of classed that the model needs to predict.
    hparams: tf.HParams object that contains additional hparams needed to
      construct the model. In this case it is the `shake_shake_widen_factor`
      that is used to determine how many filters the model has.
    is_training: Is the model training or not.

  Returns:
    The logits of the Shake-Shake model.
  """

  depth = 26
  k = hparams.shake_shake_widen_factor  # The widen factor 
  n = int((depth - 2) / 6)
  input = layers.Input(shape=images)

  x = layers.Conv2D(
      filters=16,
      kernel_size=3,
      strides=(1, 1), 
      padding='same',
      kernel_initializer='he_normal')(
      input)
  x = layers.BatchNormalization(
      momentum=0.999,
      epsilon=0.001,
      fused=True)(
      x)

  x = _shake_shake_layer(x, 16 * k, n, 1, is_training) #Shake Stage 1
  x = _shake_shake_layer(x, 32 * k, n, 2, is_training) #shake Stage 2
  x = _shake_shake_layer(x, 64 * k, n, 2, is_training) #Shake Stage 3

  x = layers.Activation('relu')(x)
  x = layers.GlobalAveragePooling2D()(x)
  x = layers.Dense(10)(x)

  # Create model.
  return models.Model(input, x, name='shake_shake')
