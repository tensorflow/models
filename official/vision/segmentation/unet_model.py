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
"""Model definition for the TF2 Keras UNet 3D Model."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import tensorflow as tf


def create_optimizer(init_learning_rate, params):
  """Creates optimizer for training."""
  learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
      initial_learning_rate=init_learning_rate,
      decay_steps=params.lr_decay_steps,
      decay_rate=params.lr_decay_rate)

  # TODO(hongjunchoi): Provide alternative optimizer options depending on model
  # config parameters.
  optimizer = tf.keras.optimizers.Adam(learning_rate)
  return optimizer


def create_convolution_block(input_layer,
                             n_filters,
                             batch_normalization=False,
                             kernel=(3, 3, 3),
                             activation=tf.nn.relu,
                             padding='SAME',
                             strides=(1, 1, 1),
                             data_format='channels_last',
                             instance_normalization=False):
  """UNet convolution block.

  Args:
    input_layer: tf.Tensor, the input tensor.
    n_filters: integer, the number of the output channels of the convolution.
    batch_normalization: boolean, use batch normalization after the convolution.
    kernel: kernel size of the convolution.
    activation: Tensorflow activation layer to use. (default is 'relu')
    padding: padding type of the convolution.
    strides: strides of the convolution.
    data_format: data format of the convolution. One of 'channels_first' or
      'channels_last'.
    instance_normalization: use Instance normalization. Exclusive with batch
      normalization.

  Returns:
    The Tensor after apply the convolution block to the input.
  """
  assert instance_normalization == 0, 'TF 2.0 does not support inst. norm.'
  layer = tf.keras.layers.Conv3D(
      filters=n_filters,
      kernel_size=kernel,
      strides=strides,
      padding=padding,
      data_format=data_format,
      activation=None,
  )(
      inputs=input_layer)
  if batch_normalization:
    layer = tf.keras.layers.BatchNormalization(axis=1)(inputs=layer)
  return activation(layer)


def apply_up_convolution(inputs,
                         num_filters,
                         pool_size,
                         kernel_size=(2, 2, 2),
                         strides=(2, 2, 2),
                         deconvolution=False):
  """Apply up convolution on inputs.

  Args:
    inputs: input feature tensor.
    num_filters: number of deconvolution output feature channels.
    pool_size: pool size of the up-scaling.
    kernel_size: kernel size of the deconvolution.
    strides: strides of the deconvolution.
    deconvolution: Use deconvolution or upsampling.

  Returns:
    The tensor of the up-scaled features.
  """
  if deconvolution:
    return tf.keras.layers.Conv3DTranspose(
        filters=num_filters, kernel_size=kernel_size, strides=strides)(
            inputs=inputs)
  else:
    return tf.keras.layers.UpSampling3D(size=pool_size)(inputs)


def unet3d_base(input_layer,
                pool_size=(2, 2, 2),
                n_labels=1,
                deconvolution=False,
                depth=4,
                n_base_filters=32,
                batch_normalization=False,
                data_format='channels_last'):
  """Builds the 3D UNet Tensorflow model and return the last layer logits.

  Args:
    input_layer: the input Tensor.
    pool_size: Pool size for the max pooling operations.
    n_labels: Number of binary labels that the model is learning.
    deconvolution: If set to True, will use transpose convolution(deconvolution)
      instead of up-sampling. This increases the amount memory required during
      training.
    depth: indicates the depth of the U-shape for the model. The greater the
      depth, the more max pooling layers will be added to the model. Lowering
      the depth may reduce the amount of memory required for training.
    n_base_filters: The number of filters that the first layer in the
      convolution network will have. Following layers will contain a multiple of
      this number. Lowering this number will likely reduce the amount of memory
      required to train the model.
    batch_normalization: boolean. True for use batch normalization after
      convolution and before activation.
    data_format: string, channel_last (default) or channel_first

  Returns:
    The last layer logits of 3D UNet.
  """
  levels = []
  current_layer = input_layer
  if data_format == 'channels_last':
    channel_dim = -1
  else:
    channel_dim = 1

  # add levels with max pooling
  for layer_depth in range(depth):
    layer1 = create_convolution_block(
        input_layer=current_layer,
        n_filters=n_base_filters * (2**layer_depth),
        batch_normalization=batch_normalization,
        kernel=(3, 3, 3),
        activation=tf.nn.relu,
        padding='SAME',
        strides=(1, 1, 1),
        data_format=data_format,
        instance_normalization=False)
    layer2 = create_convolution_block(
        input_layer=layer1,
        n_filters=n_base_filters * (2**layer_depth) * 2,
        batch_normalization=batch_normalization,
        kernel=(3, 3, 3),
        activation=tf.nn.relu,
        padding='SAME',
        strides=(1, 1, 1),
        data_format=data_format,
        instance_normalization=False)
    if layer_depth < depth - 1:
      current_layer = tf.keras.layers.MaxPool3D(
          pool_size=pool_size,
          strides=(2, 2, 2),
          padding='VALID',
          data_format=data_format)(
              inputs=layer2)
      levels.append([layer1, layer2, current_layer])
    else:
      current_layer = layer2
      levels.append([layer1, layer2])

  # add levels with up-convolution or up-sampling
  for layer_depth in range(depth - 2, -1, -1):
    up_convolution = apply_up_convolution(
        current_layer,
        pool_size=pool_size,
        deconvolution=deconvolution,
        num_filters=current_layer.get_shape().as_list()[channel_dim])
    concat = tf.concat([up_convolution, levels[layer_depth][1]],
                       axis=channel_dim)
    current_layer = create_convolution_block(
        n_filters=levels[layer_depth][1].get_shape().as_list()[channel_dim],
        input_layer=concat,
        batch_normalization=batch_normalization,
        kernel=(3, 3, 3),
        activation=tf.nn.relu,
        padding='SAME',
        strides=(1, 1, 1),
        data_format=data_format,
        instance_normalization=False)
    current_layer = create_convolution_block(
        n_filters=levels[layer_depth][1].get_shape().as_list()[channel_dim],
        input_layer=current_layer,
        batch_normalization=batch_normalization,
        kernel=(3, 3, 3),
        activation=tf.nn.relu,
        padding='SAME',
        strides=(1, 1, 1),
        data_format=data_format,
        instance_normalization=False)

  final_convolution = tf.keras.layers.Conv3D(
      filters=n_labels,
      kernel_size=(1, 1, 1),
      padding='VALID',
      data_format=data_format,
      activation=None)(
          current_layer)
  return final_convolution


def build_unet_model(params):
  """Builds the unet model, optimizer included."""
  input_shape = params.input_image_size + [1]
  input_layer = tf.keras.layers.Input(shape=input_shape)

  logits = unet3d_base(
      input_layer,
      pool_size=(2, 2, 2),
      n_labels=params.num_classes,
      deconvolution=params.deconvolution,
      depth=params.depth,
      n_base_filters=params.num_base_filters,
      batch_normalization=params.use_batch_norm,
      data_format=params.data_format)

  # Set output of softmax to float32 to avoid potential numerical overflow.
  predictions = tf.keras.layers.Softmax(dtype='float32')(logits)
  model = tf.keras.models.Model(inputs=input_layer, outputs=predictions)
  model.optimizer = create_optimizer(params.init_learning_rate, params)
  return model
