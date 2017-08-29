# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""
Contains blocks for building DenseNet-based models
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


slim = tf.contrib.slim


def densenet_arg_scope(
  weight_decay=0.0001,
  batch_norm_decay=0.997,
  batch_norm_epsilon=1e-5,
  batch_norm_scale=True,
  activation_fn=tf.nn.relu,
  use_batch_norm=True):
  """
  Args:
    weight_decay: The weight decay to use for regularizing the model.

    batch_norm_decay: The moving average decay when estimating layer activation
    statistics in batch normalization.

    batch_norm_epsilon: Small constant to prevent division by zero when
    normalizing activations by their variance in batch normalization.

    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
    activations in the batch normalization layer.

    activation_fn: The activation function which is used in ResNet.

    use_batch_norm: Whether or not to use batch normalization.

  Returns:
    An `arg_scope` to use for the densenet models.
  """
  batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'activation_fn': activation_fn,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
  }

  with slim.arg_scope(
      [slim.conv2d],
      padding='SAME',
      weights_regularizer=slim.l2_regularizer(weight_decay),
      weights_initializer=slim.variance_scaling_initializer(),
      activation_fn=None,
      normalizer_fn=slim.batch_norm if use_batch_norm else None,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
      return arg_sc


def preact_conv(inputs, n_filters, filter_size=[3, 3], dropout_p=0.2):
    """
    Basic pre-activation layer for DenseNets
    Apply successivly BatchNormalization, ReLU nonlinearity, Convolution and
    Dropout (if dropout_p > 0) on the inputs
    """
    preact = slim.batch_norm(inputs)
    conv = slim.conv2d(preact, n_filters, filter_size, normalizer_fn=None)
    if dropout_p != 0.0:
      conv = slim.dropout(conv, keep_prob=(1.0-dropout_p))
    return conv


@slim.add_arg_scope
def DenseBlock(stack, n_layers, growth_rate, dropout_p, bottleneck=False,
               scope=None, outputs_collections=None):
  """
  DenseBlock for DenseNet and FC-DenseNet

  Args:
    stack: input 4D tensor
    n_layers: number of internal layers
    growth_rate: number of feature maps per internal layer

  Returns:
    stack: current stack of feature maps (4D tensor)
    new_features: 4D tensor containing only the new feature maps generated
      in this block
  """
  with tf.name_scope(scope) as sc:
    new_features = []
    for j in range(n_layers):
      # Compute new feature maps
      # if bottleneck, do a 1x1 conv before the 3x3
      if bottleneck:
        stack = preact_conv(stack, 4*growth_rate, filter_size=[1, 1],
                            dropout_p=0.0)
      layer = preact_conv(stack, growth_rate, dropout_p=dropout_p)
      new_features.append(layer)
      # stack new layer
      stack = tf.concat([stack, layer], axis=-1)
    new_features = tf.concat(new_features, axis=-1)
    return stack, new_features

@slim.add_arg_scope
def TransitionLayer(inputs, n_filters, dropout_p=0.2, compression=1.0,
                    scope=None, outputs_collections=None):
  """
  Transition layer for DenseNet
  Apply 1x1 BN  + conv then 2x2 max pooling
  """
  with tf.name_scope(scope) as sc:
    if compression < 1.0:
      n_filters = tf.to_int32(tf.floor(n_filters*compression))
    l = preact_conv(inputs, n_filters, filter_size=[1, 1], dropout_p=dropout_p)
    l = slim.pool(l, [2, 2], stride=[2, 2], pooling_type='AVG')

    return l

@slim.add_arg_scope
def TransitionDown(inputs, n_filters, dropout_p=0.2, scope=None,
                   outputs_collections=None):
  """
  Transition Down (TD) for FC-DenseNet
  Apply 1x1 BN + ReLU + conv then 2x2 max pooling
  """
  with tf.name_scope(scope) as sc:
    l = preact_conv(inputs, n_filters, filter_size=[1, 1], dropout_p=dropout_p)
    l = slim.pool(l, [2, 2], stride=[2, 2], pooling_type='MAX')
    return l


@slim.add_arg_scope
def TransitionUp(block_to_upsample, skip_connection, n_filters_keep,
                 scope=None, outputs_collections=None):
  """
  Transition Up for FC-DenseNet
  Performs upsampling on block_to_upsample by a factor 2 and concatenates it
  with the skip_connection
  """
  with tf.name_scope(scope) as sc:
    # Upsample
    l = slim.conv2d_transpose(block_to_upsample, n_filters_keep,
                              kernel_size=[3, 3], stride=[2, 2])
    # Concatenate with skip connection
    l = tf.concat([l, skip_connection], axis=-1)
    return l
