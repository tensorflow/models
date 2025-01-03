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
"""Contains the model definition for the OverFeat network.

The definition for the network was obtained from:
  OverFeat: Integrated Recognition, Localization and Detection using
  Convolutional Networks
  Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus and
  Yann LeCun, 2014
  http://arxiv.org/abs/1312.6229

Usage:
  with slim.arg_scope(overfeat.overfeat_arg_scope()):
    outputs, end_points = overfeat.overfeat(inputs)

@@overfeat
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tf_slim as slim

# pylint: disable=g-long-lambda
trunc_normal = lambda stddev: tf.truncated_normal_initializer(
    0.0, stddev)


def overfeat_arg_scope(weight_decay=0.0005):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
      with slim.arg_scope([slim.max_pool2d], padding='VALID') as arg_sc:
        return arg_sc


def overfeat(inputs,
             num_classes=1000,
             is_training=True,
             dropout_keep_prob=0.5,
             spatial_squeeze=True,
             scope='overfeat',
             global_pool=False):
  """Contains the model definition for the OverFeat network.

  The definition for the network was obtained from:
    OverFeat: Integrated Recognition, Localization and Detection using
    Convolutional Networks
    Pierre Sermanet, David Eigen, Xiang Zhang, Michael Mathieu, Rob Fergus and
    Yann LeCun, 2014
    http://arxiv.org/abs/1312.6229

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 231x231. To use in fully
        convolutional mode, set spatial_squeeze to false.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes. If 0 or None, the logits layer is
      omitted and the input features to the logits layer are returned instead.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    global_pool: Optional boolean flag. If True, the input to the classification
      layer is avgpooled to size 1x1, for any input size. (This is not part
      of the original OverFeat.)

  Returns:
    net: the output of the logits layer (if num_classes is a non-zero integer),
      or the non-dropped-out input to the logits layer (if num_classes is 0 or
      None).
    end_points: a dict of tensors with intermediate activations.
  """
  with tf.variable_scope(scope, 'overfeat', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID',
                        scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.conv2d(net, 256, [5, 5], padding='VALID', scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.conv2d(net, 512, [3, 3], scope='conv3')
      net = slim.conv2d(net, 1024, [3, 3], scope='conv4')
      net = slim.conv2d(net, 1024, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # Use conv2d instead of fully_connected layers.
      with slim.arg_scope(
          [slim.conv2d],
          weights_initializer=trunc_normal(0.005),
          biases_initializer=tf.constant_initializer(0.1)):
        net = slim.conv2d(net, 3072, [6, 6], padding='VALID', scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(
            end_points_collection)
        if global_pool:
          net = tf.reduce_mean(
              input_tensor=net, axis=[1, 2], keepdims=True, name='global_pool')
          end_points['global_pool'] = net
        if num_classes:
          net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                             scope='dropout7')
          net = slim.conv2d(
              net,
              num_classes, [1, 1],
              activation_fn=None,
              normalizer_fn=None,
              biases_initializer=tf.zeros_initializer(),
              scope='fc8')
          if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
          end_points[sc.name + '/fc8'] = net
      return net, end_points
overfeat.default_image_size = 231
