# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Contains the definition for Inflated 3D Inception V1 (I3D).

The network architecture is proposed by:
  Joao Carreira and Andrew Zisserman,
  Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset.
  https://arxiv.org/abs/1705.07750
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import i3d_utils
from nets import s3dg

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)
conv3d_spatiotemporal = i3d_utils.conv3d_spatiotemporal


def i3d_arg_scope(weight_decay=1e-7,
                  batch_norm_decay=0.999,
                  batch_norm_epsilon=0.001,
                  use_renorm=False,
                  separable_conv3d=False):
  """Defines default arg_scope for I3D.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
    use_renorm: Whether to use batch renormalization or not.
    separable_conv3d: Whether to use separable 3d Convs.

  Returns:
    sc: An arg_scope to use for the models.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # Turns off fused batch norm.
      'fused': False,
      'renorm': use_renorm,
      # collection containing the moving mean and moving variance.
      'variables_collections': {
          'beta': None,
          'gamma': None,
          'moving_mean': ['moving_vars'],
          'moving_variance': ['moving_vars'],
      }
  }

  with slim.arg_scope(
      [slim.conv3d, conv3d_spatiotemporal],
      weights_regularizer=slim.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with slim.arg_scope(
        [conv3d_spatiotemporal], separable=separable_conv3d) as sc:
      return sc


def i3d_base(inputs, final_endpoint='Mixed_5c',
             scope='InceptionV1'):
  """Defines the I3D base architecture.

  Note that we use the names as defined in Inception V1 to facilitate checkpoint
  conversion from an image-trained Inception V1 checkpoint to I3D checkpoint.

  Args:
    inputs: A 5-D float tensor of size [batch_size, num_frames, height, width,
      channels].
    final_endpoint: Specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
      'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
      'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']
    scope: Optional variable_scope.

  Returns:
    A dictionary from components of the network to the corresponding activation.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values.
  """

  return s3dg.s3dg_base(
      inputs,
      first_temporal_kernel_size=7,
      temporal_conv_startat='Conv2d_2c_3x3',
      gating_startat=None,
      final_endpoint=final_endpoint,
      min_depth=16,
      depth_multiplier=1.0,
      data_format='NDHWC',
      scope=scope)


def i3d(inputs,
        num_classes=1000,
        dropout_keep_prob=0.8,
        is_training=True,
        prediction_fn=slim.softmax,
        spatial_squeeze=True,
        reuse=None,
        scope='InceptionV1'):
  """Defines the I3D architecture.

  The default image size used to train this network is 224x224.

  Args:
    inputs: A 5-D float tensor of size [batch_size, num_frames, height, width,
      channels].
    num_classes: number of predicted classes.
    dropout_keep_prob: the percentage of activation values that are retained.
    is_training: whether is training or not.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, num_classes]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
  # Final pooling and prediction
  with tf.variable_scope(
      scope, 'InceptionV1', [inputs, num_classes], reuse=reuse) as scope:
    with slim.arg_scope(
        [slim.batch_norm, slim.dropout], is_training=is_training):
      net, end_points = i3d_base(inputs, scope=scope)
      with tf.variable_scope('Logits'):
        kernel_size = i3d_utils.reduced_kernel_size_3d(net, [2, 7, 7])
        net = slim.avg_pool3d(
            net, kernel_size, stride=1, scope='AvgPool_0a_7x7')
        net = slim.dropout(net, dropout_keep_prob, scope='Dropout_0b')
        logits = slim.conv3d(
            net,
            num_classes, [1, 1, 1],
            activation_fn=None,
            normalizer_fn=None,
            scope='Conv2d_0c_1x1')
        # Temporal average pooling.
        logits = tf.reduce_mean(logits, axis=1)
        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points


i3d.default_image_size = 224
