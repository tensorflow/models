# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Inception-v3 expressed in TensorFlow-Slim.

  Usage:

  # Parameters for BatchNorm.
  batch_norm_params = {
      # Decay for the batch_norm moving averages.
      'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                        stddev=0.1,
                        activation=tf.nn.relu,
                        batch_norm_params=batch_norm_params):
      # Force all Variables to reside on the CPU.
      with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
        logits, endpoints = slim.inception.inception_v3(
            images,
            dropout_keep_prob=0.8,
            num_classes=num_classes,
            is_training=for_training,
            restore_logits=restore_logits,
            scope=scope)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception.slim import ops
from inception.slim import scopes


def inception_v3(inputs,
                 dropout_keep_prob=0.8,
                 num_classes=1000,
                 is_training=True,
                 restore_logits=True,
                 scope=''):
  """Latest Inception from http://arxiv.org/abs/1512.00567.

    "Rethinking the Inception Architecture for Computer Vision"

    Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
    Zbigniew Wojna

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    dropout_keep_prob: dropout keep_prob.
    num_classes: number of predicted classes.
    is_training: whether is training or not.
    restore_logits: whether or not the logits layers should be restored.
      Useful for fine-tuning a model with different num_classes.
    scope: Optional scope for name_scope.

  Returns:
    a list containing 'logits', 'aux_logits' Tensors.
  """
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}
  with tf.name_scope(scope, 'inception_v3', [inputs]):
    with scopes.arg_scope([ops.conv2d, ops.fc, ops.batch_norm, ops.dropout],
                          is_training=is_training):
      with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                            stride=1, padding='VALID'):
        # 299 x 299 x 3
        end_points['conv0'] = ops.conv2d(inputs, 32, [3, 3], stride=2,
                                         scope='conv0')
        # 149 x 149 x 32
        end_points['conv1'] = ops.conv2d(end_points['conv0'], 32, [3, 3],
                                         scope='conv1')
        # 147 x 147 x 32
        end_points['conv2'] = ops.conv2d(end_points['conv1'], 64, [3, 3],
                                         padding='SAME', scope='conv2')
        # 147 x 147 x 64
        end_points['pool1'] = ops.max_pool(end_points['conv2'], [3, 3],
                                           stride=2, scope='pool1')
        # 73 x 73 x 64
        end_points['conv3'] = ops.conv2d(end_points['pool1'], 80, [1, 1],
                                         scope='conv3')
        # 73 x 73 x 80.
        end_points['conv4'] = ops.conv2d(end_points['conv3'], 192, [3, 3],
                                         scope='conv4')
        # 71 x 71 x 192.
        end_points['pool2'] = ops.max_pool(end_points['conv4'], [3, 3],
                                           stride=2, scope='pool2')
        # 35 x 35 x 192.
        net = end_points['pool2']
      # Inception blocks
      with scopes.arg_scope([ops.conv2d, ops.max_pool, ops.avg_pool],
                            stride=1, padding='SAME'):
        # mixed: 35 x 35 x 256.
        with tf.variable_scope('mixed_35x35x256a'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 64, [1, 1])
          with tf.variable_scope('branch5x5'):
            branch5x5 = ops.conv2d(net, 48, [1, 1])
            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 32, [1, 1])
          net = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], 3)
          end_points['mixed_35x35x256a'] = net
        # mixed_1: 35 x 35 x 288.
        with tf.variable_scope('mixed_35x35x288a'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 64, [1, 1])
          with tf.variable_scope('branch5x5'):
            branch5x5 = ops.conv2d(net, 48, [1, 1])
            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 64, [1, 1])
          net = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], 3)
          end_points['mixed_35x35x288a'] = net
        # mixed_2: 35 x 35 x 288.
        with tf.variable_scope('mixed_35x35x288b'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 64, [1, 1])
          with tf.variable_scope('branch5x5'):
            branch5x5 = ops.conv2d(net, 48, [1, 1])
            branch5x5 = ops.conv2d(branch5x5, 64, [5, 5])
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 64, [1, 1])
          net = tf.concat([branch1x1, branch5x5, branch3x3dbl, branch_pool], 3)
          end_points['mixed_35x35x288b'] = net
        # mixed_3: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768a'):
          with tf.variable_scope('branch3x3'):
            branch3x3 = ops.conv2d(net, 384, [3, 3], stride=2, padding='VALID')
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 64, [1, 1])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 96, [3, 3],
                                      stride=2, padding='VALID')
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.max_pool(net, [3, 3], stride=2, padding='VALID')
          net = tf.concat([branch3x3, branch3x3dbl, branch_pool], 3)
          end_points['mixed_17x17x768a'] = net
        # mixed4: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768b'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = ops.conv2d(net, 128, [1, 1])
            branch7x7 = ops.conv2d(branch7x7, 128, [1, 7])
            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = ops.conv2d(net, 128, [1, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [1, 7])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 128, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 3)
          end_points['mixed_17x17x768b'] = net
        # mixed_5: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768c'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = ops.conv2d(net, 160, [1, 1])
            branch7x7 = ops.conv2d(branch7x7, 160, [1, 7])
            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = ops.conv2d(net, 160, [1, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [1, 7])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 3)
          end_points['mixed_17x17x768c'] = net
        # mixed_6: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768d'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = ops.conv2d(net, 160, [1, 1])
            branch7x7 = ops.conv2d(branch7x7, 160, [1, 7])
            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = ops.conv2d(net, 160, [1, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [1, 7])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 160, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 3)
          end_points['mixed_17x17x768d'] = net
        # mixed_7: 17 x 17 x 768.
        with tf.variable_scope('mixed_17x17x768e'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 192, [1, 1])
          with tf.variable_scope('branch7x7'):
            branch7x7 = ops.conv2d(net, 192, [1, 1])
            branch7x7 = ops.conv2d(branch7x7, 192, [1, 7])
            branch7x7 = ops.conv2d(branch7x7, 192, [7, 1])
          with tf.variable_scope('branch7x7dbl'):
            branch7x7dbl = ops.conv2d(net, 192, [1, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [7, 1])
            branch7x7dbl = ops.conv2d(branch7x7dbl, 192, [1, 7])
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat([branch1x1, branch7x7, branch7x7dbl, branch_pool], 3)
          end_points['mixed_17x17x768e'] = net
        # Auxiliary Head logits
        aux_logits = tf.identity(end_points['mixed_17x17x768e'])
        with tf.variable_scope('aux_logits'):
          aux_logits = ops.avg_pool(aux_logits, [5, 5], stride=3,
                                    padding='VALID')
          aux_logits = ops.conv2d(aux_logits, 128, [1, 1], scope='proj')
          # Shape of feature map before the final layer.
          shape = aux_logits.get_shape()
          aux_logits = ops.conv2d(aux_logits, 768, shape[1:3], stddev=0.01,
                                  padding='VALID')
          aux_logits = ops.flatten(aux_logits)
          aux_logits = ops.fc(aux_logits, num_classes, activation=None,
                              stddev=0.001, restore=restore_logits)
          end_points['aux_logits'] = aux_logits
        # mixed_8: 8 x 8 x 1280.
        # Note that the scope below is not changed to not void previous
        # checkpoints.
        # (TODO) Fix the scope when appropriate.
        with tf.variable_scope('mixed_17x17x1280a'):
          with tf.variable_scope('branch3x3'):
            branch3x3 = ops.conv2d(net, 192, [1, 1])
            branch3x3 = ops.conv2d(branch3x3, 320, [3, 3], stride=2,
                                   padding='VALID')
          with tf.variable_scope('branch7x7x3'):
            branch7x7x3 = ops.conv2d(net, 192, [1, 1])
            branch7x7x3 = ops.conv2d(branch7x7x3, 192, [1, 7])
            branch7x7x3 = ops.conv2d(branch7x7x3, 192, [7, 1])
            branch7x7x3 = ops.conv2d(branch7x7x3, 192, [3, 3],
                                     stride=2, padding='VALID')
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.max_pool(net, [3, 3], stride=2, padding='VALID')
          net = tf.concat([branch3x3, branch7x7x3, branch_pool], 3)
          end_points['mixed_17x17x1280a'] = net
        # mixed_9: 8 x 8 x 2048.
        with tf.variable_scope('mixed_8x8x2048a'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 320, [1, 1])
          with tf.variable_scope('branch3x3'):
            branch3x3 = ops.conv2d(net, 384, [1, 1])
            branch3x3 = tf.concat([ops.conv2d(branch3x3, 384, [1, 3]),
                                   ops.conv2d(branch3x3, 384, [3, 1])], 3)
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 448, [1, 1])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 384, [3, 3])
            branch3x3dbl = tf.concat([ops.conv2d(branch3x3dbl, 384, [1, 3]),
                                      ops.conv2d(branch3x3dbl, 384, [3, 1])], 3)
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 3)
          end_points['mixed_8x8x2048a'] = net
        # mixed_10: 8 x 8 x 2048.
        with tf.variable_scope('mixed_8x8x2048b'):
          with tf.variable_scope('branch1x1'):
            branch1x1 = ops.conv2d(net, 320, [1, 1])
          with tf.variable_scope('branch3x3'):
            branch3x3 = ops.conv2d(net, 384, [1, 1])
            branch3x3 = tf.concat([ops.conv2d(branch3x3, 384, [1, 3]),
                                   ops.conv2d(branch3x3, 384, [3, 1])], 3)
          with tf.variable_scope('branch3x3dbl'):
            branch3x3dbl = ops.conv2d(net, 448, [1, 1])
            branch3x3dbl = ops.conv2d(branch3x3dbl, 384, [3, 3])
            branch3x3dbl = tf.concat([ops.conv2d(branch3x3dbl, 384, [1, 3]),
                                      ops.conv2d(branch3x3dbl, 384, [3, 1])], 3)
          with tf.variable_scope('branch_pool'):
            branch_pool = ops.avg_pool(net, [3, 3])
            branch_pool = ops.conv2d(branch_pool, 192, [1, 1])
          net = tf.concat([branch1x1, branch3x3, branch3x3dbl, branch_pool], 3)
          end_points['mixed_8x8x2048b'] = net
        # Final pooling and prediction
        with tf.variable_scope('logits'):
          shape = net.get_shape()
          net = ops.avg_pool(net, shape[1:3], padding='VALID', scope='pool')
          # 1 x 1 x 2048
          net = ops.dropout(net, dropout_keep_prob, scope='dropout')
          net = ops.flatten(net, scope='flatten')
          # 2048
          logits = ops.fc(net, num_classes, activation=None, scope='logits',
                          restore=restore_logits)
          # 1000
          end_points['logits'] = logits
          end_points['predictions'] = tf.nn.softmax(logits, name='predictions')
      return logits, end_points


def inception_v3_parameters(weight_decay=0.00004, stddev=0.1,
                            batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
  """Yields the scope with the default parameters for inception_v3.

  Args:
    weight_decay: the weight decay for weights variables.
    stddev: standard deviation of the truncated guassian weight distribution.
    batch_norm_decay: decay for the moving average of batch_norm momentums.
    batch_norm_epsilon: small float added to variance to avoid dividing by zero.

  Yields:
    a arg_scope with the parameters needed for inception_v3.
  """
  # Set weight_decay for weights in Conv and FC layers.
  with scopes.arg_scope([ops.conv2d, ops.fc],
                        weight_decay=weight_decay):
    # Set stddev, activation and parameters for batch_norm.
    with scopes.arg_scope([ops.conv2d],
                          stddev=stddev,
                          activation=tf.nn.relu,
                          batch_norm_params={
                              'decay': batch_norm_decay,
                              'epsilon': batch_norm_epsilon}) as arg_scope:
      yield arg_scope
