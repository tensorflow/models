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

import custom_ops as ops
import tensorflow as tf


def _shake_shake_skip_connection(x, output_filters, stride):
  """Adds a residual connection to the filter x for the shake-shake model."""
  curr_filters = int(x.shape[3])
  if curr_filters == output_filters:
    return x
  stride_spec = ops.stride_arr(stride, stride)
  # Skip path 1
  path1 = tf.nn.avg_pool(
      x, [1, 1, 1, 1], stride_spec, 'VALID', data_format='NHWC')
  path1 = ops.conv2d(path1, int(output_filters / 2), 1, scope='path1_conv')

  # Skip path 2
  # First pad with 0's then crop
  pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
  path2 = tf.pad(x, pad_arr)[:, 1:, 1:, :]
  concat_axis = 3

  path2 = tf.nn.avg_pool(
      path2, [1, 1, 1, 1], stride_spec, 'VALID', data_format='NHWC')
  path2 = ops.conv2d(path2, int(output_filters / 2), 1, scope='path2_conv')

  # Concat and apply BN
  final_path = tf.concat(values=[path1, path2], axis=concat_axis)
  final_path = ops.batch_norm(final_path, scope='final_path_bn')
  return final_path


def _shake_shake_branch(x, output_filters, stride, rand_forward, rand_backward,
                        is_training):
  """Building a 2 branching convnet."""
  x = tf.nn.relu(x)
  x = ops.conv2d(x, output_filters, 3, stride=stride, scope='conv1')
  x = ops.batch_norm(x, scope='bn1')
  x = tf.nn.relu(x)
  x = ops.conv2d(x, output_filters, 3, scope='conv2')
  x = ops.batch_norm(x, scope='bn2')
  if is_training:
    x = x * rand_backward + tf.stop_gradient(x * rand_forward -
                                             x * rand_backward)
  else:
    x *= 1.0 / 2
  return x


def _shake_shake_block(x, output_filters, stride, is_training):
  """Builds a full shake-shake sub layer."""
  batch_size = tf.shape(x)[0]

  # Generate random numbers for scaling the branches
  rand_forward = [
      tf.random_uniform(
          [batch_size, 1, 1, 1], minval=0, maxval=1, dtype=tf.float32)
      for _ in range(2)
  ]
  rand_backward = [
      tf.random_uniform(
          [batch_size, 1, 1, 1], minval=0, maxval=1, dtype=tf.float32)
      for _ in range(2)
  ]
  # Normalize so that all sum to 1
  total_forward = tf.add_n(rand_forward)
  total_backward = tf.add_n(rand_backward)
  rand_forward = [samp / total_forward for samp in rand_forward]
  rand_backward = [samp / total_backward for samp in rand_backward]
  zipped_rand = zip(rand_forward, rand_backward)

  branches = []
  for branch, (r_forward, r_backward) in enumerate(zipped_rand):
    with tf.variable_scope('branch_{}'.format(branch)):
      b = _shake_shake_branch(x, output_filters, stride, r_forward, r_backward,
                              is_training)
      branches.append(b)
  res = _shake_shake_skip_connection(x, output_filters, stride)
  return res + tf.add_n(branches)


def _shake_shake_layer(x, output_filters, num_blocks, stride,
                       is_training):
  """Builds many sub layers into one full layer."""
  for block_num in range(num_blocks):
    curr_stride = stride if (block_num == 0) else 1
    with tf.variable_scope('layer_{}'.format(block_num)):
      x = _shake_shake_block(x, output_filters, curr_stride,
                             is_training)
  return x


def build_shake_shake_model(images, num_classes, hparams, is_training):
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
  x = images

  x = ops.conv2d(x, 16, 3, scope='init_conv')
  x = ops.batch_norm(x, scope='init_bn')
  with tf.variable_scope('L1'):
    x = _shake_shake_layer(x, 16 * k, n, 1, is_training)
  with tf.variable_scope('L2'):
    x = _shake_shake_layer(x, 32 * k, n, 2, is_training)
  with tf.variable_scope('L3'):
    x = _shake_shake_layer(x, 64 * k, n, 2, is_training)
  x = tf.nn.relu(x)
  x = ops.global_avg_pool(x)

  # Fully connected
  logits = ops.fc(x, num_classes)
  return logits
