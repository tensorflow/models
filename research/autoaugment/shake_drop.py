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

import math
import custom_ops as ops
import tensorflow as tf


def round_int(x):
  """Rounds `x` and then converts to an int."""
  return int(math.floor(x + 0.5))


def shortcut(x, output_filters, stride):
  """Applies strided avg pool or zero padding to make output_filters match x."""
  num_filters = int(x.shape[3])
  if stride == 2:
    x = ops.avg_pool(x, 2, stride=stride, padding='SAME')
  if num_filters != output_filters:
    diff = output_filters - num_filters
    assert diff > 0
    # Zero padd diff zeros
    padding = [[0, 0], [0, 0], [0, 0], [0, diff]]
    x = tf.pad(x, padding)
  return x


def calc_prob(curr_layer, total_layers, p_l):
  """Calculates drop prob depending on the current layer."""
  return 1 - (float(curr_layer) / total_layers) * p_l


def bottleneck_layer(x, n, stride, prob, is_training, alpha, beta):
  """Bottleneck layer for shake drop model."""
  assert alpha[1] > alpha[0]
  assert beta[1] > beta[0]
  with tf.variable_scope('bottleneck_{}'.format(prob)):
    input_layer = x
    x = ops.batch_norm(x, scope='bn_1_pre')
    x = ops.conv2d(x, n, 1, scope='1x1_conv_contract')
    x = ops.batch_norm(x, scope='bn_1_post')
    x = tf.nn.relu(x)
    x = ops.conv2d(x, n, 3, stride=stride, scope='3x3')
    x = ops.batch_norm(x, scope='bn_2')
    x = tf.nn.relu(x)
    x = ops.conv2d(x, n * 4, 1, scope='1x1_conv_expand')
    x = ops.batch_norm(x, scope='bn_3')

    # Apply regularization here
    # Sample bernoulli with prob
    if is_training:
      batch_size = tf.shape(x)[0]
      bern_shape = [batch_size, 1, 1, 1]
      random_tensor = prob
      random_tensor += tf.random_uniform(bern_shape, dtype=tf.float32)
      binary_tensor = tf.floor(random_tensor)

      alpha_values = tf.random_uniform(
          [batch_size, 1, 1, 1], minval=alpha[0], maxval=alpha[1],
          dtype=tf.float32)
      beta_values = tf.random_uniform(
          [batch_size, 1, 1, 1], minval=beta[0], maxval=beta[1],
          dtype=tf.float32)
      rand_forward = (
          binary_tensor + alpha_values - binary_tensor * alpha_values)
      rand_backward = (
          binary_tensor + beta_values - binary_tensor * beta_values)
      x = x * rand_backward + tf.stop_gradient(x * rand_forward -
                                               x * rand_backward)
    else:
      expected_alpha = (alpha[1] + alpha[0])/2
      # prob is the expectation of the bernoulli variable
      x = (prob + expected_alpha - prob * expected_alpha) * x

    res = shortcut(input_layer, n * 4, stride)
    return x + res


def build_shake_drop_model(images, num_classes, is_training):
  """Builds the PyramidNet Shake-Drop model.

  Build the PyramidNet Shake-Drop model from https://arxiv.org/abs/1802.02375.

  Args:
    images: Tensor of images that will be fed into the Wide ResNet Model.
    num_classes: Number of classed that the model needs to predict.
    is_training: Is the model training or not.

  Returns:
    The logits of the PyramidNet Shake-Drop model.
  """
  # ShakeDrop Hparams
  p_l = 0.5
  alpha_shake = [-1, 1]
  beta_shake = [0, 1]

  # PyramidNet Hparams
  alpha = 200
  depth = 272
  # This is for the bottleneck architecture specifically
  n = int((depth - 2) / 9)
  start_channel = 16
  add_channel = alpha / (3 * n)

  # Building the models
  x = images
  x = ops.conv2d(x, 16, 3, scope='init_conv')
  x = ops.batch_norm(x, scope='init_bn')

  layer_num = 1
  total_layers = n * 3
  start_channel += add_channel
  prob = calc_prob(layer_num, total_layers, p_l)
  x = bottleneck_layer(
      x, round_int(start_channel), 1, prob, is_training, alpha_shake,
      beta_shake)
  layer_num += 1
  for _ in range(1, n):
    start_channel += add_channel
    prob = calc_prob(layer_num, total_layers, p_l)
    x = bottleneck_layer(
        x, round_int(start_channel), 1, prob, is_training, alpha_shake,
        beta_shake)
    layer_num += 1

  start_channel += add_channel
  prob = calc_prob(layer_num, total_layers, p_l)
  x = bottleneck_layer(
      x, round_int(start_channel), 2, prob, is_training, alpha_shake,
      beta_shake)
  layer_num += 1
  for _ in range(1, n):
    start_channel += add_channel
    prob = calc_prob(layer_num, total_layers, p_l)
    x = bottleneck_layer(
        x, round_int(start_channel), 1, prob, is_training, alpha_shake,
        beta_shake)
    layer_num += 1

  start_channel += add_channel
  prob = calc_prob(layer_num, total_layers, p_l)
  x = bottleneck_layer(
      x, round_int(start_channel), 2, prob, is_training, alpha_shake,
      beta_shake)
  layer_num += 1
  for _ in range(1, n):
    start_channel += add_channel
    prob = calc_prob(layer_num, total_layers, p_l)
    x = bottleneck_layer(
        x, round_int(start_channel), 1, prob, is_training, alpha_shake,
        beta_shake)
    layer_num += 1

  assert layer_num - 1 == total_layers
  x = ops.batch_norm(x, scope='final_bn')
  x = tf.nn.relu(x)
  x = ops.global_avg_pool(x)
  # Fully connected
  logits = ops.fc(x, num_classes)
  return logits
