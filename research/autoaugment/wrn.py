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

"""Builds the Wide-ResNet Model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import custom_ops as ops
import numpy as np
import tensorflow as tf



def residual_block(
    x, in_filter, out_filter, stride, activate_before_residual=False):
  """Adds residual connection to `x` in addition to applying BN->ReLU->3x3 Conv.

  Args:
    x: Tensor that is the output of the previous layer in the model.
    in_filter: Number of filters `x` has.
    out_filter: Number of filters that the output of this layer will have.
    stride: Integer that specified what stride should be applied to `x`.
    activate_before_residual: Boolean on whether a BN->ReLU should be applied
      to x before the convolution is applied.

  Returns:
    A Tensor that is the result of applying two sequences of BN->ReLU->3x3 Conv
    and then adding that Tensor to `x`.
  """

  if activate_before_residual:  # Pass up RELU and BN activation for resnet
    with tf.variable_scope('shared_activation'):
      x = ops.batch_norm(x, scope='init_bn')
      x = tf.nn.relu(x)
      orig_x = x
  else:
    orig_x = x

  block_x = x
  if not activate_before_residual:
    with tf.variable_scope('residual_only_activation'):
      block_x = ops.batch_norm(block_x, scope='init_bn')
      block_x = tf.nn.relu(block_x)

  with tf.variable_scope('sub1'):
    block_x = ops.conv2d(
        block_x, out_filter, 3, stride=stride, scope='conv1')

  with tf.variable_scope('sub2'):
    block_x = ops.batch_norm(block_x, scope='bn2')
    block_x = tf.nn.relu(block_x)
    block_x = ops.conv2d(
        block_x, out_filter, 3, stride=1, scope='conv2')

  with tf.variable_scope(
      'sub_add'):  # If number of filters do not agree then zero pad them
    if in_filter != out_filter:
      orig_x = ops.avg_pool(orig_x, stride, stride)
      orig_x = ops.zero_pad(orig_x, in_filter, out_filter)
  x = orig_x + block_x
  return x


def _res_add(in_filter, out_filter, stride, x, orig_x):
  """Adds `x` with `orig_x`, both of which are layers in the model.

  Args:
    in_filter: Number of filters in `orig_x`.
    out_filter: Number of filters in `x`.
    stride: Integer specifying the stide that should be applied `orig_x`.
    x: Tensor that is the output of the previous layer.
    orig_x: Tensor that is the output of an earlier layer in the network.

  Returns:
    A Tensor that is the result of `x` and `orig_x` being added after
    zero padding and striding are applied to `orig_x` to get the shapes
    to match.
  """
  if in_filter != out_filter:
    orig_x = ops.avg_pool(orig_x, stride, stride)
    orig_x = ops.zero_pad(orig_x, in_filter, out_filter)
  x = x + orig_x
  orig_x = x
  return x, orig_x


def build_wrn_model(images, num_classes, wrn_size):
  """Builds the WRN model.

  Build the Wide ResNet model from https://arxiv.org/abs/1605.07146.

  Args:
    images: Tensor of images that will be fed into the Wide ResNet Model.
    num_classes: Number of classed that the model needs to predict.
    wrn_size: Parameter that scales the number of filters in the Wide ResNet
      model.

  Returns:
    The logits of the Wide ResNet model.
  """
  kernel_size = wrn_size
  filter_size = 3
  num_blocks_per_resnet = 4
  filters = [
      min(kernel_size, 16), kernel_size, kernel_size * 2, kernel_size * 4
  ]
  strides = [1, 2, 2]  # stride for each resblock

  # Run the first conv
  with tf.variable_scope('init'):
    x = images
    output_filters = filters[0]
    x = ops.conv2d(x, output_filters, filter_size, scope='init_conv')

  first_x = x  # Res from the beginning
  orig_x = x  # Res from previous block

  for block_num in range(1, 4):
    with tf.variable_scope('unit_{}_0'.format(block_num)):
      activate_before_residual = True if block_num == 1 else False
      x = residual_block(
          x,
          filters[block_num - 1],
          filters[block_num],
          strides[block_num - 1],
          activate_before_residual=activate_before_residual)
    for i in range(1, num_blocks_per_resnet):
      with tf.variable_scope('unit_{}_{}'.format(block_num, i)):
        x = residual_block(
            x,
            filters[block_num],
            filters[block_num],
            1,
            activate_before_residual=False)
    x, orig_x = _res_add(filters[block_num - 1], filters[block_num],
                         strides[block_num - 1], x, orig_x)
  final_stride_val = np.prod(strides)
  x, _ = _res_add(filters[0], filters[3], final_stride_val, x, first_x)
  with tf.variable_scope('unit_last'):
    x = ops.batch_norm(x, scope='final_bn')
    x = tf.nn.relu(x)
    x = ops.global_avg_pool(x)
    logits = ops.fc(x, num_classes)
  return logits
