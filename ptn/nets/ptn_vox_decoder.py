# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Training decoder as used in PTN (NIPS16)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


@tf.contrib.framework.add_arg_scope
def conv3d_transpose(inputs,
                     num_outputs,
                     kernel_size,
                     stride=1,
                     padding='SAME',
                     activation_fn=tf.nn.relu,
                     weights_initializer=tf.contrib.layers.xavier_initializer(),
                     biases_initializer=tf.zeros_initializer(),
                     reuse=None,
                     trainable=True,
                     scope=None):
  """Wrapper for conv3d_transpose layer.

  This function wraps the tf.conv3d_transpose with basic non-linearity.
  Tt creates a variable called `weights`, representing the kernel,
  that is convoled with the input. A second varibale called `biases'
  is added to the result of operation.
  """
  with tf.variable_scope(
      scope, 'Conv3d_transpose', [inputs], reuse=reuse):
    dtype = inputs.dtype.base_dtype
    kernel_d, kernel_h, kernel_w = kernel_size[0:3]
    num_filters_in = inputs.get_shape()[4]

    weights_shape = [kernel_d, kernel_h, kernel_w, num_outputs, num_filters_in]
    weights = tf.get_variable('weights',
                              shape=weights_shape,
                              dtype=dtype,
                              initializer=weights_initializer,
                              trainable=trainable)
    tf.contrib.framework.add_model_variable(weights)

    input_shape = inputs.get_shape().as_list()
    batch_size = input_shape[0]
    depth = input_shape[1]
    height = input_shape[2]
    width = input_shape[3]

    def get_deconv_dim(dim_size, stride_size):
      # Only support padding='SAME'.
      if isinstance(dim_size, tf.Tensor):
        dim_size = tf.multiply(dim_size, stride_size)
      elif dim_size is not None:
        dim_size *= stride_size
      return dim_size

    out_depth = get_deconv_dim(depth, stride)
    out_height = get_deconv_dim(height, stride)
    out_width = get_deconv_dim(width, stride)

    out_shape = [batch_size, out_depth, out_height, out_width, num_outputs]
    outputs = tf.nn.conv3d_transpose(inputs, weights, out_shape,
                                     [1, stride, stride, stride, 1],
                                     padding=padding)

    outputs.set_shape(out_shape)

    if biases_initializer is not None:
      biases = tf.get_variable('biases',
                               shape=[num_outputs,],
                               dtype=dtype,
                               initializer=biases_initializer,
                               trainable=trainable)
      tf.contrib.framework.add_model_variable(biases)
      outputs = tf.nn.bias_add(outputs, biases)

    if activation_fn:
      outputs = activation_fn(outputs)
    return outputs


def model(identities, params, is_training):
  """Model transforming embedding to voxels."""
  del is_training  # Unused
  f_dim = params.f_dim

  # Please refer to the original implementation: github.com/xcyan/nips16_PTN
  # In TF replication, we use a slightly different architecture.
  with slim.arg_scope(
      [slim.fully_connected, conv3d_transpose],
      weights_initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1)):
    h0 = slim.fully_connected(
        identities, 4 * 4 * 4 * f_dim * 8, activation_fn=tf.nn.relu)
    h1 = tf.reshape(h0, [-1, 4, 4, 4, f_dim * 8])
    h1 = conv3d_transpose(
        h1, f_dim * 4, [4, 4, 4], stride=2, activation_fn=tf.nn.relu)
    h2 = conv3d_transpose(
        h1, int(f_dim * 3 / 2), [5, 5, 5], stride=2, activation_fn=tf.nn.relu)
    h3 = conv3d_transpose(
        h2, 1, [6, 6, 6], stride=2, activation_fn=tf.nn.sigmoid)
  return h3
