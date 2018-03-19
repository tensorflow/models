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

"""Creates rotator network model.

This model performs the out-of-plane rotations given input image and action.
The action is either no-op, rotate clockwise or rotate counter-clockwise.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def bilinear(input_x, input_y, output_size):
  """Define the bilinear transformation layer."""
  shape_x = input_x.get_shape().as_list()
  shape_y = input_y.get_shape().as_list()

  weights_initializer = tf.truncated_normal_initializer(stddev=0.02,
                                                        seed=1)
  biases_initializer = tf.constant_initializer(0.0)

  matrix = tf.get_variable("Matrix", [shape_x[1], shape_y[1], output_size],
                           tf.float32, initializer=weights_initializer)
  bias = tf.get_variable("Bias", [output_size],
                         initializer=biases_initializer)
  # Add to GraphKeys.MODEL_VARIABLES
  tf.contrib.framework.add_model_variable(matrix)
  tf.contrib.framework.add_model_variable(bias)
  # Define the transformation
  h0 = tf.matmul(input_x, tf.reshape(matrix,
                                     [shape_x[1], shape_y[1]*output_size]))
  h0 = tf.reshape(h0, [-1, shape_y[1], output_size])
  h1 = tf.tile(tf.reshape(input_y, [-1, shape_y[1], 1]),
               [1, 1, output_size])
  h1 = tf.multiply(h0, h1)
  return tf.reduce_sum(h1, 1) + bias


def model(poses, actions, params, is_training):
  """Model for performing rotation."""
  del is_training  # Unused
  return bilinear(poses, actions, params.z_dim)
