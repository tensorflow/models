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

import tensorflow as tf


def fc_layer(name,
             bottom,
             output_dim,
             bias_term=True,
             weights_initializer=None,
             biases_initializer=None,
             reuse=None):
  # flatten bottom input
  shape = bottom.get_shape().as_list()
  input_dim = 1
  for d in shape[1:]:
    input_dim *= d
  flat_bottom = tf.reshape(bottom, [-1, input_dim])

  # weights and biases variables
  with tf.variable_scope(name, reuse=reuse):
    # initialize the variables
    if weights_initializer is None:
      weights_initializer = tf.contrib.layers.xavier_initializer()
    if bias_term and biases_initializer is None:
      biases_initializer = tf.constant_initializer(0.)

    # weights has shape [input_dim, output_dim]
    weights = tf.get_variable(
        'weights', [input_dim, output_dim], initializer=weights_initializer)
    if bias_term:
      biases = tf.get_variable(
          'biases', output_dim, initializer=biases_initializer)
    if not reuse:
      tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                           tf.nn.l2_loss(weights))

  if bias_term:
    fc = tf.nn.xw_plus_b(flat_bottom, weights, biases)
  else:
    fc = tf.matmul(flat_bottom, weights)
  return fc
