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

"""Code to generate a Neural Net from a given skeleton.
"""

import math
import tensorflow as tf
import skeleton_pb2


def NeuralNet(skeleton, learning_params, tf_input, tf_params=None):
  """Creates a neural net from a skeleton.

  tf_input is the input layer - the images or other input that is fed
  into the network.

  tf_params is an optional list of tf variables, these are the ones that are
  trained.  If not provided, these are set up here and returned.

  Returns: A tensorflow graph implementing the neural net specified
  by the skeleton.

  """

  output_classes = learning_params.number_of_classes
  training_nodes = []
  nodes = [None] * len(skeleton.layers)
  nodes[0] = tf_input

  tf_params_index = -1
  if tf_params:
    tf_params_index = 0

  for i, layer in enumerate(skeleton.layers):
    if i == 0:
      continue
    inp_layer = skeleton.layers[i-1]
    shape = [0, 0, inp_layer.replication, layer.replication]
    strides = [1, 1, 1, 1]
    for k in xrange(layer.width.size):
      shape[k] = layer.width[k]
      strides[k + 1] = layer.stride[k]

    padding_type = skeleton_pb2.PaddingType.Name(layer.padding)

    if layer.operator == skeleton_pb2.CONVOLUTION:
      if tf_params_index >= 0:
        assert tf_params_index < len(tf_params) - 1
        conv_filter = tf_params[tf_params_index]
        bias = tf_params[tf_params_index + 1]
        tf_params_index += 2
      else:
        # simulate xavier initialization
        if i == 0:
          stddev = math.sqrt(1.0 / (shape[0] * shape[1]))
        else:
          stddev = math.sqrt(1.0 / (shape[0] * shape[1] * shape[2]))
        conv_filter = tf.Variable(
            tf.random_normal(
                shape, stddev=stddev),
            trainable=True)
        out_shape = list(layer.dimensions)
        out_shape.append(layer.replication)
        bias = tf.Variable(
            tf.random_uniform(out_shape, maxval=learning_params.bias_max_val),
            trainable=True)

      training_nodes.append(conv_filter)
      training_nodes.append(bias)
      nodes[i] = layer.activation.Act_tf(
          tf.nn.conv2d(nodes[i-1], conv_filter, strides, padding_type) + bias)
    elif layer.operator == skeleton_pb2.MAX_POOL:
      assert shape[2] == layer.replication, "Maxpool doesnt change replication"
      shape = [1, shape[0], shape[1], 1]
      nodes[i] = tf.nn.max_pool(nodes[i-1], shape, strides, padding_type)
    elif layer.operator == skeleton_pb2.AVERAGE_POOL:
      assert shape[2] == layer.replication, "Avgpool doesnt change replication"
      shape = [1, shape[0], shape[1], 1]
      nodes[i] = tf.nn.avg_pool(nodes[i-1], shape, strides, padding_type)

  # If we only need the representation layer we are done...
  if output_classes == 0:
    return nodes[-1], training_nodes

  # Add the output layer
  last_layer = skeleton.layers[-1]
  if tf_params_index >= 0:
    assert tf_params_index == len(tf_params) - 2
    output_weights = tf_params[tf_params_index]
    output_bias = tf_params[tf_params_index + 1]
  else:
    shape = list(last_layer.dimensions)
    shape.append(last_layer.replication)
    shape.append(output_classes)
    if learning_params.last_layer_init_zeros:
      stddev = 0
    else:
      stddev = math.sqrt(1.0 / (last_layer.size * last_layer.replication))
    output_weights = tf.Variable(
        tf.random_normal(
            shape, stddev=stddev),
        trainable=True)
    output_bias = tf.Variable(
        tf.zeros([1, 1, output_classes]), trainable=True, name=("output_bias"))
  training_nodes.append(output_weights)
  training_nodes.append(output_bias)
  outputs = tf.nn.conv2d(nodes[-1],
                         output_weights, [1, 1, 1, 1], "VALID") + output_bias

  return outputs, training_nodes
