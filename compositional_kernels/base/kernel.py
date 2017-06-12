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

"""Code to compute the kernel using Tensorflow.

Takes two sets of vectors, and computes their Gram matrix.
"""

import numpy as np
import tensorflow as tf
import activations
import skeleton as sk
import skeleton_pb2
import utils


def DotProduct(x, y):
  """ x has shape (a1, b, c, d), and y has shape (a2, b, c, d).
  a1/a2 is the number of examples in the first(x) /second(y) set
  Each example is of size b x c x d
  Dot(x, y) has shape (a1, a2, b, c), and
  Dot(x, y)[a1, a2, b, c] = dot(x[a1, b, c, ...], y[a2, b, c, ...])
  1. expand x to be of shape [a1, 1, b, c, d] and y of shape [1, a2, b, c, d]
  2. mul expands x & y to be of shape [a1, a2, b, c, d] by replication
     and takes products element wise
  3. take the sum over the last dimension in order to calculate the
     inner Product between each two inputs
  4. result is of shape [a1, a2, b, c] and is reshaped to [a1*a2, b, c, 1]
  """
  xshape = tf.shape(x)
  yshape = tf.shape(y)
  prod = tf.reduce_sum(tf.expand_dims(x, 1) * tf.expand_dims(y, 0), 4)
  return tf.reshape(prod, [xshape[0] * yshape[0], xshape[1], xshape[2], 1])


def Kernel(skeleton, tf_inputs_1, tf_inputs_2):
  """ Creates a tensorflow graph that computes the kernel from a skeleton, given
  two sets of examples. """

  # nodes keeps the TF graph
  nodes = [None] * len(skeleton.layers)

  # Build the network for the hidden nodes.
  for (i, h) in enumerate(skeleton.layers):
    if i == 0:
      nodes[i] = DotProduct(tf_inputs_1, tf_inputs_2)
    else:
      filter_shape = list(h.width)
      filter_shape.extend([1, 1])
      strides = list(h.stride)
      strides.append(1)
      strides.insert(0, 1)
      padding_type = skeleton_pb2.PaddingType.Name(h.padding)
      filter = tf.constant(1.0 / np.prod(h.width),
                           shape=filter_shape,
                           dtype=tf.float32)
      if h.operator == skeleton_pb2.CONVOLUTION:
        act_input = tf.nn.conv2d(nodes[i-1], filter, strides, padding_type)
      else:
        print "Operator not supported for Kernel"

      nodes[i] = h.activation.Dual_tf(act_input)
      # Alternative using Hermite polynomials:
      # hidden_node = h.activation.HermiteDual_tf(act_input)

  return nodes[-1]
