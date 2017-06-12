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

"""Code to compute random features for a given skeleton.
"""

import os.path
import math
import numpy as np
from Queue import Queue
import random
import tensorflow as tf
import time
import cPickle
from collections import defaultdict

import config
import skeleton_pb2
import activations
import skeleton as sk

class SparseRandomFeatures(object):
  def __init__(self, features, weights):
    self.features = features
    self.weights = weights

  def save(self, path):
    with open(path, 'w') as f:
      cPickle.dump([self.features, self.weights], f)

  @staticmethod
  def load(path):
    with open(path, 'r') as f:
      features, weights = cPickle.load(f)
      return SparseRandomFeatures(features, weights)


RF_TYPE = np.int16

def InBlockCoordinates(skeleton, layer, node, raveled_index):
  '''Returns layer id and in-block coordinates.'''
  # Find the location of r in the inputs to the node
  r = raveled_index
  layer = skeleton.layers[node[0]]
  input = skeleton.layers[node[0] - 1]
  in_block = list(np.unravel_index(r, layer.width))
  for d in xrange(layer.stride.size):
    if layer.padding == skeleton_pb2.VALID:
      in_block[d] += layer.stride[d] * node[1][d]
    elif input.padding == skeleton_pb2.SAME:
      in_block[d] += layer.stride[d] * node[1][d] - layer.width[d] / 2
      in_block[d] = min(max(0, in_block[d]), input.dimensions[d] - 1)
  return in_block


def ComputeRandomFeature(skeleton, learning_params):
  """ Compute a single random feature based on the skeleton. """
  # The queue maintains pairs of node_id (layer) and index within a node
  nodes = Queue()
  inp_layer = skeleton.layers[0]
  vec = np.zeros(inp_layer.size * inp_layer.replication * 2, dtype=RF_TYPE)

  # start the iteration by initializing the queue with ONE node from
  # the last (top) layer.
  last_layer = len(skeleton.layers) -1
  assert skeleton.layers[last_layer].size == 1, (
      "For a kernel function we must have a single node as the top layer")
  nodes.put((last_layer, [0] * skeleton.layers[last_layer].dimensions.size))

  while not nodes.empty():
    node = nodes.get()
    if node[0] == 0:
      if learning_params.use_even_features_in_rf:
        channel = 2 * np.random.randint(0, inp_layer.replication)
      else:
        channel = np.random.randint(0, inp_layer.replication * 2)
      ind = np.ravel_multi_index(
          node[1], inp_layer.dimensions) * inp_layer.replication * 2 + channel
      vec[ind] += 1
    else:
      # this is a hidden layer. Select a random number of inputs to
      # this node, select each input randomly from the set of inputs
      # and add them to the queue of nodes to be explored.
      layer = skeleton.layers[node[0]]
      n = layer.activation.RandomDistr()
      for i in xrange(n):
        r = np.random.randint(0, layer.input_size)
        in_block = InBlockCoordinates(skeleton, layer, node, r)
        nodes.put((node[0] - 1, in_block))
  return vec


def SparseKernelRandomFeatures(skeleton, kernel_width, learning_params,
                               seed=57721,
                               checkpoint_file="", checkpoint_freq=20000):
  """ Generate kernel_width random features based upon the skeleton """

  inp_layer = skeleton.layers[0]
  # for each x_j in the input layer, we have a feature x_j + i x_j+1 and
  # x_j - i x_j+1
  inp_size = inp_layer.size * inp_layer.replication * 2

  start_time = time.time()
  vec_weights = defaultdict(lambda: [0, 0])
  unique_rf = 0
  non_unique_rf = 0
  sparse_len = 0

  if len(checkpoint_file) > 0 and os.path.exists(checkpoint_file):
    with file_io.FileIO(checkpoint_file, "r") as f:
      vec_weights, unique_rf, non_unique_rf, sparse_len = cPickle.load(f)
      seed += non_unique_rf


  print_freq = 1000 * math.ceil(float(kernel_width) / 10000.0)
  random.seed(seed)
  np.random.seed(seed=seed)

  class RFChunk(object):
    def __init__(self, nonzeros, values):
      self.nonzeros = nonzeros
      self.values = values
      self._hash = hash((tuple(nonzeros), tuple(values)))

    def __hash__(self):
      return self._hash

    def __eq__(self, o):
      return (self.nonzeros.size == o.nonzeros.size and
              (self.nonzeros == o.nonzeros).all() and
              (self.values == o.values).all())

  while unique_rf < kernel_width:
    non_unique_rf += 1

    vec = ComputeRandomFeature(skeleton, learning_params)
    parity = np.random.randint(2)  # in the product, take real or imaginary part

    assert len(vec) == inp_size
    nonzeros = np.nonzero(vec)[0]
    values = vec[nonzeros]

    values_key = RFChunk(nonzeros, values)
    counts = vec_weights[values_key]
    is_new = (counts[parity] == 0)
    counts[parity] += 1

    if is_new:
      unique_rf += 1
      sparse_len += nonzeros.size
      if unique_rf % print_freq == 0 or unique_rf == kernel_width:
        print 'Generated', unique_rf, \
              'unique RFs and', non_unique_rf, \
              'total RFs. Time:', time.time() - start_time, 'seconds'
      if len(checkpoint_file) > 0 and unique_rf % checkpoint_freq == 0:
        with file_io.FileIO(checkpoint_file, "w") as f:
          cPickle.dump([vec_weights, unique_rf, non_unique_rf, sparse_len], f)

  indices = np.zeros((sparse_len, 2), dtype=np.int32)
  values = np.zeros(sparse_len, dtype=np.float32)
  weights = np.zeros(unique_rf, dtype=np.float32)
  index = 0
  current = 0
  for (rf_chunk, (even_weight, odd_weight)) in sorted(vec_weights.items()):
    assert even_weight + odd_weight > 0

    rf_nonzeros = rf_chunk.nonzeros
    rf_values = rf_chunk.values
    size = rf_nonzeros.size
    for weight in (even_weight, odd_weight):
      if weight > 0:
        indices[current:current+size, 0] = index           # row
        indices[current:current+size, 1] = rf_nonzeros     # nonzero columns
        values[current:current+size] = rf_values
        weights[index] = weight
        index += 1
        current += size

  return SparseRandomFeatures(
      features=[indices, values, [kernel_width, inp_size]],
      weights=np.sqrt(weights) * math.sqrt(1.0 / weights.sum()))


def Roll(value, dim, shift):
  """ Roll a tensor value along the dimension dim by the given amount.
  Similar to numpy roll.
  E.g. If dim = 2, and the size of value in that dimension is s
     rot_value[a, b, c, d, e] = value[a, b, (c + shift) % s, d, e] """
  value_shape = tf.shape(value)
  dim_vec = tf.pad(tf.constant([1], shape=[1]),
                   [[dim, tf.rank(value) - dim - 1]])
  begin_top = tf.zeros_like(value_shape)
  size_top = - tf.ones_like(value_shape) + (shift + 1) * dim_vec

  begin_bot = begin_top + shift * dim_vec
  size_bot = - tf.ones_like(value_shape)
  return tf.concat([tf.slice(value, begin_bot, size_bot),
                    tf.slice(value, begin_top, size_top)], dim)


kEpsilon = 1e-10  # to avoid divide by zero errors

def Polar(x):
  x_R = tf.abs(x) + kEpsilon
  x_Theta = tf.acos(tf.real(x) / x_R) * tf.sign(tf.imag(x))
  return x_R, x_Theta


def RandomFeaturesGraph(skeleton,
                        output_classes,
                        tf_input,
                        kernel_width,
                        tf_rf_vectors,
                        tf_rf_params,
                        weights=np.zeros(0)):
  """ Create a tensorflow graph based on the random features.
  Input:
    skeleton - skeleton created from a proto buffer
    output_classes - number of classes
    tf_input - a TF variable of input examples
    kernel_width - number of random features
    tf_rf_vectors - a TF constant sparsetensor of the random features matrix
    tf_rf_params - a TF variable of the learned parameters
  Output:
    predictions - a TF graph which calculates the prediction of random features
  """

  inp = skeleton.layers[0]
  input = tf.reshape(tf_input, [-1, inp.size, inp.replication])
  input_offset = Roll(input, 2, 1)

  complex_input = tf.concat([tf.complex(input, input_offset),
                             tf.complex(input, -input_offset)], 2)

  inp_size = inp.size * inp.replication * 2

  inputs_reshaped = tf.reshape(complex_input, [-1, inp_size])

  inputs_reshaped = math.sqrt(inp.replication / 2.0) * inputs_reshaped
  inputs_mag, inputs_theta = Polar(inputs_reshaped)

  image_features_mag = tf.exp(tf.sparse_tensor_dense_matmul( \
      tf_rf_vectors, tf.log(inputs_mag), adjoint_b=True))

  # We take cos OR sin of each feature (real or imaginary part).
  # Since features are in random order, offsets alternate between 0 and 1.
  # However if a feature has two copies (one for real and imaginary each),
  # they occur next to each other, so get opposite offsets.
  offset = tf.reshape(tf.to_float(tf.range(kernel_width) % 2), [-1, 1])
  image_features_angle = math.sqrt(2) * tf.cos(tf.sparse_tensor_dense_matmul(
      tf_rf_vectors, inputs_theta, adjoint_b=True) + offset * (np.pi/2.0))
  image_features = tf.transpose(image_features_mag * image_features_angle)

  stddev = math.sqrt(1.0 / kernel_width)

  if weights.size > 0:
    repeated_weights = tf.constant(weights, tf.float32)
    kernel_output = image_features * repeated_weights
  else:
    kernel_output = image_features * stddev

  predictions = tf.matmul(kernel_output, tf_rf_params)
  return predictions, kernel_output


def GenerateOrLoadRF(learning_params, seed=57721):
  # Read a skeleton and create its corresponding kernel
  assert hasattr(learning_params, 'skeleton_proto'), 'Skeleton proto must exist'
  skeleton = sk.Skeleton()
  skeleton.Load(learning_params.skeleton_proto)

  # Generate random feature vectors
  kernel_width = learning_params.rf_number

  if len(learning_params.rf_file_path) > 0:
    if len(learning_params.rf_file_name) > 0:
      filename = learning_params.rf_file_name
    else:
      filename = 'features_' + str(kernel_width) + '.pkl'
    features_file = os.path.join(learning_params.rf_file_path, filename)
    if os.path.exists(features_file):
      print 'Reading features from file'
      srf = SparseRandomFeatures.load(features_file)
    else:
      print 'Generating', kernel_width, 'features'
      srf = SparseKernelRandomFeatures(
          skeleton, kernel_width, learning_params,
          checkpoint_file=learning_params.rf_checkpoint_file, seed=seed)
      srf.save(features_file)
  else:
    print 'Generating features'
    srf = SparseKernelRandomFeatures(skeleton, kernel_width, learning_params)

  return srf
