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

import math
import numpy as np
import sys

from google.protobuf import text_format
import activations
import skeleton_pb2

class Layer(object):
  """ A class to store a layer of the skeleton, with all its related
  information such as inputs, size, id etc """

  def __init__(self, layer_pb, prev_layer):
    if prev_layer == None:
      self.SetupInputLayer(layer_pb)
    else:
      self.SetupHiddenLayer(layer_pb, prev_layer)
    self.size = np.prod(self.dimensions)

  def SetupInputLayer(self, inp):
    self.dimensions = np.array(inp.dimension_size, dtype=np.int32)
    self.replication = inp.channels

  def SetupHiddenLayer(self, h, prev_layer):
    self.activation = activations.CreateActivation(h.activation)
    self.activation.SetParams(h.activation_params)
    self.padding = h.padding
    self.operator = h.operator
    self.replication = 0
    if h.HasField("replication"):
      self.replication = h.replication

    assert len(h.dim) == prev_layer.dimensions.size
    self.width = np.zeros(len(h.dim), dtype=np.int32)
    self.stride = np.ones(len(h.dim), dtype=np.int32)
    self.dimensions = np.copy(prev_layer.dimensions)
    for d, spec in enumerate(h.dim):
      if spec.fully_connected:
        self.width[d] = self.dimensions[d]
        self.dimensions[d] = 1
      else:
        assert spec.width <= self.dimensions[d]
        self.width[d] = spec.width
        self.stride[d] = spec.stride
        if self.padding == skeleton_pb2.VALID:
          self.dimensions[d] -= (self.width[d] - 1)
        self.dimensions[d] = int(math.ceil(float(self.dimensions[d]) /
                                           self.stride[d]))
    self.input_size = np.prod(self.width)


class Skeleton(object):
  """ A class to read in a skeleton description and create a skeleton,
  which can be used to create a neural network or a kernel. """

  def __init__(self):
    self.layers = []

  def Load(self, filename):
    with open(filename, "r") as f:
      skeleton_pb = skeleton_pb2.Skeleton()
      text_format.Merge(f.read(), skeleton_pb)
      self.LoadProto(skeleton_pb)

  def LoadProto(self, skeleton_pb):
    self.layers = [None] * (1 + len(skeleton_pb.hidden))
    self.layers[0] = Layer(skeleton_pb.input, None)
    for i, h in enumerate(skeleton_pb.hidden):
      self.layers[i + 1] = Layer(h, self.layers[i])

  def SetReplication(self, reps):
    """ Set the replication factors for each hidden layer, to create a NN."""
    assert len(reps) == len(self.layers) - 1
    for i, rep in enumerate(reps):
      self.layers[i + 1].replication = rep

  def RemoveDualBias(self):
    """ Ensures that the constant term in the dual activation is 0.
    Not precise mathematically, but increases feature generation
    efficiency dramatically. """
    for i, h in enumerate(self.layers):
      if i > 0:
        self.layers[i].activation.RemoveDualBias()

  def SetActivationCoeffs(self, num):
    """ Ensures that every dual activation has at least num coeffs. Useful for
    debugging, concentration experiments etc """
    for i, h in enumerate(self.layers):
      if i > 0:
        self.layers[i].activation.SetCoeffs(num)
