# Copyright 2018 Google Inc. All Rights Reserved.
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
"""Tests for biaffine_units."""


import tensorflow as tf

from google.protobuf import text_format

from dragnn.protos import spec_pb2
from dragnn.python import biaffine_units
from dragnn.python import network_units

_BATCH_SIZE = 11
_NUM_TOKENS = 22
_TOKEN_DIM = 33


class MockNetwork(object):

  def __init__(self):
    pass

  def get_layer_size(self, unused_name):
    return _TOKEN_DIM


class MockComponent(object):

  def __init__(self, master, component_spec):
    self.master = master
    self.spec = component_spec
    self.name = component_spec.name
    self.network = MockNetwork()
    self.beam_size = 1
    self.num_actions = 45
    self._attrs = {}

  def attr(self, name):
    return self._attrs[name]

  def get_variable(self, name):
    return tf.get_variable(name)


class MockMaster(object):

  def __init__(self):
    self.spec = spec_pb2.MasterSpec()
    self.hyperparams = spec_pb2.GridPoint()
    self.lookup_component = {
        'previous': MockComponent(self, spec_pb2.ComponentSpec())
    }


def _make_biaffine_spec():
  """Returns a ComponentSpec that the BiaffineDigraphNetwork works on."""
  component_spec = spec_pb2.ComponentSpec()
  text_format.Parse("""
    name: "test_component"
    backend { registered_name: "TestComponent" }
    linked_feature {
      name: "sources"
      fml: "input.focus"
      source_translator: "identity"
      source_component: "previous"
      source_layer: "sources"
      size: 1
      embedding_dim: -1
    }
    linked_feature {
      name: "targets"
      fml: "input.focus"
      source_translator: "identity"
      source_component: "previous"
      source_layer: "targets"
      size: 1
      embedding_dim: -1
    }
    network_unit {
      registered_name: "biaffine_units.BiaffineDigraphNetwork"
    }
  """, component_spec)
  return component_spec


class BiaffineDigraphNetworkTest(tf.test.TestCase):

  def setUp(self):
    # Clear the graph and all existing variables.  Otherwise, variables created
    # in different tests may collide with each other.
    tf.reset_default_graph()

  def testCanCreate(self):
    """Tests that create() works on a good spec."""
    with tf.Graph().as_default(), self.test_session():
      master = MockMaster()
      component = MockComponent(master, _make_biaffine_spec())

      with tf.variable_scope(component.name, reuse=None):
        component.network = biaffine_units.BiaffineDigraphNetwork(component)

      with tf.variable_scope(component.name, reuse=True):
        sources = network_units.NamedTensor(
            tf.zeros([_BATCH_SIZE * _NUM_TOKENS, _TOKEN_DIM]), 'sources')
        targets = network_units.NamedTensor(
            tf.zeros([_BATCH_SIZE * _NUM_TOKENS, _TOKEN_DIM]), 'targets')

        # No assertions on the result, just don't crash.
        component.network.create(
            fixed_embeddings=[],
            linked_embeddings=[sources, targets],
            context_tensor_arrays=None,
            attention_tensor=None,
            during_training=True,
            stride=_BATCH_SIZE)

  def testDerivedParametersForRuntime(self):
    """Test generation of derived parameters for the runtime."""
    with tf.Graph().as_default(), self.test_session():
      master = MockMaster()
      component = MockComponent(master, _make_biaffine_spec())

      with tf.variable_scope(component.name, reuse=None):
        component.network = biaffine_units.BiaffineDigraphNetwork(component)

      with tf.variable_scope(component.name, reuse=True):
        self.assertEqual(len(component.network.derived_params), 2)

        root_weights = component.network.derived_params[0]()
        root_bias = component.network.derived_params[1]()

        # Only check shape; values are random due to initialization.
        self.assertAllEqual(root_weights.shape.as_list(), [1, _TOKEN_DIM])
        self.assertAllEqual(root_bias.shape.as_list(), [1, 1])


if __name__ == '__main__':
  tf.test.main()
