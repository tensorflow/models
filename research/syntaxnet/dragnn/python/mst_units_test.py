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
"""Tests for DRAGNN wrappers for the MST solver."""

import math

import tensorflow as tf

from google.protobuf import text_format

from dragnn.protos import spec_pb2
from dragnn.python import mst_units
from dragnn.python import network_units

_MASTER_SPEC = r"""
  component {
    name: 'test'
    linked_feature {
      name: 'lengths'
      size: 1
      embedding_dim: -1
      fml: 'input.focus'
      source_translator: 'identity'
      source_component: 'previous'
      source_layer: 'lengths'
    }
    linked_feature {
      name: 'scores'
      size: 1
      embedding_dim: -1
      fml: 'input.focus'
      source_translator: 'identity'
      source_component: 'previous'
      source_layer: 'scores'
    }
  }
"""


class MockNetwork(object):

  def get_layer_size(self, unused_name):
    return -1


class MockComponent(object):

  def __init__(self, master, component_spec):
    self.master = master
    self.spec = component_spec
    self.name = component_spec.name
    self.beam_size = 1
    self.num_actions = -1
    self.network = MockNetwork()


class MockMaster(object):

  def __init__(self, build_runtime_graph=False):
    self.spec = spec_pb2.MasterSpec()
    text_format.Parse(_MASTER_SPEC, self.spec)
    self.hyperparams = spec_pb2.GridPoint()
    self.lookup_component = {
        'previous': MockComponent(self, spec_pb2.ComponentSpec())
    }
    self.build_runtime_graph = build_runtime_graph


class MstSolverNetworkTest(tf.test.TestCase):

  def setUp(self):
    # Clear the graph and all existing variables.  Otherwise, variables created
    # in different tests may collide with each other.
    tf.reset_default_graph()

  def testCreate(self):
    with self.test_session():
      master = MockMaster()
      component = MockComponent(master, master.spec.component[0])
      component.network = mst_units.MstSolverNetwork(component)

      stride = 1
      lengths = tf.constant([[3]], dtype=tf.int64)
      scores = tf.constant([[1.0, 0.5, 0.5],
                            [2.0, 0.5, 0.5],
                            [0.5, 3.0, 0.5]],
                           dtype=tf.float32)  # pyformat: disable

      linked_embeddings = [
          network_units.NamedTensor(lengths, 'lengths'),
          network_units.NamedTensor(scores, 'scores')
      ]
      network_tensors = component.network.create([], linked_embeddings, [],
                                                 None, False, stride)

      self.assertAllEqual(network_tensors[0].eval(), [3])
      self.assertAllEqual(network_tensors[1].eval(),
                          [[[1.0, 0.5, 0.5],
                            [2.0, 0.5, 0.5],
                            [0.5, 3.0, 0.5]]])  # pyformat: disable
      self.assertAllEqual(network_tensors[2].eval(),
                          [[1.0, 0.5, 0.5],
                           [2.0, 0.5, 0.5],
                           [0.5, 3.0, 0.5]])  # pyformat: disable
      self.assertAllEqual(network_tensors[3].eval(),
                          [[1.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])  # pyformat: disable

  def testGetBulkPredictions(self):
    with self.test_session():
      master = MockMaster()
      component = MockComponent(master, master.spec.component[0])
      component.network = mst_units.MstSolverNetwork(component)

      stride = 2
      lengths = tf.constant([[2], [3]], dtype=tf.int64)

      pad = -12345.6
      scores = tf.constant([[1.0, 2.0, pad],
                            [1.8, 2.0, pad],
                            [pad, pad, pad],
                            [3.8, 4.0, 3.9],
                            [3.9, 3.8, 4.0],
                            [3.8, 0.9, 4.0]],
                           dtype=tf.float32)  # pyformat: disable

      linked_embeddings = [
          network_units.NamedTensor(lengths, 'lengths'),
          network_units.NamedTensor(scores, 'scores')
      ]
      network_tensors = component.network.create([], linked_embeddings, [],
                                                 None, False, stride)
      predictions = component.network.get_bulk_predictions(
          stride, network_tensors)

      self.assertAllEqual(predictions.eval(),
                          [[0.0, 1.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0],
                           [0.0, 0.0, 1.0],
                           [0.0, 0.0, 1.0]])  # pyformat: disable

  def testComputeBulkLossM3n(self):
    with self.test_session():
      master = MockMaster()
      component = MockComponent(master, master.spec.component[0])
      component.spec.network_unit.parameters['loss'] = 'm3n'
      component.network = mst_units.MstSolverNetwork(component)

      stride = 2
      lengths = tf.constant([[2], [3]], dtype=tf.int64)

      # Note that these scores are large enough to overcome the +1 hamming loss
      # terms in the M3N loss.  Therefore, the score matrix determines the tree
      # that is used to compute the M3N loss.
      pad = -12345.6
      scores = tf.constant([[0.5, 2.0, pad],
                            [0.5, 2.0, pad],
                            [pad, pad, pad],
                            [2.5, 4.0, 2.5],
                            [2.5, 2.5, 4.0],
                            [2.5, 2.5, 4.0]],
                           dtype=tf.float32)  # pyformat: disable

      # For the first tree, the gold and scores agree on one arc (that index 1
      # is a root), and for the second tree, the gold and scores agree on none
      # of the arcs.  Therefore, we expect +1 and +3 for the first and second
      # trees in the M3N loss.
      gold = tf.constant([0, 1, -1, 0, 0, 1], tf.int32)
      first_gold_score = 0.5 + 2.0
      second_gold_score = 2.5 + 2.5 + 2.5
      first_tree_correct = 1
      second_tree_correct = 0
      first_tree_loss = 2 * 2.0 + 2 - first_tree_correct - first_gold_score
      second_tree_loss = 3 * 4.0 + 3 - second_tree_correct - second_gold_score

      linked_embeddings = [
          network_units.NamedTensor(lengths, 'lengths'),
          network_units.NamedTensor(scores, 'scores')
      ]
      network_tensors = component.network.create([], linked_embeddings, [],
                                                 None, False, stride)
      cost, correct, total = component.network.compute_bulk_loss(
          stride, network_tensors, gold)

      self.assertEqual(cost.eval(), first_tree_loss + second_tree_loss)
      self.assertEqual(correct.eval(), first_tree_correct + second_tree_correct)
      self.assertEqual(total.eval(), 2 + 3)

  def testComputeBulkLossCrf(self):
    with self.test_session():
      master = MockMaster()
      component = MockComponent(master, master.spec.component[0])
      component.spec.network_unit.parameters['loss'] = 'crf'
      component.network = mst_units.MstSolverNetwork(component)

      stride = 2
      lengths = tf.constant([[2], [3]], dtype=tf.int64)

      # These scores have 2.0 (in the log domain) on the gold arcs and 1.0
      # elsewhere.
      pad = -12345.6
      one = math.log(1.0)
      two = math.log(2.0)
      scores = tf.constant([[one, two, pad],
                            [one, two, pad],
                            [pad, pad, pad],
                            [one, two, one],
                            [one, one, two],
                            [one, one, two]],
                           dtype=tf.float32)  # pyformat: disable

      gold = tf.constant([1, 1, -1, 1, 2, 2], tf.int32)

      first_partition_function = (
          2.0 * 2.0 +  # 0 -> 1  (gold)
          1.0 * 1.0)  #  1 -> 0
      first_loss = -math.log(2.0 * 2.0 / first_partition_function)

      second_partition_function = (
          2.0 * 2.0 * 2.0 +  # 0 -> 1 -> 2  (gold)
          1.0 * 1.0 * 1.0 +  # 2 -> 1 -> 0
          1.0 * 1.0 * 1.0 +  # 0 -> 2 -> 1
          2.0 * 1.0 * 1.0 +  # 1 -> 2 -> 0
          2.0 * 1.0 * 1.0 +  # 1 -> 0 -> 2
          2.0 * 1.0 * 1.0 +  # 2 -> 0 -> 1
          2.0 * 2.0 * 1.0 +  # {0, 1} -> 2
          2.0 * 1.0 * 1.0 +  # {0, 2} -> 1
          1.0 * 1.0 * 1.0)  #  {1, 2} -> 0
      second_loss = -math.log(2.0 * 2.0 * 2.0 / second_partition_function)

      linked_embeddings = [
          network_units.NamedTensor(lengths, 'lengths'),
          network_units.NamedTensor(scores, 'scores')
      ]
      network_tensors = component.network.create([], linked_embeddings, [],
                                                 None, False, stride)
      cost, correct, total = component.network.compute_bulk_loss(
          stride, network_tensors, gold)

      self.assertAlmostEqual(cost.eval(), first_loss + second_loss)
      self.assertEqual(correct.eval(), 2 + 3)
      self.assertEqual(total.eval(), 2 + 3)


if __name__ == '__main__':
  tf.test.main()
