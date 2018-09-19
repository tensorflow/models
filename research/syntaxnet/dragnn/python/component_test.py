# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Tests for component.py.
"""


import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from google.protobuf import text_format

from dragnn.protos import spec_pb2
from dragnn.python import component


class MockNetworkUnit(object):

  def get_layer_size(self, unused_layer_name):
    return 64


class MockComponent(object):

  def __init__(self):
    self.name = 'mock'
    self.network = MockNetworkUnit()


class MockMaster(object):

  def __init__(self):
    self.spec = spec_pb2.MasterSpec()
    self.hyperparams = spec_pb2.GridPoint()
    self.lookup_component = {'mock': MockComponent()}
    self.build_runtime_graph = False


class ComponentTest(test_util.TensorFlowTestCase):

  def setUp(self):
    # Clear the graph and all existing variables.  Otherwise, variables created
    # in different tests may collide with each other.
    tf.reset_default_graph()
    self.master = MockMaster()
    self.master_state = component.MasterState(
        handle=tf.constant(['foo', 'bar']), current_batch_size=2)
    self.network_states = {
        'mock': component.NetworkState(),
        'test': component.NetworkState(),
    }

  def testSoftmaxCrossEntropyLoss(self):
    logits = tf.constant([[0.0, 2.0, -1.0],
                          [-5.0, 1.0, -1.0],
                          [3.0, 1.0, -2.0]])  # pyformat: disable
    gold_labels = tf.constant([1, -1, 1])
    cost, correct, total, logits, gold_labels = (
        component.build_softmax_cross_entropy_loss(logits, gold_labels))

    with self.test_session() as sess:
      cost, correct, total, logits, gold_labels = (
          sess.run([cost, correct, total, logits, gold_labels]))

      # Cost = -2 + ln(1 + exp(2) + exp(-1))
      #        -1 + ln(exp(3) + exp(1) + exp(-2))
      self.assertAlmostEqual(cost, 2.3027, 4)
      self.assertEqual(correct, 1)
      self.assertEqual(total, 2)

      # Entries corresponding to gold labels equal to -1 are skipped.
      self.assertAllEqual(logits, [[0.0, 2.0, -1.0], [3.0, 1.0, -2.0]])
      self.assertAllEqual(gold_labels, [1, 1])

  def testSigmoidCrossEntropyLoss(self):
    indices = tf.constant([0, 0, 1])
    gold_labels = tf.constant([0, 1, 2])
    probs = tf.constant([0.6, 0.7, 0.2])
    logits = tf.constant([[0.9, -0.3, 0.1], [-0.5, 0.4, 2.0]])
    cost, correct, total, gold_labels = (
        component.build_sigmoid_cross_entropy_loss(logits, gold_labels, indices,
                                                   probs))

    with self.test_session() as sess:
      cost, correct, total, gold_labels = (
          sess.run([cost, correct, total, gold_labels]))

      # The cost corresponding to the three entries is, respectively,
      # 0.7012, 0.7644, and 1.7269. Each of them is computed using the formula
      # -prob_i * log(sigmoid(logit_i)) - (1-prob_i) * log(1-sigmoid(logit_i))
      self.assertAlmostEqual(cost, 3.1924, 4)
      self.assertEqual(correct, 1)
      self.assertEqual(total, 3)
      self.assertAllEqual(gold_labels, [0, 1, 2])

  def testGraphConstruction(self):
    component_spec = spec_pb2.ComponentSpec()
    text_format.Parse("""
        name: "test"
        network_unit {
          registered_name: "IdentityNetwork"
        }
        fixed_feature {
          name: "fixed" embedding_dim: 32 size: 1
        }
        component_builder {
          registered_name: "component.DynamicComponentBuilder"
        }
        """, component_spec)
    comp = component.DynamicComponentBuilder(self.master, component_spec)
    comp.build_greedy_training(self.master_state, self.network_states)

  def testGraphConstructionWithSigmoidLoss(self):
    component_spec = spec_pb2.ComponentSpec()
    text_format.Parse("""
        name: "test"
        network_unit {
          registered_name: "IdentityNetwork"
        }
        fixed_feature {
          name: "fixed" embedding_dim: 32 size: 1
        }
        component_builder {
          registered_name: "component.DynamicComponentBuilder"
          parameters {
            key: "loss_function"
            value: "sigmoid_cross_entropy"
          }
        }
        """, component_spec)
    comp = component.DynamicComponentBuilder(self.master, component_spec)
    comp.build_greedy_training(self.master_state, self.network_states)

    # Check that the loss op is present.
    op_names = [op.name for op in tf.get_default_graph().get_operations()]
    self.assertTrue('train_test/compute_loss/'
                    'sigmoid_cross_entropy_with_logits' in op_names)


if __name__ == '__main__':
  googletest.main()
