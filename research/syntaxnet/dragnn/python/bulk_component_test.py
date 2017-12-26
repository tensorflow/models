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

"""Tests for bulk_component.

Verifies that:
1. BulkFeatureExtractor and BulkAnnotator both raise NotImplementedError when
   non-identity translator configured.
2. BulkFeatureExtractor and BulkAnnotator both raise RuntimeError when
   recurrent linked features are configured.
3. BulkAnnotator raises RuntimeError when fixed features are configured.
4. BulkFeatureIdExtractor raises ValueError when linked features are configured,
   or when the fixed features are invalid.
"""

import os.path


import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from google.protobuf import text_format

from dragnn.protos import spec_pb2
from dragnn.python import bulk_component
from dragnn.python import component
from dragnn.python import dragnn_ops
from dragnn.python import network_units
from syntaxnet import sentence_pb2

FLAGS = tf.app.flags.FLAGS


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


def _create_fake_corpus():
  """Returns a list of fake serialized sentences for tests."""
  num_docs = 4
  corpus = []
  for num_tokens in range(1, num_docs + 1):
    sentence = sentence_pb2.Sentence()
    sentence.text = 'x' * num_tokens
    for i in range(num_tokens):
      token = sentence.token.add()
      token.word = 'x'
      token.start = i
      token.end = i
    corpus.append(sentence.SerializeToString())
  return corpus


class BulkComponentTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.master = MockMaster()
    self.master_state = component.MasterState(
        handle='handle', current_batch_size=2)
    self.network_states = {
        'mock': component.NetworkState(),
        'test': component.NetworkState(),
    }

  def testFailsOnNonIdentityTranslator(self):
    component_spec = spec_pb2.ComponentSpec()
    text_format.Parse("""
        name: "test"
        network_unit {
          registered_name: "IdentityNetwork"
        }
        linked_feature {
          name: "features" embedding_dim: -1 size: 1
          source_translator: "history"
          source_component: "mock"
        }
        """, component_spec)

    # For feature extraction:
    with tf.Graph().as_default():
      comp = bulk_component.BulkFeatureExtractorComponentBuilder(
          self.master, component_spec)

      # Expect feature extraction to generate a error due to the "history"
      # translator.
      with self.assertRaises(NotImplementedError):
        comp.build_greedy_training(self.master_state, self.network_states)

    # As well as annotation:
    with tf.Graph().as_default():
      comp = bulk_component.BulkAnnotatorComponentBuilder(
          self.master, component_spec)

      with self.assertRaises(NotImplementedError):
        comp.build_greedy_training(self.master_state, self.network_states)

  def testFailsOnRecurrentLinkedFeature(self):
    component_spec = spec_pb2.ComponentSpec()
    text_format.Parse("""
        name: "test"
        network_unit {
          registered_name: "FeedForwardNetwork"
          parameters {
            key: 'hidden_layer_sizes' value: '64'
          }
        }
        linked_feature {
          name: "features" embedding_dim: -1 size: 1
          source_translator: "identity"
          source_component: "test"
          source_layer: "layer_0"
        }
        """, component_spec)

    # For feature extraction:
    with tf.Graph().as_default():
      comp = bulk_component.BulkFeatureExtractorComponentBuilder(
          self.master, component_spec)

      # Expect feature extraction to generate a error due to the "history"
      # translator.
      with self.assertRaises(RuntimeError):
        comp.build_greedy_training(self.master_state, self.network_states)

    # As well as annotation:
    with tf.Graph().as_default():
      comp = bulk_component.BulkAnnotatorComponentBuilder(
          self.master, component_spec)

      with self.assertRaises(RuntimeError):
        comp.build_greedy_training(self.master_state, self.network_states)

  def testConstantFixedFeatureFailsIfNotPretrained(self):
    component_spec = spec_pb2.ComponentSpec()
    text_format.Parse("""
        name: "test"
        network_unit {
          registered_name: "IdentityNetwork"
        }
        fixed_feature {
          name: "fixed" embedding_dim: 32 size: 1
          is_constant: true
        }
        component_builder {
          registered_name: "bulk_component.BulkFeatureExtractorComponentBuilder"
        }
        """, component_spec)
    with tf.Graph().as_default():
      comp = bulk_component.BulkFeatureExtractorComponentBuilder(
          self.master, component_spec)

      with self.assertRaisesRegexp(ValueError,
                                   'Constant embeddings must be pretrained'):
        comp.build_greedy_training(self.master_state, self.network_states)
      with self.assertRaisesRegexp(ValueError,
                                   'Constant embeddings must be pretrained'):
        comp.build_greedy_inference(
            self.master_state, self.network_states, during_training=True)
      with self.assertRaisesRegexp(ValueError,
                                   'Constant embeddings must be pretrained'):
        comp.build_greedy_inference(
            self.master_state, self.network_states, during_training=False)

  def testNormalFixedFeaturesAreDifferentiable(self):
    component_spec = spec_pb2.ComponentSpec()
    text_format.Parse("""
        name: "test"
        network_unit {
          registered_name: "IdentityNetwork"
        }
        fixed_feature {
          name: "fixed" embedding_dim: 32 size: 1
          pretrained_embedding_matrix { part {} }
          vocab { part {} }
        }
        component_builder {
          registered_name: "bulk_component.BulkFeatureExtractorComponentBuilder"
        }
        """, component_spec)
    with tf.Graph().as_default():
      comp = bulk_component.BulkFeatureExtractorComponentBuilder(
          self.master, component_spec)

      # Get embedding matrix variables.
      with tf.variable_scope(comp.name, reuse=True):
        fixed_embedding_matrix = tf.get_variable(
            network_units.fixed_embeddings_name(0))

      # Get output layer.
      comp.build_greedy_training(self.master_state, self.network_states)
      activations = self.network_states[comp.name].activations
      outputs = activations[comp.network.layers[0].name].bulk_tensor

      # Compute the gradient of the output layer w.r.t. the embedding matrix.
      # This should be well-defined for in the normal case.
      gradients = tf.gradients(outputs, fixed_embedding_matrix)
      self.assertEqual(len(gradients), 1)
      self.assertFalse(gradients[0] is None)

  def testConstantFixedFeaturesAreNotDifferentiableButOthersAre(self):
    component_spec = spec_pb2.ComponentSpec()
    text_format.Parse("""
        name: "test"
        network_unit {
          registered_name: "IdentityNetwork"
        }
        fixed_feature {
          name: "constant" embedding_dim: 32 size: 1
          is_constant: true
          pretrained_embedding_matrix { part {} }
          vocab { part {} }
        }
        fixed_feature {
          name: "trainable" embedding_dim: 32 size: 1
          pretrained_embedding_matrix { part {} }
          vocab { part {} }
        }
        component_builder {
          registered_name: "bulk_component.BulkFeatureExtractorComponentBuilder"
        }
        """, component_spec)
    with tf.Graph().as_default():
      comp = bulk_component.BulkFeatureExtractorComponentBuilder(
          self.master, component_spec)

      # Get embedding matrix variables.
      with tf.variable_scope(comp.name, reuse=True):
        constant_embedding_matrix = tf.get_variable(
            network_units.fixed_embeddings_name(0))
        trainable_embedding_matrix = tf.get_variable(
            network_units.fixed_embeddings_name(1))

      # Get output layer.
      comp.build_greedy_training(self.master_state, self.network_states)
      activations = self.network_states[comp.name].activations
      outputs = activations[comp.network.layers[0].name].bulk_tensor

      # The constant embeddings are non-differentiable.
      constant_gradients = tf.gradients(outputs, constant_embedding_matrix)
      self.assertEqual(len(constant_gradients), 1)
      self.assertTrue(constant_gradients[0] is None)

      # The trainable embeddings are differentiable.
      trainable_gradients = tf.gradients(outputs, trainable_embedding_matrix)
      self.assertEqual(len(trainable_gradients), 1)
      self.assertFalse(trainable_gradients[0] is None)

  def testFailsOnFixedFeature(self):
    component_spec = spec_pb2.ComponentSpec()
    text_format.Parse("""
        name: "annotate"
        network_unit {
          registered_name: "IdentityNetwork"
        }
        fixed_feature {
          name: "fixed" embedding_dim: 32 size: 1
        }
        """, component_spec)
    with tf.Graph().as_default():
      comp = bulk_component.BulkAnnotatorComponentBuilder(
          self.master, component_spec)

      # Expect feature extraction to generate a runtime error due to the
      # fixed feature.
      with self.assertRaises(RuntimeError):
        comp.build_greedy_training(self.master_state, self.network_states)

  def testBulkFeatureIdExtractorOkWithOneFixedFeature(self):
    component_spec = spec_pb2.ComponentSpec()
    text_format.Parse("""
        name: "test"
        network_unit {
          registered_name: "IdentityNetwork"
        }
        fixed_feature {
          name: "fixed" embedding_dim: -1 size: 1
        }
        """, component_spec)
    with tf.Graph().as_default():
      comp = bulk_component.BulkFeatureIdExtractorComponentBuilder(
          self.master, component_spec)

      # Should not raise errors.
      self.network_states[component_spec.name] = component.NetworkState()
      comp.build_greedy_training(self.master_state, self.network_states)
      self.network_states[component_spec.name] = component.NetworkState()
      comp.build_greedy_inference(self.master_state, self.network_states)

  def testBulkFeatureIdExtractorFailsOnLinkedFeature(self):
    component_spec = spec_pb2.ComponentSpec()
    text_format.Parse("""
        name: "test"
        network_unit {
          registered_name: "IdentityNetwork"
        }
        fixed_feature {
          name: "fixed" embedding_dim: -1 size: 1
        }
        linked_feature {
          name: "linked" embedding_dim: -1 size: 1
          source_translator: "identity"
          source_component: "mock"
        }
        """, component_spec)
    with tf.Graph().as_default():
      with self.assertRaises(ValueError):
        unused_comp = bulk_component.BulkFeatureIdExtractorComponentBuilder(
            self.master, component_spec)

  def testBulkFeatureIdExtractorOkWithMultipleFixedFeatures(self):
    component_spec = spec_pb2.ComponentSpec()
    text_format.Parse("""
        name: "test"
        network_unit {
          registered_name: "IdentityNetwork"
        }
        fixed_feature {
          name: "fixed1" embedding_dim: -1 size: 1
        }
        fixed_feature {
          name: "fixed2" embedding_dim: -1 size: 1
        }
        fixed_feature {
          name: "fixed3" embedding_dim: -1 size: 1
        }
        """, component_spec)
    with tf.Graph().as_default():
      comp = bulk_component.BulkFeatureIdExtractorComponentBuilder(
          self.master, component_spec)

      # Should not raise errors.
      self.network_states[component_spec.name] = component.NetworkState()
      comp.build_greedy_training(self.master_state, self.network_states)
      self.network_states[component_spec.name] = component.NetworkState()
      comp.build_greedy_inference(self.master_state, self.network_states)

  def testBulkFeatureIdExtractorFailsOnEmbeddedFixedFeature(self):
    component_spec = spec_pb2.ComponentSpec()
    text_format.Parse("""
        name: "test"
        network_unit {
          registered_name: "IdentityNetwork"
        }
        fixed_feature {
          name: "fixed" embedding_dim: 2 size: 1
        }
        """, component_spec)
    with tf.Graph().as_default():
      with self.assertRaises(ValueError):
        unused_comp = bulk_component.BulkFeatureIdExtractorComponentBuilder(
            self.master, component_spec)

  def testBulkFeatureIdExtractorExtractFocusWithOffset(self):
    path = os.path.join(tf.test.get_temp_dir(), 'label-map')
    with open(path, 'w') as label_map_file:
      label_map_file.write('0\n')

    master_spec = spec_pb2.MasterSpec()
    text_format.Parse("""
        component {
          name: "test"
          transition_system {
            registered_name: "shift-only"
          }
          resource {
            name: "label-map"
            part {
              file_pattern: "%s"
              file_format: "text"
            }
          }
          network_unit {
            registered_name: "ExportFixedFeaturesNetwork"
          }
          backend {
            registered_name: "SyntaxNetComponent"
          }
          fixed_feature {
            name: "focus1" embedding_dim: -1 size: 1 fml: "input.focus"
            predicate_map: "none"
          }
          fixed_feature {
            name: "focus2" embedding_dim: -1 size: 1 fml: "input(1).focus"
            predicate_map: "none"
          }
          fixed_feature {
            name: "focus3" embedding_dim: -1 size: 1 fml: "input(2).focus"
            predicate_map: "none"
          }
        }
        """ % path, master_spec)

    with tf.Graph().as_default():
      corpus = _create_fake_corpus()
      corpus = tf.constant(corpus, shape=[len(corpus)])
      handle = dragnn_ops.get_session(
          container='test',
          master_spec=master_spec.SerializeToString(),
          grid_point='')
      handle = dragnn_ops.attach_data_reader(handle, corpus)
      handle = dragnn_ops.init_component_data(
          handle, beam_size=1, component='test')
      batch_size = dragnn_ops.batch_size(handle, component='test')
      master_state = component.MasterState(handle, batch_size)

      extractor = bulk_component.BulkFeatureIdExtractorComponentBuilder(
          self.master, master_spec.component[0])
      network_state = component.NetworkState()
      self.network_states['test'] = network_state
      handle = extractor.build_greedy_inference(master_state,
                                                self.network_states)
      focus1 = network_state.activations['focus1'].bulk_tensor
      focus2 = network_state.activations['focus2'].bulk_tensor
      focus3 = network_state.activations['focus3'].bulk_tensor

      with self.test_session() as sess:
        focus1, focus2, focus3 = sess.run([focus1, focus2, focus3])
        tf.logging.info('focus1=\n%s', focus1)
        tf.logging.info('focus2=\n%s', focus2)
        tf.logging.info('focus3=\n%s', focus3)

        self.assertAllEqual(
            focus1,
            [[0], [-1], [-1], [-1],
             [0], [1], [-1], [-1],
             [0], [1], [2], [-1],
             [0], [1], [2], [3]])

        self.assertAllEqual(
            focus2,
            [[-1], [-1], [-1], [-1],
             [1], [-1], [-1], [-1],
             [1], [2], [-1], [-1],
             [1], [2], [3], [-1]])

        self.assertAllEqual(
            focus3,
            [[-1], [-1], [-1], [-1],
             [-1], [-1], [-1], [-1],
             [2], [-1], [-1], [-1],
             [2], [3], [-1], [-1]])

  def testBuildLossFailsOnNoExamples(self):
    with tf.Graph().as_default():
      logits = tf.constant([[0.5], [-0.5], [0.5], [-0.5]])
      gold = tf.constant([-1, -1, -1, -1])
      result = bulk_component.build_cross_entropy_loss(logits, gold)

      # Expect loss computation to generate a runtime error due to the gold
      # tensor containing no valid examples.
      with self.test_session() as sess:
        with self.assertRaises(tf.errors.InvalidArgumentError):
          sess.run(result)

if __name__ == '__main__':
  googletest.main()
