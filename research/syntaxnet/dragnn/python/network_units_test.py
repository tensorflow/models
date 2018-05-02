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
"""Tests for network_units."""


import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

from dragnn.protos import spec_pb2
from dragnn.python import network_units


class NetworkUnitsConverterTest(test_util.TensorFlowTestCase):

  def testConvertNetworkStateTensorarray(self):
    with self.test_session() as session:
      ta = tf.TensorArray(
          dtype=tf.float32,
          size=0,
          dynamic_size=True,
          clear_after_read=False,
          infer_shape=False)
      # Create a 3-step x 2-stride x 2-feature-dim source array.
      ta = ta.write(0, [[0., 0.]] * 2)  # The zeroth step will be removed.
      ta = ta.write(1, [[1., 10.]] * 2)
      ta = ta.write(2, [[2., 20.]] * 2)
      ta = ta.write(3, [[3., 30.]] * 2)
      tensor = network_units.convert_network_state_tensorarray(ta)
      actual = session.run(tensor)
      self.assertEqual(actual.shape, (6, 2))

      # The arrangement of the values is expected to be stride * steps.
      expected = [[1., 10.], [2., 20.], [3., 30.], [1., 10.], [2., 20.],
                  [3., 30.]]
      self.assertAllEqual(actual, expected)


class MockComponent(object):

  def __init__(self, master, component_spec):
    self.master = master
    self.spec = component_spec
    self.name = component_spec.name
    self.beam_size = 1
    self.num_actions = 45
    self._attrs = {}

  def attr(self, name):
    return self._attrs[name]

  def get_variable(self, name):
    return tf.get_variable(name)


class MockMaster(object):

  def __init__(self, build_runtime_graph=False):
    self.spec = spec_pb2.MasterSpec()
    self.hyperparams = spec_pb2.GridPoint()
    self.lookup_component = {
        'previous': MockComponent(self, spec_pb2.ComponentSpec())
    }
    self.build_runtime_graph = build_runtime_graph


class MockNetwork(object):

  def __init__(self, **dims):
    self._dims = dims

  def get_layer_size(self, name):
    return self._dims[name]


class NetworkUnitsLookupTest(test_util.TensorFlowTestCase):

  def setUp(self):
    # Clear the graph and all existing variables.  Otherwise, variables created
    # in different tests may collide with each other.
    tf.reset_default_graph()

    self._master = MockMaster()
    self._master.spec = spec_pb2.MasterSpec()

    # Add a component with a linked feature.
    component_spec = self._master.spec.component.add()
    component_spec.name = 'fake_linked'
    component_spec.backend.registered_name = 'FakeComponent'
    linked_feature = component_spec.linked_feature.add()
    linked_feature.source_component = 'fake_linked'
    linked_feature.source_translator = 'identity'
    linked_feature.embedding_dim = -1
    linked_feature.size = 2
    self._linked_component = MockComponent(self._master, component_spec)

    # Add a feature with a fixed feature.
    component_spec = self._master.spec.component.add()
    component_spec.name = 'fake_fixed'
    component_spec.backend.registered_name = 'FakeComponent'
    fixed_feature = component_spec.fixed_feature.add()
    fixed_feature.fml = 'input.word'
    fixed_feature.embedding_dim = 1
    fixed_feature.size = 1
    self._fixed_component = MockComponent(self._master, component_spec)

  def testExportFixedFeaturesNetworkWithEnabledEmbeddingMatrix(self):
    network = network_units.ExportFixedFeaturesNetwork(self._fixed_component)
    self.assertEqual(1, len(network.params))

  def testExportFixedFeaturesNetworkWithDisabledEmbeddingMatrix(self):
    self._fixed_component.spec.fixed_feature[0].embedding_dim = -1
    network = network_units.ExportFixedFeaturesNetwork(self._fixed_component)
    self.assertEqual(0, len(network.params))


class GetAttrsWithDefaultsTest(test_util.TensorFlowTestCase):

  def MakeAttrs(self, defaults, key=None, value=None):
    """Returns attrs based on the |defaults| and one |key|,|value| override."""
    spec = spec_pb2.RegisteredModuleSpec()
    if key and value:
      spec.parameters[key] = value
    return network_units.get_attrs_with_defaults(spec.parameters, defaults)

  def testFalseValues(self):

    def _assert_attr_is_false(value=None):
      key = 'foo'
      attrs = self.MakeAttrs({key: False}, key, value)
      self.assertFalse(attrs[key])

    _assert_attr_is_false()
    _assert_attr_is_false('false')
    _assert_attr_is_false('False')
    _assert_attr_is_false('FALSE')
    _assert_attr_is_false('no')
    _assert_attr_is_false('whatever')
    _assert_attr_is_false('   ')
    _assert_attr_is_false('')

  def testTrueValues(self):

    def _assert_attr_is_true(value=None):
      key = 'foo'
      attrs = self.MakeAttrs({key: False}, key, value)
      self.assertTrue(attrs[key])

    _assert_attr_is_true('true')
    _assert_attr_is_true('True')
    _assert_attr_is_true('TRUE')


class LstmNetworkTest(test_util.TensorFlowTestCase):
  test_spec_1 = """
      component {
        name: 'bi_lstm'
        backend { registered_name: 'TestComponent' }
        fixed_feature {
          name: 'words'
          fml: 'words'
          size: 1
          embedding_dim: 32
          vocabulary_size: 1079813,
        }
        network_unit {
          registered_name: 'LSTMNetwork'
          parameters {
            key: "hidden_layer_sizes"
            value: "128"
          }
        }
      }
    """

  test_spec_linked = """
      component {
        name: 'bi_lstm'
        backend { registered_name: 'TestComponent' }
        fixed_feature {
          name: 'words'
          fml: 'words'
          size: 1
          embedding_dim: 32
          vocabulary_size: 1079813,
        }
        linked_feature {
          name: 'lstm_h'
          fml: 'bias(0)'
          embedding_dim: -1
          size: 1
          source_component: 'bi_lstm'
          source_translator: 'history'
          source_layer: 'lstm_h'
        }
        linked_feature {
          name: 'lstm_c'
          fml: 'bias(0)'
          embedding_dim: -1
          size: 1
          source_component: 'bi_lstm'
          source_translator: 'history'
          source_layer: 'lstm_c'
        }
        network_unit {
          registered_name: 'LSTMNetwork'
          parameters {
            key: "hidden_layer_sizes"
            value: "128"
          }
        }
      }
    """

  def setUp(self):
    # Clear the graph and all existing variables.  Otherwise, variables created
    # in different tests may collide with each other.
    tf.reset_default_graph()

  def construct_lstm_network_unit(self, master):
    """Helper to construct a LSTMNetwork. Doesn't call create() yet."""
    component = MockComponent(master, master.spec.component[0])
    with tf.variable_scope('bi_lstm'):
      lstm_network_unit = network_units.LSTMNetwork(component)
    return lstm_network_unit

  def get_context_tensor_arrays(self, lstm_network_unit):
    context_tensor_arrays = []
    for context_layer in lstm_network_unit.context_layers:
      context_tensor_arrays.append(context_layer.create_array(1))
    return context_tensor_arrays

  def fixed_word_embeddings(self):
    """Helper for returning fixed embeddings, for 1 word feature."""
    words_tensor = tf.constant([[1.0] * 32], dtype=tf.float32)
    return [network_units.NamedTensor(words_tensor, 'words')]

  def testCanCreate(self):
    """Smoke test that the create() function doesn't raise errors."""
    master = MockMaster()
    master.spec = spec_pb2.MasterSpec()
    text_format.Parse(self.test_spec_1, master.spec)
    lstm_network_unit = self.construct_lstm_network_unit(master)
    with tf.variable_scope('bi_lstm', reuse=True):
      lstm_network_unit.create(
          self.fixed_word_embeddings(), [],
          self.get_context_tensor_arrays(lstm_network_unit), None, True)

  def testCanCreateLinked(self):
    """Smoke test that the create() function doesn't raise errors."""
    master = MockMaster()
    master.spec = spec_pb2.MasterSpec()
    text_format.Parse(self.test_spec_linked, master.spec)
    lstm_network_unit = self.construct_lstm_network_unit(master)
    with tf.variable_scope('bi_lstm', reuse=True):
      lstm_network_unit.create(
          self.fixed_word_embeddings(), [],
          self.get_context_tensor_arrays(lstm_network_unit), None, True)

  def testRuntimeConcatentatedMatrices(self):
    """Test generation of concatenated matrices."""
    # TODO(googleuser): Make MockComponent support runtime graph generation.
    master = MockMaster(build_runtime_graph=False)
    master.spec = spec_pb2.MasterSpec()
    text_format.Parse(self.test_spec_1, master.spec)
    lstm_network_unit = self.construct_lstm_network_unit(master)
    with tf.variable_scope('bi_lstm', reuse=True):
      lstm_network_unit.create(
          self.fixed_word_embeddings(), [],
          self.get_context_tensor_arrays(lstm_network_unit), None, False)
      x_to_ico = lstm_network_unit.derived_params[0]()
      h_to_ico = lstm_network_unit.derived_params[1]()
      ico_bias = lstm_network_unit.derived_params[2]()

      # Should be the word dimension (32) to 3x the hidden dimension (128).
      self.assertEqual(x_to_ico.shape, (32, 384))
      self.assertEqual(x_to_ico.op.name, 'bi_lstm/x_to_ico')

      # Should be the hidden dimension (128) to 3x the hidden dimension (128).
      self.assertEqual(h_to_ico.shape, (128, 384))
      self.assertEqual(h_to_ico.op.name, 'bi_lstm/h_to_ico')

      # Should be equal to the hidden dimension (128) times 3.
      self.assertEqual(ico_bias.shape, (384,))
      self.assertEqual(ico_bias.op.name, 'bi_lstm/ico_bias')

  def testRuntimeConcatentatedMatricesLinked(self):
    """Test generation of concatenated matrices."""
    # TODO(googleuser): Make MockComponent support runtime graph generation.
    master = MockMaster(build_runtime_graph=False)
    master.spec = spec_pb2.MasterSpec()
    text_format.Parse(self.test_spec_linked, master.spec)
    lstm_network_unit = self.construct_lstm_network_unit(master)
    with tf.variable_scope('bi_lstm', reuse=True):
      lstm_network_unit.create(
          self.fixed_word_embeddings(), [],
          self.get_context_tensor_arrays(lstm_network_unit), None, False)
      x_to_ico = lstm_network_unit.derived_params[0]()
      h_to_ico = lstm_network_unit.derived_params[1]()
      ico_bias = lstm_network_unit.derived_params[2]()

      # Should be the word dimension (32) to 3x the hidden dimension (128).
      self.assertEqual(x_to_ico.shape, (32, 384))

      # Should be the hidden dimension (128) to 3x the hidden dimension (128).
      self.assertEqual(h_to_ico.shape, (128, 384))

      # Should be equal to the hidden dimension (128) times 3.
      self.assertEqual(ico_bias.shape, (384,))


class GatherNetworkTest(test_util.TensorFlowTestCase):

  def setUp(self):
    # Clear the graph and all existing variables.  Otherwise, variables created
    # in different tests may collide with each other.
    tf.reset_default_graph()

    self._master = MockMaster()
    self._master.spec = spec_pb2.MasterSpec()
    text_format.Parse("""
      component {
        name: 'test'
        backend { registered_name: 'TestComponent' }
        linked_feature {
          name: 'indices'
          fml: 'input.focus'
          size: 1
          embedding_dim: -1
          source_component: 'previous'
          source_translator: 'identity'
          source_layer: 'index_layer'
        }
        linked_feature {
          name: 'features'
          fml: 'input.focus'
          size: 1
          embedding_dim: -1
          source_component: 'previous'
          source_translator: 'identity'
          source_layer: 'feature_layer'
        }
        network_unit {
          registered_name: 'GatherNetwork'
        }
      }
    """, self._master.spec)
    self._component = MockComponent(self._master,
                                    self._master.spec.component[0])
    self._master.lookup_component['previous'].network = MockNetwork(
        index_layer=1, feature_layer=2)

  def testConstantPadding(self):
    with tf.Graph().as_default(), self.test_session():
      with tf.variable_scope('test_scope'):
        network = network_units.GatherNetwork(self._component)

      # Construct a batch of two items with 3 and 2 steps, respectively.
      indices = tf.constant(
          [
              # item 1
              [1],
              [2],
              [0],
              # item 2
              [-1],
              [0],
              [-1]
          ],
          dtype=tf.int64)
      features = tf.constant(
          [
              # item 1
              [1.0, 1.5],
              [2.0, 2.5],
              [3.0, 3.5],
              # item 2
              [4.0, 4.5],
              [5.0, 5.5],
              [6.0, 6.5]
          ],
          dtype=tf.float32)

      fixed_embeddings = []
      linked_embeddings = [
          network_units.NamedTensor(indices, 'indices', 1),
          network_units.NamedTensor(features, 'features', 2)
      ]

      with tf.variable_scope('test_scope', reuse=True):
        outputs = network.create(fixed_embeddings, linked_embeddings, None,
                                 None, True, 2)
      gathered = outputs[0]

      # Zeros will be substituted for index -1.
      self.assertAllEqual(
          gathered.eval(),
          [
              [2.0, 2.5],  # gathered from 1
              [3.0, 3.5],  # gathered from 2
              [1.0, 1.5],  # gathered from 0
              [0.0, 0.0],  # gathered from -1
              [4.0, 4.5],  # gathered from 0
              [0.0, 0.0]  # gathered from -1
          ])

  def testTrainablePadding(self):
    self._component.spec.network_unit.parameters['trainable_padding'] = 'true'
    with tf.Graph().as_default(), self.test_session():
      with tf.variable_scope('test_scope'):
        network = network_units.GatherNetwork(self._component)

      # Construct a batch of two items with 3 and 2 steps, respectively.
      indices = tf.constant(
          [
              # item 1
              [1],
              [2],
              [0],
              # item 2
              [-1],
              [0],
              [-1]
          ],
          dtype=tf.int64)
      features = tf.constant(
          [
              # item 1
              [1.0, 1.5],
              [2.0, 2.5],
              [3.0, 3.5],
              # item 2
              [4.0, 4.5],
              [5.0, 5.5],
              [6.0, 6.5]
          ],
          dtype=tf.float32)

      fixed_embeddings = []
      linked_embeddings = [
          network_units.NamedTensor(indices, 'indices', 1),
          network_units.NamedTensor(features, 'features', 2)
      ]

      with tf.variable_scope('test_scope', reuse=True):
        outputs = network.create(fixed_embeddings, linked_embeddings, None,
                                 None, True, 2)
      gathered = outputs[0]

      # Ensure that the padding variable is initialized.
      tf.global_variables_initializer().run()

      # Randomly-initialized padding will be substituted for index -1.
      self.assertAllEqual(gathered[0].eval(), [2.0, 2.5])  # gathered from 1
      self.assertAllEqual(gathered[1].eval(), [3.0, 3.5])  # gathered from 2
      self.assertAllEqual(gathered[2].eval(), [1.0, 1.5])  # gathered from 0
      tf.logging.info('padding = %s', gathered[3].eval())  # gathered from -1
      self.assertAllEqual(gathered[4].eval(), [4.0, 4.5])  # gathered from 0
      tf.logging.info('padding = %s', gathered[5].eval())  # gathered from -1

      # Though random, the padding must identical.
      self.assertAllEqual(gathered[3].eval(), gathered[5].eval())


class IdentityInitializerTest(test_util.TensorFlowTestCase):

  def IdentityInitializerHelper(self, shape, expected, divisor=1.0, std=1e-4):
    """Tests identity initialization by comparing expected to actual array.

    Tests the given expected array against the result of calling
    network_units.add_var_initialized() with the given params and
    init_type='identity'.

    Args:
      shape: shape of the array
      expected: expected contents of the array to initialize
      divisor: numerator for identity initialization where the last two dims
        of the array are not equal; should divide both of the last two dims
      std: standard deviation for random normal samples
    """
    with tf.Graph().as_default(), self.test_session() as session:
      np.random.seed(4)
      tensor = network_units.add_var_initialized(
          'tensor', shape, 'identity', divisor=divisor, stddev=std)
      session.run(tf.global_variables_initializer())
      actual = session.run(tensor)
      self.assertAllClose(actual, expected, 1e-8, 1e-8)

  def IdentityInitializerSquareHelper(self, shape, middles):
    """Tests identity initialization when last two dims are equal.

    When the last two dims of the array are equal, identity initialization
    should simply set the center matrix in the last two dimensions to the
    identity, with all other entries set to zero.

    Args:
      shape: shape of the array to initialize
      middles: indices into the middle of all axes except the last two. It
          must be the case that len(middles) == len(shape) - 2.
    """
    expected = np.zeros(shape, dtype='float32')
    expected[[[m] for m in middles]] = np.eye(shape[-1])
    self.IdentityInitializerHelper(shape, expected)

  def testIdentityInitializerSquareRank2(self):
    shape = (3, 3)
    expected = np.eye(shape[-1]).astype('float32')
    self.IdentityInitializerHelper(shape, expected)

  def testIdentityInitializerSquareRank3(self):
    shape = (2, 4, 4)
    middles = [1]
    self.IdentityInitializerSquareHelper(shape, middles)

  def testIdentityInitializerSquareRank4(self):
    shape = (2, 3, 4, 4)
    middles = [1, 1]
    self.IdentityInitializerSquareHelper(shape, middles)

  def testIdentityInitializerSquareRank5(self):
    shape = (2, 3, 4, 5, 5)
    middles = [1, 1, 2]
    self.IdentityInitializerSquareHelper(shape, middles)

  def testIdentityInitializerNonSquareRank2FirstDimLarger(self):
    divisor = 3.
    std = 1e-3
    shape = (6, 3)
    m = divisor / shape[-1]
    expected = [[m, 4.99951362e-04,
                 -9.95908980e-04], [m, -4.18301526e-04, -1.58457726e-03],
                [-6.47706795e-04, m,
                 3.32250027e-04], [-1.14747661e-03, m, -8.79869258e-05],
                [4.25072387e-04, 3.32253141e-04,
                 m], [3.50997143e-04, -6.06887275e-04, m]]
    self.IdentityInitializerHelper(shape, expected, divisor, std)

  def testIdentityInitializerNonSquareRank2FirstDimSmaller(self):
    divisor = 2.
    std = 1e-3
    shape = (2, 4)
    m = divisor / shape[-1]
    expected = [[m, m, -9.95908980e-04, 6.93598529e-04],
                [-4.18301526e-04, -1.58457726e-03, m, m]]
    self.IdentityInitializerHelper(shape, expected, divisor, std)

  def testIdentityInitializerNonSquareRank3(self):
    divisor = 2.
    std = 1e-3
    shape = (2, 2, 6)
    m = divisor / shape[-1]
    expected = [[[
        5.05617063e-05, 4.99951362e-04, -9.95908980e-04, 6.93598529e-04,
        -4.18301526e-04, -1.58457726e-03
    ], [
        -6.47706795e-04, 5.98575163e-04, 3.32250027e-04, -1.14747661e-03,
        6.18669670e-04, -8.79869258e-05
    ]], [[m, m, m, 3.50997143e-04, -6.06887275e-04, 1.54697930e-03],
         [7.23341596e-04, 4.61355667e-05, -9.82991653e-04, m, m, m]]]
    self.IdentityInitializerHelper(shape, expected, divisor, std)

  def testIdentityInitializerNonSquareRank4(self):
    divisor = 2.
    std = 1e-3
    shape = (2, 3, 2, 8)
    m = divisor / float(shape[-1])
    expected = [[[[
        5.05617063e-05, 4.99951362e-04, -9.95908980e-04, 6.93598529e-04,
        -4.18301526e-04, -1.58457726e-03, -6.47706795e-04, 5.98575163e-04
    ], [
        3.32250027e-04, -1.14747661e-03, 6.18669670e-04, -8.79869258e-05,
        4.25072387e-04, 3.32253141e-04, -1.15681626e-03, 3.50997143e-04
    ]], [[
        -6.06887275e-04, 1.54697930e-03, 7.23341596e-04, 4.61355667e-05,
        -9.82991653e-04, 5.44327377e-05, 1.59892938e-04, -1.20894820e-03
    ], [
        2.22336012e-03, 3.94295203e-04, 1.69235771e-03, -1.11281220e-03,
        1.63574750e-03, -1.36096554e-03, -6.51225855e-04, 5.42451337e-04
    ]], [[
        4.80062481e-05, -2.35807360e-03, -1.10558409e-03, 8.37836356e-04,
        2.08787085e-03, 9.14840959e-04, -2.76203355e-04, 7.96511886e-04
    ], [
        -1.14379858e-03, 5.09919773e-04, -1.34746032e-03, -9.36010019e-06,
        -1.30704633e-04, 8.02086608e-04, -3.02963977e-04, 1.20200263e-03
    ]]], [[[
        -1.96745284e-04, 8.36528721e-04, 7.86602264e-04, -1.84087583e-03,
        3.75474883e-05, 3.59280530e-05, -7.78739923e-04, 1.79410708e-04
    ], [
        -1.45553437e-03, 5.56185201e-04, 5.09778853e-04, 3.00445536e-04,
        2.47658417e-03, 3.52343399e-04, 6.74710027e-05, -7.32264714e-04
    ]], [[
        m, m, m, m, 1.58469542e-04, 1.99008291e-03, 1.16418756e-03,
        2.42660157e-04
    ], [
        1.37992005e-03, -5.45587063e-05, 7.95233937e-04, 1.90899627e-05, m, m,
        m, m
    ]], [[
        -1.09712186e-03, -5.28196048e-04, -2.37977528e-03, -6.07683673e-04,
        -1.07529014e-03, 2.02240516e-03, -5.64875314e-04, -1.54292909e-03
    ], [
        8.70841788e-04, -1.75210531e-04, 4.86030076e-05, 1.88646198e-04,
        2.09313483e-04, -3.74444906e-04, 9.54698597e-04, 5.23247640e-04
    ]]]]

    self.IdentityInitializerHelper(shape, expected, divisor, std)


class FeatureIdDropoutTest(test_util.TensorFlowTestCase):

  def setUp(self):
    # Clear the graph and all existing variables.  Otherwise, variables created
    # in different tests may collide with each other.
    tf.reset_default_graph()

  def testApplyFeatureIdDropout(self):
    channel = spec_pb2.FixedFeatureChannel()
    text_format.Parse("""
      vocabulary_size: 10
      dropout_id: 8
      dropout_keep_probability: [0.0, 0.25, 0.5, 0.75, 1.0]
    """, channel)

    with tf.Graph().as_default(), self.test_session():
      with tf.variable_scope('test_scope'):
        ids = tf.constant([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=tf.int64)
        weights = tf.constant([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=tf.float32)
        tensors = network_units.apply_feature_id_dropout(ids, weights, channel)
        perturbed_ids = tensors[0].eval()
        tf.logging.info('perturbed_ids = %s', perturbed_ids)

        # Given the dropout_keep_probability values specified above:
        #   * ID 0 is never kept.
        #   * IDs 1-3 are randomly kept with varying probability.
        #   * IDs 4-9 are always kept.
        # To avoid non-determinism, we only check for specific feature IDs at
        # the extremes (never/always kept).  Behavior in between the extremes
        # should interpolate between the two extremes.
        self.assertEqual(perturbed_ids[0], channel.dropout_id)
        self.assertTrue(perturbed_ids[1] in (1, channel.dropout_id))
        self.assertTrue(perturbed_ids[2] in (2, channel.dropout_id))
        self.assertTrue(perturbed_ids[3] in (3, channel.dropout_id))
        self.assertAllEqual(perturbed_ids[4:], [4, 5, 6, 7, 8, 9])

  def testApplyFeatureIdDropoutSkip(self):
    channel = spec_pb2.FixedFeatureChannel()
    text_format.Parse("""
      vocabulary_size: 2
      dropout_id: 2
      dropout_keep_probability: [0.0, 1.0]
    """, channel)

    with tf.Graph().as_default(), self.test_session():
      with tf.variable_scope('test_scope'):
        ids = tf.constant([0, 1], dtype=tf.int64)
        weights = tf.constant([1, 1], dtype=tf.float32)
        tensors = network_units.apply_feature_id_dropout(ids, weights, channel)
        perturbed_ids, perturbed_weights = tensors[0].eval(), tensors[1].eval()
        tf.logging.info('perturbed_ids = %s', perturbed_ids)
        tf.logging.info('perturbed_weights = %s', perturbed_weights)

        # Given the dropout_keep_probability values specified above:
        #   * ID 0 is never kept, its weight is set to 0.
        #   * IDs 1 are always kept.
        # To avoid non-determinism, we only check for specific feature IDs at
        # the extremes (never/always kept).
        self.assertEqual(perturbed_ids[0], channel.dropout_id)
        self.assertEqual(perturbed_weights[0], 0)
        self.assertEqual(perturbed_ids[1], 1)
        self.assertEqual(perturbed_weights[1], 1)


if __name__ == '__main__':
  googletest.main()
