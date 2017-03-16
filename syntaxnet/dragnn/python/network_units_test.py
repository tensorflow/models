"""Tests for network_units."""


import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

from dragnn.protos import spec_pb2
from dragnn.python import network_units

import dragnn.python.load_dragnn_cc_impl
import syntaxnet.load_parser_ops

FLAGS = tf.app.flags.FLAGS


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
    self._attrs = {}

  def attr(self, name):
    return self._attrs[name]


class MockMaster(object):

  def __init__(self):
    self.spec = spec_pb2.MasterSpec()
    self.hyperparams = spec_pb2.GridPoint()
    self.lookup_component = {
        'previous': MockComponent(self, spec_pb2.ComponentSpec())
    }


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


if __name__ == '__main__':
  googletest.main()
