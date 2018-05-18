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
"""Tests for the runtime support utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from dragnn.protos import export_pb2
from dragnn.protos import spec_pb2
from dragnn.python import network_units
from dragnn.python import runtime_support


class MockNetwork(object):
  """Mock for tests."""

  def __init__(self):
    self.params = [
        tf.get_variable('rank2', [64, 127], tf.float32),
        tf.get_variable('rank3', [64, 127, 250], tf.float32)
    ]
    self.derived_params = [
        self._fake_derived_vector, self._fake_derived_parameter
    ]

  def _fake_derived_vector(self):
    value = tf.constant([1, 2, 3], dtype=tf.float32)
    with tf.name_scope(None):
      return tf.identity(value, name='derived/vector')

  def _fake_derived_parameter(self):
    # Use absolute scoping to put the derived parameter in the same namespace.
    base_name = self.params[0].op.name.rsplit('/', 1)[0]
    with tf.name_scope(None):
      return tf.concat(
          [self.params[0], self.params[0]],
          axis=0,
          name='{}/derived'.format(base_name))


class MockComponent(object):
  """Mock for tests."""

  def __init__(self):
    self.name = 'test_component'
    self.spec = spec_pb2.ComponentSpec()
    with tf.variable_scope(self.name):
      self.network = MockNetwork()

  def get_variable(self, var_name=None, var_params=None):
    if var_name:
      return tf.get_variable(var_name)
    else:
      return var_params


class RuntimeSupportTest(tf.test.TestCase):
  """Testing rig."""

  def testAddLinkedHooks(self):
    component = MockComponent()
    link0 = component.spec.linked_feature.add()
    link1 = component.spec.linked_feature.add()
    link0.embedding_dim = -1  # direct link
    link1.embedding_dim = 32  # transformed link
    link0_matrix_name = network_units.linked_embeddings_name(0)
    link1_matrix_name = network_units.linked_embeddings_name(1)

    with self.test_session() as session:
      graph = session.graph

      # Create linked embedding matrices.  Only channel 1 uses one.
      with tf.variable_scope(component.name):
        tf.get_variable(link1_matrix_name, shape=[64 + 1, 32], dtype=tf.float32)

      # Add hooks.  This should ignore channel 0 and add hooks for channel 1.
      with tf.variable_scope(component.name, reuse=True):
        runtime_support.add_hooks(component, export_pb2.CellSubgraphSpec())

      # Check that no hooks were added for channel 0.
      with self.assertRaises(KeyError):
        graph.get_tensor_by_name(
            '{}/{}/weights:0'.format(component.name, link0_matrix_name))
      with self.assertRaises(KeyError):
        graph.get_tensor_by_name('{}/{}/weights/transposed:0'.format(
            component.name, link0_matrix_name))
      with self.assertRaises(KeyError):
        graph.get_tensor_by_name('{}/{}/weights/transposed/shape:0'.format(
            component.name, link0_matrix_name))
      with self.assertRaises(KeyError):
        graph.get_tensor_by_name('{}/{}/weights/transposed/blocked32:0'.format(
            component.name, link0_matrix_name))
      with self.assertRaises(KeyError):
        graph.get_tensor_by_name('{}/{}/weights/transposed/blocked48:0'.format(
            component.name, link0_matrix_name))
      with self.assertRaises(KeyError):
        graph.get_tensor_by_name(
            '{}/{}/out_of_bounds:0'.format(component.name, link0_matrix_name))

      # Get the hooks added for channel 1.
      weights = graph.get_tensor_by_name(
          '{}/{}/weights:0'.format(component.name, link1_matrix_name))
      transposed = graph.get_tensor_by_name('{}/{}/weights/transposed:0'.format(
          component.name, link1_matrix_name))
      transposed_shape = graph.get_tensor_by_name(
          '{}/{}/weights/transposed/shape:0'.format(component.name,
                                                    link1_matrix_name))
      transposed32 = graph.get_tensor_by_name(
          '{}/{}/weights/transposed/blocked32:0'.format(component.name,
                                                        link1_matrix_name))
      transposed48 = graph.get_tensor_by_name(
          '{}/{}/weights/transposed/blocked48:0'.format(component.name,
                                                        link1_matrix_name))
      out_of_bounds = graph.get_tensor_by_name(
          '{}/{}/out_of_bounds:0'.format(component.name, link1_matrix_name))

      # Check dimensions of the hooks.
      tf.global_variables_initializer().run()
      self.assertAllEqual(tf.shape(weights).eval(), [64, 32])
      self.assertAllEqual(tf.shape(transposed).eval(), [32, 64])
      self.assertAllEqual(transposed_shape.eval(), [32, 64])
      self.assertAllEqual(tf.shape(transposed32).eval(), [2, 32, 32])
      self.assertAllEqual(tf.shape(transposed48).eval(), [2, 32, 48])
      self.assertAllEqual(tf.shape(out_of_bounds).eval(), [1, 32])

  def testAddFixedHooks(self):
    component = MockComponent()
    fixed0 = component.spec.fixed_feature.add()
    fixed1 = component.spec.fixed_feature.add()
    fixed0.embedding_dim = -1
    fixed1.embedding_dim = 32
    fixed0.vocabulary_size = 100
    fixed1.vocabulary_size = 1000
    fixed0_matrix_name = network_units.fixed_embeddings_name(0)
    fixed1_matrix_name = network_units.fixed_embeddings_name(1)

    with self.test_session() as session:
      graph = session.graph

      # Create fixed embedding matrices.  Only channel 1 uses one.
      with tf.variable_scope(component.name):
        tf.get_variable(
            fixed1_matrix_name, shape=[1000 + 1, 32], dtype=tf.float32)

      # Add hooks.  This should ignore channel 0 and add hooks for channel 1.
      with tf.variable_scope(component.name, reuse=True):
        runtime_support.add_hooks(component, export_pb2.CellSubgraphSpec())

      # Check that no hooks were added for channel 0.
      with self.assertRaises(KeyError):
        graph.get_tensor_by_name(
            '{}/{}/trimmed:0'.format(component.name, fixed0_matrix_name))

      # Get the hooks added for channel 1.
      trimmed = graph.get_tensor_by_name(
          '{}/{}/trimmed:0'.format(component.name, fixed1_matrix_name))

      # Check dimensions of the hooks.
      tf.global_variables_initializer().run()
      self.assertAllEqual(tf.shape(trimmed).eval(), [1000, 32])

  def testAddParamsHooks(self):
    component = MockComponent()
    rank2_name = 'rank2'
    rank3_name = 'rank3'

    with self.test_session() as session:
      graph = session.graph

      # Add hooks.  This should add hooks for all rank-2 params.
      with tf.variable_scope(component.name, reuse=True):
        runtime_support.add_hooks(component, export_pb2.CellSubgraphSpec())

      # Check that no hooks were added for the rank-3 params.
      with self.assertRaises(KeyError):
        graph.get_tensor_by_name(
            '{}/{}/matrix:0'.format(component.name, rank3_name))
      with self.assertRaises(KeyError):
        graph.get_tensor_by_name(
            '{}/{}/transposed:0'.format(component.name, rank3_name))
      with self.assertRaises(KeyError):
        graph.get_tensor_by_name(
            '{}/{}/matrix/blocked32:0'.format(component.name, rank3_name))
      with self.assertRaises(KeyError):
        graph.get_tensor_by_name(
            '{}/{}/matrix/blocked48:0'.format(component.name, rank3_name))
      with self.assertRaises(KeyError):
        graph.get_tensor_by_name(
            '{}/{}/transposed/blocked32:0'.format(component.name, rank3_name))
      with self.assertRaises(KeyError):
        graph.get_tensor_by_name(
            '{}/{}/transposed/blocked48:0'.format(component.name, rank3_name))
      with self.assertRaises(KeyError):
        graph.get_tensor_by_name(
            '{}/{}/matrix/shape:0'.format(component.name, rank3_name))
      with self.assertRaises(KeyError):
        graph.get_tensor_by_name(
            '{}/{}/transposed/shape:0'.format(component.name, rank3_name))

      # Get the hooks added for each variable.
      matrix = graph.get_tensor_by_name(
          '{}/{}/matrix:0'.format(component.name, rank2_name))
      transposed = graph.get_tensor_by_name(
          '{}/{}/transposed:0'.format(component.name, rank2_name))
      matrix32 = graph.get_tensor_by_name(
          '{}/{}/matrix/blocked32:0'.format(component.name, rank2_name))
      matrix48 = graph.get_tensor_by_name(
          '{}/{}/matrix/blocked48:0'.format(component.name, rank2_name))
      transposed32 = graph.get_tensor_by_name(
          '{}/{}/transposed/blocked32:0'.format(component.name, rank2_name))
      transposed48 = graph.get_tensor_by_name(
          '{}/{}/transposed/blocked48:0'.format(component.name, rank2_name))
      matrix_shape = graph.get_tensor_by_name(
          '{}/{}/matrix/shape:0'.format(component.name, rank2_name))
      transposed_shape = graph.get_tensor_by_name(
          '{}/{}/transposed/shape:0'.format(component.name, rank2_name))

      # Check dimensions of the hooks.
      tf.global_variables_initializer().run()
      self.assertAllEqual(tf.shape(matrix).eval(), [64, 127])
      self.assertAllEqual(tf.shape(transposed).eval(), [127, 64])
      self.assertAllEqual(matrix_shape.eval(), [64, 127])
      self.assertAllEqual(transposed_shape.eval(), [127, 64])
      self.assertAllEqual(tf.shape(matrix32).eval(), [4, 64, 32])
      self.assertAllEqual(tf.shape(matrix48).eval(), [3, 64, 48])
      self.assertAllEqual(tf.shape(transposed32).eval(), [2, 127, 32])
      self.assertAllEqual(tf.shape(transposed48).eval(), [2, 127, 48])

  def testAddDerivedParamHooks(self):
    component = MockComponent()
    derived_name = 'derived'

    with self.test_session() as session:
      graph = session.graph

      # Add hooks.
      with tf.variable_scope(component.name, reuse=True):
        runtime_support.add_hooks(component, export_pb2.CellSubgraphSpec())

      session.run(tf.global_variables_initializer())

      # Get hooks for the derived vector.
      vector = graph.get_tensor_by_name('derived/vector:0')
      self.assertEqual(vector.shape, (3,))

      # Get the hooks for the derived variable.
      matrix = graph.get_tensor_by_name(
          '{}/{}/matrix/blocked32:0'.format(component.name, derived_name))
      self.assertAllEqual(tf.shape(matrix).eval(), [4, 128, 32])

      # Check the bfloat16 version. It should have the same shape.
      bfloat16_matrix = graph.get_tensor_by_name(
          '{}/{}/matrix/blocked32/bfloat16:0'.format(component.name,
                                                     derived_name))
      self.assertAllEqual(tf.shape(bfloat16_matrix).eval(), [4, 128, 32])

  def testMakePaddedBlockedMatrix(self):
    with self.test_session():
      matrix = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20]]
      expected_blocked = [[[1, 2], [6, 7], [11, 12],
                           [16, 17]], [[3, 4], [8, 9], [13, 14], [18, 19]],
                          [[5, 0], [10, 0], [15, 0], [20, 0]]]

      matrix = tf.constant(matrix, tf.float32)
      actual_blocked = runtime_support.make_padded_blocked_matrix(matrix, 2)
      self.assertAllEqual(actual_blocked.eval(), expected_blocked)

  def testBfloat16Permutation(self):
    with self.test_session():
      matrix = [list(range(16))]
      expected_permuted = [[
          0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15
      ]]
      matrix = tf.constant(matrix, tf.float32)
      actual_permuted = runtime_support.bfloat16_permutation(matrix)
      self.assertAllEqual(actual_permuted.eval(), expected_permuted)

  def testLargerBfloat16Permutation(self):
    with self.test_session() as session:
      matrix = tf.random_uniform((3, 4, 32))
      permuted = runtime_support.bfloat16_permutation(matrix)
      matrix, actual_permuted = session.run([matrix, permuted])

      # Just check a few items for now, hopefully that's sufficient to ensure
      # the permutation is okay.
      self.assertEqual(matrix[0, 0, 0], actual_permuted[0, 0, 0])
      self.assertEqual(matrix[0, 0, 1], actual_permuted[0, 0, 1])
      self.assertEqual(matrix[1, 1, 16], actual_permuted[1, 1, 16])
      self.assertEqual(matrix[2, 0, 4], actual_permuted[2, 0, 8])
      self.assertEqual(matrix[2, 0, 5], actual_permuted[2, 0, 9])
      self.assertEqual(matrix[2, 1, 8], actual_permuted[2, 1, 4])
      self.assertEqual(matrix[2, 1, 8 + 16], actual_permuted[2, 1, 4 + 16])

  def testAddCellSubgraphSpecHook(self):
    component = MockComponent()
    cell = export_pb2.CellSubgraphSpec()
    cell.input.add(
        name='feature',
        tensor='feature_tensor',
        type=export_pb2.CellSubgraphSpec.Input.TYPE_FEATURE)
    cell.input.add(
        name='recurrent',
        tensor='recurrent_tensor',
        type=export_pb2.CellSubgraphSpec.Input.TYPE_RECURRENT)
    cell.output.add(name='layer_0', tensor='layer_0_tensor')
    cell.output.add(name='logits', tensor='logits_tensor')

    with self.test_session() as session:
      graph = session.graph

      # Add hooks for the cell constructed above.
      with tf.variable_scope(component.name, reuse=True):
        runtime_support.add_hooks(component, cell)

      # Get the hook containing the wire-format proto.
      cell_wire_format = graph.get_tensor_by_name(
          '{}/EXPORT/CellSubgraphSpec:0'.format(component.name))

      # Check that the hook matches the cell.
      tf.global_variables_initializer().run()
      self.assertEqual(cell_wire_format.eval(), cell.SerializeToString())


if __name__ == '__main__':
  tf.test.main()
