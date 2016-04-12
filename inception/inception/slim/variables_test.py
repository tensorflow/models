# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for slim.variables."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception.slim import scopes
from inception.slim import variables


class VariablesTest(tf.test.TestCase):

  def testCreateVariable(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = variables.variable('a', [5])
        self.assertEquals(a.op.name, 'A/a')
        self.assertListEqual(a.get_shape().as_list(), [5])

  def testGetVariables(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = variables.variable('a', [5])
      with tf.variable_scope('B'):
        b = variables.variable('a', [5])
      self.assertEquals([a, b], variables.get_variables())
      self.assertEquals([a], variables.get_variables('A'))
      self.assertEquals([b], variables.get_variables('B'))

  def testGetVariablesSuffix(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = variables.variable('a', [5])
      with tf.variable_scope('A'):
        b = variables.variable('b', [5])
      self.assertEquals([a], variables.get_variables(suffix='a'))
      self.assertEquals([b], variables.get_variables(suffix='b'))

  def testGetVariableWithSingleVar(self):
    with self.test_session():
      with tf.variable_scope('parent'):
        a = variables.variable('child', [5])
      self.assertEquals(a, variables.get_unique_variable('parent/child'))

  def testGetVariableWithDistractors(self):
    with self.test_session():
      with tf.variable_scope('parent'):
        a = variables.variable('child', [5])
        with tf.variable_scope('child'):
          variables.variable('grandchild1', [7])
          variables.variable('grandchild2', [9])
      self.assertEquals(a, variables.get_unique_variable('parent/child'))

  def testGetVariableThrowsExceptionWithNoMatch(self):
    var_name = 'cant_find_me'
    with self.test_session():
      with self.assertRaises(ValueError):
        variables.get_unique_variable(var_name)

  def testGetThrowsExceptionWithChildrenButNoMatch(self):
    var_name = 'parent/child'
    with self.test_session():
      with tf.variable_scope(var_name):
        variables.variable('grandchild1', [7])
        variables.variable('grandchild2', [9])
      with self.assertRaises(ValueError):
        variables.get_unique_variable(var_name)

  def testGetVariablesToRestore(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = variables.variable('a', [5])
      with tf.variable_scope('B'):
        b = variables.variable('a', [5])
      self.assertEquals([a, b], variables.get_variables_to_restore())

  def testNoneGetVariablesToRestore(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = variables.variable('a', [5], restore=False)
      with tf.variable_scope('B'):
        b = variables.variable('a', [5], restore=False)
      self.assertEquals([], variables.get_variables_to_restore())
      self.assertEquals([a, b], variables.get_variables())

  def testGetMixedVariablesToRestore(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = variables.variable('a', [5])
        b = variables.variable('b', [5], restore=False)
      with tf.variable_scope('B'):
        c = variables.variable('c', [5])
        d = variables.variable('d', [5], restore=False)
      self.assertEquals([a, b, c, d], variables.get_variables())
      self.assertEquals([a, c], variables.get_variables_to_restore())

  def testReuseVariable(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = variables.variable('a', [])
      with tf.variable_scope('A', reuse=True):
        b = variables.variable('a', [])
      self.assertEquals(a, b)
      self.assertListEqual([a], variables.get_variables())

  def testVariableWithDevice(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = variables.variable('a', [], device='cpu:0')
        b = variables.variable('b', [], device='cpu:1')
      self.assertDeviceEqual(a.device, 'cpu:0')
      self.assertDeviceEqual(b.device, 'cpu:1')

  def testVariableWithDeviceFromScope(self):
    with self.test_session():
      with tf.device('/cpu:0'):
        a = variables.variable('a', [])
        b = variables.variable('b', [], device='cpu:1')
      self.assertDeviceEqual(a.device, 'cpu:0')
      self.assertDeviceEqual(b.device, 'cpu:1')

  def testVariableWithDeviceFunction(self):
    class DevFn(object):

      def __init__(self):
        self.counter = -1

      def __call__(self, op):
        self.counter += 1
        return 'cpu:%d' % self.counter

    with self.test_session():
      with scopes.arg_scope([variables.variable], device=DevFn()):
        a = variables.variable('a', [])
        b = variables.variable('b', [])
        c = variables.variable('c', [], device='cpu:12')
        d = variables.variable('d', [])
        with tf.device('cpu:99'):
          e_init = tf.constant(12)
        e = variables.variable('e', initializer=e_init)
      self.assertDeviceEqual(a.device, 'cpu:0')
      self.assertDeviceEqual(a.initial_value.device, 'cpu:0')
      self.assertDeviceEqual(b.device, 'cpu:1')
      self.assertDeviceEqual(b.initial_value.device, 'cpu:1')
      self.assertDeviceEqual(c.device, 'cpu:12')
      self.assertDeviceEqual(c.initial_value.device, 'cpu:12')
      self.assertDeviceEqual(d.device, 'cpu:2')
      self.assertDeviceEqual(d.initial_value.device, 'cpu:2')
      self.assertDeviceEqual(e.device, 'cpu:3')
      self.assertDeviceEqual(e.initial_value.device, 'cpu:99')

  def testVariableWithReplicaDeviceSetter(self):
    with self.test_session():
      with tf.device(tf.train.replica_device_setter(ps_tasks=2)):
        a = variables.variable('a', [])
        b = variables.variable('b', [])
        c = variables.variable('c', [], device='cpu:12')
        d = variables.variable('d', [])
        with tf.device('cpu:99'):
          e_init = tf.constant(12)
        e = variables.variable('e', initializer=e_init)
      # The values below highlight how the replica_device_setter puts initial
      # values on the worker job, and how it merges explicit devices.
      self.assertDeviceEqual(a.device, '/job:ps/task:0/cpu:0')
      self.assertDeviceEqual(a.initial_value.device, '/job:worker/cpu:0')
      self.assertDeviceEqual(b.device, '/job:ps/task:1/cpu:0')
      self.assertDeviceEqual(b.initial_value.device, '/job:worker/cpu:0')
      self.assertDeviceEqual(c.device, '/job:ps/task:0/cpu:12')
      self.assertDeviceEqual(c.initial_value.device, '/job:worker/cpu:12')
      self.assertDeviceEqual(d.device, '/job:ps/task:1/cpu:0')
      self.assertDeviceEqual(d.initial_value.device, '/job:worker/cpu:0')
      self.assertDeviceEqual(e.device, '/job:ps/task:0/cpu:0')
      self.assertDeviceEqual(e.initial_value.device, '/job:worker/cpu:99')

  def testVariableWithVariableDeviceChooser(self):

    with tf.Graph().as_default():
      device_fn = variables.VariableDeviceChooser(num_parameter_servers=2)
      with scopes.arg_scope([variables.variable], device=device_fn):
        a = variables.variable('a', [])
        b = variables.variable('b', [])
        c = variables.variable('c', [], device='cpu:12')
        d = variables.variable('d', [])
        with tf.device('cpu:99'):
          e_init = tf.constant(12)
        e = variables.variable('e', initializer=e_init)
      # The values below highlight how the VariableDeviceChooser puts initial
      # values on the same device as the variable job.
      self.assertDeviceEqual(a.device, '/job:ps/task:0/cpu:0')
      self.assertDeviceEqual(a.initial_value.device, a.device)
      self.assertDeviceEqual(b.device, '/job:ps/task:1/cpu:0')
      self.assertDeviceEqual(b.initial_value.device, b.device)
      self.assertDeviceEqual(c.device, '/cpu:12')
      self.assertDeviceEqual(c.initial_value.device, c.device)
      self.assertDeviceEqual(d.device, '/job:ps/task:0/cpu:0')
      self.assertDeviceEqual(d.initial_value.device, d.device)
      self.assertDeviceEqual(e.device, '/job:ps/task:1/cpu:0')
      self.assertDeviceEqual(e.initial_value.device, '/cpu:99')

  def testVariableGPUPlacement(self):

    with tf.Graph().as_default():
      device_fn = variables.VariableDeviceChooser(placement='gpu:0')
      with scopes.arg_scope([variables.variable], device=device_fn):
        a = variables.variable('a', [])
        b = variables.variable('b', [])
        c = variables.variable('c', [], device='cpu:12')
        d = variables.variable('d', [])
        with tf.device('cpu:99'):
          e_init = tf.constant(12)
        e = variables.variable('e', initializer=e_init)
      # The values below highlight how the VariableDeviceChooser puts initial
      # values on the same device as the variable job.
      self.assertDeviceEqual(a.device, '/gpu:0')
      self.assertDeviceEqual(a.initial_value.device, a.device)
      self.assertDeviceEqual(b.device, '/gpu:0')
      self.assertDeviceEqual(b.initial_value.device, b.device)
      self.assertDeviceEqual(c.device, '/cpu:12')
      self.assertDeviceEqual(c.initial_value.device, c.device)
      self.assertDeviceEqual(d.device, '/gpu:0')
      self.assertDeviceEqual(d.initial_value.device, d.device)
      self.assertDeviceEqual(e.device, '/gpu:0')
      self.assertDeviceEqual(e.initial_value.device, '/cpu:99')

  def testVariableCollection(self):
    with self.test_session():
      a = variables.variable('a', [], collections='A')
      b = variables.variable('b', [], collections='B')
      self.assertEquals(a, tf.get_collection('A')[0])
      self.assertEquals(b, tf.get_collection('B')[0])

  def testVariableCollections(self):
    with self.test_session():
      a = variables.variable('a', [], collections=['A', 'C'])
      b = variables.variable('b', [], collections=['B', 'C'])
      self.assertEquals(a, tf.get_collection('A')[0])
      self.assertEquals(b, tf.get_collection('B')[0])

  def testVariableCollectionsWithArgScope(self):
    with self.test_session():
      with scopes.arg_scope([variables.variable], collections='A'):
        a = variables.variable('a', [])
        b = variables.variable('b', [])
      self.assertListEqual([a, b], tf.get_collection('A'))

  def testVariableCollectionsWithArgScopeNested(self):
    with self.test_session():
      with scopes.arg_scope([variables.variable], collections='A'):
        a = variables.variable('a', [])
        with scopes.arg_scope([variables.variable], collections='B'):
          b = variables.variable('b', [])
      self.assertEquals(a, tf.get_collection('A')[0])
      self.assertEquals(b, tf.get_collection('B')[0])

  def testVariableCollectionsWithArgScopeNonNested(self):
    with self.test_session():
      with scopes.arg_scope([variables.variable], collections='A'):
        a = variables.variable('a', [])
      with scopes.arg_scope([variables.variable], collections='B'):
        b = variables.variable('b', [])
      variables.variable('c', [])
      self.assertListEqual([a], tf.get_collection('A'))
      self.assertListEqual([b], tf.get_collection('B'))

  def testVariableRestoreWithArgScopeNested(self):
    with self.test_session():
      with scopes.arg_scope([variables.variable], restore=True):
        a = variables.variable('a', [])
        with scopes.arg_scope([variables.variable],
                              trainable=False,
                              collections=['A', 'B']):
          b = variables.variable('b', [])
        c = variables.variable('c', [])
      self.assertListEqual([a, b, c], variables.get_variables_to_restore())
      self.assertListEqual([a, c], tf.trainable_variables())
      self.assertListEqual([b], tf.get_collection('A'))
      self.assertListEqual([b], tf.get_collection('B'))


class GetVariablesByNameTest(tf.test.TestCase):

  def testGetVariableGivenNameScoped(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = variables.variable('a', [5])
        b = variables.variable('b', [5])
        self.assertEquals([a], variables.get_variables_by_name('a'))
        self.assertEquals([b], variables.get_variables_by_name('b'))

  def testGetVariablesByNameReturnsByValueWithScope(self):
    with self.test_session():
      with tf.variable_scope('A'):
        a = variables.variable('a', [5])
        matched_variables = variables.get_variables_by_name('a')

        # If variables.get_variables_by_name returns the list by reference, the
        # following append should persist, and be returned, in subsequent calls
        # to variables.get_variables_by_name('a').
        matched_variables.append(4)

        matched_variables = variables.get_variables_by_name('a')
        self.assertEquals([a], matched_variables)

  def testGetVariablesByNameReturnsByValueWithoutScope(self):
    with self.test_session():
      a = variables.variable('a', [5])
      matched_variables = variables.get_variables_by_name('a')

      # If variables.get_variables_by_name returns the list by reference, the
      # following append should persist, and be returned, in subsequent calls
      # to variables.get_variables_by_name('a').
      matched_variables.append(4)

      matched_variables = variables.get_variables_by_name('a')
      self.assertEquals([a], matched_variables)


class GlobalStepTest(tf.test.TestCase):

  def testStable(self):
    with tf.Graph().as_default():
      gs = variables.global_step()
      gs2 = variables.global_step()
      self.assertTrue(gs is gs2)

  def testDevice(self):
    with tf.Graph().as_default():
      with scopes.arg_scope([variables.global_step], device='/gpu:0'):
        gs = variables.global_step()
      self.assertDeviceEqual(gs.device, '/gpu:0')

  def testDeviceFn(self):
    class DevFn(object):

      def __init__(self):
        self.counter = -1

      def __call__(self, op):
        self.counter += 1
        return '/cpu:%d' % self.counter

    with tf.Graph().as_default():
      with scopes.arg_scope([variables.global_step], device=DevFn()):
        gs = variables.global_step()
        gs2 = variables.global_step()
      self.assertDeviceEqual(gs.device, '/cpu:0')
      self.assertEquals(gs, gs2)
      self.assertDeviceEqual(gs2.device, '/cpu:0')

  def testReplicaDeviceSetter(self):
    device_fn = tf.train.replica_device_setter(2)
    with tf.Graph().as_default():
      with scopes.arg_scope([variables.global_step], device=device_fn):
        gs = variables.global_step()
        gs2 = variables.global_step()
        self.assertEquals(gs, gs2)
        self.assertDeviceEqual(gs.device, '/job:ps/task:0')
        self.assertDeviceEqual(gs.initial_value.device, '/job:ps/task:0')
        self.assertDeviceEqual(gs2.device, '/job:ps/task:0')
        self.assertDeviceEqual(gs2.initial_value.device, '/job:ps/task:0')

  def testVariableWithVariableDeviceChooser(self):

    with tf.Graph().as_default():
      device_fn = variables.VariableDeviceChooser()
      with scopes.arg_scope([variables.global_step], device=device_fn):
        gs = variables.global_step()
        gs2 = variables.global_step()
        self.assertEquals(gs, gs2)
        self.assertDeviceEqual(gs.device, 'cpu:0')
        self.assertDeviceEqual(gs.initial_value.device, gs.device)
        self.assertDeviceEqual(gs2.device, 'cpu:0')
        self.assertDeviceEqual(gs2.initial_value.device, gs2.device)


if __name__ == '__main__':
  tf.test.main()
