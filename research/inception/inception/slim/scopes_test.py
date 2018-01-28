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
"""Tests slim.scopes."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from inception.slim import scopes


@scopes.add_arg_scope
def func1(*args, **kwargs):
  return (args, kwargs)


@scopes.add_arg_scope
def func2(*args, **kwargs):
  return (args, kwargs)


class ArgScopeTest(tf.test.TestCase):

  def testEmptyArgScope(self):
    with self.test_session():
      self.assertEqual(scopes._current_arg_scope(), {})

  def testCurrentArgScope(self):
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    key_op = (func1.__module__, func1.__name__)
    current_scope = {key_op: func1_kwargs.copy()}
    with self.test_session():
      with scopes.arg_scope([func1], a=1, b=None, c=[1]) as scope:
        self.assertDictEqual(scope, current_scope)

  def testCurrentArgScopeNested(self):
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    func2_kwargs = {'b': 2, 'd': [2]}
    key = lambda f: (f.__module__, f.__name__)
    current_scope = {key(func1): func1_kwargs.copy(),
                     key(func2): func2_kwargs.copy()}
    with self.test_session():
      with scopes.arg_scope([func1], a=1, b=None, c=[1]):
        with scopes.arg_scope([func2], b=2, d=[2]) as scope:
          self.assertDictEqual(scope, current_scope)

  def testReuseArgScope(self):
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    key_op = (func1.__module__, func1.__name__)
    current_scope = {key_op: func1_kwargs.copy()}
    with self.test_session():
      with scopes.arg_scope([func1], a=1, b=None, c=[1]) as scope1:
        pass
      with scopes.arg_scope(scope1) as scope:
        self.assertDictEqual(scope, current_scope)

  def testReuseArgScopeNested(self):
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    func2_kwargs = {'b': 2, 'd': [2]}
    key = lambda f: (f.__module__, f.__name__)
    current_scope1 = {key(func1): func1_kwargs.copy()}
    current_scope2 = {key(func1): func1_kwargs.copy(),
                      key(func2): func2_kwargs.copy()}
    with self.test_session():
      with scopes.arg_scope([func1], a=1, b=None, c=[1]) as scope1:
        with scopes.arg_scope([func2], b=2, d=[2]) as scope2:
          pass
      with scopes.arg_scope(scope1):
        self.assertDictEqual(scopes._current_arg_scope(), current_scope1)
      with scopes.arg_scope(scope2):
        self.assertDictEqual(scopes._current_arg_scope(), current_scope2)

  def testSimpleArgScope(self):
    func1_args = (0,)
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    with self.test_session():
      with scopes.arg_scope([func1], a=1, b=None, c=[1]):
        args, kwargs = func1(0)
        self.assertTupleEqual(args, func1_args)
        self.assertDictEqual(kwargs, func1_kwargs)

  def testSimpleArgScopeWithTuple(self):
    func1_args = (0,)
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    with self.test_session():
      with scopes.arg_scope((func1,), a=1, b=None, c=[1]):
        args, kwargs = func1(0)
        self.assertTupleEqual(args, func1_args)
        self.assertDictEqual(kwargs, func1_kwargs)

  def testOverwriteArgScope(self):
    func1_args = (0,)
    func1_kwargs = {'a': 1, 'b': 2, 'c': [1]}
    with scopes.arg_scope([func1], a=1, b=None, c=[1]):
      args, kwargs = func1(0, b=2)
      self.assertTupleEqual(args, func1_args)
      self.assertDictEqual(kwargs, func1_kwargs)

  def testNestedArgScope(self):
    func1_args = (0,)
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    with scopes.arg_scope([func1], a=1, b=None, c=[1]):
      args, kwargs = func1(0)
      self.assertTupleEqual(args, func1_args)
      self.assertDictEqual(kwargs, func1_kwargs)
      func1_kwargs['b'] = 2
      with scopes.arg_scope([func1], b=2):
        args, kwargs = func1(0)
        self.assertTupleEqual(args, func1_args)
        self.assertDictEqual(kwargs, func1_kwargs)

  def testSharedArgScope(self):
    func1_args = (0,)
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    with scopes.arg_scope([func1, func2], a=1, b=None, c=[1]):
      args, kwargs = func1(0)
      self.assertTupleEqual(args, func1_args)
      self.assertDictEqual(kwargs, func1_kwargs)
      args, kwargs = func2(0)
      self.assertTupleEqual(args, func1_args)
      self.assertDictEqual(kwargs, func1_kwargs)

  def testSharedArgScopeTuple(self):
    func1_args = (0,)
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    with scopes.arg_scope((func1, func2), a=1, b=None, c=[1]):
      args, kwargs = func1(0)
      self.assertTupleEqual(args, func1_args)
      self.assertDictEqual(kwargs, func1_kwargs)
      args, kwargs = func2(0)
      self.assertTupleEqual(args, func1_args)
      self.assertDictEqual(kwargs, func1_kwargs)

  def testPartiallySharedArgScope(self):
    func1_args = (0,)
    func1_kwargs = {'a': 1, 'b': None, 'c': [1]}
    func2_args = (1,)
    func2_kwargs = {'a': 1, 'b': None, 'd': [2]}
    with scopes.arg_scope([func1, func2], a=1, b=None):
      with scopes.arg_scope([func1], c=[1]), scopes.arg_scope([func2], d=[2]):
        args, kwargs = func1(0)
        self.assertTupleEqual(args, func1_args)
        self.assertDictEqual(kwargs, func1_kwargs)
        args, kwargs = func2(1)
        self.assertTupleEqual(args, func2_args)
        self.assertDictEqual(kwargs, func2_kwargs)

if __name__ == '__main__':
  tf.test.main()
