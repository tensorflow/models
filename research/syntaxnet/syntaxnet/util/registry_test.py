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

"""Tests for registry system.

This test uses two other modules:
  * registry_test_base, which defines a registered base class.
  * registry_test_impl, which defines a subclass.

Critically, although we use both as build dependencies, we do not explicitly
import registry_test_impl.
"""

import traceback

from tensorflow.python.platform import googletest

from syntaxnet.util import registry_test_base

PATH = 'syntaxnet.util.'


class RegistryTest(googletest.TestCase):
  """Testing rig."""

  def testCanCreateImpl(self):
    """Tests that Create can create the Impl subclass."""
    try:
      impl = registry_test_base.Base.Create(PATH + 'registry_test_impl.Impl',
                                            'hello world')
    except ValueError:
      self.fail('Create raised ValueError: %s' % traceback.format_exc())
    self.assertEqual('hello world', impl.Get())

  def testCanCreateByAlias(self):
    """Tests that Create can create an Impl subclass via Alias."""
    try:
      impl = registry_test_base.Base.Create(PATH + 'registry_test_impl.Alias',
                                            'hello world')
    except ValueError:
      self.fail('Create raised ValueError: %s' % traceback.format_exc())
    self.assertEqual('hello world', impl.Get())

  def testCannotCreateNonSubclass(self):
    """Tests that Create fails if the class is not a subclass of Base."""
    with self.assertRaisesRegexp(ValueError, 'Failed to create'):
      registry_test_base.Base.Create(PATH + 'registry_test_impl.NonSubclass',
                                     'hello world')

  def testCannotCreateNonClass(self):
    """Tests that Create fails if the name does not identify a class."""
    with self.assertRaisesRegexp(ValueError, 'Failed to create'):
      registry_test_base.Base.Create(PATH + 'registry_test_impl.variable',
                                     'hello world')
    with self.assertRaisesRegexp(ValueError, 'Failed to create'):
      registry_test_base.Base.Create(PATH + 'registry_test_impl.Function',
                                     'hello world')

  def testCannotCreateMissingClass(self):
    """Tests that Create fails if the class does not exist in the module."""
    with self.assertRaisesRegexp(ValueError, 'Failed to create'):
      registry_test_base.Base.Create(PATH + 'registry_test_impl.MissingClass',
                                     'hello world')

  def testCannotCreateMissingModule(self):
    """Tests that Create fails if the module does not exist."""
    with self.assertRaisesRegexp(ValueError, 'Failed to create'):
      registry_test_base.Base.Create(PATH + 'missing.SomeClass', 'hello world')

  def testCannotCreateMissingPackage(self):
    """Tests that Create fails if the package does not exist."""
    with self.assertRaisesRegexp(ValueError, 'Failed to create'):
      registry_test_base.Base.Create('missing.package.path.module.SomeClass',
                                     'hello world')

  def testCannotCreateMalformedType(self):
    """Tests that Create fails on malformed type names."""
    with self.assertRaisesRegexp(ValueError, 'Failed to create'):
      registry_test_base.Base.Create('oneword', 'hello world')
    with self.assertRaisesRegexp(ValueError, 'Failed to create'):
      registry_test_base.Base.Create('hyphen-ated', 'hello world')
    with self.assertRaisesRegexp(ValueError, 'Failed to create'):
      registry_test_base.Base.Create('has space', 'hello world')
    with self.assertRaisesRegexp(ValueError, 'Failed to create'):
      registry_test_base.Base.Create(' ', 'hello world')
    with self.assertRaisesRegexp(ValueError, 'Failed to create'):
      registry_test_base.Base.Create('', 'hello world')

  def testCanCreateWithRelativePath(self):
    """Tests that Create can create the Impl subclass using a relative path."""
    for name in [
        PATH + 'registry_test_impl.Impl',


        'syntaxnet.util.registry_test_impl.Impl',
        'util.registry_test_impl.Impl',
        'registry_test_impl.Impl'
    ]:
      value = 'created via %s' % name
      try:
        impl = registry_test_base.Base.Create(name, value)
      except ValueError:
        self.fail('Create raised ValueError: %s' % traceback.format_exc())
      self.assertTrue(impl is not None)
      self.assertEqual(value, impl.Get())

  def testCannotResolveRelativeName(self):
    """Tests that Create fails if a relative path cannot be resolved."""
    for name in [
        'bad.syntaxnet.util.registry_test_base.Impl',
        'syntaxnet.bad.registry_test_impl.Impl',
        'missing.registry_test_impl.Impl',
        'registry_test_impl.Bad',
        'Impl'
    ]:
      with self.assertRaisesRegexp(ValueError, 'Failed to create'):
        registry_test_base.Base.Create(name, 'hello world')


if __name__ == '__main__':
  googletest.main()
