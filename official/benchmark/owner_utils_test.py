# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for official.benchmark.owner_utils."""

from absl.testing import absltest

from official.benchmark import owner_utils


@owner_utils.Owner('static_owner')
def static_function(foo=5):
  return foo


def static_function_without_owner(foo=5):
  return foo


class BenchmarkClassWithoutOwner:

  def method_without_owner(self):
    return 100

  @owner_utils.Owner('method_owner')
  def method_with_owner(self):
    return 200


@owner_utils.Owner('class_owner')
class SomeBenchmarkClass:

  def method_inherited_owner(self):
    return 123

  @owner_utils.Owner('method_owner')
  def method_override_owner(self):
    return 345


@owner_utils.Owner('new_class_owner')
class InheritedClass(SomeBenchmarkClass):

  def method_inherited_owner(self):
    return 456

  @owner_utils.Owner('new_method_owner')
  def method_override_owner(self):
    return 567


class OwnerUtilsTest(absltest.TestCase):
  """Tests to assert for owner decorator functionality."""

  def test_owner_tag_missing(self):
    self.assertEqual(None, owner_utils.GetOwner(static_function_without_owner))

    benchmark_class = BenchmarkClassWithoutOwner()
    self.assertEqual(None,
                     owner_utils.GetOwner(benchmark_class.method_without_owner))
    self.assertEqual(100, benchmark_class.method_without_owner())

    self.assertEqual('method_owner',
                     owner_utils.GetOwner(benchmark_class.method_with_owner))
    self.assertEqual(200, benchmark_class.method_with_owner())

  def test_owner_attributes_static(self):
    self.assertEqual('static_owner', owner_utils.GetOwner(static_function))
    self.assertEqual(5, static_function(5))

  def test_owner_attributes_per_class(self):
    level1 = SomeBenchmarkClass()
    self.assertEqual('class_owner',
                     owner_utils.GetOwner(level1.method_inherited_owner))
    self.assertEqual(123, level1.method_inherited_owner())

    self.assertEqual('method_owner',
                     owner_utils.GetOwner(level1.method_override_owner))
    self.assertEqual(345, level1.method_override_owner())

  def test_owner_attributes_inherited_class(self):
    level2 = InheritedClass()
    self.assertEqual('new_class_owner',
                     owner_utils.GetOwner(level2.method_inherited_owner))
    self.assertEqual(456, level2.method_inherited_owner())

    self.assertEqual('new_method_owner',
                     owner_utils.GetOwner(level2.method_override_owner))
    self.assertEqual(567, level2.method_override_owner())


if __name__ == '__main__':
  absltest.main()
