# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for tf_example_feature_key."""
import dataclasses
import inspect
from absl.testing import absltest
from absl.testing import parameterized

from official.core import tf_example_feature_key


@tf_example_feature_key.dataclass
class TestFeatureKey(tf_example_feature_key.TfExampleFeatureKeyBase):
  test: str = 'foo/bar'


class TfExampleFeatureKeyTest(parameterized.TestCase):

  def test_add_prefix_success(self):
    test_key = TestFeatureKey('prefix')
    self.assertEqual(test_key.test, 'prefix/foo/bar')

  @parameterized.parameters(None, '')
  def test_add_prefix_skip_success(self, prefix):
    test_key = TestFeatureKey(prefix)
    self.assertEqual(test_key.test, 'foo/bar')

  def test_all_feature_key_classes_are_valid(self):
    for _, obj in inspect.getmembers(tf_example_feature_key):
      if inspect.isclass(obj):
        self.assertTrue(dataclasses.is_dataclass(obj))
        self.assertTrue(
            issubclass(obj, tf_example_feature_key.TfExampleFeatureKeyBase))


if __name__ == '__main__':
  absltest.main()
