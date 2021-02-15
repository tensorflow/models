# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for official.core.train_utils."""

import tensorflow as tf

from official.core import train_utils


class TrainUtilsTest(tf.test.TestCase):

  def test_get_leaf_nested_dict(self):
    d = {'a': {'i': {'x': 5}}}
    self.assertEqual(train_utils.get_leaf_nested_dict(d, ['a', 'i', 'x']), 5)

  def test_get_leaf_nested_dict_not_leaf(self):
    with self.assertRaisesRegex(KeyError, 'The value extracted with keys.*'):
      d = {'a': {'i': {'x': 5}}}
      train_utils.get_leaf_nested_dict(d, ['a', 'i'])

  def test_get_leaf_nested_dict_path_not_exist_missing_key(self):
    with self.assertRaisesRegex(KeyError, 'Path not exist while traversing .*'):
      d = {'a': {'i': {'x': 5}}}
      train_utils.get_leaf_nested_dict(d, ['a', 'i', 'y'])

  def test_get_leaf_nested_dict_path_not_exist_out_of_range(self):
    with self.assertRaisesRegex(KeyError, 'Path not exist while traversing .*'):
      d = {'a': {'i': {'x': 5}}}
      train_utils.get_leaf_nested_dict(d, ['a', 'i', 'z'])

  def test_get_leaf_nested_dict_path_not_exist_meets_leaf(self):
    with self.assertRaisesRegex(KeyError, 'Path not exist while traversing .*'):
      d = {'a': {'i': 5}}
      train_utils.get_leaf_nested_dict(d, ['a', 'i', 'z'])

  def test_cast_leaf_nested_dict(self):
    d = {'a': {'i': {'x': '123'}}, 'b': 456.5}
    d = train_utils.cast_leaf_nested_dict(d, int)
    self.assertEqual(d['a']['i']['x'], 123)
    self.assertEqual(d['b'], 456)


if __name__ == '__main__':
  tf.test.main()
