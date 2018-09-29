# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Tests for fivo.nested_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
nest = tf.contrib.framework.nest

from fivo import nested_utils

# An example namedtuple for use in the following tests.
ExampleTuple = collections.namedtuple('ExampleTuple', ['a', 'b'])


class NestedUtilsTest(tf.test.TestCase):

  def test_map_nested_works_on_nested_structures(self):
    """Check that map_nested works with nested structures."""
    original = [1, (2, 3.2, (4., ExampleTuple(5, 6)))]
    expected = [2, (3, 4.2, (5., ExampleTuple(6, 7)))]
    out = nested_utils.map_nested(lambda x: x+1, original)
    self.assertEqual(expected, out)

  def test_map_nested_works_on_single_objects(self):
    """Check that map_nested works with raw objects."""
    original = 1
    expected = 2
    out = nested_utils.map_nested(lambda x: x+1, original)
    self.assertEqual(expected, out)

  def test_map_nested_works_on_flat_lists(self):
    """Check that map_nested works with a flat list."""
    original = [1, 2, 3]
    expected = [2, 3, 4]
    out = nested_utils.map_nested(lambda x: x+1, original)
    self.assertEqual(expected, out)

  def test_tile_tensors(self):
    """Checks that tile_tensors correctly tiles tensors of different ranks."""
    a = tf.range(20)
    b = tf.reshape(a, [2, 10])
    c = tf.reshape(a, [2, 2, 5])
    a_tiled = tf.tile(a, [3])
    b_tiled = tf.tile(b, [3, 1])
    c_tiled = tf.tile(c, [3, 1, 1])
    tensors = [a, (b, ExampleTuple(c, c))]
    expected_tensors = [a_tiled, (b_tiled, ExampleTuple(c_tiled, c_tiled))]
    tiled = nested_utils.tile_tensors(tensors, [3])
    nest.assert_same_structure(expected_tensors, tiled)
    with self.test_session() as sess:
      expected, out = sess.run([expected_tensors, tiled])
      expected = nest.flatten(expected)
      out = nest.flatten(out)
      # Check that the tiling is correct.
      for x, y in zip(expected, out):
        self.assertAllClose(x, y)

  def test_gather_tensors(self):
    a = tf.reshape(tf.range(20), [5, 4])
    inds = [0, 0, 1, 4]
    a_gathered = tf.gather(a, inds)
    tensors = [a, (a, ExampleTuple(a, a))]
    gt_gathered = [a_gathered, (a_gathered,
                                ExampleTuple(a_gathered, a_gathered))]
    gathered = nested_utils.gather_tensors(tensors, inds)
    nest.assert_same_structure(gt_gathered, gathered)
    with self.test_session() as sess:
      gt, out = sess.run([gt_gathered, gathered])
      gt = nest.flatten(gt)
      out = nest.flatten(out)
      # Check that the gathering is correct.
      for x, y in zip(gt, out):
        self.assertAllClose(x, y)

  def test_tas_for_tensors(self):
    a = tf.reshape(tf.range(20), [5, 4])
    tensors = [a, (a, ExampleTuple(a, a))]
    tas = nested_utils.tas_for_tensors(tensors, 5)
    nest.assert_same_structure(tensors, tas)
    # We can't pass TensorArrays to sess.run so instead we turn then back into
    # tensors to check that they were created correctly.
    stacked = nested_utils.map_nested(lambda x: x.stack(), tas)
    with self.test_session() as sess:
      gt, out = sess.run([tensors, stacked])
      gt = nest.flatten(gt)
      out = nest.flatten(out)
      # Check that the tas were created correctly.
      for x, y in zip(gt, out):
        self.assertAllClose(x, y)

  def test_read_tas(self):
    a = tf.reshape(tf.range(20), [5, 4])
    a_read = a[3, :]
    tensors = [a, (a, ExampleTuple(a, a))]
    gt_read = [a_read, (a_read, ExampleTuple(a_read, a_read))]
    tas = nested_utils.tas_for_tensors(tensors, 5)
    tas_read = nested_utils.read_tas(tas, 3)
    nest.assert_same_structure(tas, tas_read)
    with self.test_session() as sess:
      gt, out = sess.run([gt_read, tas_read])
      gt = nest.flatten(gt)
      out = nest.flatten(out)
      # Check that the tas were read correctly.
      for x, y in zip(gt, out):
        self.assertAllClose(x, y)

if __name__ == '__main__':
  tf.test.main()
