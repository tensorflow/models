# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for Model Helper functions."""

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.misc import model_helpers


class PastStopThresholdTest(tf.test.TestCase):
  """Tests for past_stop_threshold."""

  def setUp(self):
    super(PastStopThresholdTest, self).setUp()
    tf.compat.v1.disable_eager_execution()

  def test_past_stop_threshold(self):
    """Tests for normal operating conditions."""
    self.assertTrue(model_helpers.past_stop_threshold(0.54, 1))
    self.assertTrue(model_helpers.past_stop_threshold(54, 100))
    self.assertFalse(model_helpers.past_stop_threshold(0.54, 0.1))
    self.assertFalse(model_helpers.past_stop_threshold(-0.54, -1.5))
    self.assertTrue(model_helpers.past_stop_threshold(-0.54, 0))
    self.assertTrue(model_helpers.past_stop_threshold(0, 0))
    self.assertTrue(model_helpers.past_stop_threshold(0.54, 0.54))

  def test_past_stop_threshold_none_false(self):
    """Tests that check None returns false."""
    self.assertFalse(model_helpers.past_stop_threshold(None, -1.5))
    self.assertFalse(model_helpers.past_stop_threshold(None, None))
    self.assertFalse(model_helpers.past_stop_threshold(None, 1.5))
    # Zero should be okay, though.
    self.assertTrue(model_helpers.past_stop_threshold(0, 1.5))

  def test_past_stop_threshold_not_number(self):
    """Tests for error conditions."""
    with self.assertRaises(ValueError):
      model_helpers.past_stop_threshold('str', 1)

    with self.assertRaises(ValueError):
      model_helpers.past_stop_threshold('str', tf.constant(5))

    with self.assertRaises(ValueError):
      model_helpers.past_stop_threshold('str', 'another')

    with self.assertRaises(ValueError):
      model_helpers.past_stop_threshold(0, None)

    with self.assertRaises(ValueError):
      model_helpers.past_stop_threshold(0.7, 'str')

    with self.assertRaises(ValueError):
      model_helpers.past_stop_threshold(tf.constant(4), None)


class SyntheticDataTest(tf.test.TestCase):
  """Tests for generate_synthetic_data."""

  def test_generate_synethetic_data(self):
    input_element, label_element = tf.compat.v1.data.make_one_shot_iterator(
        model_helpers.generate_synthetic_data(
            input_shape=tf.TensorShape([5]),
            input_value=123,
            input_dtype=tf.float32,
            label_shape=tf.TensorShape([]),
            label_value=456,
            label_dtype=tf.int32)).get_next()

    with self.session() as sess:
      for n in range(5):
        inp, lab = sess.run((input_element, label_element))
        self.assertAllClose(inp, [123., 123., 123., 123., 123.])
        self.assertEquals(lab, 456)

  def test_generate_only_input_data(self):
    d = model_helpers.generate_synthetic_data(
        input_shape=tf.TensorShape([4]),
        input_value=43.5,
        input_dtype=tf.float32)

    element = tf.compat.v1.data.make_one_shot_iterator(d).get_next()
    self.assertFalse(isinstance(element, tuple))

    with self.session() as sess:
      inp = sess.run(element)
      self.assertAllClose(inp, [43.5, 43.5, 43.5, 43.5])

  def test_generate_nested_data(self):
    d = model_helpers.generate_synthetic_data(
        input_shape={
            'a': tf.TensorShape([2]),
            'b': {
                'c': tf.TensorShape([3]),
                'd': tf.TensorShape([])
            }
        },
        input_value=1.1)

    element = tf.compat.v1.data.make_one_shot_iterator(d).get_next()
    self.assertIn('a', element)
    self.assertIn('b', element)
    self.assertEquals(len(element['b']), 2)
    self.assertIn('c', element['b'])
    self.assertIn('d', element['b'])
    self.assertNotIn('c', element)

    with self.session() as sess:
      inp = sess.run(element)
      self.assertAllClose(inp['a'], [1.1, 1.1])
      self.assertAllClose(inp['b']['c'], [1.1, 1.1, 1.1])
      self.assertAllClose(inp['b']['d'], 1.1)


if __name__ == '__main__':
  tf.test.main()
