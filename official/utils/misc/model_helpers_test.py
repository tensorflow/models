# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
""" Tests for Model Helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.misc import model_helpers


class PastStopThresholdTest(tf.test.TestCase):
  """Tests for past_stop_threshold."""

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
      model_helpers.past_stop_threshold("str", 1)

    with self.assertRaises(ValueError):
      model_helpers.past_stop_threshold("str", tf.constant(5))

    with self.assertRaises(ValueError):
      model_helpers.past_stop_threshold("str", "another")

    with self.assertRaises(ValueError):
      model_helpers.past_stop_threshold(0, None)

    with self.assertRaises(ValueError):
      model_helpers.past_stop_threshold(0.7, "str")

    with self.assertRaises(ValueError):
      model_helpers.past_stop_threshold(tf.constant(4), None)

  def test_random_seed(self):
    """It is unclear if this test is a good idea or stable.
    If tests are run in parallel, this could be flakey."""
    model_helpers.set_random_seed(42)
    expected_py_random = [int(random.random() * 1000) for i in range(10)]
    tf_random = []
    with tf.Session() as sess:
      for i in range(10):
          a = tf.random_uniform([1])
          tf_random.append(int(sess.run(a)[0] * 1000))

    model_helpers.set_random_seed(42)
    py_random = [int(random.random() * 1000) for i in range(10)]

    # Instead of concerning ourselves with the particular results, we simply
    # want to ensure that the results are reproducible. So, we seed, read,
    # re-seed, re-read.
    self.assertAllEqual(expected_py_random, py_random)

    # TF does not accept being re-seeded.
    expected_tf_random = [637, 689, 961, 969, 321, 390, 919, 681, 112, 187]
    self.assertAllEqual(expected_tf_random, tf_random)


if __name__ == "__main__":
  tf.test.main()
