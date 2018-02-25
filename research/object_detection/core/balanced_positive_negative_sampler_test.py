# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for object_detection.core.balanced_positive_negative_sampler."""

import numpy as np
import tensorflow as tf

from object_detection.core import balanced_positive_negative_sampler


class BalancedPositiveNegativeSamplerTest(tf.test.TestCase):

  def test_subsample_all_examples(self):
    numpy_labels = np.random.permutation(300)
    indicator = tf.constant(np.ones(300) == 1)
    numpy_labels = (numpy_labels - 200) > 0

    labels = tf.constant(numpy_labels)

    sampler = (balanced_positive_negative_sampler.
               BalancedPositiveNegativeSampler())
    is_sampled = sampler.subsample(indicator, 64, labels)
    with self.test_session() as sess:
      is_sampled = sess.run(is_sampled)
      self.assertTrue(sum(is_sampled) == 64)
      self.assertTrue(sum(np.logical_and(numpy_labels, is_sampled)) == 32)
      self.assertTrue(sum(np.logical_and(
          np.logical_not(numpy_labels), is_sampled)) == 32)

  def test_subsample_selection(self):
    # Test random sampling when only some examples can be sampled:
    # 100 samples, 20 positives, 10 positives cannot be sampled
    numpy_labels = np.arange(100)
    numpy_indicator = numpy_labels < 90
    indicator = tf.constant(numpy_indicator)
    numpy_labels = (numpy_labels - 80) >= 0

    labels = tf.constant(numpy_labels)

    sampler = (balanced_positive_negative_sampler.
               BalancedPositiveNegativeSampler())
    is_sampled = sampler.subsample(indicator, 64, labels)
    with self.test_session() as sess:
      is_sampled = sess.run(is_sampled)
      self.assertTrue(sum(is_sampled) == 64)
      self.assertTrue(sum(np.logical_and(numpy_labels, is_sampled)) == 10)
      self.assertTrue(sum(np.logical_and(
          np.logical_not(numpy_labels), is_sampled)) == 54)
      self.assertAllEqual(is_sampled, np.logical_and(is_sampled,
                                                     numpy_indicator))

  def test_raises_error_with_incorrect_label_shape(self):
    labels = tf.constant([[True, False, False]])
    indicator = tf.constant([True, False, True])
    sampler = (balanced_positive_negative_sampler.
               BalancedPositiveNegativeSampler())
    with self.assertRaises(ValueError):
      sampler.subsample(indicator, 64, labels)

  def test_raises_error_with_incorrect_indicator_shape(self):
    labels = tf.constant([True, False, False])
    indicator = tf.constant([[True, False, True]])
    sampler = (balanced_positive_negative_sampler.
               BalancedPositiveNegativeSampler())
    with self.assertRaises(ValueError):
      sampler.subsample(indicator, 64, labels)


if __name__ == '__main__':
  tf.test.main()
