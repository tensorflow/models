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
from object_detection.utils import test_case


class BalancedPositiveNegativeSamplerTest(test_case.TestCase):

  def _test_subsample_all_examples(self, is_static=False):
    numpy_labels = np.random.permutation(300)
    indicator = tf.constant(np.ones(300) == 1)
    numpy_labels = (numpy_labels - 200) > 0

    labels = tf.constant(numpy_labels)

    sampler = (
        balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(
            is_static=is_static))
    is_sampled = sampler.subsample(indicator, 64, labels)
    with self.test_session() as sess:
      is_sampled = sess.run(is_sampled)
      self.assertTrue(sum(is_sampled) == 64)
      self.assertTrue(sum(np.logical_and(numpy_labels, is_sampled)) == 32)
      self.assertTrue(sum(np.logical_and(
          np.logical_not(numpy_labels), is_sampled)) == 32)

  def test_subsample_all_examples_dynamic(self):
    self._test_subsample_all_examples()

  def test_subsample_all_examples_static(self):
    self._test_subsample_all_examples(is_static=True)

  def _test_subsample_selection(self, is_static=False):
    # Test random sampling when only some examples can be sampled:
    # 100 samples, 20 positives, 10 positives cannot be sampled
    numpy_labels = np.arange(100)
    numpy_indicator = numpy_labels < 90
    indicator = tf.constant(numpy_indicator)
    numpy_labels = (numpy_labels - 80) >= 0

    labels = tf.constant(numpy_labels)

    sampler = (
        balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(
            is_static=is_static))
    is_sampled = sampler.subsample(indicator, 64, labels)
    with self.test_session() as sess:
      is_sampled = sess.run(is_sampled)
      self.assertTrue(sum(is_sampled) == 64)
      self.assertTrue(sum(np.logical_and(numpy_labels, is_sampled)) == 10)
      self.assertTrue(sum(np.logical_and(
          np.logical_not(numpy_labels), is_sampled)) == 54)
      self.assertAllEqual(is_sampled, np.logical_and(is_sampled,
                                                     numpy_indicator))

  def test_subsample_selection_dynamic(self):
    self._test_subsample_selection()

  def test_subsample_selection_static(self):
    self._test_subsample_selection(is_static=True)

  def _test_subsample_selection_larger_batch_size(self, is_static=False):
    # Test random sampling when total number of examples that can be sampled are
    # less than batch size:
    # 100 samples, 50 positives, 40 positives cannot be sampled, batch size 64.
    numpy_labels = np.arange(100)
    numpy_indicator = numpy_labels < 60
    indicator = tf.constant(numpy_indicator)
    numpy_labels = (numpy_labels - 50) >= 0

    labels = tf.constant(numpy_labels)

    sampler = (
        balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(
            is_static=is_static))
    is_sampled = sampler.subsample(indicator, 64, labels)
    with self.test_session() as sess:
      is_sampled = sess.run(is_sampled)
      self.assertTrue(sum(is_sampled) == 60)
      self.assertTrue(sum(np.logical_and(numpy_labels, is_sampled)) == 10)
      self.assertTrue(
          sum(np.logical_and(np.logical_not(numpy_labels), is_sampled)) == 50)
      self.assertAllEqual(is_sampled, np.logical_and(is_sampled,
                                                     numpy_indicator))

  def test_subsample_selection_larger_batch_size_dynamic(self):
    self._test_subsample_selection_larger_batch_size()

  def test_subsample_selection_larger_batch_size_static(self):
    self._test_subsample_selection_larger_batch_size(is_static=True)

  def test_subsample_selection_no_batch_size(self):
    # Test random sampling when only some examples can be sampled:
    # 1000 samples, 6 positives (5 can be sampled).
    numpy_labels = np.arange(1000)
    numpy_indicator = numpy_labels < 999
    indicator = tf.constant(numpy_indicator)
    numpy_labels = (numpy_labels - 994) >= 0

    labels = tf.constant(numpy_labels)

    sampler = (balanced_positive_negative_sampler.
               BalancedPositiveNegativeSampler(0.01))
    is_sampled = sampler.subsample(indicator, None, labels)
    with self.test_session() as sess:
      is_sampled = sess.run(is_sampled)
      self.assertTrue(sum(is_sampled) == 500)
      self.assertTrue(sum(np.logical_and(numpy_labels, is_sampled)) == 5)
      self.assertTrue(sum(np.logical_and(
          np.logical_not(numpy_labels), is_sampled)) == 495)
      self.assertAllEqual(is_sampled, np.logical_and(is_sampled,
                                                     numpy_indicator))

  def test_subsample_selection_no_batch_size_static(self):
    labels = tf.constant([[True, False, False]])
    indicator = tf.constant([True, False, True])
    sampler = (
        balanced_positive_negative_sampler.BalancedPositiveNegativeSampler())
    with self.assertRaises(ValueError):
      sampler.subsample(indicator, None, labels)

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
