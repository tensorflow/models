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
import tensorflow.compat.v1 as tf

from object_detection.core import balanced_positive_negative_sampler
from object_detection.utils import test_case


class BalancedPositiveNegativeSamplerTest(test_case.TestCase):

  def test_subsample_all_examples(self):
    if self.has_tpu(): return
    numpy_labels = np.random.permutation(300)
    indicator = np.array(np.ones(300) == 1, np.bool)
    numpy_labels = (numpy_labels - 200) > 0

    labels = np.array(numpy_labels, np.bool)

    def graph_fn(indicator, labels):
      sampler = (
          balanced_positive_negative_sampler.BalancedPositiveNegativeSampler())
      return sampler.subsample(indicator, 64, labels)

    is_sampled = self.execute_cpu(graph_fn, [indicator, labels])
    self.assertEqual(sum(is_sampled), 64)
    self.assertEqual(sum(np.logical_and(numpy_labels, is_sampled)), 32)
    self.assertEqual(sum(np.logical_and(
        np.logical_not(numpy_labels), is_sampled)), 32)

  def test_subsample_all_examples_static(self):
    if not self.has_tpu(): return
    numpy_labels = np.random.permutation(300)
    indicator = np.array(np.ones(300) == 1, np.bool)
    numpy_labels = (numpy_labels - 200) > 0

    labels = np.array(numpy_labels, np.bool)

    def graph_fn(indicator, labels):
      sampler = (
          balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(
              is_static=True))
      return sampler.subsample(indicator, 64, labels)

    is_sampled = self.execute_tpu(graph_fn, [indicator, labels])
    self.assertEqual(sum(is_sampled), 64)
    self.assertEqual(sum(np.logical_and(numpy_labels, is_sampled)), 32)
    self.assertEqual(sum(np.logical_and(
        np.logical_not(numpy_labels), is_sampled)), 32)

  def test_subsample_selection(self):
    if self.has_tpu(): return
    # Test random sampling when only some examples can be sampled:
    # 100 samples, 20 positives, 10 positives cannot be sampled.
    numpy_labels = np.arange(100)
    numpy_indicator = numpy_labels < 90
    indicator = np.array(numpy_indicator, np.bool)
    numpy_labels = (numpy_labels - 80) >= 0

    labels = np.array(numpy_labels, np.bool)

    def graph_fn(indicator, labels):
      sampler = (
          balanced_positive_negative_sampler.BalancedPositiveNegativeSampler())
      return sampler.subsample(indicator, 64, labels)

    is_sampled = self.execute_cpu(graph_fn, [indicator, labels])
    self.assertEqual(sum(is_sampled), 64)
    self.assertEqual(sum(np.logical_and(numpy_labels, is_sampled)), 10)
    self.assertEqual(sum(np.logical_and(
        np.logical_not(numpy_labels), is_sampled)), 54)
    self.assertAllEqual(is_sampled, np.logical_and(is_sampled, numpy_indicator))

  def test_subsample_selection_static(self):
    if not self.has_tpu(): return
    # Test random sampling when only some examples can be sampled:
    # 100 samples, 20 positives, 10 positives cannot be sampled.
    numpy_labels = np.arange(100)
    numpy_indicator = numpy_labels < 90
    indicator = np.array(numpy_indicator, np.bool)
    numpy_labels = (numpy_labels - 80) >= 0

    labels = np.array(numpy_labels, np.bool)

    def graph_fn(indicator, labels):
      sampler = (
          balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(
              is_static=True))
      return sampler.subsample(indicator, 64, labels)

    is_sampled = self.execute_tpu(graph_fn, [indicator, labels])
    self.assertEqual(sum(is_sampled), 64)
    self.assertEqual(sum(np.logical_and(numpy_labels, is_sampled)), 10)
    self.assertEqual(sum(np.logical_and(
        np.logical_not(numpy_labels), is_sampled)), 54)
    self.assertAllEqual(is_sampled, np.logical_and(is_sampled, numpy_indicator))

  def test_subsample_selection_larger_batch_size(self):
    if self.has_tpu(): return
    # Test random sampling when total number of examples that can be sampled are
    # less than batch size:
    # 100 samples, 50 positives, 40 positives cannot be sampled, batch size 64.
    # It should still return 64 samples, with 4 of them that couldn't have been
    # sampled.
    numpy_labels = np.arange(100)
    numpy_indicator = numpy_labels < 60
    indicator = np.array(numpy_indicator, np.bool)
    numpy_labels = (numpy_labels - 50) >= 0

    labels = np.array(numpy_labels, np.bool)

    def graph_fn(indicator, labels):
      sampler = (
          balanced_positive_negative_sampler.BalancedPositiveNegativeSampler())
      return sampler.subsample(indicator, 64, labels)

    is_sampled = self.execute_cpu(graph_fn, [indicator, labels])
    self.assertEqual(sum(is_sampled), 60)
    self.assertGreaterEqual(sum(np.logical_and(numpy_labels, is_sampled)), 10)
    self.assertGreaterEqual(
        sum(np.logical_and(np.logical_not(numpy_labels), is_sampled)), 50)
    self.assertEqual(sum(np.logical_and(is_sampled, numpy_indicator)), 60)

  def test_subsample_selection_larger_batch_size_static(self):
    if not self.has_tpu(): return
    # Test random sampling when total number of examples that can be sampled are
    # less than batch size:
    # 100 samples, 50 positives, 40 positives cannot be sampled, batch size 64.
    # It should still return 64 samples, with 4 of them that couldn't have been
    # sampled.
    numpy_labels = np.arange(100)
    numpy_indicator = numpy_labels < 60
    indicator = np.array(numpy_indicator, np.bool)
    numpy_labels = (numpy_labels - 50) >= 0

    labels = np.array(numpy_labels, np.bool)

    def graph_fn(indicator, labels):
      sampler = (
          balanced_positive_negative_sampler.BalancedPositiveNegativeSampler(
              is_static=True))
      return sampler.subsample(indicator, 64, labels)

    is_sampled = self.execute_tpu(graph_fn, [indicator, labels])
    self.assertEqual(sum(is_sampled), 64)
    self.assertGreaterEqual(sum(np.logical_and(numpy_labels, is_sampled)), 10)
    self.assertGreaterEqual(
        sum(np.logical_and(np.logical_not(numpy_labels), is_sampled)), 50)
    self.assertEqual(sum(np.logical_and(is_sampled, numpy_indicator)), 60)

  def test_subsample_selection_no_batch_size(self):
    if self.has_tpu(): return
    # Test random sampling when only some examples can be sampled:
    # 1000 samples, 6 positives (5 can be sampled).
    numpy_labels = np.arange(1000)
    numpy_indicator = numpy_labels < 999
    numpy_labels = (numpy_labels - 994) >= 0

    def graph_fn(indicator, labels):
      sampler = (balanced_positive_negative_sampler.
                 BalancedPositiveNegativeSampler(0.01))
      is_sampled = sampler.subsample(indicator, None, labels)
      return is_sampled
    is_sampled_out = self.execute_cpu(graph_fn, [numpy_indicator, numpy_labels])
    self.assertEqual(sum(is_sampled_out), 500)
    self.assertEqual(sum(np.logical_and(numpy_labels, is_sampled_out)), 5)
    self.assertEqual(sum(np.logical_and(
        np.logical_not(numpy_labels), is_sampled_out)), 495)
    self.assertAllEqual(is_sampled_out, np.logical_and(is_sampled_out,
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
