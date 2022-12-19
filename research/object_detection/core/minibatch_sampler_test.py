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

"""Tests for google3.research.vale.object_detection.minibatch_sampler."""

import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.core import minibatch_sampler
from object_detection.utils import test_case


class MinibatchSamplerTest(test_case.TestCase):

  def test_subsample_indicator_when_more_true_elements_than_num_samples(self):
    np_indicator = np.array([True, False, True, False, True, True, False])
    def graph_fn(indicator):
      samples = minibatch_sampler.MinibatchSampler.subsample_indicator(
          indicator, 3)
      return samples
    samples_out = self.execute(graph_fn, [np_indicator])
    self.assertTrue(np.sum(samples_out), 3)
    self.assertAllEqual(samples_out,
                        np.logical_and(samples_out, np_indicator))

  def test_subsample_indicator_when_less_true_elements_than_num_samples(self):
    np_indicator = np.array([True, False, True, False, True, True, False])
    def graph_fn(indicator):
      samples = minibatch_sampler.MinibatchSampler.subsample_indicator(
          indicator, 5)
      return samples
    samples_out = self.execute(graph_fn, [np_indicator])
    self.assertTrue(np.sum(samples_out), 4)
    self.assertAllEqual(samples_out,
                        np.logical_and(samples_out, np_indicator))

  def test_subsample_indicator_when_num_samples_is_zero(self):
    np_indicator = np.array([True, False, True, False, True, True, False])
    def graph_fn(indicator):
      samples_none = minibatch_sampler.MinibatchSampler.subsample_indicator(
          indicator, 0)
      return samples_none
    samples_out = self.execute(graph_fn, [np_indicator])
    self.assertAllEqual(
        np.zeros_like(samples_out, dtype=bool),
        samples_out)

  def test_subsample_indicator_when_indicator_all_false(self):
    indicator_empty = np.zeros([0], dtype=bool)
    def graph_fn(indicator):
      samples_empty = minibatch_sampler.MinibatchSampler.subsample_indicator(
          indicator, 4)
      return samples_empty
    samples_out = self.execute(graph_fn, [indicator_empty])
    self.assertEqual(0, samples_out.size)


if __name__ == '__main__':
  tf.test.main()
