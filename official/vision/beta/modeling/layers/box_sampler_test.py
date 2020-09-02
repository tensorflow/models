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
"""Tests for roi_sampler.py."""

# Import libraries
import numpy as np
import tensorflow as tf

from official.vision.beta.modeling.layers import box_sampler


class BoxSamplerTest(tf.test.TestCase):

  def test_box_sampler(self):
    positive_matches = np.array(
        [[True, False, False, False, True, True, False],
         [False, False, False, False, False, True, True]])
    negative_matches = np.array(
        [[False, True, True, True, False, False, False],
         [True, True, True, True, False, False, False]])
    ignored_matches = np.array(
        [[False, False, False, False, False, False, True],
         [False, False, False, False, True, False, False]])

    sampler = box_sampler.BoxSampler(num_samples=2, foreground_fraction=0.5)

    # Runs on TPU.
    strategy = tf.distribute.experimental.TPUStrategy()
    with strategy.scope():
      selected_indices_tpu = sampler(
          positive_matches, negative_matches, ignored_matches)

    self.assertEqual(2, tf.shape(selected_indices_tpu)[1])

    # Runs on CPU.
    selected_indices_cpu = sampler(
        positive_matches, negative_matches, ignored_matches)
    self.assertEqual(2, tf.shape(selected_indices_cpu)[1])

  def test_serialize_deserialize(self):
    kwargs = dict(
        num_samples=512,
        foreground_fraction=0.25,
    )
    sampler = box_sampler.BoxSampler(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(sampler.get_config(), expected_config)

    new_sampler = box_sampler.BoxSampler.from_config(
        sampler.get_config())

    self.assertAllEqual(sampler.get_config(), new_sampler.get_config())


if __name__ == '__main__':
  tf.test.main()
