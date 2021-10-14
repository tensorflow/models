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

"""Tests for segmentation_losses.py."""

from absl.testing import parameterized
import tensorflow as tf

from official.projects.volumetric_models.losses import segmentation_losses


class SegmentationLossDiceScoreTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters((None, 0.5, 0.3), ('generalized', 0.5, 0.3),
                            ('adaptive', 0.5, 0.07))
  def test_supported_loss(self, metric_type, output, expected_score):
    loss = segmentation_losses.SegmentationLossDiceScore(
        metric_type=metric_type)
    logits = tf.constant(output, shape=[2, 128, 128, 128, 1], dtype=tf.float32)
    labels = tf.ones(shape=[2, 128, 128, 128, 1], dtype=tf.float32)
    actual_score = loss(logits=logits, labels=labels)
    self.assertAlmostEqual(actual_score.numpy(), expected_score, places=1)

  @parameterized.parameters((None, 0, 0), ('generalized', 0, 0),
                            ('adaptive', 0, 0))
  def test_supported_loss_zero_labels_logits(self, metric_type, output,
                                             expected_score):
    loss = segmentation_losses.SegmentationLossDiceScore(
        metric_type=metric_type)
    logits = tf.constant(output, shape=[2, 128, 128, 128, 1], dtype=tf.float32)
    labels = tf.zeros(shape=[2, 128, 128, 128, 1], dtype=tf.float32)
    actual_score = loss(logits=logits, labels=labels)
    self.assertAlmostEqual(actual_score.numpy(), expected_score, places=1)


if __name__ == '__main__':
  tf.test.main()
