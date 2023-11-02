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

"""Tests for segmentation_losses.py."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.projects.volumetric_models.evaluation import segmentation_metrics


class SegmentationMetricsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters((1, 'generalized', 0.5, [0.67, 0.67]),
                            (1, 'adaptive', 0.5, [0.93, 0.67]),
                            (2, None, 0.5, [0.67, 0.67, 0.67]),
                            (3, 'generalized', 0.5, [0.67, 0.67, 0.67, 0.67]))
  def test_forward_dice_score(self, num_classes, metric_type, output,
                              expected_score):
    metric = segmentation_metrics.DiceScore(
        num_classes=num_classes, metric_type=metric_type, per_class_metric=True)
    y_pred = tf.constant(
        output, shape=[2, 128, 128, 128, num_classes], dtype=tf.float32)
    y_true = tf.ones(shape=[2, 128, 128, 128, num_classes], dtype=tf.float32)
    metric.update_state(y_true=y_true, y_pred=y_pred)
    actual_score = metric.result().numpy()
    self.assertAllClose(
        actual_score,
        expected_score,
        atol=1e-2,
        msg='Output metric {} does not match expected metric {}.'.format(
            actual_score, expected_score))

  def test_num_classes_not_equal(self):
    metric = segmentation_metrics.DiceScore(num_classes=4)
    y_pred = tf.constant(0.5, shape=[2, 128, 128, 128, 2], dtype=tf.float32)
    y_true = tf.ones(shape=[2, 128, 128, 128, 2], dtype=tf.float32)
    with self.assertRaisesRegex(
        ValueError,
        'The number of classes from groundtruth labels and `num_classes` '
        'should equal'):
      metric.update_state(y_true=y_true, y_pred=y_pred)


if __name__ == '__main__':
  tf.test.main()
