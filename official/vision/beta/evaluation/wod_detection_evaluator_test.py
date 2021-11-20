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

"""Tests for wod_detection_evaluator."""

import tensorflow as tf

from official.vision.beta.evaluation import wod_detection_evaluator


class WodDetectionEvaluatorTest(tf.test.TestCase):

  def _create_test_data(self):
    y_pred = {
        'source_id':
            tf.convert_to_tensor([1], dtype=tf.int64),
        'image_info':
            tf.convert_to_tensor([[[100, 100], [50, 50], [0.5, 0.5], [0, 0]]],
                                 dtype=tf.float32),
        'num_detections':
            tf.convert_to_tensor([4], dtype=tf.int64),
        'detection_boxes':
            tf.convert_to_tensor(
                [[[0.1, 0.15, 0.2, 0.25], [0.35, 0.18, 0.43, 0.4],
                  [0.2, 0.1, 0.3, 0.2], [0.65, 0.55, 0.75, 0.85]]],
                dtype=tf.float32),
        'detection_classes':
            tf.convert_to_tensor([[1, 1, 2, 2]], dtype=tf.int64),
        'detection_scores':
            tf.convert_to_tensor([[0.95, 0.5, 0.1, 0.7]], dtype=tf.float32)
    }

    y_true = {
        'source_id':
            tf.convert_to_tensor([1], dtype=tf.int64),
        'num_detections':
            tf.convert_to_tensor([4], dtype=tf.int64),
        'boxes':
            tf.convert_to_tensor([[[0.1, 0.15, 0.2, 0.25], [0.3, 0.2, 0.4, 0.3],
                                   [0.4, 0.3, 0.5, 0.6], [0.6, 0.5, 0.7, 0.8]]],
                                 dtype=tf.float32),
        'classes':
            tf.convert_to_tensor([[1, 1, 1, 2]], dtype=tf.int64),
        'difficulties':
            tf.zeros([1, 4], dtype=tf.int64)
    }
    return y_pred, y_true

  def test_wod_detection_evaluator(self):
    wod_detection_metric = wod_detection_evaluator.WOD2dDetectionEvaluator()
    y_pred, y_true = self._create_test_data()
    wod_detection_metric.update_state(groundtruths=y_true, predictions=y_pred)
    metrics = wod_detection_metric.evaluate()
    for _, metric_value in metrics.items():
      self.assertAlmostEqual(metric_value.numpy(), 0.0, places=3)


if __name__ == '__main__':
  tf.test.main()
