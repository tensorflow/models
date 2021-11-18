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

"""Tests for segmentation_metrics."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.vision.beta.evaluation import segmentation_metrics


class SegmentationMetricsTest(parameterized.TestCase, tf.test.TestCase):

  def _create_test_data(self):
    y_pred_cls0 = np.expand_dims(
        np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint16),
        axis=(0, -1))
    y_pred_cls1 = np.expand_dims(
        np.array([[0, 0, 0], [0, 0, 1], [0, 0, 1]], dtype=np.uint16),
        axis=(0, -1))
    y_pred = np.concatenate((y_pred_cls0, y_pred_cls1), axis=-1)

    y_true = {
        'masks':
            np.expand_dims(
                np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1],
                          [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]],
                         dtype=np.uint16),
                axis=(0, -1)),
        'valid_masks':
            np.ones([1, 6, 6, 1], dtype=np.uint16),
        'image_info':
            np.array([[[6, 6], [3, 3], [0.5, 0.5], [0, 0]]], dtype=np.float32)
    }
    return y_pred, y_true

  @parameterized.parameters(True, False)
  def test_mean_iou_metric(self, rescale_predictions):
    tf.config.experimental_run_functions_eagerly(True)
    mean_iou_metric = segmentation_metrics.MeanIoU(
        num_classes=2, rescale_predictions=rescale_predictions)
    y_pred, y_true = self._create_test_data()
    # Disable autograph for correct coverage statistics.
    update_fn = tf.autograph.experimental.do_not_convert(
        mean_iou_metric.update_state)
    update_fn(y_true=y_true, y_pred=y_pred)
    miou = mean_iou_metric.result()
    self.assertAlmostEqual(miou.numpy(), 0.762, places=3)

  @parameterized.parameters(True, False)
  def test_per_class_mean_iou_metric(self, rescale_predictions):
    per_class_iou_metric = segmentation_metrics.PerClassIoU(
        num_classes=2, rescale_predictions=rescale_predictions)
    y_pred, y_true = self._create_test_data()
    # Disable autograph for correct coverage statistics.
    update_fn = tf.autograph.experimental.do_not_convert(
        per_class_iou_metric.update_state)
    update_fn(y_true=y_true, y_pred=y_pred)
    per_class_miou = per_class_iou_metric.result()
    self.assertAllClose(per_class_miou.numpy(), [0.857, 0.667], atol=1e-3)


if __name__ == '__main__':
  tf.test.main()
