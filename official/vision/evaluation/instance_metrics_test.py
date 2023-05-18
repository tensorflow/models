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

"""Tests for metrics.py."""

from absl.testing import parameterized
import tensorflow as tf

from official.vision.evaluation import instance_metrics


class InstanceMetricsTest(tf.test.TestCase, parameterized.TestCase):

  def test_compute_coco_ap(self):
    precisions = [1.0, 1.0, 0.5, 0.8, 0.4, 0.5, 0.2, 0.3]
    recalls = [0.0, 0.1, 0.1, 0.5, 0.5, 0.7, 0.7, 1.0]
    self.assertAllClose(
        instance_metrics.COCOAveragePrecision(recalls_desc=False)(
            precisions, recalls
        ),
        0.613861,
        atol=1e-4,
    )

    precisions.reverse()
    recalls.reverse()
    self.assertAllClose(
        instance_metrics.COCOAveragePrecision(recalls_desc=True)(
            precisions, recalls
        ),
        0.613861,
        atol=1e-4,
    )

  def test_compute_voc10_ap(self):
    precisions = [1.0, 1.0, 0.5, 0.8, 0.4, 0.5, 0.2, 0.3]
    recalls = [0.0, 0.1, 0.1, 0.5, 0.5, 0.7, 0.7, 1.0]
    self.assertAllClose(
        instance_metrics.VOC2010AveragePrecision(recalls_desc=False)(
            precisions, recalls
        ),
        0.61,
        atol=1e-4,
    )

    precisions.reverse()
    recalls.reverse()
    self.assertAllClose(
        instance_metrics.VOC2010AveragePrecision(recalls_desc=True)(
            precisions, recalls
        ),
        0.61,
        atol=1e-4,
    )

  def test_match_detections_to_gts(self):
    coco_matching_algorithm = instance_metrics.COCOMatchingAlgorithm(
        iou_thresholds=(0.5, 0.85)
    )

    detection_is_tp, gt_is_tp = coco_matching_algorithm(
        detection_to_gt_ious=tf.constant([[[0.8, 0.7, 0.95], [0.9, 0.6, 0.3]]]),
        detection_classes=tf.constant([[1, 1]]),
        detection_scores=tf.constant([[0.6, 0.8]]),
        gt_classes=tf.constant([[1, 1, 2]]),
    )
    self.assertAllEqual(detection_is_tp, [[[True, False], [True, True]]])
    self.assertAllEqual(
        gt_is_tp, [[[True, True], [True, False], [False, False]]]
    )

  def test_shift_and_rescale_boxes(self):
    self.assertAllClose(
        instance_metrics._shift_and_rescale_boxes(
            boxes=[[[2, 3, 4, 9], [15, 17, 18, 23]]], output_boundary=(20, 20)
        ),
        [[[0.0, 0.0, 2.0, 6.0], [13.0, 14.0, 16.0, 20.0]]],
        atol=1e-4,
    )
    self.assertAllClose(
        instance_metrics._shift_and_rescale_boxes(
            boxes=[[[-2, -1, 0, 5], [11, 13, 14, 19]]], output_boundary=(20, 20)
        ),
        [[[0.0, 0.0, 2.0, 6.0], [13.0, 14.0, 16.0, 20.0]]],
        atol=1e-4,
    )
    self.assertAllClose(
        instance_metrics._shift_and_rescale_boxes(
            boxes=[[[2, 3, 4, 9], [15, 17, 18, 23]]], output_boundary=(10, 10)
        ),
        [[[0.0, 0.0, 1.0, 3.0], [6.5, 7.0, 8.0, 10.0]]],
        atol=1e-4,
    )
    self.assertAllClose(
        instance_metrics._shift_and_rescale_boxes(
            boxes=[[[-2, -1, 0, 5], [11, 13, 14, 19]]], output_boundary=(10, 10)
        ),
        [[[0.0, 0.0, 1.0, 3.0], [6.5, 7.0, 8.0, 10.0]]],
        atol=1e-4,
    )
    self.assertAllClose(
        instance_metrics._shift_and_rescale_boxes(
            boxes=[[[2, 3, 4, 9], [-1, -1, -1, -1]]], output_boundary=(10, 10)
        ),
        [[[0.0, 0.0, 2.0, 6.0], [0.0, 0.0, 0.0, 0.0]]],
        atol=1e-4,
    )

  def test_count_detection_type(self):
    result = instance_metrics._count_detection_type(
        detection_type_mask=tf.constant(
            [[[True], [True], [False]], [[True], [True], [False]]]
        ),
        detection_classes=tf.constant([[1, 2, 3], [2, 3, 4]]),
        flattened_binned_confidence_one_hot=tf.constant([
            [False, True, False],
            [True, False, False],
            [False, True, False],
            [True, False, False],
            [False, False, True],
            [False, False, True],
        ]),
        num_classes=5,
    )
    self.assertAllClose(
        result,
        [[
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]],
        atol=1e-4,
    )

  @parameterized.parameters(True, False)
  def test_instance_metrics(self, use_mask):
    metrics = instance_metrics.InstanceMetrics(
        name='per_class_ap',
        num_classes=3,
        use_masks=use_mask,
        iou_thresholds=(0.1, 0.5),
        confidence_thresholds=(0.2, 0.7),
        mask_output_boundary=(32, 32),
        average_precision_algorithms={
            'ap_coco': instance_metrics.COCOAveragePrecision(),
            'ap_voc10': instance_metrics.VOC2010AveragePrecision(),
        },
    )
    y_true = {
        'boxes': [[
            [12, 12, 15, 15],
            [16, 16, 20, 20],
            [0, 0, 5, 5],
            [6, 6, 10, 10],
        ]],
        # 1x1 mask
        'masks': [[[[1.0]], [[0.9]], [[0.8]], [[0.7]]]],
        'classes': [[2, 1, 1, 1]],
        'image_info': tf.constant(
            [[[32, 32], [32, 32], [1, 1], [0, 0]]], dtype=tf.float32
        ),
    }
    y_pred = {
        'detection_boxes': [[
            [12, 12, 15, 15],
            # The duplicate detection with lower score won't be counted as TP.
            [12, 12, 15, 15],
            [16, 19, 20, 20],
            [1, 1, 6, 6],
            [6, 6, 11, 11],
        ]],
        # 1x1 mask
        'detection_masks': [[[[1.0]], [[0.9]], [[0.8]], [[0.7]], [[0.6]]]],
        'detection_classes': [[1, 1, 1, 2, 1]],
        'detection_scores': [[0.3, 0.25, 0.4, 0.6, 0.8]],
    }
    metrics.update_state(y_true, y_pred)
    result = metrics.result()
    self.assertAllClose(
        result['ap_coco'],
        [[0.0, 0.663366, 0.0], [0.0, 0.336634, 0.0]],
        atol=1e-4,
    )
    self.assertAllClose(
        result['ap_voc10'],
        [[0.0, 2.0 / 3.0, 0.0], [0.0, 1.0 / 3.0, 0.0]],
        atol=1e-4,
    )
    self.assertAllClose(
        result['precisions'],
        [
            [[0.0, 0.5, 0.0], [0.0, 0.25, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        ],
        atol=1e-4,
    )
    self.assertAllClose(
        result['recalls'],
        [
            [[0.0, 2.0 / 3.0, 0.0], [0.0, 1.0 / 3.0, 0.0]],
            [[0.0, 1.0 / 3, 0.0], [0.0, 1.0 / 3, 0.0]],
        ],
        atol=1e-4,
    )
    self.assertAllEqual(result['valid_classes'], [False, True, True])

  def test_mask_metrics_with_instance_rescaled(self):
    metrics = instance_metrics.InstanceMetrics(
        name='per_class_ap',
        use_masks=True,
        num_classes=3,
        iou_thresholds=(0.5,),
        confidence_thresholds=(0.5,),
        mask_output_boundary=(10, 10),
        average_precision_algorithms={
            'ap_coco': instance_metrics.COCOAveragePrecision(),
            'ap_voc10': instance_metrics.VOC2010AveragePrecision(),
        },
    )
    y_true = {
        # Instances are rescaled to (10, 10) boundary.
        'boxes': [[[0, 0, 8, 8], [10, 10, 20, 20]]],
        'masks': [[
            [
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 0],
                [1, 1, 1, 0],
            ],
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
            ],
        ]],
        'classes': [[1, 2]],
        'image_info': tf.constant(
            [[[20, 20], [20, 20], [1, 1], [0, 0]]], dtype=tf.float32
        ),
    }
    y_pred = {
        # Instances are rescaled to (10, 10) boundary.
        'detection_boxes': [[[0, 1, 8, 9], [10, 10, 20, 20]]],
        'detection_masks': [[
            [
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
        ]],
        'detection_classes': [[1, 2]],
        'detection_scores': [[0.9, 0.8]],
    }
    metrics.update_state(y_true, y_pred)
    result = metrics.result()
    self.assertAllClose(
        result['precisions'],
        [[[0.0, 1.0, 0.0]]],
        atol=1e-4,
    )
    self.assertAllClose(
        result['recalls'],
        [[[0.0, 1.0, 0.0]]],
        atol=1e-4,
    )
    self.assertAllClose(
        result['ap_coco'],
        [[0.0, 1.0, 0.0]],
        atol=1e-4,
    )
    self.assertAllClose(
        result['ap_voc10'],
        [[0.0, 1.0, 0.0]],
        atol=1e-4,
    )
    self.assertAllEqual(result['valid_classes'], [False, True, True])

  @parameterized.parameters(True, False)
  def test_instance_metrics_with_crowd(self, use_mask):
    metrics = instance_metrics.InstanceMetrics(
        name='per_class_ap',
        use_masks=use_mask,
        num_classes=2,
        iou_thresholds=(0.5,),
        confidence_thresholds=(0.5,),
        mask_output_boundary=(20, 20),
        average_precision_algorithms={
            'ap_coco': instance_metrics.COCOAveragePrecision(),
            'ap_voc10': instance_metrics.VOC2010AveragePrecision(),
        },
    )
    y_true = {
        'boxes': [[[0, 1, 4, 10], [0, 5, 4, 11]]],
        'masks': [[[[1]], [[1]]]],
        'classes': [[1, 1]],
        'image_info': tf.constant(
            [[[20, 20], [20, 20], [1, 1], [0, 0]]], dtype=tf.float32
        ),
        'is_crowds': [[True, False]],
    }
    y_pred = {
        # Over 50% of first box [0, 0, 4, 4] matches the crowd instance
        # [0, 1, 4, 10], so it's excluded from the false positives.
        'detection_boxes': [[[0, 0, 4, 4], [1, 5, 5, 11]]],
        'detection_masks': [[[[1]], [[1]]]],
        'detection_classes': [[1, 1]],
        'detection_scores': [[0.9, 0.8]],
    }
    metrics.update_state(y_true, y_pred)
    result = metrics.result()
    self.assertAllClose(
        result['precisions'],
        [[[0.0, 1.0]]],
        atol=1e-4,
    )
    self.assertAllClose(
        result['recalls'],
        [[[0.0, 1.0]]],
        atol=1e-4,
    )
    self.assertAllClose(
        result['ap_coco'],
        [[0.0, 1.0]],
        atol=1e-4,
    )
    self.assertAllClose(
        result['ap_voc10'],
        [[0.0, 1.0]],
        atol=1e-4,
    )
    self.assertAllEqual(result['valid_classes'], [False, True])


if __name__ == '__main__':
  tf.test.main()
