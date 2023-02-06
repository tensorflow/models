# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for Panoptic Quality metric.

Note that this metric test class is branched from
https://github.com/tensorflow/models/blob/master/research/deeplab/evaluation/panoptic_quality_test.py
"""


from absl.testing import absltest
import numpy as np
import tensorflow as tf

from official.vision.evaluation import panoptic_quality


class PanopticQualityTest(absltest.TestCase):

  def test_perfect_match(self):
    category_mask = np.zeros([6, 6], np.uint16)
    instance_mask = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 1],
        [1, 2, 2, 1, 1, 1],
        [1, 2, 1, 1, 1, 1],
    ],
                             dtype=np.uint16)

    groundtruths = {
        'category_mask': category_mask,
        'instance_mask': instance_mask
    }
    predictions = {
        'category_mask': category_mask,
        'instance_mask': instance_mask
    }
    pq_metric = panoptic_quality.PanopticQuality(
        num_categories=1,
        ignored_label=2,
        max_instances_per_category=16,
        offset=16)
    pq_metric.compare_and_accumulate(groundtruths, predictions)

    np.testing.assert_array_equal(pq_metric.iou_per_class, [2.0])
    np.testing.assert_array_equal(pq_metric.tp_per_class, [2])
    np.testing.assert_array_equal(pq_metric.fn_per_class, [0])
    np.testing.assert_array_equal(pq_metric.fp_per_class, [0])
    results = pq_metric.result()
    np.testing.assert_array_equal(results['pq_per_class'], [1.0])
    np.testing.assert_array_equal(results['rq_per_class'], [1.0])
    np.testing.assert_array_equal(results['sq_per_class'], [1.0])
    self.assertAlmostEqual(results['All_pq'], 1.0)
    self.assertAlmostEqual(results['All_rq'], 1.0)
    self.assertAlmostEqual(results['All_sq'], 1.0)
    self.assertEqual(results['All_num_categories'], 1)

  def test_totally_wrong(self):
    category_mask = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
                             dtype=np.uint16)
    instance_mask = np.zeros([6, 6], np.uint16)

    groundtruths = {
        'category_mask': category_mask,
        'instance_mask': instance_mask
    }
    predictions = {
        'category_mask': 1 - category_mask,
        'instance_mask': instance_mask
    }

    pq_metric = panoptic_quality.PanopticQuality(
        num_categories=2,
        ignored_label=2,
        max_instances_per_category=1,
        offset=16)
    pq_metric.compare_and_accumulate(groundtruths, predictions)
    np.testing.assert_array_equal(pq_metric.iou_per_class, [0.0, 0.0])
    np.testing.assert_array_equal(pq_metric.tp_per_class, [0, 0])
    np.testing.assert_array_equal(pq_metric.fn_per_class, [1, 1])
    np.testing.assert_array_equal(pq_metric.fp_per_class, [1, 1])
    results = pq_metric.result()
    np.testing.assert_array_equal(results['pq_per_class'], [0.0, 0.0])
    np.testing.assert_array_equal(results['rq_per_class'], [0.0, 0.0])
    np.testing.assert_array_equal(results['sq_per_class'], [0.0, 0.0])
    self.assertAlmostEqual(results['All_pq'], 0.0)
    self.assertAlmostEqual(results['All_rq'], 0.0)
    self.assertAlmostEqual(results['All_sq'], 0.0)
    self.assertEqual(results['All_num_categories'], 2)

  def test_matches_by_iou(self):
    groundtruth_instance_mask = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 2, 2, 2, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=np.uint16)

    good_det_instance_mask = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 2, 2, 2, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=np.uint16)

    groundtruths = {
        'category_mask': np.zeros_like(groundtruth_instance_mask),
        'instance_mask': groundtruth_instance_mask
    }
    predictions = {
        'category_mask': np.zeros_like(good_det_instance_mask),
        'instance_mask': good_det_instance_mask
    }
    pq_metric = panoptic_quality.PanopticQuality(
        num_categories=1,
        ignored_label=2,
        max_instances_per_category=16,
        offset=16)
    pq_metric.compare_and_accumulate(groundtruths, predictions)

    # iou(1, 1) = 28/30
    # iou(2, 2) = 6 / 8
    np.testing.assert_array_almost_equal(pq_metric.iou_per_class,
                                         [28 / 30 + 6 / 8])
    np.testing.assert_array_equal(pq_metric.tp_per_class, [2])
    np.testing.assert_array_equal(pq_metric.fn_per_class, [0])
    np.testing.assert_array_equal(pq_metric.fp_per_class, [0])
    results = pq_metric.result()
    np.testing.assert_array_equal(results['pq_per_class'],
                                  [(28 / 30 + 6 / 8) / 2])
    np.testing.assert_array_equal(results['rq_per_class'], [1.0])
    np.testing.assert_array_equal(results['sq_per_class'],
                                  [(28 / 30 + 6 / 8) / 2])
    self.assertAlmostEqual(results['All_pq'], (28 / 30 + 6 / 8) / 2)
    self.assertAlmostEqual(results['All_rq'], 1.0)
    self.assertAlmostEqual(results['All_sq'], (28 / 30 + 6 / 8) / 2)
    self.assertEqual(results['All_num_categories'], 1)

    bad_det_instance_mask = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 2, 2, 1],
            [1, 1, 1, 2, 2, 1],
            [1, 1, 1, 2, 2, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=np.uint16)
    predictions['instance_mask'] = bad_det_instance_mask

    pq_metric.reset()
    pq_metric.compare_and_accumulate(groundtruths, predictions)

    # iou(1, 1) = 27/32
    np.testing.assert_array_almost_equal(pq_metric.iou_per_class, [27 / 32])
    np.testing.assert_array_equal(pq_metric.tp_per_class, [1])
    np.testing.assert_array_equal(pq_metric.fn_per_class, [1])
    np.testing.assert_array_equal(pq_metric.fp_per_class, [1])
    results = pq_metric.result()
    np.testing.assert_array_equal(results['pq_per_class'], [27 / 32 / 2])
    np.testing.assert_array_equal(results['rq_per_class'], [0.5])
    np.testing.assert_array_equal(results['sq_per_class'], [27 / 32])
    self.assertAlmostEqual(results['All_pq'], 27 / 32 / 2)
    self.assertAlmostEqual(results['All_rq'], 0.5)
    self.assertAlmostEqual(results['All_sq'], 27 / 32)
    self.assertEqual(results['All_num_categories'], 1)

  def test_wrong_instances(self):
    category_mask = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 2, 2, 1, 2, 2],
        [1, 2, 2, 1, 2, 2],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ],
                             dtype=np.uint16)
    groundtruth_instance_mask = np.zeros([6, 6], dtype=np.uint16)
    predicted_instance_mask = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
                                       dtype=np.uint16)

    groundtruths = {
        'category_mask': category_mask,
        'instance_mask': groundtruth_instance_mask
    }
    predictions = {
        'category_mask': category_mask,
        'instance_mask': predicted_instance_mask
    }

    pq_metric = panoptic_quality.PanopticQuality(
        num_categories=3,
        ignored_label=0,
        max_instances_per_category=10,
        offset=100)
    pq_metric.compare_and_accumulate(groundtruths, predictions)

    np.testing.assert_array_equal(pq_metric.iou_per_class, [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(pq_metric.tp_per_class, [0, 1, 0])
    np.testing.assert_array_equal(pq_metric.fn_per_class, [0, 0, 1])
    np.testing.assert_array_equal(pq_metric.fp_per_class, [0, 0, 2])
    results = pq_metric.result()
    np.testing.assert_array_equal(results['pq_per_class'], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(results['rq_per_class'], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(results['sq_per_class'], [0.0, 1.0, 0.0])
    self.assertAlmostEqual(results['All_pq'], 0.5)
    self.assertAlmostEqual(results['All_rq'], 0.5)
    self.assertAlmostEqual(results['All_sq'], 0.5)
    self.assertEqual(results['All_num_categories'], 2)

  def test_instance_order_is_arbitrary(self):
    category_mask = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 2, 2, 1, 2, 2],
        [1, 2, 2, 1, 2, 2],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ],
                             dtype=np.uint16)
    groundtruth_instance_mask = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
                                         dtype=np.uint16)
    predicted_instance_mask = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
                                       dtype=np.uint16)

    groundtruths = {
        'category_mask': category_mask,
        'instance_mask': groundtruth_instance_mask
    }
    predictions = {
        'category_mask': category_mask,
        'instance_mask': predicted_instance_mask
    }

    pq_metric = panoptic_quality.PanopticQuality(
        num_categories=3,
        ignored_label=0,
        max_instances_per_category=10,
        offset=100)
    pq_metric.compare_and_accumulate(groundtruths, predictions)

    np.testing.assert_array_equal(pq_metric.iou_per_class, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(pq_metric.tp_per_class, [0, 1, 2])
    np.testing.assert_array_equal(pq_metric.fn_per_class, [0, 0, 0])
    np.testing.assert_array_equal(pq_metric.fp_per_class, [0, 0, 0])
    results = pq_metric.result()
    np.testing.assert_array_equal(results['pq_per_class'], [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(results['rq_per_class'], [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(results['sq_per_class'], [0.0, 1.0, 1.0])
    self.assertAlmostEqual(results['All_pq'], 1.0)
    self.assertAlmostEqual(results['All_rq'], 1.0)
    self.assertAlmostEqual(results['All_sq'], 1.0)
    self.assertEqual(results['All_num_categories'], 2)


class PanopticQualityV2Test(tf.test.TestCase):

  def test_perfect_match(self):
    panoptic_metrics = panoptic_quality.PanopticQualityV2(
        name='panoptic_metrics',
        num_categories=2,
    )
    y_true = {
        'category_mask': tf.ones([1, 6, 6], dtype=tf.int32),
        'instance_mask': [[
            [1, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 2, 2, 1, 1, 1],
            [1, 2, 1, 1, 1, 1],
        ]],
        'image_info': tf.constant(
            [[[6, 6], [6, 6], [1, 1], [0, 0]]], dtype=tf.float32
        ),
    }
    y_pred = y_true
    panoptic_metrics.update_state(y_true, y_pred)

    result = panoptic_metrics.result()
    self.assertAllEqual(result['valid_thing_classes'], [False, True])
    self.assertAllEqual(result['valid_stuff_classes'], [False, False])
    self.assertAllClose(result['sq_per_class'], [0.0, 1.0], atol=1e-4)
    self.assertAllClose(result['rq_per_class'], [0.0, 1.0], atol=1e-4)
    self.assertAllClose(result['pq_per_class'], [0.0, 1.0], atol=1e-4)

  def test_totally_wrong(self):
    panoptic_metrics = panoptic_quality.PanopticQualityV2(
        name='panoptic_metrics',
        num_categories=4,
    )
    y_true = {
        'category_mask': [[
            [1, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 2, 2, 1, 1, 1],
            [1, 2, 1, 1, 1, 1],
        ]],
        'instance_mask': [[
            [1, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 2, 2, 1, 1, 1],
            [1, 2, 1, 1, 1, 1],
        ]],
        'image_info': tf.constant(
            [[[6, 6], [6, 6], [1, 1], [0, 0]]], dtype=tf.float32
        ),
    }
    y_pred = {
        'category_mask': tf.constant(y_true['category_mask']) + 1,
        'instance_mask': y_true['instance_mask'],
    }
    panoptic_metrics.update_state(y_true, y_pred)
    result = panoptic_metrics.result()
    self.assertAllEqual(
        result['valid_thing_classes'], [False, True, True, True]
    )
    self.assertAllEqual(
        result['valid_stuff_classes'], [False, False, False, False]
    )
    self.assertAllClose(result['sq_per_class'], [0.0, 0.0, 0.0, 0.0], atol=1e-4)
    self.assertAllClose(result['rq_per_class'], [0.0, 0.0, 0.0, 0.0], atol=1e-4)
    self.assertAllClose(result['pq_per_class'], [0.0, 0.0, 0.0, 0.0], atol=1e-4)

  def test_matches_by_iou(self):
    panoptic_metrics = panoptic_quality.PanopticQualityV2(
        name='panoptic_metrics',
        num_categories=2,
    )
    y_true = {
        'category_mask': tf.ones([1, 6, 6], dtype=tf.int32),
        'instance_mask': [[
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 2, 2, 2, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]],
        'image_info': tf.constant(
            [[[6, 6], [6, 6], [1, 1], [0, 0]]], dtype=tf.float32
        ),
    }
    y_pred1 = {
        'category_mask': tf.ones([1, 6, 6], dtype=tf.int32),
        'instance_mask': [[
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 2, 2, 2, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]],
    }
    panoptic_metrics.update_state(y_true, y_pred1)
    result1 = panoptic_metrics.result()
    self.assertAllEqual(result1['valid_thing_classes'], [False, True])
    self.assertAllEqual(result1['valid_stuff_classes'], [False, False])
    self.assertAllClose(
        result1['sq_per_class'], [0.0, (28 / 30 + 6 / 8) / 2], atol=1e-4
    )
    self.assertAllClose(result1['rq_per_class'], [0.0, 1.0], atol=1e-4)
    self.assertAllClose(
        result1['pq_per_class'], [0.0, (28 / 30 + 6 / 8) / 2], atol=1e-4
    )

    panoptic_metrics.reset_state()
    y_pred2 = {
        'category_mask': tf.ones([1, 6, 6], dtype=tf.int32),
        'instance_mask': [[
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 2, 2, 1],
            [1, 1, 1, 2, 2, 1],
            [1, 1, 1, 2, 2, 1],
            [1, 1, 1, 1, 1, 1],
        ]],
    }
    panoptic_metrics.update_state(y_true, y_pred2)
    result2 = panoptic_metrics.result()
    self.assertAllEqual(result2['valid_thing_classes'], [False, True])
    self.assertAllEqual(result2['valid_stuff_classes'], [False, False])
    self.assertAllClose(result2['sq_per_class'], [0.0, 27 / 32], atol=1e-4)
    self.assertAllClose(result2['rq_per_class'], [0.0, 1 / 2], atol=1e-4)
    self.assertAllClose(result2['pq_per_class'], [0.0, 27 / 64], atol=1e-4)

  def test_thing_and_stuff(self):
    panoptic_metrics = panoptic_quality.PanopticQualityV2(
        name='panoptic_metrics',
        num_categories=10,
        is_thing=[
            False,
            True,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
        ],
        max_num_instances=15,
        ignored_label=255,
    )
    y_true = {
        'category_mask': [[
            [6, 6, 4, 6, 2, 5, 5],
            [6, 8, 4, 3, 2, 5, 5],
        ]],
        'instance_mask': [[
            [1, 1, 2, 5, 3, 0, 0],
            [1, 6, 2, 0, 4, 0, 0],
        ]],
        'image_info': tf.constant(
            [[[2, 7], [2, 7], [1, 1], [0, 0]]], dtype=tf.float32
        ),
    }
    y_pred = {
        'category_mask': [[
            [6, 4, 4, 6, 2, 255, 255],
            [6, 6, 4, 3, 255, 255, 7],
        ]],
        'instance_mask': [[
            [1, 2, 2, 5, 0, 0, 0],
            [1, 6, 2, 0, 0, 0, 0],
        ]],
    }
    panoptic_metrics.update_state(y_true, y_pred)
    result = panoptic_metrics.result()

    self.assertAllEqual(
        result['valid_thing_classes'],
        [False, False, True, False, True, False, True, False, True, False],
    )
    self.assertAllEqual(
        result['valid_stuff_classes'],
        [False, False, False, True, False, True, False, True, False, False],
    )
    self.assertAllClose(
        result['sq_per_class'],
        [0.0, 0.0, 0.0, 1.0, 0.666667, 0.0, 0.833333, 0.0, 0.0, 0.0],
        atol=1e-4,
    )
    self.assertAllClose(
        result['rq_per_class'],
        [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.8, 0.0, 0.0, 0.0],
        atol=1e-4,
    )
    self.assertAllClose(
        result['pq_per_class'],
        [0.0, 0.0, 0.0, 1.0, 0.666667, 0.0, 0.666667, 0.0, 0.0, 0.0],
        atol=1e-4,
    )


if __name__ == '__main__':
  absltest.main()
