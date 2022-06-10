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


if __name__ == '__main__':
  absltest.main()
