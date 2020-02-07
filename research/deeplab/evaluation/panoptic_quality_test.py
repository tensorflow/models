# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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
"""Tests for Panoptic Quality metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from absl.testing import absltest
import numpy as np
import six

from deeplab.evaluation import panoptic_quality
from deeplab.evaluation import test_utils

# See the definition of the color names at:
#   https://en.wikipedia.org/wiki/Web_colors.
_CLASS_COLOR_MAP = {
    (0, 0, 0): 0,
    (0, 0, 255): 1,  # Person (blue).
    (255, 0, 0): 2,  # Bear (red).
    (0, 255, 0): 3,  # Tree (lime).
    (255, 0, 255): 4,  # Bird (fuchsia).
    (0, 255, 255): 5,  # Sky (aqua).
    (255, 255, 0): 6,  # Cat (yellow).
}


class PanopticQualityTest(absltest.TestCase):

  def test_perfect_match(self):
    categories = np.zeros([6, 6], np.uint16)
    instances = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 1],
        [1, 2, 2, 1, 1, 1],
        [1, 2, 1, 1, 1, 1],
    ],
                         dtype=np.uint16)

    pq = panoptic_quality.PanopticQuality(
        num_categories=1,
        ignored_label=2,
        max_instances_per_category=16,
        offset=16)
    pq.compare_and_accumulate(categories, instances, categories, instances)
    np.testing.assert_array_equal(pq.iou_per_class, [2.0])
    np.testing.assert_array_equal(pq.tp_per_class, [2])
    np.testing.assert_array_equal(pq.fn_per_class, [0])
    np.testing.assert_array_equal(pq.fp_per_class, [0])
    np.testing.assert_array_equal(pq.result_per_category(), [1.0])
    self.assertEqual(pq.result(), 1.0)

  def test_totally_wrong(self):
    det_categories = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
                              dtype=np.uint16)
    gt_categories = 1 - det_categories
    instances = np.zeros([6, 6], np.uint16)

    pq = panoptic_quality.PanopticQuality(
        num_categories=2,
        ignored_label=2,
        max_instances_per_category=1,
        offset=16)
    pq.compare_and_accumulate(gt_categories, instances, det_categories,
                              instances)
    np.testing.assert_array_equal(pq.iou_per_class, [0.0, 0.0])
    np.testing.assert_array_equal(pq.tp_per_class, [0, 0])
    np.testing.assert_array_equal(pq.fn_per_class, [1, 1])
    np.testing.assert_array_equal(pq.fp_per_class, [1, 1])
    np.testing.assert_array_equal(pq.result_per_category(), [0.0, 0.0])
    self.assertEqual(pq.result(), 0.0)

  def test_matches_by_iou(self):
    good_det_labels = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 2, 2, 2, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=np.uint16)
    gt_labels = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 2, 2, 2, 1],
            [1, 2, 2, 2, 2, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=np.uint16)

    pq = panoptic_quality.PanopticQuality(
        num_categories=1,
        ignored_label=2,
        max_instances_per_category=16,
        offset=16)
    pq.compare_and_accumulate(
        np.zeros_like(gt_labels), gt_labels, np.zeros_like(good_det_labels),
        good_det_labels)

    # iou(1, 1) = 28/30
    # iou(2, 2) = 6/8
    np.testing.assert_array_almost_equal(pq.iou_per_class, [28 / 30 + 6 / 8])
    np.testing.assert_array_equal(pq.tp_per_class, [2])
    np.testing.assert_array_equal(pq.fn_per_class, [0])
    np.testing.assert_array_equal(pq.fp_per_class, [0])
    self.assertAlmostEqual(pq.result(), (28 / 30 + 6 / 8) / 2)

    bad_det_labels = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 2, 2, 1],
            [1, 1, 1, 2, 2, 1],
            [1, 1, 1, 2, 2, 1],
            [1, 1, 1, 1, 1, 1],
        ],
        dtype=np.uint16)

    pq.reset()
    pq.compare_and_accumulate(
        np.zeros_like(gt_labels), gt_labels, np.zeros_like(bad_det_labels),
        bad_det_labels)

    # iou(1, 1) = 27/32
    np.testing.assert_array_almost_equal(pq.iou_per_class, [27 / 32])
    np.testing.assert_array_equal(pq.tp_per_class, [1])
    np.testing.assert_array_equal(pq.fn_per_class, [1])
    np.testing.assert_array_equal(pq.fp_per_class, [1])
    self.assertAlmostEqual(pq.result(), (27 / 32) * (1 / 2))

  def test_wrong_instances(self):
    categories = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 2, 2, 1, 2, 2],
        [1, 2, 2, 1, 2, 2],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ],
                          dtype=np.uint16)
    predicted_instances = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
                                   dtype=np.uint16)
    groundtruth_instances = np.zeros([6, 6], dtype=np.uint16)

    pq = panoptic_quality.PanopticQuality(
        num_categories=3,
        ignored_label=0,
        max_instances_per_category=10,
        offset=100)
    pq.compare_and_accumulate(categories, groundtruth_instances, categories,
                              predicted_instances)

    np.testing.assert_array_equal(pq.iou_per_class, [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(pq.tp_per_class, [0, 1, 0])
    np.testing.assert_array_equal(pq.fn_per_class, [0, 0, 1])
    np.testing.assert_array_equal(pq.fp_per_class, [0, 0, 2])
    np.testing.assert_array_equal(pq.result_per_category(), [0, 1, 0])
    self.assertAlmostEqual(pq.result(), 0.5)

  def test_instance_order_is_arbitrary(self):
    categories = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 2, 2, 1, 2, 2],
        [1, 2, 2, 1, 2, 2],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ],
                          dtype=np.uint16)
    predicted_instances = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
                                   dtype=np.uint16)
    groundtruth_instances = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
                                     dtype=np.uint16)

    pq = panoptic_quality.PanopticQuality(
        num_categories=3,
        ignored_label=0,
        max_instances_per_category=10,
        offset=100)
    pq.compare_and_accumulate(categories, groundtruth_instances, categories,
                              predicted_instances)

    np.testing.assert_array_equal(pq.iou_per_class, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(pq.tp_per_class, [0, 1, 2])
    np.testing.assert_array_equal(pq.fn_per_class, [0, 0, 0])
    np.testing.assert_array_equal(pq.fp_per_class, [0, 0, 0])
    np.testing.assert_array_equal(pq.result_per_category(), [0, 1, 1])
    self.assertAlmostEqual(pq.result(), 1.0)

  def test_matches_expected(self):
    pred_classes = test_utils.read_segmentation_with_rgb_color_map(
        'team_pred_class.png', _CLASS_COLOR_MAP)
    pred_instances = test_utils.read_test_image(
        'team_pred_instance.png', mode='L')

    instance_class_map = {
        0: 0,
        47: 1,
        97: 1,
        133: 1,
        150: 1,
        174: 1,
        198: 2,
        215: 1,
        244: 1,
        255: 1,
    }
    gt_instances, gt_classes = test_utils.panoptic_segmentation_with_class_map(
        'team_gt_instance.png', instance_class_map)

    pq = panoptic_quality.PanopticQuality(
        num_categories=3,
        ignored_label=0,
        max_instances_per_category=256,
        offset=256 * 256)
    pq.compare_and_accumulate(gt_classes, gt_instances, pred_classes,
                              pred_instances)
    np.testing.assert_array_almost_equal(
        pq.iou_per_class, [2.06104, 5.26827, 0.54069], decimal=4)
    np.testing.assert_array_equal(pq.tp_per_class, [1, 7, 1])
    np.testing.assert_array_equal(pq.fn_per_class, [0, 1, 0])
    np.testing.assert_array_equal(pq.fp_per_class, [0, 0, 0])
    np.testing.assert_array_almost_equal(pq.result_per_category(),
                                         [2.061038, 0.702436, 0.54069])
    self.assertAlmostEqual(pq.result(), 0.62156287)

  def test_merge_accumulates_all_across_instances(self):
    categories = np.zeros([6, 6], np.uint16)
    good_det_labels = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 2, 2, 2, 2, 1],
        [1, 2, 2, 2, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ],
                               dtype=np.uint16)
    gt_labels = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 2, 2, 2, 1],
        [1, 2, 2, 2, 2, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ],
                         dtype=np.uint16)

    good_pq = panoptic_quality.PanopticQuality(
        num_categories=1,
        ignored_label=2,
        max_instances_per_category=16,
        offset=16)
    for _ in six.moves.range(2):
      good_pq.compare_and_accumulate(categories, gt_labels, categories,
                                     good_det_labels)

    bad_det_labels = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 2, 2, 1],
        [1, 1, 1, 2, 2, 1],
        [1, 1, 1, 2, 2, 1],
        [1, 1, 1, 1, 1, 1],
    ],
                              dtype=np.uint16)

    bad_pq = panoptic_quality.PanopticQuality(
        num_categories=1,
        ignored_label=2,
        max_instances_per_category=16,
        offset=16)
    for _ in six.moves.range(2):
      bad_pq.compare_and_accumulate(categories, gt_labels, categories,
                                    bad_det_labels)

    good_pq.merge(bad_pq)

    np.testing.assert_array_almost_equal(
        good_pq.iou_per_class, [2 * (28 / 30 + 6 / 8) + 2 * (27 / 32)])
    np.testing.assert_array_equal(good_pq.tp_per_class, [2 * 2 + 2])
    np.testing.assert_array_equal(good_pq.fn_per_class, [2])
    np.testing.assert_array_equal(good_pq.fp_per_class, [2])
    self.assertAlmostEqual(good_pq.result(), 0.63177083)


if __name__ == '__main__':
  absltest.main()
