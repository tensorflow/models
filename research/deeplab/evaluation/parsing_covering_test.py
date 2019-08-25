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
"""Tests for Parsing Covering metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from absl.testing import absltest
import numpy as np

from deeplab.evaluation import parsing_covering
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


class CoveringConveringTest(absltest.TestCase):

  def test_perfect_match(self):
    categories = np.zeros([6, 6], np.uint16)
    instances = np.array([
        [2, 2, 2, 2, 2, 2],
        [2, 4, 4, 4, 4, 2],
        [2, 4, 4, 4, 4, 2],
        [2, 4, 4, 4, 4, 2],
        [2, 4, 4, 2, 2, 2],
        [2, 4, 2, 2, 2, 2],
    ],
                         dtype=np.uint16)

    pc = parsing_covering.ParsingCovering(
        num_categories=3,
        ignored_label=2,
        max_instances_per_category=2,
        offset=16,
        normalize_by_image_size=False)
    pc.compare_and_accumulate(categories, instances, categories, instances)
    np.testing.assert_array_equal(pc.weighted_iou_per_class, [0.0, 21.0, 0.0])
    np.testing.assert_array_equal(pc.gt_area_per_class, [0.0, 21.0, 0.0])
    np.testing.assert_array_equal(pc.result_per_category(), [0.0, 1.0, 0.0])
    self.assertEqual(pc.result(), 1.0)

  def test_totally_wrong(self):
    categories = np.zeros([6, 6], np.uint16)
    gt_instances = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
                            dtype=np.uint16)
    pred_instances = 1 - gt_instances

    pc = parsing_covering.ParsingCovering(
        num_categories=2,
        ignored_label=0,
        max_instances_per_category=1,
        offset=16,
        normalize_by_image_size=False)
    pc.compare_and_accumulate(categories, gt_instances, categories,
                              pred_instances)
    np.testing.assert_array_equal(pc.weighted_iou_per_class, [0.0, 0.0])
    np.testing.assert_array_equal(pc.gt_area_per_class, [0.0, 10.0])
    np.testing.assert_array_equal(pc.result_per_category(), [0.0, 0.0])
    self.assertEqual(pc.result(), 0.0)

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

    pc = parsing_covering.ParsingCovering(
        num_categories=3,
        ignored_label=0,
        max_instances_per_category=256,
        offset=256 * 256,
        normalize_by_image_size=False)
    pc.compare_and_accumulate(gt_classes, gt_instances, pred_classes,
                              pred_instances)
    np.testing.assert_array_almost_equal(
        pc.weighted_iou_per_class, [0.0, 39864.14634, 3136], decimal=4)
    np.testing.assert_array_equal(pc.gt_area_per_class, [0.0, 56870, 5800])
    np.testing.assert_array_almost_equal(
        pc.result_per_category(), [0.0, 0.70097, 0.54069], decimal=4)
    self.assertAlmostEqual(pc.result(), 0.6208296732)

  def test_matches_expected_normalize_by_size(self):
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

    pc = parsing_covering.ParsingCovering(
        num_categories=3,
        ignored_label=0,
        max_instances_per_category=256,
        offset=256 * 256,
        normalize_by_image_size=True)
    pc.compare_and_accumulate(gt_classes, gt_instances, pred_classes,
                              pred_instances)
    np.testing.assert_array_almost_equal(
        pc.weighted_iou_per_class, [0.0, 0.5002088756, 0.03935002196],
        decimal=4)
    np.testing.assert_array_almost_equal(
        pc.gt_area_per_class, [0.0, 0.7135955832, 0.07277746408], decimal=4)
    # Note that the per-category and overall PCs are identical to those without
    # normalization in the previous test, because we only have a single image.
    np.testing.assert_array_almost_equal(
        pc.result_per_category(), [0.0, 0.70097, 0.54069], decimal=4)
    self.assertAlmostEqual(pc.result(), 0.6208296732)


if __name__ == '__main__':
  absltest.main()
