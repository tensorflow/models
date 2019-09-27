# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for object_detection.utils.np_box_mask_list_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from object_detection.utils import np_box_mask_list
from object_detection.utils import np_box_mask_list_ops


class AreaRelatedTest(tf.test.TestCase):

  def setUp(self):
    boxes1 = np.array([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]],
                      dtype=float)
    masks1_0 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 0, 0, 0, 0],
                         [1, 1, 1, 1, 0, 0, 0, 0]],
                        dtype=np.uint8)
    masks1_1 = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]],
                        dtype=np.uint8)
    masks1 = np.stack([masks1_0, masks1_1])
    boxes2 = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                       [0.0, 0.0, 20.0, 20.0]],
                      dtype=float)
    masks2_0 = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 0, 0, 0, 0],
                         [1, 1, 1, 1, 0, 0, 0, 0]],
                        dtype=np.uint8)
    masks2_1 = np.array([[1, 1, 1, 1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]],
                        dtype=np.uint8)
    masks2_2 = np.array([[1, 1, 1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 1, 1, 0, 0, 0],
                         [1, 1, 1, 1, 1, 0, 0, 0]],
                        dtype=np.uint8)
    masks2 = np.stack([masks2_0, masks2_1, masks2_2])
    self.box_mask_list1 = np_box_mask_list.BoxMaskList(
        box_data=boxes1, mask_data=masks1)
    self.box_mask_list2 = np_box_mask_list.BoxMaskList(
        box_data=boxes2, mask_data=masks2)

  def test_area(self):
    areas = np_box_mask_list_ops.area(self.box_mask_list1)
    expected_areas = np.array([8.0, 10.0], dtype=float)
    self.assertAllClose(expected_areas, areas)

  def test_intersection(self):
    intersection = np_box_mask_list_ops.intersection(self.box_mask_list1,
                                                     self.box_mask_list2)
    expected_intersection = np.array([[8.0, 0.0, 8.0], [0.0, 9.0, 7.0]],
                                     dtype=float)
    self.assertAllClose(intersection, expected_intersection)

  def test_iou(self):
    iou = np_box_mask_list_ops.iou(self.box_mask_list1, self.box_mask_list2)
    expected_iou = np.array(
        [[1.0, 0.0, 8.0 / 25.0], [0.0, 9.0 / 16.0, 7.0 / 28.0]], dtype=float)
    self.assertAllClose(iou, expected_iou)

  def test_ioa(self):
    ioa21 = np_box_mask_list_ops.ioa(self.box_mask_list1, self.box_mask_list2)
    expected_ioa21 = np.array([[1.0, 0.0, 8.0/25.0],
                               [0.0, 9.0/15.0, 7.0/25.0]],
                              dtype=np.float32)
    self.assertAllClose(ioa21, expected_ioa21)


class NonMaximumSuppressionTest(tf.test.TestCase):

  def setUp(self):
    boxes1 = np.array(
        [[4.0, 3.0, 7.0, 6.0], [5.0, 6.0, 10.0, 10.0]], dtype=float)
    boxes2 = np.array(
        [[3.0, 4.0, 6.0, 8.0], [5.0, 6.0, 10.0, 10.0], [1.0, 1.0, 10.0, 10.0]],
        dtype=float)
    masks1 = np.array(
        [[[0, 1, 0], [1, 1, 0], [0, 0, 0]], [[0, 1, 1], [0, 1, 1], [0, 1, 1]]],
        dtype=np.uint8)
    masks2 = np.array(
        [[[0, 1, 0], [1, 1, 1], [0, 0, 0]], [[0, 1, 0], [0, 0, 1], [0, 1, 1]],
         [[0, 1, 1], [0, 1, 1], [0, 1, 1]]],
        dtype=np.uint8)
    self.boxes1 = boxes1
    self.boxes2 = boxes2
    self.masks1 = masks1
    self.masks2 = masks2

  def test_with_no_scores_field(self):
    box_mask_list = np_box_mask_list.BoxMaskList(
        box_data=self.boxes1, mask_data=self.masks1)
    max_output_size = 3
    iou_threshold = 0.5

    with self.assertRaises(ValueError):
      np_box_mask_list_ops.non_max_suppression(
          box_mask_list, max_output_size, iou_threshold)

  def test_nms_disabled_max_output_size_equals_one(self):
    box_mask_list = np_box_mask_list.BoxMaskList(
        box_data=self.boxes2, mask_data=self.masks2)
    box_mask_list.add_field('scores',
                            np.array([.9, .75, .6], dtype=float))
    max_output_size = 1
    iou_threshold = 1.  # No NMS
    expected_boxes = np.array([[3.0, 4.0, 6.0, 8.0]], dtype=float)
    expected_masks = np.array(
        [[[0, 1, 0], [1, 1, 1], [0, 0, 0]]], dtype=np.uint8)
    nms_box_mask_list = np_box_mask_list_ops.non_max_suppression(
        box_mask_list, max_output_size, iou_threshold)
    self.assertAllClose(nms_box_mask_list.get(), expected_boxes)
    self.assertAllClose(nms_box_mask_list.get_masks(), expected_masks)

  def test_multiclass_nms(self):
    boxes = np.array(
        [[0.2, 0.4, 0.8, 0.8], [0.4, 0.2, 0.8, 0.8], [0.6, 0.0, 1.0, 1.0]],
        dtype=np.float32)
    mask0 = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0],
                      [0, 0, 1, 1, 0],
                      [0, 0, 1, 1, 0],
                      [0, 0, 0, 0, 0]],
                     dtype=np.uint8)
    mask1 = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 1, 1, 1, 0],
                      [0, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0]],
                     dtype=np.uint8)
    mask2 = np.array([[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1]],
                     dtype=np.uint8)
    masks = np.stack([mask0, mask1, mask2])
    box_mask_list = np_box_mask_list.BoxMaskList(
        box_data=boxes, mask_data=masks)
    scores = np.array([[-0.2, 0.1, 0.5, -0.4, 0.3],
                       [0.7, -0.7, 0.6, 0.2, -0.9],
                       [0.4, 0.34, -0.9, 0.2, 0.31]],
                      dtype=np.float32)
    box_mask_list.add_field('scores', scores)
    box_mask_list_clean = np_box_mask_list_ops.multi_class_non_max_suppression(
        box_mask_list, score_thresh=0.25, iou_thresh=0.1, max_output_size=3)

    scores_clean = box_mask_list_clean.get_field('scores')
    classes_clean = box_mask_list_clean.get_field('classes')
    boxes = box_mask_list_clean.get()
    masks = box_mask_list_clean.get_masks()
    expected_scores = np.array([0.7, 0.6, 0.34, 0.31])
    expected_classes = np.array([0, 2, 1, 4])
    expected_boxes = np.array([[0.4, 0.2, 0.8, 0.8],
                               [0.4, 0.2, 0.8, 0.8],
                               [0.6, 0.0, 1.0, 1.0],
                               [0.6, 0.0, 1.0, 1.0]],
                              dtype=np.float32)
    self.assertAllClose(scores_clean, expected_scores)
    self.assertAllClose(classes_clean, expected_classes)
    self.assertAllClose(boxes, expected_boxes)


if __name__ == '__main__':
  tf.test.main()
