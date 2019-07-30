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

"""Tests for object_detection.utils.np_box_list_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from object_detection.utils import np_box_list
from object_detection.utils import np_box_list_ops


class AreaRelatedTest(tf.test.TestCase):

  def setUp(self):
    boxes1 = np.array([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]],
                      dtype=float)
    boxes2 = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                       [0.0, 0.0, 20.0, 20.0]],
                      dtype=float)
    self.boxlist1 = np_box_list.BoxList(boxes1)
    self.boxlist2 = np_box_list.BoxList(boxes2)

  def test_area(self):
    areas = np_box_list_ops.area(self.boxlist1)
    expected_areas = np.array([6.0, 5.0], dtype=float)
    self.assertAllClose(expected_areas, areas)

  def test_intersection(self):
    intersection = np_box_list_ops.intersection(self.boxlist1, self.boxlist2)
    expected_intersection = np.array([[2.0, 0.0, 6.0], [1.0, 0.0, 5.0]],
                                     dtype=float)
    self.assertAllClose(intersection, expected_intersection)

  def test_iou(self):
    iou = np_box_list_ops.iou(self.boxlist1, self.boxlist2)
    expected_iou = np.array([[2.0 / 16.0, 0.0, 6.0 / 400.0],
                             [1.0 / 16.0, 0.0, 5.0 / 400.0]],
                            dtype=float)
    self.assertAllClose(iou, expected_iou)

  def test_ioa(self):
    boxlist1 = np_box_list.BoxList(
        np.array(
            [[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 0.5, 0.75]], dtype=
            np.float32))
    boxlist2 = np_box_list.BoxList(
        np.array(
            [[0.5, 0.25, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32))
    ioa21 = np_box_list_ops.ioa(boxlist2, boxlist1)
    expected_ioa21 = np.array([[0.5, 0.0],
                               [1.0, 1.0]],
                              dtype=np.float32)
    self.assertAllClose(ioa21, expected_ioa21)

  def test_scale(self):
    boxlist = np_box_list.BoxList(
        np.array(
            [[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 0.5, 0.75]], dtype=
            np.float32))
    boxlist_scaled = np_box_list_ops.scale(boxlist, 2.0, 3.0)
    expected_boxlist_scaled = np_box_list.BoxList(
        np.array(
            [[0.5, 0.75, 1.5, 2.25], [0.0, 0.0, 1.0, 2.25]], dtype=np.float32))
    self.assertAllClose(expected_boxlist_scaled.get(), boxlist_scaled.get())

  def test_clip_to_window(self):
    boxlist = np_box_list.BoxList(
        np.array(
            [[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 0.5, 0.75],
             [-0.2, -0.3, 0.7, 1.5]],
            dtype=np.float32))
    boxlist_clipped = np_box_list_ops.clip_to_window(boxlist,
                                                     [0.0, 0.0, 1.0, 1.0])
    expected_boxlist_clipped = np_box_list.BoxList(
        np.array(
            [[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 0.5, 0.75],
             [0.0, 0.0, 0.7, 1.0]],
            dtype=np.float32))
    self.assertAllClose(expected_boxlist_clipped.get(), boxlist_clipped.get())

  def test_prune_outside_window(self):
    boxlist = np_box_list.BoxList(
        np.array(
            [[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 0.5, 0.75],
             [-0.2, -0.3, 0.7, 1.5]],
            dtype=np.float32))
    boxlist_pruned, _ = np_box_list_ops.prune_outside_window(
        boxlist, [0.0, 0.0, 1.0, 1.0])
    expected_boxlist_pruned = np_box_list.BoxList(
        np.array(
            [[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 0.5, 0.75]], dtype=
            np.float32))
    self.assertAllClose(expected_boxlist_pruned.get(), boxlist_pruned.get())

  def test_concatenate(self):
    boxlist1 = np_box_list.BoxList(
        np.array(
            [[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 0.5, 0.75]], dtype=
            np.float32))
    boxlist2 = np_box_list.BoxList(
        np.array(
            [[0.5, 0.25, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32))
    boxlists = [boxlist1, boxlist2]
    boxlist_concatenated = np_box_list_ops.concatenate(boxlists)
    boxlist_concatenated_expected = np_box_list.BoxList(
        np.array(
            [[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 0.5, 0.75],
             [0.5, 0.25, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]],
            dtype=np.float32))
    self.assertAllClose(boxlist_concatenated_expected.get(),
                        boxlist_concatenated.get())

  def test_change_coordinate_frame(self):
    boxlist = np_box_list.BoxList(
        np.array(
            [[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 0.5, 0.75]], dtype=
            np.float32))
    boxlist_coord = np_box_list_ops.change_coordinate_frame(
        boxlist, np.array([0, 0, 0.5, 0.5], dtype=np.float32))
    expected_boxlist_coord = np_box_list.BoxList(
        np.array([[0.5, 0.5, 1.5, 1.5], [0, 0, 1.0, 1.5]], dtype=np.float32))
    self.assertAllClose(boxlist_coord.get(), expected_boxlist_coord.get())

  def test_filter_scores_greater_than(self):
    boxlist = np_box_list.BoxList(
        np.array(
            [[0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 0.5, 0.75]], dtype=
            np.float32))
    boxlist.add_field('scores', np.array([0.8, 0.2], np.float32))
    boxlist_greater = np_box_list_ops.filter_scores_greater_than(boxlist, 0.5)

    expected_boxlist_greater = np_box_list.BoxList(
        np.array([[0.25, 0.25, 0.75, 0.75]], dtype=np.float32))

    self.assertAllClose(boxlist_greater.get(), expected_boxlist_greater.get())


class GatherOpsTest(tf.test.TestCase):

  def setUp(self):
    boxes = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                      [0.0, 0.0, 20.0, 20.0]],
                     dtype=float)
    self.boxlist = np_box_list.BoxList(boxes)
    self.boxlist.add_field('scores', np.array([0.5, 0.7, 0.9], dtype=float))
    self.boxlist.add_field('labels',
                           np.array([[0, 0, 0, 1, 0], [0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1]],
                                    dtype=int))

  def test_gather_with_out_of_range_indices(self):
    indices = np.array([3, 1], dtype=int)
    boxlist = self.boxlist
    with self.assertRaises(ValueError):
      np_box_list_ops.gather(boxlist, indices)

  def test_gather_with_invalid_multidimensional_indices(self):
    indices = np.array([[0, 1], [1, 2]], dtype=int)
    boxlist = self.boxlist
    with self.assertRaises(ValueError):
      np_box_list_ops.gather(boxlist, indices)

  def test_gather_without_fields_specified(self):
    indices = np.array([2, 0, 1], dtype=int)
    boxlist = self.boxlist
    subboxlist = np_box_list_ops.gather(boxlist, indices)

    expected_scores = np.array([0.9, 0.5, 0.7], dtype=float)
    self.assertAllClose(expected_scores, subboxlist.get_field('scores'))

    expected_boxes = np.array([[0.0, 0.0, 20.0, 20.0], [3.0, 4.0, 6.0, 8.0],
                               [14.0, 14.0, 15.0, 15.0]],
                              dtype=float)
    self.assertAllClose(expected_boxes, subboxlist.get())

    expected_labels = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 1, 0],
                                [0, 1, 0, 0, 0]],
                               dtype=int)
    self.assertAllClose(expected_labels, subboxlist.get_field('labels'))

  def test_gather_with_invalid_field_specified(self):
    indices = np.array([2, 0, 1], dtype=int)
    boxlist = self.boxlist

    with self.assertRaises(ValueError):
      np_box_list_ops.gather(boxlist, indices, 'labels')

    with self.assertRaises(ValueError):
      np_box_list_ops.gather(boxlist, indices, ['objectness'])

  def test_gather_with_fields_specified(self):
    indices = np.array([2, 0, 1], dtype=int)
    boxlist = self.boxlist
    subboxlist = np_box_list_ops.gather(boxlist, indices, ['labels'])

    self.assertFalse(subboxlist.has_field('scores'))

    expected_boxes = np.array([[0.0, 0.0, 20.0, 20.0], [3.0, 4.0, 6.0, 8.0],
                               [14.0, 14.0, 15.0, 15.0]],
                              dtype=float)
    self.assertAllClose(expected_boxes, subboxlist.get())

    expected_labels = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 1, 0],
                                [0, 1, 0, 0, 0]],
                               dtype=int)
    self.assertAllClose(expected_labels, subboxlist.get_field('labels'))


class SortByFieldTest(tf.test.TestCase):

  def setUp(self):
    boxes = np.array([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                      [0.0, 0.0, 20.0, 20.0]],
                     dtype=float)
    self.boxlist = np_box_list.BoxList(boxes)
    self.boxlist.add_field('scores', np.array([0.5, 0.9, 0.4], dtype=float))
    self.boxlist.add_field('labels',
                           np.array([[0, 0, 0, 1, 0], [0, 1, 0, 0, 0],
                                     [0, 0, 0, 0, 1]],
                                    dtype=int))

  def test_with_invalid_field(self):
    with self.assertRaises(ValueError):
      np_box_list_ops.sort_by_field(self.boxlist, 'objectness')
    with self.assertRaises(ValueError):
      np_box_list_ops.sort_by_field(self.boxlist, 'labels')

  def test_with_invalid_sorting_order(self):
    with self.assertRaises(ValueError):
      np_box_list_ops.sort_by_field(self.boxlist, 'scores', 'Descending')

  def test_with_descending_sorting(self):
    sorted_boxlist = np_box_list_ops.sort_by_field(self.boxlist, 'scores')

    expected_boxes = np.array([[14.0, 14.0, 15.0, 15.0], [3.0, 4.0, 6.0, 8.0],
                               [0.0, 0.0, 20.0, 20.0]],
                              dtype=float)
    self.assertAllClose(expected_boxes, sorted_boxlist.get())

    expected_scores = np.array([0.9, 0.5, 0.4], dtype=float)
    self.assertAllClose(expected_scores, sorted_boxlist.get_field('scores'))

  def test_with_ascending_sorting(self):
    sorted_boxlist = np_box_list_ops.sort_by_field(
        self.boxlist, 'scores', np_box_list_ops.SortOrder.ASCEND)

    expected_boxes = np.array([[0.0, 0.0, 20.0, 20.0],
                               [3.0, 4.0, 6.0, 8.0],
                               [14.0, 14.0, 15.0, 15.0],],
                              dtype=float)
    self.assertAllClose(expected_boxes, sorted_boxlist.get())

    expected_scores = np.array([0.4, 0.5, 0.9], dtype=float)
    self.assertAllClose(expected_scores, sorted_boxlist.get_field('scores'))


class NonMaximumSuppressionTest(tf.test.TestCase):

  def setUp(self):
    self._boxes = np.array([[0, 0, 1, 1],
                            [0, 0.1, 1, 1.1],
                            [0, -0.1, 1, 0.9],
                            [0, 10, 1, 11],
                            [0, 10.1, 1, 11.1],
                            [0, 100, 1, 101]],
                           dtype=float)
    self._boxlist = np_box_list.BoxList(self._boxes)

  def test_with_no_scores_field(self):
    boxlist = np_box_list.BoxList(self._boxes)
    max_output_size = 3
    iou_threshold = 0.5

    with self.assertRaises(ValueError):
      np_box_list_ops.non_max_suppression(
          boxlist, max_output_size, iou_threshold)

  def test_nms_disabled_max_output_size_equals_three(self):
    boxlist = np_box_list.BoxList(self._boxes)
    boxlist.add_field('scores',
                      np.array([.9, .75, .6, .95, .2, .3], dtype=float))
    max_output_size = 3
    iou_threshold = 1.  # No NMS

    expected_boxes = np.array([[0, 10, 1, 11], [0, 0, 1, 1], [0, 0.1, 1, 1.1]],
                              dtype=float)
    nms_boxlist = np_box_list_ops.non_max_suppression(
        boxlist, max_output_size, iou_threshold)
    self.assertAllClose(nms_boxlist.get(), expected_boxes)

  def test_select_from_three_clusters(self):
    boxlist = np_box_list.BoxList(self._boxes)
    boxlist.add_field('scores',
                      np.array([.9, .75, .6, .95, .2, .3], dtype=float))
    max_output_size = 3
    iou_threshold = 0.5

    expected_boxes = np.array([[0, 10, 1, 11], [0, 0, 1, 1], [0, 100, 1, 101]],
                              dtype=float)
    nms_boxlist = np_box_list_ops.non_max_suppression(
        boxlist, max_output_size, iou_threshold)
    self.assertAllClose(nms_boxlist.get(), expected_boxes)

  def test_select_at_most_two_from_three_clusters(self):
    boxlist = np_box_list.BoxList(self._boxes)
    boxlist.add_field('scores',
                      np.array([.9, .75, .6, .95, .5, .3], dtype=float))
    max_output_size = 2
    iou_threshold = 0.5

    expected_boxes = np.array([[0, 10, 1, 11], [0, 0, 1, 1]], dtype=float)
    nms_boxlist = np_box_list_ops.non_max_suppression(
        boxlist, max_output_size, iou_threshold)
    self.assertAllClose(nms_boxlist.get(), expected_boxes)

  def test_select_at_most_thirty_from_three_clusters(self):
    boxlist = np_box_list.BoxList(self._boxes)
    boxlist.add_field('scores',
                      np.array([.9, .75, .6, .95, .5, .3], dtype=float))
    max_output_size = 30
    iou_threshold = 0.5

    expected_boxes = np.array([[0, 10, 1, 11], [0, 0, 1, 1], [0, 100, 1, 101]],
                              dtype=float)
    nms_boxlist = np_box_list_ops.non_max_suppression(
        boxlist, max_output_size, iou_threshold)
    self.assertAllClose(nms_boxlist.get(), expected_boxes)

  def test_select_from_ten_indentical_boxes(self):
    boxes = np.array(10 * [[0, 0, 1, 1]], dtype=float)
    boxlist = np_box_list.BoxList(boxes)
    boxlist.add_field('scores', np.array(10 * [0.8]))
    iou_threshold = .5
    max_output_size = 3
    expected_boxes = np.array([[0, 0, 1, 1]], dtype=float)
    nms_boxlist = np_box_list_ops.non_max_suppression(
        boxlist, max_output_size, iou_threshold)
    self.assertAllClose(nms_boxlist.get(), expected_boxes)

  def test_different_iou_threshold(self):
    boxes = np.array([[0, 0, 20, 100], [0, 0, 20, 80], [200, 200, 210, 300],
                      [200, 200, 210, 250]],
                     dtype=float)
    boxlist = np_box_list.BoxList(boxes)
    boxlist.add_field('scores', np.array([0.9, 0.8, 0.7, 0.6]))
    max_output_size = 4

    iou_threshold = .4
    expected_boxes = np.array([[0, 0, 20, 100],
                               [200, 200, 210, 300],],
                              dtype=float)
    nms_boxlist = np_box_list_ops.non_max_suppression(
        boxlist, max_output_size, iou_threshold)
    self.assertAllClose(nms_boxlist.get(), expected_boxes)

    iou_threshold = .5
    expected_boxes = np.array([[0, 0, 20, 100], [200, 200, 210, 300],
                               [200, 200, 210, 250]],
                              dtype=float)
    nms_boxlist = np_box_list_ops.non_max_suppression(
        boxlist, max_output_size, iou_threshold)
    self.assertAllClose(nms_boxlist.get(), expected_boxes)

    iou_threshold = .8
    expected_boxes = np.array([[0, 0, 20, 100], [0, 0, 20, 80],
                               [200, 200, 210, 300], [200, 200, 210, 250]],
                              dtype=float)
    nms_boxlist = np_box_list_ops.non_max_suppression(
        boxlist, max_output_size, iou_threshold)
    self.assertAllClose(nms_boxlist.get(), expected_boxes)

  def test_multiclass_nms(self):
    boxlist = np_box_list.BoxList(
        np.array(
            [[0.2, 0.4, 0.8, 0.8], [0.4, 0.2, 0.8, 0.8], [0.6, 0.0, 1.0, 1.0]],
            dtype=np.float32))
    scores = np.array([[-0.2, 0.1, 0.5, -0.4, 0.3],
                       [0.7, -0.7, 0.6, 0.2, -0.9],
                       [0.4, 0.34, -0.9, 0.2, 0.31]],
                      dtype=np.float32)
    boxlist.add_field('scores', scores)
    boxlist_clean = np_box_list_ops.multi_class_non_max_suppression(
        boxlist, score_thresh=0.25, iou_thresh=0.1, max_output_size=3)

    scores_clean = boxlist_clean.get_field('scores')
    classes_clean = boxlist_clean.get_field('classes')
    boxes = boxlist_clean.get()
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
