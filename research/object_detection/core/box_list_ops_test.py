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

"""Tests for object_detection.core.box_list_ops."""
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.utils import test_case


class BoxListOpsTest(test_case.TestCase):
  """Tests for common bounding box operations."""

  def test_area(self):
    def graph_fn():
      corners = tf.constant([[0.0, 0.0, 10.0, 20.0], [1.0, 2.0, 3.0, 4.0]])
      boxes = box_list.BoxList(corners)
      areas = box_list_ops.area(boxes)
      return areas
    areas_out = self.execute(graph_fn, [])
    exp_output = [200.0, 4.0]
    self.assertAllClose(areas_out, exp_output)

  def test_height_width(self):
    def graph_fn():
      corners = tf.constant([[0.0, 0.0, 10.0, 20.0], [1.0, 2.0, 3.0, 4.0]])
      boxes = box_list.BoxList(corners)
      return box_list_ops.height_width(boxes)
    heights_out, widths_out = self.execute(graph_fn, [])
    exp_output_heights = [10., 2.]
    exp_output_widths = [20., 2.]
    self.assertAllClose(heights_out, exp_output_heights)
    self.assertAllClose(widths_out, exp_output_widths)

  def test_scale(self):
    def graph_fn():
      corners = tf.constant([[0, 0, 100, 200], [50, 120, 100, 140]],
                            dtype=tf.float32)
      boxes = box_list.BoxList(corners)
      boxes.add_field('extra_data', tf.constant([[1], [2]]))

      y_scale = tf.constant(1.0/100)
      x_scale = tf.constant(1.0/200)
      scaled_boxes = box_list_ops.scale(boxes, y_scale, x_scale)
      return scaled_boxes.get(), scaled_boxes.get_field('extra_data')
    scaled_corners_out, extra_data_out = self.execute(graph_fn, [])
    exp_output = [[0, 0, 1, 1], [0.5, 0.6, 1.0, 0.7]]
    self.assertAllClose(scaled_corners_out, exp_output)
    self.assertAllEqual(extra_data_out, [[1], [2]])

  def test_scale_height_width(self):
    def graph_fn():
      corners = tf.constant([[-10, -20, 10, 20], [0, 100, 100, 200]],
                            dtype=tf.float32)
      boxes = box_list.BoxList(corners)
      boxes.add_field('extra_data', tf.constant([[1], [2]]))

      y_scale = tf.constant(2.)
      x_scale = tf.constant(0.5)
      scaled_boxes = box_list_ops.scale_height_width(boxes, y_scale, x_scale)
      return scaled_boxes.get(), scaled_boxes.get_field('extra_data')
    exp_output = [
        [-20., -10, 20., 10],
        [-50., 125, 150., 175.]]
    scaled_corners_out, extra_data_out = self.execute(graph_fn, [])
    self.assertAllClose(scaled_corners_out, exp_output)
    self.assertAllEqual(extra_data_out, [[1], [2]])

  def test_clip_to_window_filter_boxes_which_fall_outside_the_window(
      self):
    def graph_fn():
      window = tf.constant([0, 0, 9, 14], tf.float32)
      corners = tf.constant([[5.0, 5.0, 6.0, 6.0],
                             [-1.0, -2.0, 4.0, 5.0],
                             [2.0, 3.0, 5.0, 9.0],
                             [0.0, 0.0, 9.0, 14.0],
                             [-100.0, -100.0, 300.0, 600.0],
                             [-10.0, -10.0, -9.0, -9.0]])
      boxes = box_list.BoxList(corners)
      boxes.add_field('extra_data', tf.constant([[1], [2], [3], [4], [5], [6]]))
      pruned = box_list_ops.clip_to_window(
          boxes, window, filter_nonoverlapping=True)
      return pruned.get(), pruned.get_field('extra_data')
    exp_output = [[5.0, 5.0, 6.0, 6.0], [0.0, 0.0, 4.0, 5.0],
                  [2.0, 3.0, 5.0, 9.0], [0.0, 0.0, 9.0, 14.0],
                  [0.0, 0.0, 9.0, 14.0]]
    pruned_output, extra_data_out = self.execute_cpu(graph_fn, [])
    self.assertAllClose(pruned_output, exp_output)
    self.assertAllEqual(extra_data_out, [[1], [2], [3], [4], [5]])

  def test_clip_to_window_without_filtering_boxes_which_fall_outside_the_window(
      self):
    def graph_fn():
      window = tf.constant([0, 0, 9, 14], tf.float32)
      corners = tf.constant([[5.0, 5.0, 6.0, 6.0],
                             [-1.0, -2.0, 4.0, 5.0],
                             [2.0, 3.0, 5.0, 9.0],
                             [0.0, 0.0, 9.0, 14.0],
                             [-100.0, -100.0, 300.0, 600.0],
                             [-10.0, -10.0, -9.0, -9.0]])
      boxes = box_list.BoxList(corners)
      boxes.add_field('extra_data', tf.constant([[1], [2], [3], [4], [5], [6]]))
      pruned = box_list_ops.clip_to_window(
          boxes, window, filter_nonoverlapping=False)
      return pruned.get(), pruned.get_field('extra_data')
    pruned_output, extra_data_out = self.execute(graph_fn, [])
    exp_output = [[5.0, 5.0, 6.0, 6.0], [0.0, 0.0, 4.0, 5.0],
                  [2.0, 3.0, 5.0, 9.0], [0.0, 0.0, 9.0, 14.0],
                  [0.0, 0.0, 9.0, 14.0], [0.0, 0.0, 0.0, 0.0]]
    self.assertAllClose(pruned_output, exp_output)
    self.assertAllEqual(extra_data_out, [[1], [2], [3], [4], [5], [6]])

  def test_prune_outside_window_filters_boxes_which_fall_outside_the_window(
      self):
    def graph_fn():
      window = tf.constant([0, 0, 9, 14], tf.float32)
      corners = tf.constant([[5.0, 5.0, 6.0, 6.0],
                             [-1.0, -2.0, 4.0, 5.0],
                             [2.0, 3.0, 5.0, 9.0],
                             [0.0, 0.0, 9.0, 14.0],
                             [-10.0, -10.0, -9.0, -9.0],
                             [-100.0, -100.0, 300.0, 600.0]])
      boxes = box_list.BoxList(corners)
      boxes.add_field('extra_data', tf.constant([[1], [2], [3], [4], [5], [6]]))
      pruned, keep_indices = box_list_ops.prune_outside_window(boxes, window)
      return pruned.get(), pruned.get_field('extra_data'), keep_indices
    pruned_output, extra_data_out, keep_indices_out = self.execute_cpu(graph_fn,
                                                                       [])
    exp_output = [[5.0, 5.0, 6.0, 6.0],
                  [2.0, 3.0, 5.0, 9.0],
                  [0.0, 0.0, 9.0, 14.0]]
    self.assertAllClose(pruned_output, exp_output)
    self.assertAllEqual(keep_indices_out, [0, 2, 3])
    self.assertAllEqual(extra_data_out, [[1], [3], [4]])

  def test_prune_completely_outside_window(self):
    def graph_fn():
      window = tf.constant([0, 0, 9, 14], tf.float32)
      corners = tf.constant([[5.0, 5.0, 6.0, 6.0],
                             [-1.0, -2.0, 4.0, 5.0],
                             [2.0, 3.0, 5.0, 9.0],
                             [0.0, 0.0, 9.0, 14.0],
                             [-10.0, -10.0, -9.0, -9.0],
                             [-100.0, -100.0, 300.0, 600.0]])
      boxes = box_list.BoxList(corners)
      boxes.add_field('extra_data', tf.constant([[1], [2], [3], [4], [5], [6]]))
      pruned, keep_indices = box_list_ops.prune_completely_outside_window(
          boxes, window)
      return pruned.get(), pruned.get_field('extra_data'), keep_indices
    pruned_output, extra_data_out, keep_indices_out = self.execute(graph_fn, [])
    exp_output = [[5.0, 5.0, 6.0, 6.0],
                  [-1.0, -2.0, 4.0, 5.0],
                  [2.0, 3.0, 5.0, 9.0],
                  [0.0, 0.0, 9.0, 14.0],
                  [-100.0, -100.0, 300.0, 600.0]]
    self.assertAllClose(pruned_output, exp_output)
    self.assertAllEqual(keep_indices_out, [0, 1, 2, 3, 5])
    self.assertAllEqual(extra_data_out, [[1], [2], [3], [4], [6]])

  def test_prune_completely_outside_window_with_empty_boxlist(self):
    def graph_fn():
      window = tf.constant([0, 0, 9, 14], tf.float32)
      corners = tf.zeros(shape=[0, 4], dtype=tf.float32)
      boxes = box_list.BoxList(corners)
      boxes.add_field('extra_data', tf.zeros(shape=[0], dtype=tf.int32))
      pruned, keep_indices = box_list_ops.prune_completely_outside_window(
          boxes, window)
      pruned_boxes = pruned.get()
      extra = pruned.get_field('extra_data')
      return pruned_boxes, extra, keep_indices

    pruned_boxes_out, extra_out, keep_indices_out = self.execute(graph_fn, [])
    exp_pruned_boxes = np.zeros(shape=[0, 4], dtype=np.float32)
    exp_extra = np.zeros(shape=[0], dtype=np.int32)
    self.assertAllClose(exp_pruned_boxes, pruned_boxes_out)
    self.assertAllEqual([], keep_indices_out)
    self.assertAllEqual(exp_extra, extra_out)

  def test_intersection(self):
    def graph_fn():
      corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
      corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                              [0.0, 0.0, 20.0, 20.0]])
      boxes1 = box_list.BoxList(corners1)
      boxes2 = box_list.BoxList(corners2)
      intersect = box_list_ops.intersection(boxes1, boxes2)
      return intersect
    exp_output = [[2.0, 0.0, 6.0], [1.0, 0.0, 5.0]]
    intersect_out = self.execute(graph_fn, [])
    self.assertAllClose(intersect_out, exp_output)

  def test_matched_intersection(self):
    def graph_fn():
      corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
      corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
      boxes1 = box_list.BoxList(corners1)
      boxes2 = box_list.BoxList(corners2)
      intersect = box_list_ops.matched_intersection(boxes1, boxes2)
      return intersect
    exp_output = [2.0, 0.0]
    intersect_out = self.execute(graph_fn, [])
    self.assertAllClose(intersect_out, exp_output)

  def test_iou(self):
    def graph_fn():
      corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
      corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
      corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                              [0.0, 0.0, 20.0, 20.0]])
      boxes1 = box_list.BoxList(corners1)
      boxes2 = box_list.BoxList(corners2)
      iou = box_list_ops.iou(boxes1, boxes2)
      return iou
    exp_output = [[2.0 / 16.0, 0, 6.0 / 400.0], [1.0 / 16.0, 0.0, 5.0 / 400.0]]
    iou_output = self.execute(graph_fn, [])
    self.assertAllClose(iou_output, exp_output)

  def test_l1(self):
    def graph_fn():
      corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
      corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                              [0.0, 0.0, 20.0, 20.0]])
      boxes1 = box_list.BoxList(corners1)
      boxes2 = box_list.BoxList(corners2)
      l1 = box_list_ops.l1(boxes1, boxes2)
      return l1
    exp_output = [[5.0, 22.5, 45.5], [8.5, 19.0, 40.0]]
    l1_output = self.execute(graph_fn, [])
    self.assertAllClose(l1_output, exp_output)

  def test_giou(self):
    def graph_fn():
      corners1 = tf.constant([[5.0, 7.0, 7.0, 9.0]])
      corners2 = tf.constant([[5.0, 7.0, 7.0, 9.0], [5.0, 11.0, 7.0, 13.0]])
      boxes1 = box_list.BoxList(corners1)
      boxes2 = box_list.BoxList(corners2)
      giou = box_list_ops.giou(boxes1, boxes2)
      return giou
    exp_output = [[1.0, -1.0 / 3.0]]
    giou_output = self.execute(graph_fn, [])
    self.assertAllClose(giou_output, exp_output)

  def test_matched_iou(self):
    def graph_fn():
      corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
      corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
      boxes1 = box_list.BoxList(corners1)
      boxes2 = box_list.BoxList(corners2)
      iou = box_list_ops.matched_iou(boxes1, boxes2)
      return iou
    exp_output = [2.0 / 16.0, 0]
    iou_output = self.execute(graph_fn, [])
    self.assertAllClose(iou_output, exp_output)

  def test_iouworks_on_empty_inputs(self):
    def graph_fn():
      corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
      corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                              [0.0, 0.0, 20.0, 20.0]])
      boxes1 = box_list.BoxList(corners1)
      boxes2 = box_list.BoxList(corners2)
      boxes_empty = box_list.BoxList(tf.zeros((0, 4)))
      iou_empty_1 = box_list_ops.iou(boxes1, boxes_empty)
      iou_empty_2 = box_list_ops.iou(boxes_empty, boxes2)
      iou_empty_3 = box_list_ops.iou(boxes_empty, boxes_empty)
      return iou_empty_1, iou_empty_2, iou_empty_3
    iou_output_1, iou_output_2, iou_output_3 = self.execute(graph_fn, [])
    self.assertAllEqual(iou_output_1.shape, (2, 0))
    self.assertAllEqual(iou_output_2.shape, (0, 3))
    self.assertAllEqual(iou_output_3.shape, (0, 0))

  def test_ioa(self):
    def graph_fn():
      corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
      corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                              [0.0, 0.0, 20.0, 20.0]])
      boxes1 = box_list.BoxList(corners1)
      boxes2 = box_list.BoxList(corners2)
      ioa_1 = box_list_ops.ioa(boxes1, boxes2)
      ioa_2 = box_list_ops.ioa(boxes2, boxes1)
      return ioa_1, ioa_2
    exp_output_1 = [[2.0 / 12.0, 0, 6.0 / 400.0],
                    [1.0 / 12.0, 0.0, 5.0 / 400.0]]
    exp_output_2 = [[2.0 / 6.0, 1.0 / 5.0],
                    [0, 0],
                    [6.0 / 6.0, 5.0 / 5.0]]
    ioa_output_1, ioa_output_2 = self.execute(graph_fn, [])
    self.assertAllClose(ioa_output_1, exp_output_1)
    self.assertAllClose(ioa_output_2, exp_output_2)

  def test_prune_non_overlapping_boxes(self):
    def graph_fn():
      corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
      corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                              [0.0, 0.0, 20.0, 20.0]])
      boxes1 = box_list.BoxList(corners1)
      boxes2 = box_list.BoxList(corners2)
      minoverlap = 0.5

      exp_output_1 = boxes1
      exp_output_2 = box_list.BoxList(tf.constant(0.0, shape=[0, 4]))
      output_1, keep_indices_1 = box_list_ops.prune_non_overlapping_boxes(
          boxes1, boxes2, min_overlap=minoverlap)
      output_2, keep_indices_2 = box_list_ops.prune_non_overlapping_boxes(
          boxes2, boxes1, min_overlap=minoverlap)
      return (output_1.get(), keep_indices_1, output_2.get(), keep_indices_2,
              exp_output_1.get(), exp_output_2.get())

    (output_1_, keep_indices_1_, output_2_, keep_indices_2_, exp_output_1_,
     exp_output_2_) = self.execute_cpu(graph_fn, [])
    self.assertAllClose(output_1_, exp_output_1_)
    self.assertAllClose(output_2_, exp_output_2_)
    self.assertAllEqual(keep_indices_1_, [0, 1])
    self.assertAllEqual(keep_indices_2_, [])

  def test_prune_small_boxes(self):
    def graph_fn():
      boxes = tf.constant([[4.0, 3.0, 7.0, 5.0],
                           [5.0, 6.0, 10.0, 7.0],
                           [3.0, 4.0, 6.0, 8.0],
                           [14.0, 14.0, 15.0, 15.0],
                           [0.0, 0.0, 20.0, 20.0]])
      boxes = box_list.BoxList(boxes)
      pruned_boxes = box_list_ops.prune_small_boxes(boxes, 3)
      return pruned_boxes.get()
    exp_boxes = [[3.0, 4.0, 6.0, 8.0],
                 [0.0, 0.0, 20.0, 20.0]]
    pruned_boxes = self.execute(graph_fn, [])
    self.assertAllEqual(pruned_boxes, exp_boxes)

  def test_prune_small_boxes_prunes_boxes_with_negative_side(self):
    def graph_fn():
      boxes = tf.constant([[4.0, 3.0, 7.0, 5.0],
                           [5.0, 6.0, 10.0, 7.0],
                           [3.0, 4.0, 6.0, 8.0],
                           [14.0, 14.0, 15.0, 15.0],
                           [0.0, 0.0, 20.0, 20.0],
                           [2.0, 3.0, 1.5, 7.0],  # negative height
                           [2.0, 3.0, 5.0, 1.7]])  # negative width
      boxes = box_list.BoxList(boxes)
      pruned_boxes = box_list_ops.prune_small_boxes(boxes, 3)
      return pruned_boxes.get()
    exp_boxes = [[3.0, 4.0, 6.0, 8.0],
                 [0.0, 0.0, 20.0, 20.0]]
    pruned_boxes = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(pruned_boxes, exp_boxes)

  def test_change_coordinate_frame(self):
    def graph_fn():
      corners = tf.constant([[0.25, 0.5, 0.75, 0.75], [0.5, 0.0, 1.0, 1.0]])
      window = tf.constant([0.25, 0.25, 0.75, 0.75])
      boxes = box_list.BoxList(corners)

      expected_corners = tf.constant([[0, 0.5, 1.0, 1.0],
                                      [0.5, -0.5, 1.5, 1.5]])
      expected_boxes = box_list.BoxList(expected_corners)
      output = box_list_ops.change_coordinate_frame(boxes, window)
      return output.get(), expected_boxes.get()
    output_, expected_boxes_ = self.execute(graph_fn, [])
    self.assertAllClose(output_, expected_boxes_)

  def test_ioaworks_on_empty_inputs(self):
    def graph_fn():
      corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
      corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                              [0.0, 0.0, 20.0, 20.0]])
      boxes1 = box_list.BoxList(corners1)
      boxes2 = box_list.BoxList(corners2)
      boxes_empty = box_list.BoxList(tf.zeros((0, 4)))
      ioa_empty_1 = box_list_ops.ioa(boxes1, boxes_empty)
      ioa_empty_2 = box_list_ops.ioa(boxes_empty, boxes2)
      ioa_empty_3 = box_list_ops.ioa(boxes_empty, boxes_empty)
      return ioa_empty_1, ioa_empty_2, ioa_empty_3
    ioa_output_1, ioa_output_2, ioa_output_3 = self.execute(graph_fn, [])
    self.assertAllEqual(ioa_output_1.shape, (2, 0))
    self.assertAllEqual(ioa_output_2.shape, (0, 3))
    self.assertAllEqual(ioa_output_3.shape, (0, 0))

  def test_pairwise_distances(self):
    def graph_fn():
      corners1 = tf.constant([[0.0, 0.0, 0.0, 0.0],
                              [1.0, 1.0, 0.0, 2.0]])
      corners2 = tf.constant([[3.0, 4.0, 1.0, 0.0],
                              [-4.0, 0.0, 0.0, 3.0],
                              [0.0, 0.0, 0.0, 0.0]])
      boxes1 = box_list.BoxList(corners1)
      boxes2 = box_list.BoxList(corners2)
      dist_matrix = box_list_ops.sq_dist(boxes1, boxes2)
      return dist_matrix
    exp_output = [[26, 25, 0], [18, 27, 6]]
    dist_output = self.execute(graph_fn, [])
    self.assertAllClose(dist_output, exp_output)

  def test_boolean_mask(self):
    def graph_fn():
      corners = tf.constant(
          [4 * [0.0], 4 * [1.0], 4 * [2.0], 4 * [3.0], 4 * [4.0]])
      indicator = tf.constant([True, False, True, False, True], tf.bool)
      boxes = box_list.BoxList(corners)
      subset = box_list_ops.boolean_mask(boxes, indicator)
      return subset.get()
    expected_subset = [4 * [0.0], 4 * [2.0], 4 * [4.0]]
    subset_output = self.execute_cpu(graph_fn, [])
    self.assertAllClose(subset_output, expected_subset)

  def test_static_boolean_mask_with_field(self):

    def graph_fn(corners, weights, indicator):
      boxes = box_list.BoxList(corners)
      boxes.add_field('weights', weights)
      subset = box_list_ops.boolean_mask(
          boxes,
          indicator, ['weights'],
          use_static_shapes=True,
          indicator_sum=3)
      return (subset.get_field('boxes'), subset.get_field('weights'))

    corners = np.array(
        [4 * [0.0], 4 * [1.0], 4 * [2.0], 4 * [3.0], 4 * [4.0]],
        dtype=np.float32)
    indicator = np.array([True, False, True, False, True], dtype=np.bool)
    weights = np.array([[.1], [.3], [.5], [.7], [.9]], dtype=np.float32)
    result_boxes, result_weights = self.execute_cpu(
        graph_fn, [corners, weights, indicator])
    expected_boxes = [4 * [0.0], 4 * [2.0], 4 * [4.0]]
    expected_weights = [[.1], [.5], [.9]]

    self.assertAllClose(result_boxes, expected_boxes)
    self.assertAllClose(result_weights, expected_weights)

  def test_gather(self):
    def graph_fn():
      corners = tf.constant(
          [4 * [0.0], 4 * [1.0], 4 * [2.0], 4 * [3.0], 4 * [4.0]])
      indices = tf.constant([0, 2, 4], tf.int32)
      boxes = box_list.BoxList(corners)
      subset = box_list_ops.gather(boxes, indices)
      return subset.get()
    expected_subset = [4 * [0.0], 4 * [2.0], 4 * [4.0]]
    subset_output = self.execute(graph_fn, [])
    self.assertAllClose(subset_output, expected_subset)

  def test_static_gather_with_field(self):

    def graph_fn(corners, weights, indices):
      boxes = box_list.BoxList(corners)
      boxes.add_field('weights', weights)
      subset = box_list_ops.gather(
          boxes, indices, ['weights'], use_static_shapes=True)
      return (subset.get_field('boxes'), subset.get_field('weights'))

    corners = np.array([4 * [0.0], 4 * [1.0], 4 * [2.0], 4 * [3.0],
                        4 * [4.0]], dtype=np.float32)
    weights = np.array([[.1], [.3], [.5], [.7], [.9]], dtype=np.float32)
    indices = np.array([0, 2, 4], dtype=np.int32)

    result_boxes, result_weights = self.execute(graph_fn,
                                                [corners, weights, indices])
    expected_boxes = [4 * [0.0], 4 * [2.0], 4 * [4.0]]
    expected_weights = [[.1], [.5], [.9]]
    self.assertAllClose(result_boxes, expected_boxes)
    self.assertAllClose(result_weights, expected_weights)

  def test_gather_with_invalid_field(self):
    corners = tf.constant([4 * [0.0], 4 * [1.0]])
    indices = tf.constant([0, 1], tf.int32)
    weights = tf.constant([[.1], [.3]], tf.float32)

    boxes = box_list.BoxList(corners)
    boxes.add_field('weights', weights)
    with self.assertRaises(ValueError):
      box_list_ops.gather(boxes, indices, ['foo', 'bar'])

  def test_gather_with_invalid_inputs(self):
    corners = tf.constant(
        [4 * [0.0], 4 * [1.0], 4 * [2.0], 4 * [3.0], 4 * [4.0]])
    indices_float32 = tf.constant([0, 2, 4], tf.float32)
    boxes = box_list.BoxList(corners)
    with self.assertRaises(ValueError):
      _ = box_list_ops.gather(boxes, indices_float32)
    indices_2d = tf.constant([[0, 2, 4]], tf.int32)
    boxes = box_list.BoxList(corners)
    with self.assertRaises(ValueError):
      _ = box_list_ops.gather(boxes, indices_2d)

  def test_gather_with_dynamic_indexing(self):
    def graph_fn():
      corners = tf.constant(
          [4 * [0.0], 4 * [1.0], 4 * [2.0], 4 * [3.0], 4 * [4.0]])
      weights = tf.constant([.5, .3, .7, .1, .9], tf.float32)
      indices = tf.reshape(tf.where(tf.greater(weights, 0.4)), [-1])
      boxes = box_list.BoxList(corners)
      boxes.add_field('weights', weights)
      subset = box_list_ops.gather(boxes, indices, ['weights'])
      return subset.get(), subset.get_field('weights')
    expected_subset = [4 * [0.0], 4 * [2.0], 4 * [4.0]]
    expected_weights = [.5, .7, .9]
    subset_output, weights_output = self.execute(graph_fn, [])
    self.assertAllClose(subset_output, expected_subset)
    self.assertAllClose(weights_output, expected_weights)

  def test_sort_by_field_ascending_order(self):
    exp_corners = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                   [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    exp_scores = [.95, .9, .75, .6, .5, .3]
    exp_weights = [.2, .45, .6, .75, .8, .92]

    def graph_fn():
      shuffle = [2, 4, 0, 5, 1, 3]
      corners = tf.constant([exp_corners[i] for i in shuffle], tf.float32)
      boxes = box_list.BoxList(corners)
      boxes.add_field('scores', tf.constant(
          [exp_scores[i] for i in shuffle], tf.float32))
      boxes.add_field('weights', tf.constant(
          [exp_weights[i] for i in shuffle], tf.float32))
      sort_by_weight = box_list_ops.sort_by_field(
          boxes,
          'weights',
          order=box_list_ops.SortOrder.ascend)
      return [sort_by_weight.get(), sort_by_weight.get_field('scores'),
              sort_by_weight.get_field('weights')]
    corners_out, scores_out, weights_out = self.execute(graph_fn, [])
    self.assertAllClose(corners_out, exp_corners)
    self.assertAllClose(scores_out, exp_scores)
    self.assertAllClose(weights_out, exp_weights)

  def test_sort_by_field_descending_order(self):
    exp_corners = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                   [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    exp_scores = [.95, .9, .75, .6, .5, .3]
    exp_weights = [.2, .45, .6, .75, .8, .92]

    def graph_fn():
      shuffle = [2, 4, 0, 5, 1, 3]
      corners = tf.constant([exp_corners[i] for i in shuffle], tf.float32)
      boxes = box_list.BoxList(corners)
      boxes.add_field('scores', tf.constant(
          [exp_scores[i] for i in shuffle], tf.float32))
      boxes.add_field('weights', tf.constant(
          [exp_weights[i] for i in shuffle], tf.float32))
      sort_by_score = box_list_ops.sort_by_field(boxes, 'scores')
      return (sort_by_score.get(), sort_by_score.get_field('scores'),
              sort_by_score.get_field('weights'))

    corners_out, scores_out, weights_out = self.execute(graph_fn, [])
    self.assertAllClose(corners_out, exp_corners)
    self.assertAllClose(scores_out, exp_scores)
    self.assertAllClose(weights_out, exp_weights)

  def test_sort_by_field_invalid_inputs(self):
    corners = tf.constant([4 * [0.0], 4 * [0.5], 4 * [1.0], 4 * [2.0], 4 *
                           [3.0], 4 * [4.0]])
    misc = tf.constant([[.95, .9], [.5, .3]], tf.float32)
    weights = tf.constant([[.1, .2]], tf.float32)
    boxes = box_list.BoxList(corners)
    boxes.add_field('misc', misc)
    boxes.add_field('weights', weights)

    with self.assertRaises(ValueError):
      box_list_ops.sort_by_field(boxes, 'area')

    with self.assertRaises(ValueError):
      box_list_ops.sort_by_field(boxes, 'misc')

    with self.assertRaises(ValueError):
      box_list_ops.sort_by_field(boxes, 'weights')

  def test_visualize_boxes_in_image(self):
    def graph_fn():
      image = tf.zeros((6, 4, 3))
      corners = tf.constant([[0, 0, 5, 3],
                             [0, 0, 3, 2]], tf.float32)
      boxes = box_list.BoxList(corners)
      image_and_boxes = box_list_ops.visualize_boxes_in_image(image, boxes)
      image_and_boxes_bw = tf.cast(
          tf.greater(tf.reduce_sum(image_and_boxes, 2), 0.0), dtype=tf.float32)
      return image_and_boxes_bw
    exp_result = [[1, 1, 1, 0],
                  [1, 1, 1, 0],
                  [1, 1, 1, 0],
                  [1, 0, 1, 0],
                  [1, 1, 1, 0],
                  [0, 0, 0, 0]]
    output = self.execute_cpu(graph_fn, [])
    self.assertAllEqual(output.astype(int), exp_result)

  def test_filter_field_value_equals(self):
    def graph_fn():
      corners = tf.constant([[0, 0, 1, 1],
                             [0, 0.1, 1, 1.1],
                             [0, -0.1, 1, 0.9],
                             [0, 10, 1, 11],
                             [0, 10.1, 1, 11.1],
                             [0, 100, 1, 101]], tf.float32)
      boxes = box_list.BoxList(corners)
      boxes.add_field('classes', tf.constant([1, 2, 1, 2, 2, 1]))
      filtered_boxes1 = box_list_ops.filter_field_value_equals(
          boxes, 'classes', 1)
      filtered_boxes2 = box_list_ops.filter_field_value_equals(
          boxes, 'classes', 2)
      return filtered_boxes1.get(), filtered_boxes2.get()
    exp_output1 = [[0, 0, 1, 1], [0, -0.1, 1, 0.9], [0, 100, 1, 101]]
    exp_output2 = [[0, 0.1, 1, 1.1], [0, 10, 1, 11], [0, 10.1, 1, 11.1]]
    filtered_output1, filtered_output2 = self.execute_cpu(graph_fn, [])
    self.assertAllClose(filtered_output1, exp_output1)
    self.assertAllClose(filtered_output2, exp_output2)

  def test_filter_greater_than(self):
    def graph_fn():
      corners = tf.constant([[0, 0, 1, 1],
                             [0, 0.1, 1, 1.1],
                             [0, -0.1, 1, 0.9],
                             [0, 10, 1, 11],
                             [0, 10.1, 1, 11.1],
                             [0, 100, 1, 101]], tf.float32)
      boxes = box_list.BoxList(corners)
      boxes.add_field('scores', tf.constant([.1, .75, .9, .5, .5, .8]))
      thresh = .6
      filtered_boxes = box_list_ops.filter_greater_than(boxes, thresh)
      return filtered_boxes.get()
    exp_output = [[0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9], [0, 100, 1, 101]]
    filtered_output = self.execute_cpu(graph_fn, [])
    self.assertAllClose(filtered_output, exp_output)

  def test_clip_box_list(self):
    def graph_fn():
      boxlist = box_list.BoxList(
          tf.constant([[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5],
                       [0.6, 0.6, 0.8, 0.8], [0.2, 0.2, 0.3, 0.3]], tf.float32))
      boxlist.add_field('classes', tf.constant([0, 0, 1, 1]))
      boxlist.add_field('scores', tf.constant([0.75, 0.65, 0.3, 0.2]))
      num_boxes = 2
      clipped_boxlist = box_list_ops.pad_or_clip_box_list(boxlist, num_boxes)
      return (clipped_boxlist.get(), clipped_boxlist.get_field('classes'),
              clipped_boxlist.get_field('scores'))

    expected_boxes = [[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5]]
    expected_classes = [0, 0]
    expected_scores = [0.75, 0.65]
    boxes_out, classes_out, scores_out = self.execute(graph_fn, [])

    self.assertAllClose(expected_boxes, boxes_out)
    self.assertAllEqual(expected_classes, classes_out)
    self.assertAllClose(expected_scores, scores_out)

  def test_pad_box_list(self):
    def graph_fn():
      boxlist = box_list.BoxList(
          tf.constant([[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5]], tf.float32))
      boxlist.add_field('classes', tf.constant([0, 1]))
      boxlist.add_field('scores', tf.constant([0.75, 0.2]))
      num_boxes = 4
      padded_boxlist = box_list_ops.pad_or_clip_box_list(boxlist, num_boxes)
      return (padded_boxlist.get(), padded_boxlist.get_field('classes'),
              padded_boxlist.get_field('scores'))
    expected_boxes = [[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5],
                      [0, 0, 0, 0], [0, 0, 0, 0]]
    expected_classes = [0, 1, 0, 0]
    expected_scores = [0.75, 0.2, 0, 0]
    boxes_out, classes_out, scores_out = self.execute(graph_fn, [])
    self.assertAllClose(expected_boxes, boxes_out)
    self.assertAllEqual(expected_classes, classes_out)
    self.assertAllClose(expected_scores, scores_out)

  def test_select_random_box(self):
    boxes = [[0., 0., 1., 1.],
             [0., 1., 2., 3.],
             [0., 2., 3., 4.]]
    def graph_fn():
      corners = tf.constant(boxes, dtype=tf.float32)
      boxlist = box_list.BoxList(corners)
      random_bbox, valid = box_list_ops.select_random_box(boxlist)
      return random_bbox, valid
    random_bbox_out, valid_out = self.execute(graph_fn, [])
    norm_small = any(
        [np.linalg.norm(random_bbox_out - box) < 1e-6 for box in boxes])
    self.assertTrue(norm_small)
    self.assertTrue(valid_out)

  def test_select_random_box_with_empty_boxlist(self):
    def graph_fn():
      corners = tf.constant([], shape=[0, 4], dtype=tf.float32)
      boxlist = box_list.BoxList(corners)
      random_bbox, valid = box_list_ops.select_random_box(boxlist)
      return random_bbox, valid
    random_bbox_out, valid_out = self.execute_cpu(graph_fn, [])
    expected_bbox_out = np.array([[-1., -1., -1., -1.]], dtype=np.float32)
    self.assertAllEqual(expected_bbox_out, random_bbox_out)
    self.assertFalse(valid_out)

  def test_get_minimal_coverage_box(self):
    def graph_fn():
      boxes = [[0., 0., 1., 1.],
               [-1., 1., 2., 3.],
               [0., 2., 3., 4.]]
      corners = tf.constant(boxes, dtype=tf.float32)
      boxlist = box_list.BoxList(corners)
      coverage_box = box_list_ops.get_minimal_coverage_box(boxlist)
      return coverage_box
    coverage_box_out = self.execute(graph_fn, [])
    expected_coverage_box = [[-1., 0., 3., 4.]]
    self.assertAllClose(expected_coverage_box, coverage_box_out)

  def test_get_minimal_coverage_box_with_empty_boxlist(self):
    def graph_fn():
      corners = tf.constant([], shape=[0, 4], dtype=tf.float32)
      boxlist = box_list.BoxList(corners)
      coverage_box = box_list_ops.get_minimal_coverage_box(boxlist)
      return coverage_box
    coverage_box_out = self.execute(graph_fn, [])
    self.assertAllClose([[0.0, 0.0, 1.0, 1.0]], coverage_box_out)


class ConcatenateTest(test_case.TestCase):

  def test_invalid_input_box_list_list(self):
    with self.assertRaises(ValueError):
      box_list_ops.concatenate(None)
    with self.assertRaises(ValueError):
      box_list_ops.concatenate([])
    with self.assertRaises(ValueError):
      corners = tf.constant([[0, 0, 0, 0]], tf.float32)
      boxlist = box_list.BoxList(corners)
      box_list_ops.concatenate([boxlist, 2])

  def test_concatenate_with_missing_fields(self):
    corners1 = tf.constant([[0, 0, 0, 0], [1, 2, 3, 4]], tf.float32)
    scores1 = tf.constant([1.0, 2.1])
    corners2 = tf.constant([[0, 3, 1, 6], [2, 4, 3, 8]], tf.float32)
    boxlist1 = box_list.BoxList(corners1)
    boxlist1.add_field('scores', scores1)
    boxlist2 = box_list.BoxList(corners2)
    with self.assertRaises(ValueError):
      box_list_ops.concatenate([boxlist1, boxlist2])

  def test_concatenate_with_incompatible_field_shapes(self):
    corners1 = tf.constant([[0, 0, 0, 0], [1, 2, 3, 4]], tf.float32)
    scores1 = tf.constant([1.0, 2.1])
    corners2 = tf.constant([[0, 3, 1, 6], [2, 4, 3, 8]], tf.float32)
    scores2 = tf.constant([[1.0, 1.0], [2.1, 3.2]])
    boxlist1 = box_list.BoxList(corners1)
    boxlist1.add_field('scores', scores1)
    boxlist2 = box_list.BoxList(corners2)
    boxlist2.add_field('scores', scores2)
    with self.assertRaises(ValueError):
      box_list_ops.concatenate([boxlist1, boxlist2])

  def test_concatenate_is_correct(self):
    def graph_fn():
      corners1 = tf.constant([[0, 0, 0, 0], [1, 2, 3, 4]], tf.float32)
      scores1 = tf.constant([1.0, 2.1])
      corners2 = tf.constant([[0, 3, 1, 6], [2, 4, 3, 8], [1, 0, 5, 10]],
                             tf.float32)
      scores2 = tf.constant([1.0, 2.1, 5.6])
      boxlist1 = box_list.BoxList(corners1)
      boxlist1.add_field('scores', scores1)
      boxlist2 = box_list.BoxList(corners2)
      boxlist2.add_field('scores', scores2)
      result = box_list_ops.concatenate([boxlist1, boxlist2])
      return result.get(), result.get_field('scores')
    exp_corners = [[0, 0, 0, 0],
                   [1, 2, 3, 4],
                   [0, 3, 1, 6],
                   [2, 4, 3, 8],
                   [1, 0, 5, 10]]
    exp_scores = [1.0, 2.1, 1.0, 2.1, 5.6]
    corners_output, scores_output = self.execute(graph_fn, [])
    self.assertAllClose(corners_output, exp_corners)
    self.assertAllClose(scores_output, exp_scores)


class NonMaxSuppressionTest(test_case.TestCase):

  def test_select_from_three_clusters(self):
    def graph_fn():
      corners = tf.constant([[0, 0, 1, 1],
                             [0, 0.1, 1, 1.1],
                             [0, -0.1, 1, 0.9],
                             [0, 10, 1, 11],
                             [0, 10.1, 1, 11.1],
                             [0, 100, 1, 101]], tf.float32)
      boxes = box_list.BoxList(corners)
      boxes.add_field('scores', tf.constant([.9, .75, .6, .95, .5, .3]))
      iou_thresh = .5
      max_output_size = 3
      nms = box_list_ops.non_max_suppression(
          boxes, iou_thresh, max_output_size)
      return nms.get()
    exp_nms = [[0, 10, 1, 11],
               [0, 0, 1, 1],
               [0, 100, 1, 101]]
    nms_output = self.execute_cpu(graph_fn, [])
    self.assertAllClose(nms_output, exp_nms)

  def test_select_at_most_two_boxes_from_three_clusters(self):
    def graph_fn():
      corners = tf.constant([[0, 0, 1, 1],
                             [0, 0.1, 1, 1.1],
                             [0, -0.1, 1, 0.9],
                             [0, 10, 1, 11],
                             [0, 10.1, 1, 11.1],
                             [0, 100, 1, 101]], tf.float32)
      boxes = box_list.BoxList(corners)
      boxes.add_field('scores', tf.constant([.9, .75, .6, .95, .5, .3]))
      iou_thresh = .5
      max_output_size = 2
      nms = box_list_ops.non_max_suppression(
          boxes, iou_thresh, max_output_size)
      return nms.get()
    exp_nms = [[0, 10, 1, 11],
               [0, 0, 1, 1]]
    nms_output = self.execute_cpu(graph_fn, [])
    self.assertAllClose(nms_output, exp_nms)

  def test_select_at_most_thirty_boxes_from_three_clusters(self):
    def graph_fn():
      corners = tf.constant([[0, 0, 1, 1],
                             [0, 0.1, 1, 1.1],
                             [0, -0.1, 1, 0.9],
                             [0, 10, 1, 11],
                             [0, 10.1, 1, 11.1],
                             [0, 100, 1, 101]], tf.float32)
      boxes = box_list.BoxList(corners)
      boxes.add_field('scores', tf.constant([.9, .75, .6, .95, .5, .3]))
      iou_thresh = .5
      max_output_size = 30
      nms = box_list_ops.non_max_suppression(
          boxes, iou_thresh, max_output_size)
      return nms.get()
    exp_nms = [[0, 10, 1, 11],
               [0, 0, 1, 1],
               [0, 100, 1, 101]]
    nms_output = self.execute_cpu(graph_fn, [])
    self.assertAllClose(nms_output, exp_nms)

  def test_select_single_box(self):
    def graph_fn():
      corners = tf.constant([[0, 0, 1, 1]], tf.float32)
      boxes = box_list.BoxList(corners)
      boxes.add_field('scores', tf.constant([.9]))
      iou_thresh = .5
      max_output_size = 3
      nms = box_list_ops.non_max_suppression(
          boxes, iou_thresh, max_output_size)
      return nms.get()
    exp_nms = [[0, 0, 1, 1]]
    nms_output = self.execute_cpu(graph_fn, [])
    self.assertAllClose(nms_output, exp_nms)

  def test_select_from_ten_identical_boxes(self):
    def graph_fn():
      corners = tf.constant(10 * [[0, 0, 1, 1]], tf.float32)
      boxes = box_list.BoxList(corners)
      boxes.add_field('scores', tf.constant(10 * [.9]))
      iou_thresh = .5
      max_output_size = 3
      nms = box_list_ops.non_max_suppression(
          boxes, iou_thresh, max_output_size)
      return nms.get()
    exp_nms = [[0, 0, 1, 1]]
    nms_output = self.execute_cpu(graph_fn, [])
    self.assertAllClose(nms_output, exp_nms)

  def test_copy_extra_fields(self):
    tensor1 = np.array([[1], [4]])
    tensor2 = np.array([[1, 1], [2, 2]])
    def graph_fn():
      corners = tf.constant([[0, 0, 1, 1],
                             [0, 0.1, 1, 1.1]], tf.float32)
      boxes = box_list.BoxList(corners)

      boxes.add_field('tensor1', tf.constant(tensor1))
      boxes.add_field('tensor2', tf.constant(tensor2))
      new_boxes = box_list.BoxList(tf.constant([[0, 0, 10, 10],
                                                [1, 3, 5, 5]], tf.float32))
      new_boxes = box_list_ops._copy_extra_fields(new_boxes, boxes)
      return new_boxes.get_field('tensor1'), new_boxes.get_field('tensor2')
    tensor1_out, tensor2_out = self.execute_cpu(graph_fn, [])
    self.assertAllClose(tensor1, tensor1_out)
    self.assertAllClose(tensor2, tensor2_out)


class CoordinatesConversionTest(test_case.TestCase):

  def test_to_normalized_coordinates(self):
    def graph_fn():
      coordinates = tf.constant([[0, 0, 100, 100],
                                 [25, 25, 75, 75]], tf.float32)
      img = tf.ones((128, 100, 100, 3))
      boxlist = box_list.BoxList(coordinates)
      normalized_boxlist = box_list_ops.to_normalized_coordinates(
          boxlist, tf.shape(img)[1], tf.shape(img)[2])
      return normalized_boxlist.get()
    expected_boxes = [[0, 0, 1, 1],
                      [0.25, 0.25, 0.75, 0.75]]
    normalized_boxes = self.execute(graph_fn, [])
    self.assertAllClose(normalized_boxes, expected_boxes)

  def test_to_normalized_coordinates_already_normalized(self):
    def graph_fn():
      coordinates = tf.constant([[0, 0, 1, 1],
                                 [0.25, 0.25, 0.75, 0.75]], tf.float32)
      img = tf.ones((128, 100, 100, 3))
      boxlist = box_list.BoxList(coordinates)
      normalized_boxlist = box_list_ops.to_normalized_coordinates(
          boxlist, tf.shape(img)[1], tf.shape(img)[2])
      return normalized_boxlist.get()
    with self.assertRaisesOpError('assertion failed'):
      self.execute_cpu(graph_fn, [])

  def test_to_absolute_coordinates(self):
    def graph_fn():
      coordinates = tf.constant([[0, 0, 1, 1],
                                 [0.25, 0.25, 0.75, 0.75]], tf.float32)
      img = tf.ones((128, 100, 100, 3))
      boxlist = box_list.BoxList(coordinates)
      absolute_boxlist = box_list_ops.to_absolute_coordinates(boxlist,
                                                              tf.shape(img)[1],
                                                              tf.shape(img)[2])
      return absolute_boxlist.get()
    expected_boxes = [[0, 0, 100, 100],
                      [25, 25, 75, 75]]
    absolute_boxes = self.execute(graph_fn, [])
    self.assertAllClose(absolute_boxes, expected_boxes)

  def test_to_absolute_coordinates_already_abolute(self):
    def graph_fn():
      coordinates = tf.constant([[0, 0, 100, 100],
                                 [25, 25, 75, 75]], tf.float32)
      img = tf.ones((128, 100, 100, 3))
      boxlist = box_list.BoxList(coordinates)
      absolute_boxlist = box_list_ops.to_absolute_coordinates(boxlist,
                                                              tf.shape(img)[1],
                                                              tf.shape(img)[2])
      return absolute_boxlist.get()
    with self.assertRaisesOpError('assertion failed'):
      self.execute_cpu(graph_fn, [])

  def test_convert_to_normalized_and_back(self):
    coordinates = np.random.uniform(size=(100, 4))
    coordinates = np.round(np.sort(coordinates) * 200)
    coordinates[:, 2:4] += 1
    coordinates[99, :] = [0, 0, 201, 201]
    def graph_fn():
      img = tf.ones((128, 202, 202, 3))

      boxlist = box_list.BoxList(tf.constant(coordinates, tf.float32))
      boxlist = box_list_ops.to_normalized_coordinates(boxlist,
                                                       tf.shape(img)[1],
                                                       tf.shape(img)[2])
      boxlist = box_list_ops.to_absolute_coordinates(boxlist,
                                                     tf.shape(img)[1],
                                                     tf.shape(img)[2])
      return boxlist.get()
    out = self.execute(graph_fn, [])
    self.assertAllClose(out, coordinates)

  def test_convert_to_absolute_and_back(self):
    coordinates = np.random.uniform(size=(100, 4))
    coordinates = np.sort(coordinates)
    coordinates[99, :] = [0, 0, 1, 1]
    def graph_fn():
      img = tf.ones((128, 202, 202, 3))
      boxlist = box_list.BoxList(tf.constant(coordinates, tf.float32))
      boxlist = box_list_ops.to_absolute_coordinates(boxlist,
                                                     tf.shape(img)[1],
                                                     tf.shape(img)[2])
      boxlist = box_list_ops.to_normalized_coordinates(boxlist,
                                                       tf.shape(img)[1],
                                                       tf.shape(img)[2])
      return boxlist.get()
    out = self.execute(graph_fn, [])
    self.assertAllClose(out, coordinates)

  def test_to_absolute_coordinates_maximum_coordinate_check(self):
    def graph_fn():
      coordinates = tf.constant([[0, 0, 1.2, 1.2],
                                 [0.25, 0.25, 0.75, 0.75]], tf.float32)
      img = tf.ones((128, 100, 100, 3))
      boxlist = box_list.BoxList(coordinates)
      absolute_boxlist = box_list_ops.to_absolute_coordinates(
          boxlist,
          tf.shape(img)[1],
          tf.shape(img)[2],
          maximum_normalized_coordinate=1.1)
      return absolute_boxlist.get()
    with self.assertRaisesOpError('assertion failed'):
      self.execute_cpu(graph_fn, [])


class BoxRefinementTest(test_case.TestCase):

  def test_box_voting(self):
    def graph_fn():
      candidates = box_list.BoxList(
          tf.constant([[0.1, 0.1, 0.4, 0.4], [0.6, 0.6, 0.8, 0.8]], tf.float32))
      candidates.add_field('ExtraField', tf.constant([1, 2]))
      pool = box_list.BoxList(
          tf.constant([[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5],
                       [0.6, 0.6, 0.8, 0.8]], tf.float32))
      pool.add_field('scores', tf.constant([0.75, 0.25, 0.3]))
      averaged_boxes = box_list_ops.box_voting(candidates, pool)
      return (averaged_boxes.get(), averaged_boxes.get_field('scores'),
              averaged_boxes.get_field('ExtraField'))

    expected_boxes = [[0.1, 0.1, 0.425, 0.425], [0.6, 0.6, 0.8, 0.8]]
    expected_scores = [0.5, 0.3]
    boxes_out, scores_out, extra_field_out = self.execute(graph_fn, [])
    self.assertAllClose(expected_boxes, boxes_out)
    self.assertAllClose(expected_scores, scores_out)
    self.assertAllEqual(extra_field_out, [1, 2])

  def test_box_voting_fails_with_negative_scores(self):
    def graph_fn():
      candidates = box_list.BoxList(
          tf.constant([[0.1, 0.1, 0.4, 0.4]], tf.float32))
      pool = box_list.BoxList(tf.constant([[0.1, 0.1, 0.4, 0.4]], tf.float32))
      pool.add_field('scores', tf.constant([-0.2]))
      averaged_boxes = box_list_ops.box_voting(candidates, pool)
      return averaged_boxes.get()

    with self.assertRaisesOpError('Scores must be non negative'):
      self.execute_cpu(graph_fn, [])

  def test_box_voting_fails_when_unmatched(self):
    def graph_fn():
      candidates = box_list.BoxList(
          tf.constant([[0.1, 0.1, 0.4, 0.4]], tf.float32))
      pool = box_list.BoxList(tf.constant([[0.6, 0.6, 0.8, 0.8]], tf.float32))
      pool.add_field('scores', tf.constant([0.2]))
      averaged_boxes = box_list_ops.box_voting(candidates, pool)
      return averaged_boxes.get()
    with self.assertRaisesOpError('Each box in selected_boxes must match '
                                  'with at least one box in pool_boxes.'):
      self.execute_cpu(graph_fn, [])

  def test_refine_boxes(self):
    def graph_fn():
      pool = box_list.BoxList(
          tf.constant([[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5],
                       [0.6, 0.6, 0.8, 0.8]], tf.float32))
      pool.add_field('ExtraField', tf.constant([1, 2, 3]))
      pool.add_field('scores', tf.constant([0.75, 0.25, 0.3]))
      averaged_boxes = box_list_ops.refine_boxes(pool, 0.5, 10)
      return (averaged_boxes.get(), averaged_boxes.get_field('scores'),
              averaged_boxes.get_field('ExtraField'))
    boxes_out, scores_out, extra_field_out = self.execute_cpu(graph_fn, [])
    expected_boxes = [[0.1, 0.1, 0.425, 0.425], [0.6, 0.6, 0.8, 0.8]]
    expected_scores = [0.5, 0.3]
    self.assertAllClose(expected_boxes, boxes_out)
    self.assertAllClose(expected_scores, scores_out)
    self.assertAllEqual(extra_field_out, [1, 3])

  def test_refine_boxes_multi_class(self):
    def graph_fn():
      pool = box_list.BoxList(
          tf.constant([[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5],
                       [0.6, 0.6, 0.8, 0.8], [0.2, 0.2, 0.3, 0.3]], tf.float32))
      pool.add_field('classes', tf.constant([0, 0, 1, 1]))
      pool.add_field('scores', tf.constant([0.75, 0.25, 0.3, 0.2]))
      averaged_boxes = box_list_ops.refine_boxes_multi_class(pool, 3, 0.5, 10)
      return (averaged_boxes.get(), averaged_boxes.get_field('scores'),
              averaged_boxes.get_field('classes'))
    boxes_out, scores_out, extra_field_out = self.execute_cpu(graph_fn, [])
    expected_boxes = [[0.1, 0.1, 0.425, 0.425], [0.6, 0.6, 0.8, 0.8],
                      [0.2, 0.2, 0.3, 0.3]]
    expected_scores = [0.5, 0.3, 0.2]
    self.assertAllClose(expected_boxes, boxes_out)
    self.assertAllClose(expected_scores, scores_out)
    self.assertAllEqual(extra_field_out, [0, 1, 1])

  def test_sample_boxes_by_jittering(self):
    def graph_fn():
      boxes = box_list.BoxList(
          tf.constant([[0.1, 0.1, 0.4, 0.4],
                       [0.1, 0.1, 0.5, 0.5],
                       [0.6, 0.6, 0.8, 0.8],
                       [0.2, 0.2, 0.3, 0.3]], tf.float32))
      sampled_boxes = box_list_ops.sample_boxes_by_jittering(
          boxlist=boxes, num_boxes_to_sample=10)
      iou = box_list_ops.iou(boxes, sampled_boxes)
      iou_max = tf.reduce_max(iou, axis=0)
      return sampled_boxes.get(), iou_max
    np_sampled_boxes, np_iou_max = self.execute(graph_fn, [])
    self.assertAllEqual(np_sampled_boxes.shape, [10, 4])
    self.assertAllGreater(np_iou_max, 0.3)


if __name__ == '__main__':
  tf.test.main()
