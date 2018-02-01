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
import tensorflow as tf
from tensorflow.python.framework import errors

from object_detection.core import box_list
from object_detection.core import box_list_ops


class BoxListOpsTest(tf.test.TestCase):
  """Tests for common bounding box operations."""

  def test_area(self):
    corners = tf.constant([[0.0, 0.0, 10.0, 20.0], [1.0, 2.0, 3.0, 4.0]])
    exp_output = [200.0, 4.0]
    boxes = box_list.BoxList(corners)
    areas = box_list_ops.area(boxes)
    with self.test_session() as sess:
      areas_output = sess.run(areas)
      self.assertAllClose(areas_output, exp_output)

  def test_height_width(self):
    corners = tf.constant([[0.0, 0.0, 10.0, 20.0], [1.0, 2.0, 3.0, 4.0]])
    exp_output_heights = [10., 2.]
    exp_output_widths = [20., 2.]
    boxes = box_list.BoxList(corners)
    heights, widths = box_list_ops.height_width(boxes)
    with self.test_session() as sess:
      output_heights, output_widths = sess.run([heights, widths])
      self.assertAllClose(output_heights, exp_output_heights)
      self.assertAllClose(output_widths, exp_output_widths)

  def test_scale(self):
    corners = tf.constant([[0, 0, 100, 200], [50, 120, 100, 140]],
                          dtype=tf.float32)
    boxes = box_list.BoxList(corners)
    boxes.add_field('extra_data', tf.constant([[1], [2]]))

    y_scale = tf.constant(1.0/100)
    x_scale = tf.constant(1.0/200)
    scaled_boxes = box_list_ops.scale(boxes, y_scale, x_scale)
    exp_output = [[0, 0, 1, 1], [0.5, 0.6, 1.0, 0.7]]
    with self.test_session() as sess:
      scaled_corners_out = sess.run(scaled_boxes.get())
      self.assertAllClose(scaled_corners_out, exp_output)
      extra_data_out = sess.run(scaled_boxes.get_field('extra_data'))
      self.assertAllEqual(extra_data_out, [[1], [2]])

  def test_clip_to_window_filter_boxes_which_fall_outside_the_window(
      self):
    window = tf.constant([0, 0, 9, 14], tf.float32)
    corners = tf.constant([[5.0, 5.0, 6.0, 6.0],
                           [-1.0, -2.0, 4.0, 5.0],
                           [2.0, 3.0, 5.0, 9.0],
                           [0.0, 0.0, 9.0, 14.0],
                           [-100.0, -100.0, 300.0, 600.0],
                           [-10.0, -10.0, -9.0, -9.0]])
    boxes = box_list.BoxList(corners)
    boxes.add_field('extra_data', tf.constant([[1], [2], [3], [4], [5], [6]]))
    exp_output = [[5.0, 5.0, 6.0, 6.0], [0.0, 0.0, 4.0, 5.0],
                  [2.0, 3.0, 5.0, 9.0], [0.0, 0.0, 9.0, 14.0],
                  [0.0, 0.0, 9.0, 14.0]]
    pruned = box_list_ops.clip_to_window(
        boxes, window, filter_nonoverlapping=True)
    with self.test_session() as sess:
      pruned_output = sess.run(pruned.get())
      self.assertAllClose(pruned_output, exp_output)
      extra_data_out = sess.run(pruned.get_field('extra_data'))
      self.assertAllEqual(extra_data_out, [[1], [2], [3], [4], [5]])

  def test_clip_to_window_without_filtering_boxes_which_fall_outside_the_window(
      self):
    window = tf.constant([0, 0, 9, 14], tf.float32)
    corners = tf.constant([[5.0, 5.0, 6.0, 6.0],
                           [-1.0, -2.0, 4.0, 5.0],
                           [2.0, 3.0, 5.0, 9.0],
                           [0.0, 0.0, 9.0, 14.0],
                           [-100.0, -100.0, 300.0, 600.0],
                           [-10.0, -10.0, -9.0, -9.0]])
    boxes = box_list.BoxList(corners)
    boxes.add_field('extra_data', tf.constant([[1], [2], [3], [4], [5], [6]]))
    exp_output = [[5.0, 5.0, 6.0, 6.0], [0.0, 0.0, 4.0, 5.0],
                  [2.0, 3.0, 5.0, 9.0], [0.0, 0.0, 9.0, 14.0],
                  [0.0, 0.0, 9.0, 14.0], [0.0, 0.0, 0.0, 0.0]]
    pruned = box_list_ops.clip_to_window(
        boxes, window, filter_nonoverlapping=False)
    with self.test_session() as sess:
      pruned_output = sess.run(pruned.get())
      self.assertAllClose(pruned_output, exp_output)
      extra_data_out = sess.run(pruned.get_field('extra_data'))
      self.assertAllEqual(extra_data_out, [[1], [2], [3], [4], [5], [6]])

  def test_prune_outside_window_filters_boxes_which_fall_outside_the_window(
      self):
    window = tf.constant([0, 0, 9, 14], tf.float32)
    corners = tf.constant([[5.0, 5.0, 6.0, 6.0],
                           [-1.0, -2.0, 4.0, 5.0],
                           [2.0, 3.0, 5.0, 9.0],
                           [0.0, 0.0, 9.0, 14.0],
                           [-10.0, -10.0, -9.0, -9.0],
                           [-100.0, -100.0, 300.0, 600.0]])
    boxes = box_list.BoxList(corners)
    boxes.add_field('extra_data', tf.constant([[1], [2], [3], [4], [5], [6]]))
    exp_output = [[5.0, 5.0, 6.0, 6.0],
                  [2.0, 3.0, 5.0, 9.0],
                  [0.0, 0.0, 9.0, 14.0]]
    pruned, keep_indices = box_list_ops.prune_outside_window(boxes, window)
    with self.test_session() as sess:
      pruned_output = sess.run(pruned.get())
      self.assertAllClose(pruned_output, exp_output)
      keep_indices_out = sess.run(keep_indices)
      self.assertAllEqual(keep_indices_out, [0, 2, 3])
      extra_data_out = sess.run(pruned.get_field('extra_data'))
      self.assertAllEqual(extra_data_out, [[1], [3], [4]])

  def test_prune_completely_outside_window(self):
    window = tf.constant([0, 0, 9, 14], tf.float32)
    corners = tf.constant([[5.0, 5.0, 6.0, 6.0],
                           [-1.0, -2.0, 4.0, 5.0],
                           [2.0, 3.0, 5.0, 9.0],
                           [0.0, 0.0, 9.0, 14.0],
                           [-10.0, -10.0, -9.0, -9.0],
                           [-100.0, -100.0, 300.0, 600.0]])
    boxes = box_list.BoxList(corners)
    boxes.add_field('extra_data', tf.constant([[1], [2], [3], [4], [5], [6]]))
    exp_output = [[5.0, 5.0, 6.0, 6.0],
                  [-1.0, -2.0, 4.0, 5.0],
                  [2.0, 3.0, 5.0, 9.0],
                  [0.0, 0.0, 9.0, 14.0],
                  [-100.0, -100.0, 300.0, 600.0]]
    pruned, keep_indices = box_list_ops.prune_completely_outside_window(boxes,
                                                                        window)
    with self.test_session() as sess:
      pruned_output = sess.run(pruned.get())
      self.assertAllClose(pruned_output, exp_output)
      keep_indices_out = sess.run(keep_indices)
      self.assertAllEqual(keep_indices_out, [0, 1, 2, 3, 5])
      extra_data_out = sess.run(pruned.get_field('extra_data'))
      self.assertAllEqual(extra_data_out, [[1], [2], [3], [4], [6]])

  def test_intersection(self):
    corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                            [0.0, 0.0, 20.0, 20.0]])
    exp_output = [[2.0, 0.0, 6.0], [1.0, 0.0, 5.0]]
    boxes1 = box_list.BoxList(corners1)
    boxes2 = box_list.BoxList(corners2)
    intersect = box_list_ops.intersection(boxes1, boxes2)
    with self.test_session() as sess:
      intersect_output = sess.run(intersect)
      self.assertAllClose(intersect_output, exp_output)

  def test_matched_intersection(self):
    corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
    exp_output = [2.0, 0.0]
    boxes1 = box_list.BoxList(corners1)
    boxes2 = box_list.BoxList(corners2)
    intersect = box_list_ops.matched_intersection(boxes1, boxes2)
    with self.test_session() as sess:
      intersect_output = sess.run(intersect)
      self.assertAllClose(intersect_output, exp_output)

  def test_iou(self):
    corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                            [0.0, 0.0, 20.0, 20.0]])
    exp_output = [[2.0 / 16.0, 0, 6.0 / 400.0], [1.0 / 16.0, 0.0, 5.0 / 400.0]]
    boxes1 = box_list.BoxList(corners1)
    boxes2 = box_list.BoxList(corners2)
    iou = box_list_ops.iou(boxes1, boxes2)
    with self.test_session() as sess:
      iou_output = sess.run(iou)
      self.assertAllClose(iou_output, exp_output)

  def test_matched_iou(self):
    corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0]])
    exp_output = [2.0 / 16.0, 0]
    boxes1 = box_list.BoxList(corners1)
    boxes2 = box_list.BoxList(corners2)
    iou = box_list_ops.matched_iou(boxes1, boxes2)
    with self.test_session() as sess:
      iou_output = sess.run(iou)
      self.assertAllClose(iou_output, exp_output)

  def test_iouworks_on_empty_inputs(self):
    corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                            [0.0, 0.0, 20.0, 20.0]])
    boxes1 = box_list.BoxList(corners1)
    boxes2 = box_list.BoxList(corners2)
    boxes_empty = box_list.BoxList(tf.zeros((0, 4)))
    iou_empty_1 = box_list_ops.iou(boxes1, boxes_empty)
    iou_empty_2 = box_list_ops.iou(boxes_empty, boxes2)
    iou_empty_3 = box_list_ops.iou(boxes_empty, boxes_empty)
    with self.test_session() as sess:
      iou_output_1, iou_output_2, iou_output_3 = sess.run(
          [iou_empty_1, iou_empty_2, iou_empty_3])
      self.assertAllEqual(iou_output_1.shape, (2, 0))
      self.assertAllEqual(iou_output_2.shape, (0, 3))
      self.assertAllEqual(iou_output_3.shape, (0, 0))

  def test_ioa(self):
    corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                            [0.0, 0.0, 20.0, 20.0]])
    exp_output_1 = [[2.0 / 12.0, 0, 6.0 / 400.0],
                    [1.0 / 12.0, 0.0, 5.0 / 400.0]]
    exp_output_2 = [[2.0 / 6.0, 1.0 / 5.0],
                    [0, 0],
                    [6.0 / 6.0, 5.0 / 5.0]]
    boxes1 = box_list.BoxList(corners1)
    boxes2 = box_list.BoxList(corners2)
    ioa_1 = box_list_ops.ioa(boxes1, boxes2)
    ioa_2 = box_list_ops.ioa(boxes2, boxes1)
    with self.test_session() as sess:
      ioa_output_1, ioa_output_2 = sess.run([ioa_1, ioa_2])
      self.assertAllClose(ioa_output_1, exp_output_1)
      self.assertAllClose(ioa_output_2, exp_output_2)

  def test_prune_non_overlapping_boxes(self):
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
    with self.test_session() as sess:
      (output_1_, keep_indices_1_, output_2_, keep_indices_2_, exp_output_1_,
       exp_output_2_) = sess.run(
           [output_1.get(), keep_indices_1,
            output_2.get(), keep_indices_2,
            exp_output_1.get(), exp_output_2.get()])
      self.assertAllClose(output_1_, exp_output_1_)
      self.assertAllClose(output_2_, exp_output_2_)
      self.assertAllEqual(keep_indices_1_, [0, 1])
      self.assertAllEqual(keep_indices_2_, [])

  def test_prune_small_boxes(self):
    boxes = tf.constant([[4.0, 3.0, 7.0, 5.0],
                         [5.0, 6.0, 10.0, 7.0],
                         [3.0, 4.0, 6.0, 8.0],
                         [14.0, 14.0, 15.0, 15.0],
                         [0.0, 0.0, 20.0, 20.0]])
    exp_boxes = [[3.0, 4.0, 6.0, 8.0],
                 [0.0, 0.0, 20.0, 20.0]]
    boxes = box_list.BoxList(boxes)
    pruned_boxes = box_list_ops.prune_small_boxes(boxes, 3)
    with self.test_session() as sess:
      pruned_boxes = sess.run(pruned_boxes.get())
      self.assertAllEqual(pruned_boxes, exp_boxes)

  def test_prune_small_boxes_prunes_boxes_with_negative_side(self):
    boxes = tf.constant([[4.0, 3.0, 7.0, 5.0],
                         [5.0, 6.0, 10.0, 7.0],
                         [3.0, 4.0, 6.0, 8.0],
                         [14.0, 14.0, 15.0, 15.0],
                         [0.0, 0.0, 20.0, 20.0],
                         [2.0, 3.0, 1.5, 7.0],  # negative height
                         [2.0, 3.0, 5.0, 1.7]])  # negative width
    exp_boxes = [[3.0, 4.0, 6.0, 8.0],
                 [0.0, 0.0, 20.0, 20.0]]
    boxes = box_list.BoxList(boxes)
    pruned_boxes = box_list_ops.prune_small_boxes(boxes, 3)
    with self.test_session() as sess:
      pruned_boxes = sess.run(pruned_boxes.get())
      self.assertAllEqual(pruned_boxes, exp_boxes)

  def test_change_coordinate_frame(self):
    corners = tf.constant([[0.25, 0.5, 0.75, 0.75], [0.5, 0.0, 1.0, 1.0]])
    window = tf.constant([0.25, 0.25, 0.75, 0.75])
    boxes = box_list.BoxList(corners)

    expected_corners = tf.constant([[0, 0.5, 1.0, 1.0], [0.5, -0.5, 1.5, 1.5]])
    expected_boxes = box_list.BoxList(expected_corners)
    output = box_list_ops.change_coordinate_frame(boxes, window)

    with self.test_session() as sess:
      output_, expected_boxes_ = sess.run([output.get(), expected_boxes.get()])
      self.assertAllClose(output_, expected_boxes_)

  def test_ioaworks_on_empty_inputs(self):
    corners1 = tf.constant([[4.0, 3.0, 7.0, 5.0], [5.0, 6.0, 10.0, 7.0]])
    corners2 = tf.constant([[3.0, 4.0, 6.0, 8.0], [14.0, 14.0, 15.0, 15.0],
                            [0.0, 0.0, 20.0, 20.0]])
    boxes1 = box_list.BoxList(corners1)
    boxes2 = box_list.BoxList(corners2)
    boxes_empty = box_list.BoxList(tf.zeros((0, 4)))
    ioa_empty_1 = box_list_ops.ioa(boxes1, boxes_empty)
    ioa_empty_2 = box_list_ops.ioa(boxes_empty, boxes2)
    ioa_empty_3 = box_list_ops.ioa(boxes_empty, boxes_empty)
    with self.test_session() as sess:
      ioa_output_1, ioa_output_2, ioa_output_3 = sess.run(
          [ioa_empty_1, ioa_empty_2, ioa_empty_3])
      self.assertAllEqual(ioa_output_1.shape, (2, 0))
      self.assertAllEqual(ioa_output_2.shape, (0, 3))
      self.assertAllEqual(ioa_output_3.shape, (0, 0))

  def test_pairwise_distances(self):
    corners1 = tf.constant([[0.0, 0.0, 0.0, 0.0],
                            [1.0, 1.0, 0.0, 2.0]])
    corners2 = tf.constant([[3.0, 4.0, 1.0, 0.0],
                            [-4.0, 0.0, 0.0, 3.0],
                            [0.0, 0.0, 0.0, 0.0]])
    exp_output = [[26, 25, 0], [18, 27, 6]]
    boxes1 = box_list.BoxList(corners1)
    boxes2 = box_list.BoxList(corners2)
    dist_matrix = box_list_ops.sq_dist(boxes1, boxes2)
    with self.test_session() as sess:
      dist_output = sess.run(dist_matrix)
      self.assertAllClose(dist_output, exp_output)

  def test_boolean_mask(self):
    corners = tf.constant(
        [4 * [0.0], 4 * [1.0], 4 * [2.0], 4 * [3.0], 4 * [4.0]])
    indicator = tf.constant([True, False, True, False, True], tf.bool)
    expected_subset = [4 * [0.0], 4 * [2.0], 4 * [4.0]]
    boxes = box_list.BoxList(corners)
    subset = box_list_ops.boolean_mask(boxes, indicator)
    with self.test_session() as sess:
      subset_output = sess.run(subset.get())
      self.assertAllClose(subset_output, expected_subset)

  def test_boolean_mask_with_field(self):
    corners = tf.constant(
        [4 * [0.0], 4 * [1.0], 4 * [2.0], 4 * [3.0], 4 * [4.0]])
    indicator = tf.constant([True, False, True, False, True], tf.bool)
    weights = tf.constant([[.1], [.3], [.5], [.7], [.9]], tf.float32)
    expected_subset = [4 * [0.0], 4 * [2.0], 4 * [4.0]]
    expected_weights = [[.1], [.5], [.9]]

    boxes = box_list.BoxList(corners)
    boxes.add_field('weights', weights)
    subset = box_list_ops.boolean_mask(boxes, indicator, ['weights'])
    with self.test_session() as sess:
      subset_output, weights_output = sess.run(
          [subset.get(), subset.get_field('weights')])
      self.assertAllClose(subset_output, expected_subset)
      self.assertAllClose(weights_output, expected_weights)

  def test_gather(self):
    corners = tf.constant(
        [4 * [0.0], 4 * [1.0], 4 * [2.0], 4 * [3.0], 4 * [4.0]])
    indices = tf.constant([0, 2, 4], tf.int32)
    expected_subset = [4 * [0.0], 4 * [2.0], 4 * [4.0]]
    boxes = box_list.BoxList(corners)
    subset = box_list_ops.gather(boxes, indices)
    with self.test_session() as sess:
      subset_output = sess.run(subset.get())
      self.assertAllClose(subset_output, expected_subset)

  def test_gather_with_field(self):
    corners = tf.constant([4*[0.0], 4*[1.0], 4*[2.0], 4*[3.0], 4*[4.0]])
    indices = tf.constant([0, 2, 4], tf.int32)
    weights = tf.constant([[.1], [.3], [.5], [.7], [.9]], tf.float32)
    expected_subset = [4 * [0.0], 4 * [2.0], 4 * [4.0]]
    expected_weights = [[.1], [.5], [.9]]

    boxes = box_list.BoxList(corners)
    boxes.add_field('weights', weights)
    subset = box_list_ops.gather(boxes, indices, ['weights'])
    with self.test_session() as sess:
      subset_output, weights_output = sess.run(
          [subset.get(), subset.get_field('weights')])
      self.assertAllClose(subset_output, expected_subset)
      self.assertAllClose(weights_output, expected_weights)

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
    corners = tf.constant([4 * [0.0], 4 * [1.0], 4 * [2.0], 4 * [3.0], 4 * [4.0]
                          ])
    weights = tf.constant([.5, .3, .7, .1, .9], tf.float32)
    indices = tf.reshape(tf.where(tf.greater(weights, 0.4)), [-1])
    expected_subset = [4 * [0.0], 4 * [2.0], 4 * [4.0]]
    expected_weights = [.5, .7, .9]

    boxes = box_list.BoxList(corners)
    boxes.add_field('weights', weights)
    subset = box_list_ops.gather(boxes, indices, ['weights'])
    with self.test_session() as sess:
      subset_output, weights_output = sess.run([subset.get(), subset.get_field(
          'weights')])
      self.assertAllClose(subset_output, expected_subset)
      self.assertAllClose(weights_output, expected_weights)

  def test_sort_by_field_ascending_order(self):
    exp_corners = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                   [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    exp_scores = [.95, .9, .75, .6, .5, .3]
    exp_weights = [.2, .45, .6, .75, .8, .92]
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
    with self.test_session() as sess:
      corners_out, scores_out, weights_out = sess.run([
          sort_by_weight.get(),
          sort_by_weight.get_field('scores'),
          sort_by_weight.get_field('weights')])
      self.assertAllClose(corners_out, exp_corners)
      self.assertAllClose(scores_out, exp_scores)
      self.assertAllClose(weights_out, exp_weights)

  def test_sort_by_field_descending_order(self):
    exp_corners = [[0, 0, 1, 1], [0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9],
                   [0, 10, 1, 11], [0, 10.1, 1, 11.1], [0, 100, 1, 101]]
    exp_scores = [.95, .9, .75, .6, .5, .3]
    exp_weights = [.2, .45, .6, .75, .8, .92]
    shuffle = [2, 4, 0, 5, 1, 3]

    corners = tf.constant([exp_corners[i] for i in shuffle], tf.float32)
    boxes = box_list.BoxList(corners)
    boxes.add_field('scores', tf.constant(
        [exp_scores[i] for i in shuffle], tf.float32))
    boxes.add_field('weights', tf.constant(
        [exp_weights[i] for i in shuffle], tf.float32))

    sort_by_score = box_list_ops.sort_by_field(boxes, 'scores')
    with self.test_session() as sess:
      corners_out, scores_out, weights_out = sess.run([sort_by_score.get(
      ), sort_by_score.get_field('scores'), sort_by_score.get_field('weights')])
      self.assertAllClose(corners_out, exp_corners)
      self.assertAllClose(scores_out, exp_scores)
      self.assertAllClose(weights_out, exp_weights)

  def test_sort_by_field_invalid_inputs(self):
    corners = tf.constant([4 * [0.0], 4 * [0.5], 4 * [1.0], 4 * [2.0], 4 *
                           [3.0], 4 * [4.0]])
    misc = tf.constant([[.95, .9], [.5, .3]], tf.float32)
    weights = tf.constant([.1, .2], tf.float32)
    boxes = box_list.BoxList(corners)
    boxes.add_field('misc', misc)
    boxes.add_field('weights', weights)

    with self.test_session() as sess:
      with self.assertRaises(ValueError):
        box_list_ops.sort_by_field(boxes, 'area')

      with self.assertRaises(ValueError):
        box_list_ops.sort_by_field(boxes, 'misc')

      with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError,
                                               'Incorrect field size'):
        sess.run(box_list_ops.sort_by_field(boxes, 'weights').get())

  def test_visualize_boxes_in_image(self):
    image = tf.zeros((6, 4, 3))
    corners = tf.constant([[0, 0, 5, 3],
                           [0, 0, 3, 2]], tf.float32)
    boxes = box_list.BoxList(corners)
    image_and_boxes = box_list_ops.visualize_boxes_in_image(image, boxes)
    image_and_boxes_bw = tf.to_float(
        tf.greater(tf.reduce_sum(image_and_boxes, 2), 0.0))
    exp_result = [[1, 1, 1, 0],
                  [1, 1, 1, 0],
                  [1, 1, 1, 0],
                  [1, 0, 1, 0],
                  [1, 1, 1, 0],
                  [0, 0, 0, 0]]
    with self.test_session() as sess:
      output = sess.run(image_and_boxes_bw)
      self.assertAllEqual(output.astype(int), exp_result)

  def test_filter_field_value_equals(self):
    corners = tf.constant([[0, 0, 1, 1],
                           [0, 0.1, 1, 1.1],
                           [0, -0.1, 1, 0.9],
                           [0, 10, 1, 11],
                           [0, 10.1, 1, 11.1],
                           [0, 100, 1, 101]], tf.float32)
    boxes = box_list.BoxList(corners)
    boxes.add_field('classes', tf.constant([1, 2, 1, 2, 2, 1]))
    exp_output1 = [[0, 0, 1, 1], [0, -0.1, 1, 0.9], [0, 100, 1, 101]]
    exp_output2 = [[0, 0.1, 1, 1.1], [0, 10, 1, 11], [0, 10.1, 1, 11.1]]

    filtered_boxes1 = box_list_ops.filter_field_value_equals(
        boxes, 'classes', 1)
    filtered_boxes2 = box_list_ops.filter_field_value_equals(
        boxes, 'classes', 2)
    with self.test_session() as sess:
      filtered_output1, filtered_output2 = sess.run([filtered_boxes1.get(),
                                                     filtered_boxes2.get()])
      self.assertAllClose(filtered_output1, exp_output1)
      self.assertAllClose(filtered_output2, exp_output2)

  def test_filter_greater_than(self):
    corners = tf.constant([[0, 0, 1, 1],
                           [0, 0.1, 1, 1.1],
                           [0, -0.1, 1, 0.9],
                           [0, 10, 1, 11],
                           [0, 10.1, 1, 11.1],
                           [0, 100, 1, 101]], tf.float32)
    boxes = box_list.BoxList(corners)
    boxes.add_field('scores', tf.constant([.1, .75, .9, .5, .5, .8]))
    thresh = .6
    exp_output = [[0, 0.1, 1, 1.1], [0, -0.1, 1, 0.9], [0, 100, 1, 101]]

    filtered_boxes = box_list_ops.filter_greater_than(boxes, thresh)
    with self.test_session() as sess:
      filtered_output = sess.run(filtered_boxes.get())
      self.assertAllClose(filtered_output, exp_output)

  def test_clip_box_list(self):
    boxlist = box_list.BoxList(
        tf.constant([[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5],
                     [0.6, 0.6, 0.8, 0.8], [0.2, 0.2, 0.3, 0.3]], tf.float32))
    boxlist.add_field('classes', tf.constant([0, 0, 1, 1]))
    boxlist.add_field('scores', tf.constant([0.75, 0.65, 0.3, 0.2]))
    num_boxes = 2
    clipped_boxlist = box_list_ops.pad_or_clip_box_list(boxlist, num_boxes)

    expected_boxes = [[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5]]
    expected_classes = [0, 0]
    expected_scores = [0.75, 0.65]
    with self.test_session() as sess:
      boxes_out, classes_out, scores_out = sess.run(
          [clipped_boxlist.get(), clipped_boxlist.get_field('classes'),
           clipped_boxlist.get_field('scores')])

      self.assertAllClose(expected_boxes, boxes_out)
      self.assertAllEqual(expected_classes, classes_out)
      self.assertAllClose(expected_scores, scores_out)

  def test_pad_box_list(self):
    boxlist = box_list.BoxList(
        tf.constant([[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5]], tf.float32))
    boxlist.add_field('classes', tf.constant([0, 1]))
    boxlist.add_field('scores', tf.constant([0.75, 0.2]))
    num_boxes = 4
    padded_boxlist = box_list_ops.pad_or_clip_box_list(boxlist, num_boxes)

    expected_boxes = [[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5],
                      [0, 0, 0, 0], [0, 0, 0, 0]]
    expected_classes = [0, 1, 0, 0]
    expected_scores = [0.75, 0.2, 0, 0]
    with self.test_session() as sess:
      boxes_out, classes_out, scores_out = sess.run(
          [padded_boxlist.get(), padded_boxlist.get_field('classes'),
           padded_boxlist.get_field('scores')])

      self.assertAllClose(expected_boxes, boxes_out)
      self.assertAllEqual(expected_classes, classes_out)
      self.assertAllClose(expected_scores, scores_out)


class ConcatenateTest(tf.test.TestCase):

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
    corners1 = tf.constant([[0, 0, 0, 0], [1, 2, 3, 4]], tf.float32)
    scores1 = tf.constant([1.0, 2.1])
    corners2 = tf.constant([[0, 3, 1, 6], [2, 4, 3, 8], [1, 0, 5, 10]],
                           tf.float32)
    scores2 = tf.constant([1.0, 2.1, 5.6])

    exp_corners = [[0, 0, 0, 0],
                   [1, 2, 3, 4],
                   [0, 3, 1, 6],
                   [2, 4, 3, 8],
                   [1, 0, 5, 10]]
    exp_scores = [1.0, 2.1, 1.0, 2.1, 5.6]

    boxlist1 = box_list.BoxList(corners1)
    boxlist1.add_field('scores', scores1)
    boxlist2 = box_list.BoxList(corners2)
    boxlist2.add_field('scores', scores2)
    result = box_list_ops.concatenate([boxlist1, boxlist2])
    with self.test_session() as sess:
      corners_output, scores_output = sess.run(
          [result.get(), result.get_field('scores')])
      self.assertAllClose(corners_output, exp_corners)
      self.assertAllClose(scores_output, exp_scores)


class NonMaxSuppressionTest(tf.test.TestCase):

  def test_with_invalid_scores_field(self):
    corners = tf.constant([[0, 0, 1, 1],
                           [0, 0.1, 1, 1.1],
                           [0, -0.1, 1, 0.9],
                           [0, 10, 1, 11],
                           [0, 10.1, 1, 11.1],
                           [0, 100, 1, 101]], tf.float32)
    boxes = box_list.BoxList(corners)
    boxes.add_field('scores', tf.constant([.9, .75, .6, .95, .5]))
    iou_thresh = .5
    max_output_size = 3
    nms = box_list_ops.non_max_suppression(
        boxes, iou_thresh, max_output_size)
    with self.test_session() as sess:
      with self.assertRaisesWithPredicateMatch(
          errors.InvalidArgumentError, 'scores has incompatible shape'):
        sess.run(nms.get())

  def test_select_from_three_clusters(self):
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

    exp_nms = [[0, 10, 1, 11],
               [0, 0, 1, 1],
               [0, 100, 1, 101]]
    nms = box_list_ops.non_max_suppression(
        boxes, iou_thresh, max_output_size)
    with self.test_session() as sess:
      nms_output = sess.run(nms.get())
      self.assertAllClose(nms_output, exp_nms)

  def test_select_at_most_two_boxes_from_three_clusters(self):
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

    exp_nms = [[0, 10, 1, 11],
               [0, 0, 1, 1]]
    nms = box_list_ops.non_max_suppression(
        boxes, iou_thresh, max_output_size)
    with self.test_session() as sess:
      nms_output = sess.run(nms.get())
      self.assertAllClose(nms_output, exp_nms)

  def test_select_at_most_thirty_boxes_from_three_clusters(self):
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

    exp_nms = [[0, 10, 1, 11],
               [0, 0, 1, 1],
               [0, 100, 1, 101]]
    nms = box_list_ops.non_max_suppression(
        boxes, iou_thresh, max_output_size)
    with self.test_session() as sess:
      nms_output = sess.run(nms.get())
      self.assertAllClose(nms_output, exp_nms)

  def test_select_single_box(self):
    corners = tf.constant([[0, 0, 1, 1]], tf.float32)
    boxes = box_list.BoxList(corners)
    boxes.add_field('scores', tf.constant([.9]))
    iou_thresh = .5
    max_output_size = 3

    exp_nms = [[0, 0, 1, 1]]
    nms = box_list_ops.non_max_suppression(
        boxes, iou_thresh, max_output_size)
    with self.test_session() as sess:
      nms_output = sess.run(nms.get())
      self.assertAllClose(nms_output, exp_nms)

  def test_select_from_ten_identical_boxes(self):
    corners = tf.constant(10 * [[0, 0, 1, 1]], tf.float32)
    boxes = box_list.BoxList(corners)
    boxes.add_field('scores', tf.constant(10 * [.9]))
    iou_thresh = .5
    max_output_size = 3

    exp_nms = [[0, 0, 1, 1]]
    nms = box_list_ops.non_max_suppression(
        boxes, iou_thresh, max_output_size)
    with self.test_session() as sess:
      nms_output = sess.run(nms.get())
      self.assertAllClose(nms_output, exp_nms)

  def test_copy_extra_fields(self):
    corners = tf.constant([[0, 0, 1, 1],
                           [0, 0.1, 1, 1.1]], tf.float32)
    boxes = box_list.BoxList(corners)
    tensor1 = np.array([[1], [4]])
    tensor2 = np.array([[1, 1], [2, 2]])
    boxes.add_field('tensor1', tf.constant(tensor1))
    boxes.add_field('tensor2', tf.constant(tensor2))
    new_boxes = box_list.BoxList(tf.constant([[0, 0, 10, 10],
                                              [1, 3, 5, 5]], tf.float32))
    new_boxes = box_list_ops._copy_extra_fields(new_boxes, boxes)
    with self.test_session() as sess:
      self.assertAllClose(tensor1, sess.run(new_boxes.get_field('tensor1')))
      self.assertAllClose(tensor2, sess.run(new_boxes.get_field('tensor2')))


class CoordinatesConversionTest(tf.test.TestCase):

  def test_to_normalized_coordinates(self):
    coordinates = tf.constant([[0, 0, 100, 100],
                               [25, 25, 75, 75]], tf.float32)
    img = tf.ones((128, 100, 100, 3))
    boxlist = box_list.BoxList(coordinates)
    normalized_boxlist = box_list_ops.to_normalized_coordinates(
        boxlist, tf.shape(img)[1], tf.shape(img)[2])
    expected_boxes = [[0, 0, 1, 1],
                      [0.25, 0.25, 0.75, 0.75]]

    with self.test_session() as sess:
      normalized_boxes = sess.run(normalized_boxlist.get())
      self.assertAllClose(normalized_boxes, expected_boxes)

  def test_to_normalized_coordinates_already_normalized(self):
    coordinates = tf.constant([[0, 0, 1, 1],
                               [0.25, 0.25, 0.75, 0.75]], tf.float32)
    img = tf.ones((128, 100, 100, 3))
    boxlist = box_list.BoxList(coordinates)
    normalized_boxlist = box_list_ops.to_normalized_coordinates(
        boxlist, tf.shape(img)[1], tf.shape(img)[2])

    with self.test_session() as sess:
      with self.assertRaisesOpError('assertion failed'):
        sess.run(normalized_boxlist.get())

  def test_to_absolute_coordinates(self):
    coordinates = tf.constant([[0, 0, 1, 1],
                               [0.25, 0.25, 0.75, 0.75]], tf.float32)
    img = tf.ones((128, 100, 100, 3))
    boxlist = box_list.BoxList(coordinates)
    absolute_boxlist = box_list_ops.to_absolute_coordinates(boxlist,
                                                            tf.shape(img)[1],
                                                            tf.shape(img)[2])
    expected_boxes = [[0, 0, 100, 100],
                      [25, 25, 75, 75]]

    with self.test_session() as sess:
      absolute_boxes = sess.run(absolute_boxlist.get())
      self.assertAllClose(absolute_boxes, expected_boxes)

  def test_to_absolute_coordinates_already_abolute(self):
    coordinates = tf.constant([[0, 0, 100, 100],
                               [25, 25, 75, 75]], tf.float32)
    img = tf.ones((128, 100, 100, 3))
    boxlist = box_list.BoxList(coordinates)
    absolute_boxlist = box_list_ops.to_absolute_coordinates(boxlist,
                                                            tf.shape(img)[1],
                                                            tf.shape(img)[2])

    with self.test_session() as sess:
      with self.assertRaisesOpError('assertion failed'):
        sess.run(absolute_boxlist.get())

  def test_convert_to_normalized_and_back(self):
    coordinates = np.random.uniform(size=(100, 4))
    coordinates = np.round(np.sort(coordinates) * 200)
    coordinates[:, 2:4] += 1
    coordinates[99, :] = [0, 0, 201, 201]
    img = tf.ones((128, 202, 202, 3))

    boxlist = box_list.BoxList(tf.constant(coordinates, tf.float32))
    boxlist = box_list_ops.to_normalized_coordinates(boxlist,
                                                     tf.shape(img)[1],
                                                     tf.shape(img)[2])
    boxlist = box_list_ops.to_absolute_coordinates(boxlist,
                                                   tf.shape(img)[1],
                                                   tf.shape(img)[2])

    with self.test_session() as sess:
      out = sess.run(boxlist.get())
      self.assertAllClose(out, coordinates)

  def test_convert_to_absolute_and_back(self):
    coordinates = np.random.uniform(size=(100, 4))
    coordinates = np.sort(coordinates)
    coordinates[99, :] = [0, 0, 1, 1]
    img = tf.ones((128, 202, 202, 3))

    boxlist = box_list.BoxList(tf.constant(coordinates, tf.float32))
    boxlist = box_list_ops.to_absolute_coordinates(boxlist,
                                                   tf.shape(img)[1],
                                                   tf.shape(img)[2])
    boxlist = box_list_ops.to_normalized_coordinates(boxlist,
                                                     tf.shape(img)[1],
                                                     tf.shape(img)[2])

    with self.test_session() as sess:
      out = sess.run(boxlist.get())
      self.assertAllClose(out, coordinates)


class BoxRefinementTest(tf.test.TestCase):

  def test_box_voting(self):
    candidates = box_list.BoxList(
        tf.constant([[0.1, 0.1, 0.4, 0.4], [0.6, 0.6, 0.8, 0.8]], tf.float32))
    candidates.add_field('ExtraField', tf.constant([1, 2]))
    pool = box_list.BoxList(
        tf.constant([[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5],
                     [0.6, 0.6, 0.8, 0.8]], tf.float32))
    pool.add_field('scores', tf.constant([0.75, 0.25, 0.3]))
    averaged_boxes = box_list_ops.box_voting(candidates, pool)
    expected_boxes = [[0.1, 0.1, 0.425, 0.425], [0.6, 0.6, 0.8, 0.8]]
    expected_scores = [0.5, 0.3]
    with self.test_session() as sess:
      boxes_out, scores_out, extra_field_out = sess.run(
          [averaged_boxes.get(), averaged_boxes.get_field('scores'),
           averaged_boxes.get_field('ExtraField')])

      self.assertAllClose(expected_boxes, boxes_out)
      self.assertAllClose(expected_scores, scores_out)
      self.assertAllEqual(extra_field_out, [1, 2])

  def test_box_voting_fails_with_negative_scores(self):
    candidates = box_list.BoxList(
        tf.constant([[0.1, 0.1, 0.4, 0.4]], tf.float32))
    pool = box_list.BoxList(tf.constant([[0.1, 0.1, 0.4, 0.4]], tf.float32))
    pool.add_field('scores', tf.constant([-0.2]))
    averaged_boxes = box_list_ops.box_voting(candidates, pool)

    with self.test_session() as sess:
      with self.assertRaisesOpError('Scores must be non negative'):
        sess.run([averaged_boxes.get()])

  def test_box_voting_fails_when_unmatched(self):
    candidates = box_list.BoxList(
        tf.constant([[0.1, 0.1, 0.4, 0.4]], tf.float32))
    pool = box_list.BoxList(tf.constant([[0.6, 0.6, 0.8, 0.8]], tf.float32))
    pool.add_field('scores', tf.constant([0.2]))
    averaged_boxes = box_list_ops.box_voting(candidates, pool)

    with self.test_session() as sess:
      with self.assertRaisesOpError('Each box in selected_boxes must match '
                                    'with at least one box in pool_boxes.'):
        sess.run([averaged_boxes.get()])

  def test_refine_boxes(self):
    pool = box_list.BoxList(
        tf.constant([[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5],
                     [0.6, 0.6, 0.8, 0.8]], tf.float32))
    pool.add_field('ExtraField', tf.constant([1, 2, 3]))
    pool.add_field('scores', tf.constant([0.75, 0.25, 0.3]))
    refined_boxes = box_list_ops.refine_boxes(pool, 0.5, 10)

    expected_boxes = [[0.1, 0.1, 0.425, 0.425], [0.6, 0.6, 0.8, 0.8]]
    expected_scores = [0.5, 0.3]
    with self.test_session() as sess:
      boxes_out, scores_out, extra_field_out = sess.run(
          [refined_boxes.get(), refined_boxes.get_field('scores'),
           refined_boxes.get_field('ExtraField')])

      self.assertAllClose(expected_boxes, boxes_out)
      self.assertAllClose(expected_scores, scores_out)
      self.assertAllEqual(extra_field_out, [1, 3])

  def test_refine_boxes_multi_class(self):
    pool = box_list.BoxList(
        tf.constant([[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.5, 0.5],
                     [0.6, 0.6, 0.8, 0.8], [0.2, 0.2, 0.3, 0.3]], tf.float32))
    pool.add_field('classes', tf.constant([0, 0, 1, 1]))
    pool.add_field('scores', tf.constant([0.75, 0.25, 0.3, 0.2]))
    refined_boxes = box_list_ops.refine_boxes_multi_class(pool, 3, 0.5, 10)

    expected_boxes = [[0.1, 0.1, 0.425, 0.425], [0.6, 0.6, 0.8, 0.8],
                      [0.2, 0.2, 0.3, 0.3]]
    expected_scores = [0.5, 0.3, 0.2]
    with self.test_session() as sess:
      boxes_out, scores_out, extra_field_out = sess.run(
          [refined_boxes.get(), refined_boxes.get_field('scores'),
           refined_boxes.get_field('classes')])

      self.assertAllClose(expected_boxes, boxes_out)
      self.assertAllClose(expected_scores, scores_out)
      self.assertAllEqual(extra_field_out, [0, 1, 1])

if __name__ == '__main__':
  tf.test.main()
