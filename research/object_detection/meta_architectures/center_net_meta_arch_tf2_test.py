# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the CenterNet Meta architecture code."""

from __future__ import division

import functools
import re
import unittest

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.core import losses
from object_detection.core import preprocessor
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner as cn_assigner
from object_detection.meta_architectures import center_net_meta_arch as cnma
from object_detection.models import center_net_resnet_feature_extractor
from object_detection.utils import test_case
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class CenterNetMetaArchPredictionHeadTest(test_case.TestCase):
  """Test CenterNet meta architecture prediction head."""

  def test_prediction_head(self):
    head = cnma.make_prediction_net(num_out_channels=7)
    output = head(np.zeros((4, 128, 128, 8)))

    self.assertEqual((4, 128, 128, 7), output.shape)


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class CenterNetMetaArchHelpersTest(test_case.TestCase, parameterized.TestCase):
  """Test for CenterNet meta architecture related functions."""

  def test_row_col_indices_from_flattened_indices(self):
    """Tests that the computation of row, col, channel indices is correct."""

    r_grid, c_grid, ch_grid = (np.zeros((5, 4, 3), dtype=np.int),
                               np.zeros((5, 4, 3), dtype=np.int),
                               np.zeros((5, 4, 3), dtype=np.int))

    r_grid[..., 0] = r_grid[..., 1] = r_grid[..., 2] = np.array(
        [[0, 0, 0, 0],
         [1, 1, 1, 1],
         [2, 2, 2, 2],
         [3, 3, 3, 3],
         [4, 4, 4, 4]]
    )

    c_grid[..., 0] = c_grid[..., 1] = c_grid[..., 2] = np.array(
        [[0, 1, 2, 3],
         [0, 1, 2, 3],
         [0, 1, 2, 3],
         [0, 1, 2, 3],
         [0, 1, 2, 3]]
    )

    for i in range(3):
      ch_grid[..., i] = i

    indices = np.arange(60)
    ri, ci, chi = cnma.row_col_channel_indices_from_flattened_indices(
        indices, 4, 3)

    np.testing.assert_array_equal(ri, r_grid.flatten())
    np.testing.assert_array_equal(ci, c_grid.flatten())
    np.testing.assert_array_equal(chi, ch_grid.flatten())

  def test_flattened_indices_from_row_col_indices(self):

    r = np.array(
        [[0, 0, 0, 0],
         [1, 1, 1, 1],
         [2, 2, 2, 2]]
    )

    c = np.array(
        [[0, 1, 2, 3],
         [0, 1, 2, 3],
         [0, 1, 2, 3]]
    )

    idx = cnma.flattened_indices_from_row_col_indices(r, c, 4)
    np.testing.assert_array_equal(np.arange(12), idx.flatten())

  def test_get_valid_anchor_weights_in_flattened_image(self):
    """Tests that the anchor weights are valid upon flattening out."""

    valid_weights = np.zeros((2, 5, 5), dtype=np.float)

    valid_weights[0, :3, :4] = 1.0
    valid_weights[1, :2, :2] = 1.0

    def graph_fn():
      true_image_shapes = tf.constant([[3, 4], [2, 2]])
      w = cnma.get_valid_anchor_weights_in_flattened_image(
          true_image_shapes, 5, 5)
      return w

    w = self.execute(graph_fn, [])
    np.testing.assert_allclose(w, valid_weights.reshape(2, -1))
    self.assertEqual((2, 25), w.shape)

  def test_convert_strided_predictions_to_normalized_boxes(self):
    """Tests that boxes have correct coordinates in normalized input space."""

    def graph_fn():
      boxes = np.zeros((2, 3, 4), dtype=np.float32)

      boxes[0] = [[10, 20, 30, 40], [20, 30, 50, 100], [50, 60, 100, 180]]
      boxes[1] = [[-5, -5, 5, 5], [45, 60, 110, 120], [150, 150, 200, 250]]

      true_image_shapes = tf.constant([[100, 90, 3], [150, 150, 3]])

      clipped_boxes = (
          cnma.convert_strided_predictions_to_normalized_boxes(
              boxes, 2, true_image_shapes))
      return clipped_boxes

    clipped_boxes = self.execute(graph_fn, [])

    expected_boxes = np.zeros((2, 3, 4), dtype=np.float32)
    expected_boxes[0] = [[0.2, 4./9, 0.6, 8./9], [0.4, 2./3, 1, 1],
                         [1, 1, 1, 1]]
    expected_boxes[1] = [[0., 0, 1./15, 1./15], [3./5, 4./5, 1, 1],
                         [1, 1, 1, 1]]

    np.testing.assert_allclose(expected_boxes, clipped_boxes)

  @parameterized.parameters(
      {'clip_to_window': True},
      {'clip_to_window': False}
  )
  def test_convert_strided_predictions_to_normalized_keypoints(
      self, clip_to_window):
    """Tests that keypoints have correct coordinates in normalized coords."""

    keypoint_coords_np = np.array(
        [
            # Example 0.
            [
                [[-10., 8.], [60., 22.], [60., 120.]],
                [[20., 20.], [0., 0.], [0., 0.]],
            ],
            # Example 1.
            [
                [[40., 50.], [20., 160.], [200., 150.]],
                [[10., 0.], [40., 10.], [0., 0.]],
            ],
        ], dtype=np.float32)
    keypoint_scores_np = np.array(
        [
            # Example 0.
            [
                [1.0, 0.9, 0.2],
                [0.7, 0.0, 0.0],
            ],
            # Example 1.
            [
                [1.0, 1.0, 0.2],
                [0.7, 0.6, 0.0],
            ],
        ], dtype=np.float32)

    def graph_fn():
      keypoint_coords = tf.constant(keypoint_coords_np, dtype=tf.float32)
      keypoint_scores = tf.constant(keypoint_scores_np, dtype=tf.float32)
      true_image_shapes = tf.constant([[320, 400, 3], [640, 640, 3]])
      stride = 4

      keypoint_coords_out, keypoint_scores_out = (
          cnma.convert_strided_predictions_to_normalized_keypoints(
              keypoint_coords, keypoint_scores, stride, true_image_shapes,
              clip_to_window))
      return keypoint_coords_out, keypoint_scores_out

    keypoint_coords_out, keypoint_scores_out = self.execute(graph_fn, [])

    if clip_to_window:
      expected_keypoint_coords_np = np.array(
          [
              # Example 0.
              [
                  [[0.0, 0.08], [0.75, 0.22], [0.75, 1.0]],
                  [[0.25, 0.2], [0., 0.], [0.0, 0.0]],
              ],
              # Example 1.
              [
                  [[0.25, 0.3125], [0.125, 1.0], [1.0, 0.9375]],
                  [[0.0625, 0.], [0.25, 0.0625], [0., 0.]],
              ],
          ], dtype=np.float32)
      expected_keypoint_scores_np = np.array(
          [
              # Example 0.
              [
                  [0.0, 0.9, 0.0],
                  [0.7, 0.0, 0.0],
              ],
              # Example 1.
              [
                  [1.0, 1.0, 0.0],
                  [0.7, 0.6, 0.0],
              ],
          ], dtype=np.float32)
    else:
      expected_keypoint_coords_np = np.array(
          [
              # Example 0.
              [
                  [[-0.125, 0.08], [0.75, 0.22], [0.75, 1.2]],
                  [[0.25, 0.2], [0., 0.], [0., 0.]],
              ],
              # Example 1.
              [
                  [[0.25, 0.3125], [0.125, 1.0], [1.25, 0.9375]],
                  [[0.0625, 0.], [0.25, 0.0625], [0., 0.]],
              ],
          ], dtype=np.float32)
      expected_keypoint_scores_np = np.array(
          [
              # Example 0.
              [
                  [1.0, 0.9, 0.2],
                  [0.7, 0.0, 0.0],
              ],
              # Example 1.
              [
                  [1.0, 1.0, 0.2],
                  [0.7, 0.6, 0.0],
              ],
          ], dtype=np.float32)
    np.testing.assert_allclose(expected_keypoint_coords_np, keypoint_coords_out)
    np.testing.assert_allclose(expected_keypoint_scores_np, keypoint_scores_out)

  def test_convert_strided_predictions_to_instance_masks(self):

    def graph_fn():
      boxes = tf.constant(
          [
              [[0.5, 0.5, 1.0, 1.0],
               [0.0, 0.5, 0.5, 1.0],
               [0.0, 0.0, 0.0, 0.0]],
          ], tf.float32)
      classes = tf.constant(
          [
              [0, 1, 0],
          ], tf.int32)
      masks_np = np.zeros((1, 4, 4, 2), dtype=np.float32)
      masks_np[0, :, 2:, 0] = 1  # Class 0.
      masks_np[0, :, :3, 1] = 1  # Class 1.
      masks = tf.constant(masks_np)
      true_image_shapes = tf.constant([[6, 8, 3]])
      instance_masks, _ = cnma.convert_strided_predictions_to_instance_masks(
          boxes, classes, masks, stride=2, mask_height=2, mask_width=2,
          true_image_shapes=true_image_shapes)
      return instance_masks

    instance_masks = self.execute_cpu(graph_fn, [])

    expected_instance_masks = np.array(
        [
            [
                # Mask 0 (class 0).
                [[1, 1],
                 [1, 1]],
                # Mask 1 (class 1).
                [[1, 0],
                 [1, 0]],
                # Mask 2 (class 0).
                [[0, 0],
                 [0, 0]],
            ]
        ])
    np.testing.assert_array_equal(expected_instance_masks, instance_masks)

  def test_convert_strided_predictions_raises_error_with_one_tensor(self):
    def graph_fn():
      boxes = tf.constant(
          [
              [[0.5, 0.5, 1.0, 1.0],
               [0.0, 0.5, 0.5, 1.0],
               [0.0, 0.0, 0.0, 0.0]],
          ], tf.float32)
      classes = tf.constant(
          [
              [0, 1, 0],
          ], tf.int32)
      masks_np = np.zeros((1, 4, 4, 2), dtype=np.float32)
      masks_np[0, :, 2:, 0] = 1  # Class 0.
      masks_np[0, :, :3, 1] = 1  # Class 1.
      masks = tf.constant(masks_np)
      true_image_shapes = tf.constant([[6, 8, 3]])
      densepose_part_heatmap = tf.random.uniform(
          [1, 4, 4, 24])
      instance_masks, _ = cnma.convert_strided_predictions_to_instance_masks(
          boxes, classes, masks, true_image_shapes,
          densepose_part_heatmap=densepose_part_heatmap,
          densepose_surface_coords=None)
      return instance_masks

    with self.assertRaises(ValueError):
      self.execute_cpu(graph_fn, [])

  def test_crop_and_threshold_masks(self):
    boxes_np = np.array(
        [[0., 0., 0.5, 0.5],
         [0.25, 0.25, 1.0, 1.0]], dtype=np.float32)
    classes_np = np.array([0, 2], dtype=np.int32)
    masks_np = np.zeros((4, 4, _NUM_CLASSES), dtype=np.float32)
    masks_np[0, 0, 0] = 0.8
    masks_np[1, 1, 0] = 0.6
    masks_np[3, 3, 2] = 0.7
    part_heatmap_np = np.zeros((4, 4, _DENSEPOSE_NUM_PARTS), dtype=np.float32)
    part_heatmap_np[0, 0, 4] = 1
    part_heatmap_np[0, 0, 2] = 0.6  # Lower scoring.
    part_heatmap_np[1, 1, 8] = 0.2
    part_heatmap_np[3, 3, 4] = 0.5
    surf_coords_np = np.zeros((4, 4, 2 * _DENSEPOSE_NUM_PARTS),
                              dtype=np.float32)
    surf_coords_np[:, :, 8:10] = 0.2, 0.9
    surf_coords_np[:, :, 16:18] = 0.3, 0.5
    true_height, true_width = 10, 10
    input_height, input_width = 10, 10
    mask_height = 4
    mask_width = 4
    def graph_fn():
      elems = [
          tf.constant(boxes_np),
          tf.constant(classes_np),
          tf.constant(masks_np),
          tf.constant(part_heatmap_np),
          tf.constant(surf_coords_np),
          tf.constant(true_height, dtype=tf.int32),
          tf.constant(true_width, dtype=tf.int32)
      ]
      part_masks, surface_coords = cnma.crop_and_threshold_masks(
          elems, input_height, input_width, mask_height=mask_height,
          mask_width=mask_width, densepose_class_index=0)
      return part_masks, surface_coords

    part_masks, surface_coords = self.execute_cpu(graph_fn, [])

    expected_part_masks = np.zeros((2, 4, 4), dtype=np.uint8)
    expected_part_masks[0, 0, 0] = 5  # Recall classes are 1-indexed in output.
    expected_part_masks[0, 2, 2] = 9  # Recall classes are 1-indexed in output.
    expected_part_masks[1, 3, 3] = 1  # Standard instance segmentation mask.
    expected_surface_coords = np.zeros((2, 4, 4, 2), dtype=np.float32)
    expected_surface_coords[0, 0, 0, :] = 0.2, 0.9
    expected_surface_coords[0, 2, 2, :] = 0.3, 0.5
    np.testing.assert_allclose(expected_part_masks, part_masks)
    np.testing.assert_allclose(expected_surface_coords, surface_coords)

  def test_gather_surface_coords_for_parts(self):
    surface_coords_cropped_np = np.zeros((2, 5, 5, _DENSEPOSE_NUM_PARTS, 2),
                                         dtype=np.float32)
    surface_coords_cropped_np[0, 0, 0, 5] = 0.3, 0.4
    surface_coords_cropped_np[0, 1, 0, 9] = 0.5, 0.6
    highest_scoring_part_np = np.zeros((2, 5, 5), dtype=np.int32)
    highest_scoring_part_np[0, 0, 0] = 5
    highest_scoring_part_np[0, 1, 0] = 9
    def graph_fn():
      surface_coords_cropped = tf.constant(surface_coords_cropped_np,
                                           tf.float32)
      highest_scoring_part = tf.constant(highest_scoring_part_np, tf.int32)
      surface_coords_gathered = cnma.gather_surface_coords_for_parts(
          surface_coords_cropped, highest_scoring_part)
      return surface_coords_gathered

    surface_coords_gathered = self.execute_cpu(graph_fn, [])

    np.testing.assert_allclose([0.3, 0.4], surface_coords_gathered[0, 0, 0])
    np.testing.assert_allclose([0.5, 0.6], surface_coords_gathered[0, 1, 0])

  def test_top_k_feature_map_locations(self):
    feature_map_np = np.zeros((2, 3, 3, 2), dtype=np.float32)
    feature_map_np[0, 2, 0, 1] = 1.0
    feature_map_np[0, 2, 1, 1] = 0.9  # Get's filtered due to max pool.
    feature_map_np[0, 0, 1, 0] = 0.7
    feature_map_np[0, 2, 2, 0] = 0.5
    feature_map_np[0, 2, 2, 1] = -0.3
    feature_map_np[1, 2, 1, 1] = 0.7
    feature_map_np[1, 1, 0, 0] = 0.4
    feature_map_np[1, 1, 2, 0] = 0.1

    def graph_fn():
      feature_map = tf.constant(feature_map_np)
      scores, y_inds, x_inds, channel_inds = (
          cnma.top_k_feature_map_locations(
              feature_map, max_pool_kernel_size=3, k=3))
      return scores, y_inds, x_inds, channel_inds

    scores, y_inds, x_inds, channel_inds = self.execute(graph_fn, [])

    np.testing.assert_allclose([1.0, 0.7, 0.5], scores[0])
    np.testing.assert_array_equal([2, 0, 2], y_inds[0])
    np.testing.assert_array_equal([0, 1, 2], x_inds[0])
    np.testing.assert_array_equal([1, 0, 0], channel_inds[0])

    np.testing.assert_allclose([0.7, 0.4, 0.1], scores[1])
    np.testing.assert_array_equal([2, 1, 1], y_inds[1])
    np.testing.assert_array_equal([1, 0, 2], x_inds[1])
    np.testing.assert_array_equal([1, 0, 0], channel_inds[1])

  def test_top_k_feature_map_locations_no_pooling(self):
    feature_map_np = np.zeros((2, 3, 3, 2), dtype=np.float32)
    feature_map_np[0, 2, 0, 1] = 1.0
    feature_map_np[0, 2, 1, 1] = 0.9
    feature_map_np[0, 0, 1, 0] = 0.7
    feature_map_np[0, 2, 2, 0] = 0.5
    feature_map_np[0, 2, 2, 1] = -0.3
    feature_map_np[1, 2, 1, 1] = 0.7
    feature_map_np[1, 1, 0, 0] = 0.4
    feature_map_np[1, 1, 2, 0] = 0.1

    def graph_fn():
      feature_map = tf.constant(feature_map_np)
      scores, y_inds, x_inds, channel_inds = (
          cnma.top_k_feature_map_locations(
              feature_map, max_pool_kernel_size=1, k=3))
      return scores, y_inds, x_inds, channel_inds

    scores, y_inds, x_inds, channel_inds = self.execute(graph_fn, [])

    np.testing.assert_allclose([1.0, 0.9, 0.7], scores[0])
    np.testing.assert_array_equal([2, 2, 0], y_inds[0])
    np.testing.assert_array_equal([0, 1, 1], x_inds[0])
    np.testing.assert_array_equal([1, 1, 0], channel_inds[0])

    np.testing.assert_allclose([0.7, 0.4, 0.1], scores[1])
    np.testing.assert_array_equal([2, 1, 1], y_inds[1])
    np.testing.assert_array_equal([1, 0, 2], x_inds[1])
    np.testing.assert_array_equal([1, 0, 0], channel_inds[1])

  def test_top_k_feature_map_locations_per_channel(self):
    feature_map_np = np.zeros((2, 3, 3, 2), dtype=np.float32)
    feature_map_np[0, 2, 0, 0] = 1.0  # Selected.
    feature_map_np[0, 2, 1, 0] = 0.9  # Get's filtered due to max pool.
    feature_map_np[0, 0, 1, 0] = 0.7  # Selected.
    feature_map_np[0, 2, 2, 1] = 0.5  # Selected.
    feature_map_np[0, 0, 0, 1] = 0.3  # Selected.
    feature_map_np[1, 2, 1, 0] = 0.7  # Selected.
    feature_map_np[1, 1, 0, 0] = 0.4  # Get's filtered due to max pool.
    feature_map_np[1, 1, 2, 0] = 0.3  # Get's filtered due to max pool.
    feature_map_np[1, 1, 0, 1] = 0.8  # Selected.
    feature_map_np[1, 1, 2, 1] = 0.3  # Selected.

    def graph_fn():
      feature_map = tf.constant(feature_map_np)
      scores, y_inds, x_inds, channel_inds = (
          cnma.top_k_feature_map_locations(
              feature_map, max_pool_kernel_size=3, k=2, per_channel=True))
      return scores, y_inds, x_inds, channel_inds

    scores, y_inds, x_inds, channel_inds = self.execute(graph_fn, [])

    np.testing.assert_allclose([1.0, 0.7, 0.5, 0.3], scores[0])
    np.testing.assert_array_equal([2, 0, 2, 0], y_inds[0])
    np.testing.assert_array_equal([0, 1, 2, 0], x_inds[0])
    np.testing.assert_array_equal([0, 0, 1, 1], channel_inds[0])

    np.testing.assert_allclose([0.7, 0.0, 0.8, 0.3], scores[1])
    np.testing.assert_array_equal([2, 0, 1, 1], y_inds[1])
    np.testing.assert_array_equal([1, 0, 0, 2], x_inds[1])
    np.testing.assert_array_equal([0, 0, 1, 1], channel_inds[1])

  def test_box_prediction(self):

    class_pred = np.zeros((3, 128, 128, 5), dtype=np.float32)
    hw_pred = np.zeros((3, 128, 128, 2), dtype=np.float32)
    offset_pred = np.zeros((3, 128, 128, 2), dtype=np.float32)

    # Sample 1, 2 boxes
    class_pred[0, 10, 20] = [0.3, .7, 0.0, 0.0, 0.0]
    hw_pred[0, 10, 20] = [40, 60]
    offset_pred[0, 10, 20] = [1, 2]

    class_pred[0, 50, 60] = [0.55, 0.0, 0.0, 0.0, 0.45]
    hw_pred[0, 50, 60] = [50, 50]
    offset_pred[0, 50, 60] = [0, 0]

    # Sample 2, 2 boxes (at same location)
    class_pred[1, 100, 100] = [0.0, 0.1, 0.9, 0.0, 0.0]
    hw_pred[1, 100, 100] = [10, 10]
    offset_pred[1, 100, 100] = [1, 3]

    # Sample 3, 3 boxes
    class_pred[2, 60, 90] = [0.0, 0.0, 0.0, 0.2, 0.8]
    hw_pred[2, 60, 90] = [40, 30]
    offset_pred[2, 60, 90] = [0, 0]

    class_pred[2, 65, 95] = [0.0, 0.7, 0.3, 0.0, 0.0]
    hw_pred[2, 65, 95] = [20, 20]
    offset_pred[2, 65, 95] = [1, 2]

    class_pred[2, 75, 85] = [1.0, 0.0, 0.0, 0.0, 0.0]
    hw_pred[2, 75, 85] = [21, 25]
    offset_pred[2, 75, 85] = [5, 2]

    def graph_fn():
      class_pred_tensor = tf.constant(class_pred)
      hw_pred_tensor = tf.constant(hw_pred)
      offset_pred_tensor = tf.constant(offset_pred)

      detection_scores, y_indices, x_indices, channel_indices = (
          cnma.top_k_feature_map_locations(
              class_pred_tensor, max_pool_kernel_size=3, k=2))

      boxes, classes, scores, num_dets = cnma.prediction_tensors_to_boxes(
          detection_scores, y_indices, x_indices, channel_indices,
          hw_pred_tensor, offset_pred_tensor)
      return boxes, classes, scores, num_dets

    boxes, classes, scores, num_dets = self.execute(graph_fn, [])

    np.testing.assert_array_equal(num_dets, [2, 2, 2])

    np.testing.assert_allclose(
        [[-9, -8, 31, 52], [25, 35, 75, 85]], boxes[0])
    np.testing.assert_allclose(
        [[96, 98, 106, 108], [96, 98, 106, 108]], boxes[1])
    np.testing.assert_allclose(
        [[69.5, 74.5, 90.5, 99.5], [40, 75, 80, 105]], boxes[2])

    np.testing.assert_array_equal(classes[0], [1, 0])
    np.testing.assert_array_equal(classes[1], [2, 1])
    np.testing.assert_array_equal(classes[2], [0, 4])

    np.testing.assert_allclose(scores[0], [.7, .55])
    np.testing.assert_allclose(scores[1][:1], [.9])
    np.testing.assert_allclose(scores[2], [1., .8])

  def test_keypoint_candidate_prediction(self):
    keypoint_heatmap_np = np.zeros((2, 3, 3, 2), dtype=np.float32)
    keypoint_heatmap_np[0, 0, 0, 0] = 1.0
    keypoint_heatmap_np[0, 2, 1, 0] = 0.7
    keypoint_heatmap_np[0, 1, 1, 0] = 0.6
    keypoint_heatmap_np[0, 0, 2, 1] = 0.7
    keypoint_heatmap_np[0, 1, 1, 1] = 0.3  # Filtered by low score.
    keypoint_heatmap_np[0, 2, 2, 1] = 0.2
    keypoint_heatmap_np[1, 1, 0, 0] = 0.6
    keypoint_heatmap_np[1, 2, 1, 0] = 0.5
    keypoint_heatmap_np[1, 0, 0, 0] = 0.4
    keypoint_heatmap_np[1, 0, 0, 1] = 1.0
    keypoint_heatmap_np[1, 0, 1, 1] = 0.9
    keypoint_heatmap_np[1, 2, 0, 1] = 0.8

    keypoint_heatmap_offsets_np = np.zeros((2, 3, 3, 2), dtype=np.float32)
    keypoint_heatmap_offsets_np[0, 0, 0] = [0.5, 0.25]
    keypoint_heatmap_offsets_np[0, 2, 1] = [-0.25, 0.5]
    keypoint_heatmap_offsets_np[0, 1, 1] = [0.0, 0.0]
    keypoint_heatmap_offsets_np[0, 0, 2] = [1.0, 0.0]
    keypoint_heatmap_offsets_np[0, 2, 2] = [1.0, 1.0]
    keypoint_heatmap_offsets_np[1, 1, 0] = [0.25, 0.5]
    keypoint_heatmap_offsets_np[1, 2, 1] = [0.5, 0.0]
    keypoint_heatmap_offsets_np[1, 0, 0] = [0.0, -0.5]
    keypoint_heatmap_offsets_np[1, 0, 1] = [0.5, -0.5]
    keypoint_heatmap_offsets_np[1, 2, 0] = [-1.0, -0.5]

    def graph_fn():
      keypoint_heatmap = tf.constant(keypoint_heatmap_np, dtype=tf.float32)
      keypoint_heatmap_offsets = tf.constant(
          keypoint_heatmap_offsets_np, dtype=tf.float32)

      keypoint_cands, keypoint_scores, num_keypoint_candidates = (
          cnma.prediction_tensors_to_keypoint_candidates(
              keypoint_heatmap,
              keypoint_heatmap_offsets,
              keypoint_score_threshold=0.5,
              max_pool_kernel_size=1,
              max_candidates=2))
      return keypoint_cands, keypoint_scores, num_keypoint_candidates

    (keypoint_cands, keypoint_scores,
     num_keypoint_candidates) = self.execute(graph_fn, [])

    expected_keypoint_candidates = [
        [  # Example 0.
            [[0.5, 0.25], [1.0, 2.0]],  # Keypoint 1.
            [[1.75, 1.5], [1.0, 1.0]],  # Keypoint 2.
        ],
        [  # Example 1.
            [[1.25, 0.5], [0.0, -0.5]],  # Keypoint 1.
            [[2.5, 1.0], [0.5, 0.5]],  # Keypoint 2.
        ],
    ]
    expected_keypoint_scores = [
        [  # Example 0.
            [1.0, 0.7],  # Keypoint 1.
            [0.7, 0.3],  # Keypoint 2.
        ],
        [  # Example 1.
            [0.6, 1.0],  # Keypoint 1.
            [0.5, 0.9],  # Keypoint 2.
        ],
    ]
    expected_num_keypoint_candidates = [
        [2, 1],
        [2, 2]
    ]
    np.testing.assert_allclose(expected_keypoint_candidates, keypoint_cands)
    np.testing.assert_allclose(expected_keypoint_scores, keypoint_scores)
    np.testing.assert_array_equal(expected_num_keypoint_candidates,
                                  num_keypoint_candidates)

  def test_keypoint_candidate_prediction_per_keypoints(self):
    keypoint_heatmap_np = np.zeros((2, 3, 3, 2), dtype=np.float32)
    keypoint_heatmap_np[0, 0, 0, 0] = 1.0
    keypoint_heatmap_np[0, 2, 1, 0] = 0.7
    keypoint_heatmap_np[0, 1, 1, 0] = 0.6
    keypoint_heatmap_np[0, 0, 2, 1] = 0.7
    keypoint_heatmap_np[0, 1, 1, 1] = 0.3  # Filtered by low score.
    keypoint_heatmap_np[0, 2, 2, 1] = 0.2
    keypoint_heatmap_np[1, 1, 0, 0] = 0.6
    keypoint_heatmap_np[1, 2, 1, 0] = 0.5
    keypoint_heatmap_np[1, 0, 0, 0] = 0.4
    keypoint_heatmap_np[1, 0, 0, 1] = 1.0
    keypoint_heatmap_np[1, 0, 1, 1] = 0.9
    keypoint_heatmap_np[1, 2, 0, 1] = 0.8

    # Note that the keypoint offsets are now per keypoint (as opposed to
    # keypoint agnostic, in the test test_keypoint_candidate_prediction).
    keypoint_heatmap_offsets_np = np.zeros((2, 3, 3, 4), dtype=np.float32)
    keypoint_heatmap_offsets_np[0, 0, 0] = [0.5, 0.25, 0.0, 0.0]
    keypoint_heatmap_offsets_np[0, 2, 1] = [-0.25, 0.5, 0.0, 0.0]
    keypoint_heatmap_offsets_np[0, 1, 1] = [0.0, 0.0, 0.0, 0.0]
    keypoint_heatmap_offsets_np[0, 0, 2] = [0.0, 0.0, 1.0, 0.0]
    keypoint_heatmap_offsets_np[0, 2, 2] = [0.0, 0.0, 1.0, 1.0]
    keypoint_heatmap_offsets_np[1, 1, 0] = [0.25, 0.5, 0.0, 0.0]
    keypoint_heatmap_offsets_np[1, 2, 1] = [0.5, 0.0, 0.0, 0.0]
    keypoint_heatmap_offsets_np[1, 0, 0] = [0.0, 0.0, 0.0, -0.5]
    keypoint_heatmap_offsets_np[1, 0, 1] = [0.0, 0.0, 0.5, -0.5]
    keypoint_heatmap_offsets_np[1, 2, 0] = [0.0, 0.0, -1.0, -0.5]

    def graph_fn():
      keypoint_heatmap = tf.constant(keypoint_heatmap_np, dtype=tf.float32)
      keypoint_heatmap_offsets = tf.constant(
          keypoint_heatmap_offsets_np, dtype=tf.float32)

      keypoint_cands, keypoint_scores, num_keypoint_candidates = (
          cnma.prediction_tensors_to_keypoint_candidates(
              keypoint_heatmap,
              keypoint_heatmap_offsets,
              keypoint_score_threshold=0.5,
              max_pool_kernel_size=1,
              max_candidates=2))
      return keypoint_cands, keypoint_scores, num_keypoint_candidates

    (keypoint_cands, keypoint_scores,
     num_keypoint_candidates) = self.execute(graph_fn, [])

    expected_keypoint_candidates = [
        [  # Example 0.
            [[0.5, 0.25], [1.0, 2.0]],  # Candidate 1 of keypoint 1, 2.
            [[1.75, 1.5], [1.0, 1.0]],  # Candidate 2 of keypoint 1, 2.
        ],
        [  # Example 1.
            [[1.25, 0.5], [0.0, -0.5]],  # Candidate 1 of keypoint 1, 2.
            [[2.5, 1.0], [0.5, 0.5]],    # Candidate 2 of keypoint 1, 2.
        ],
    ]
    expected_keypoint_scores = [
        [  # Example 0.
            [1.0, 0.7],  # Candidate 1 scores of keypoint 1, 2.
            [0.7, 0.3],  # Candidate 2 scores of keypoint 1, 2.
        ],
        [  # Example 1.
            [0.6, 1.0],  # Candidate 1 scores of keypoint 1, 2.
            [0.5, 0.9],  # Candidate 2 scores of keypoint 1, 2.
        ],
    ]
    expected_num_keypoint_candidates = [
        [2, 1],
        [2, 2]
    ]
    np.testing.assert_allclose(expected_keypoint_candidates, keypoint_cands)
    np.testing.assert_allclose(expected_keypoint_scores, keypoint_scores)
    np.testing.assert_array_equal(expected_num_keypoint_candidates,
                                  num_keypoint_candidates)

  def test_regressed_keypoints_at_object_centers(self):
    batch_size = 2
    num_keypoints = 5
    num_instances = 6
    regressed_keypoint_feature_map_np = np.random.randn(
        batch_size, 10, 10, 2 * num_keypoints).astype(np.float32)
    y_indices = np.random.choice(10, (batch_size, num_instances))
    x_indices = np.random.choice(10, (batch_size, num_instances))
    offsets = np.stack([y_indices, x_indices], axis=2).astype(np.float32)

    def graph_fn():
      regressed_keypoint_feature_map = tf.constant(
          regressed_keypoint_feature_map_np, dtype=tf.float32)

      gathered_regressed_keypoints = (
          cnma.regressed_keypoints_at_object_centers(
              regressed_keypoint_feature_map,
              tf.constant(y_indices, dtype=tf.int32),
              tf.constant(x_indices, dtype=tf.int32)))
      return gathered_regressed_keypoints

    gathered_regressed_keypoints = self.execute(graph_fn, [])

    expected_gathered_keypoints_0 = regressed_keypoint_feature_map_np[
        0, y_indices[0], x_indices[0], :]
    expected_gathered_keypoints_1 = regressed_keypoint_feature_map_np[
        1, y_indices[1], x_indices[1], :]
    expected_gathered_keypoints = np.stack([
        expected_gathered_keypoints_0,
        expected_gathered_keypoints_1], axis=0)
    expected_gathered_keypoints = np.reshape(
        expected_gathered_keypoints,
        [batch_size, num_instances, num_keypoints, 2])
    expected_gathered_keypoints += np.expand_dims(offsets, axis=2)
    expected_gathered_keypoints = np.reshape(
        expected_gathered_keypoints,
        [batch_size, num_instances, -1])
    np.testing.assert_allclose(expected_gathered_keypoints,
                               gathered_regressed_keypoints)

  @parameterized.parameters(
      {'candidate_ranking_mode': 'min_distance'},
      {'candidate_ranking_mode': 'score_distance_ratio'},
  )
  def test_refine_keypoints(self, candidate_ranking_mode):
    regressed_keypoints_np = np.array(
        [
            # Example 0.
            [
                [[2.0, 2.0], [6.0, 10.0], [14.0, 7.0]],  # Instance 0.
                [[0.0, 6.0], [3.0, 3.0], [5.0, 7.0]],  # Instance 1.
            ],
            # Example 1.
            [
                [[6.0, 2.0], [0.0, 0.0], [0.1, 0.1]],  # Instance 0.
                [[6.0, 2.5], [5.0, 5.0], [9.0, 3.0]],  # Instance 1.
            ],
        ], dtype=np.float32)
    keypoint_candidates_np = np.array(
        [
            # Example 0.
            [
                [[2.0, 2.5], [6.0, 10.5], [4.0, 7.0]],  # Candidate 0.
                [[1.0, 8.0], [0.0, 0.0], [2.0, 2.0]],  # Candidate 1.
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],  # Candidate 2.
            ],
            # Example 1.
            [
                [[6.0, 1.5], [0.1, 0.4], [0.0, 0.0]],  # Candidate 0.
                [[1.0, 4.0], [0.0, 0.3], [0.0, 0.0]],  # Candidate 1.
                [[0.0, 0.0], [0.1, 0.3], [0.0, 0.0]],  # Candidate 2.
            ]
        ], dtype=np.float32)
    keypoint_scores_np = np.array(
        [
            # Example 0.
            [
                [0.8, 0.9, 1.0],  # Candidate 0.
                [0.6, 0.1, 0.9],  # Candidate 1.
                [0.0, 0.0, 0.0],  # Candidate 1.
            ],
            # Example 1.
            [
                [0.7, 0.3, 0.0],  # Candidate 0.
                [0.6, 0.1, 0.0],  # Candidate 1.
                [0.0, 0.28, 0.0],  # Candidate 1.
            ]
        ], dtype=np.float32)
    num_keypoints_candidates_np = np.array(
        [
            # Example 0.
            [2, 2, 2],
            # Example 1.
            [2, 3, 0],
        ], dtype=np.int32)
    unmatched_keypoint_score = 0.1

    def graph_fn():
      regressed_keypoints = tf.constant(
          regressed_keypoints_np, dtype=tf.float32)
      keypoint_candidates = tf.constant(
          keypoint_candidates_np, dtype=tf.float32)
      keypoint_scores = tf.constant(keypoint_scores_np, dtype=tf.float32)
      num_keypoint_candidates = tf.constant(num_keypoints_candidates_np,
                                            dtype=tf.int32)
      refined_keypoints, refined_scores = cnma.refine_keypoints(
          regressed_keypoints, keypoint_candidates, keypoint_scores,
          num_keypoint_candidates, bboxes=None,
          unmatched_keypoint_score=unmatched_keypoint_score,
          box_scale=1.2, candidate_search_scale=0.3,
          candidate_ranking_mode=candidate_ranking_mode)
      return refined_keypoints, refined_scores

    refined_keypoints, refined_scores = self.execute(graph_fn, [])

    if candidate_ranking_mode == 'min_distance':
      expected_refined_keypoints = np.array(
          [
              # Example 0.
              [
                  [[2.0, 2.5], [6.0, 10.5], [14.0, 7.0]],  # Instance 0.
                  [[0.0, 6.0], [3.0, 3.0], [4.0, 7.0]],  # Instance 1.
              ],
              # Example 1.
              [
                  [[6.0, 1.5], [0.0, 0.3], [0.1, 0.1]],  # Instance 0.
                  [[6.0, 2.5], [5.0, 5.0], [9.0, 3.0]],  # Instance 1.
              ],
          ], dtype=np.float32)
      expected_refined_scores = np.array(
          [
              # Example 0.
              [
                  [0.8, 0.9, unmatched_keypoint_score],  # Instance 0.
                  [unmatched_keypoint_score,  # Instance 1.
                   unmatched_keypoint_score, 1.0],
              ],
              # Example 1.
              [
                  [0.7, 0.1, unmatched_keypoint_score],  # Instance 0.
                  [unmatched_keypoint_score,  # Instance 1.
                   0.1, unmatched_keypoint_score],
              ],
          ], dtype=np.float32)
    else:
      expected_refined_keypoints = np.array(
          [
              # Example 0.
              [
                  [[2.0, 2.5], [6.0, 10.5], [14.0, 7.0]],  # Instance 0.
                  [[0.0, 6.0], [3.0, 3.0], [4.0, 7.0]],  # Instance 1.
              ],
              # Example 1.
              [
                  [[6.0, 1.5], [0.1, 0.3], [0.1, 0.1]],  # Instance 0.
                  [[6.0, 2.5], [5.0, 5.0], [9.0, 3.0]],  # Instance 1.
              ],
          ], dtype=np.float32)
      expected_refined_scores = np.array(
          [
              # Example 0.
              [
                  [0.8, 0.9, unmatched_keypoint_score],  # Instance 0.
                  [unmatched_keypoint_score,  # Instance 1.
                   unmatched_keypoint_score, 1.0],
              ],
              # Example 1.
              [
                  [0.7, 0.28, unmatched_keypoint_score],  # Instance 0.
                  [unmatched_keypoint_score,  # Instance 1.
                   0.1, unmatched_keypoint_score],
              ],
          ], dtype=np.float32)

    np.testing.assert_allclose(expected_refined_keypoints, refined_keypoints)
    np.testing.assert_allclose(expected_refined_scores, refined_scores)

  def test_refine_keypoints_with_bboxes(self):
    regressed_keypoints_np = np.array(
        [
            # Example 0.
            [
                [[2.0, 2.0], [6.0, 10.0], [14.0, 7.0]],  # Instance 0.
                [[0.0, 6.0], [3.0, 3.0], [5.0, 7.0]],  # Instance 1.
            ],
            # Example 1.
            [
                [[6.0, 2.0], [0.0, 0.0], [0.1, 0.1]],  # Instance 0.
                [[6.0, 2.5], [5.0, 5.0], [9.0, 3.0]],  # Instance 1.
            ],
        ], dtype=np.float32)
    keypoint_candidates_np = np.array(
        [
            # Example 0.
            [
                [[2.0, 2.5], [6.0, 10.5], [4.0, 7.0]],  # Candidate 0.
                [[1.0, 8.0], [0.0, 0.0], [2.0, 2.0]],  # Candidate 1.
            ],
            # Example 1.
            [
                [[6.0, 1.5], [5.0, 5.0], [0.0, 0.0]],  # Candidate 0.
                [[1.0, 4.0], [0.0, 0.3], [0.0, 0.0]],  # Candidate 1.
            ]
        ], dtype=np.float32)
    keypoint_scores_np = np.array(
        [
            # Example 0.
            [
                [0.8, 0.9, 1.0],  # Candidate 0.
                [0.6, 0.1, 0.9],  # Candidate 1.
            ],
            # Example 1.
            [
                [0.7, 0.4, 0.0],  # Candidate 0.
                [0.6, 0.1, 0.0],  # Candidate 1.
            ]
        ], dtype=np.float32)
    num_keypoints_candidates_np = np.array(
        [
            # Example 0.
            [2, 2, 2],
            # Example 1.
            [2, 2, 0],
        ], dtype=np.int32)
    bboxes_np = np.array(
        [
            # Example 0.
            [
                [2.0, 2.0, 14.0, 10.0],  # Instance 0.
                [0.0, 3.0, 5.0, 7.0],  # Instance 1.
            ],
            # Example 1.
            [
                [0.0, 0.0, 6.0, 2.0],  # Instance 0.
                [5.0, 1.4, 9.0, 5.0],  # Instance 1.
            ],
        ], dtype=np.float32)
    unmatched_keypoint_score = 0.1

    def graph_fn():
      regressed_keypoints = tf.constant(
          regressed_keypoints_np, dtype=tf.float32)
      keypoint_candidates = tf.constant(
          keypoint_candidates_np, dtype=tf.float32)
      keypoint_scores = tf.constant(keypoint_scores_np, dtype=tf.float32)
      num_keypoint_candidates = tf.constant(num_keypoints_candidates_np,
                                            dtype=tf.int32)
      bboxes = tf.constant(bboxes_np, dtype=tf.float32)
      refined_keypoints, refined_scores = cnma.refine_keypoints(
          regressed_keypoints, keypoint_candidates, keypoint_scores,
          num_keypoint_candidates, bboxes=bboxes,
          unmatched_keypoint_score=unmatched_keypoint_score,
          box_scale=1.0, candidate_search_scale=0.3)
      return refined_keypoints, refined_scores

    refined_keypoints, refined_scores = self.execute(graph_fn, [])

    expected_refined_keypoints = np.array(
        [
            # Example 0.
            [
                [[2.0, 2.5], [6.0, 10.0], [14.0, 7.0]],  # Instance 0.
                [[0.0, 6.0], [3.0, 3.0], [4.0, 7.0]],  # Instance 1.
            ],
            # Example 1.
            [
                [[6.0, 1.5], [0.0, 0.3], [0.1, 0.1]],  # Instance 0.
                [[6.0, 1.5], [5.0, 5.0], [9.0, 3.0]],  # Instance 1.
            ],
        ], dtype=np.float32)
    expected_refined_scores = np.array(
        [
            # Example 0.
            [
                [0.8, unmatched_keypoint_score,  # Instance 0.
                 unmatched_keypoint_score],
                [unmatched_keypoint_score,  # Instance 1.
                 unmatched_keypoint_score, 1.0],
            ],
            # Example 1.
            [
                [0.7, 0.1, unmatched_keypoint_score],  # Instance 0.
                [0.7, 0.4, unmatched_keypoint_score],  # Instance 1.
            ],
        ], dtype=np.float32)

    np.testing.assert_allclose(expected_refined_keypoints, refined_keypoints)
    np.testing.assert_allclose(expected_refined_scores, refined_scores)

  def test_pad_to_full_keypoint_dim(self):
    batch_size = 4
    num_instances = 8
    num_keypoints = 2
    keypoint_inds = [1, 3]
    num_total_keypoints = 5

    kpt_coords_np = np.random.randn(batch_size, num_instances, num_keypoints, 2)
    kpt_scores_np = np.random.randn(batch_size, num_instances, num_keypoints)

    def graph_fn():
      kpt_coords = tf.constant(kpt_coords_np)
      kpt_scores = tf.constant(kpt_scores_np)
      kpt_coords_padded, kpt_scores_padded = (
          cnma._pad_to_full_keypoint_dim(
              kpt_coords, kpt_scores, keypoint_inds, num_total_keypoints))
      return kpt_coords_padded, kpt_scores_padded

    kpt_coords_padded, kpt_scores_padded = self.execute(graph_fn, [])

    self.assertAllEqual([batch_size, num_instances, num_total_keypoints, 2],
                        kpt_coords_padded.shape)
    self.assertAllEqual([batch_size, num_instances, num_total_keypoints],
                        kpt_scores_padded.shape)

    for i, kpt_ind in enumerate(keypoint_inds):
      np.testing.assert_allclose(kpt_coords_np[:, :, i, :],
                                 kpt_coords_padded[:, :, kpt_ind, :])
      np.testing.assert_allclose(kpt_scores_np[:, :, i],
                                 kpt_scores_padded[:, :, kpt_ind])

  def test_pad_to_full_instance_dim(self):
    batch_size = 4
    max_instances = 8
    num_keypoints = 6
    num_instances = 2
    instance_inds = [1, 3]

    kpt_coords_np = np.random.randn(batch_size, num_instances, num_keypoints, 2)
    kpt_scores_np = np.random.randn(batch_size, num_instances, num_keypoints)

    def graph_fn():
      kpt_coords = tf.constant(kpt_coords_np)
      kpt_scores = tf.constant(kpt_scores_np)
      kpt_coords_padded, kpt_scores_padded = (
          cnma._pad_to_full_instance_dim(
              kpt_coords, kpt_scores, instance_inds, max_instances))
      return kpt_coords_padded, kpt_scores_padded

    kpt_coords_padded, kpt_scores_padded = self.execute(graph_fn, [])

    self.assertAllEqual([batch_size, max_instances, num_keypoints, 2],
                        kpt_coords_padded.shape)
    self.assertAllEqual([batch_size, max_instances, num_keypoints],
                        kpt_scores_padded.shape)

    for i, inst_ind in enumerate(instance_inds):
      np.testing.assert_allclose(kpt_coords_np[:, i, :, :],
                                 kpt_coords_padded[:, inst_ind, :, :])
      np.testing.assert_allclose(kpt_scores_np[:, i, :],
                                 kpt_scores_padded[:, inst_ind, :])


# Common parameters for setting up testing examples across tests.
_NUM_CLASSES = 10
_KEYPOINT_INDICES = [0, 1, 2, 3]
_NUM_KEYPOINTS = len(_KEYPOINT_INDICES)
_DENSEPOSE_NUM_PARTS = 24
_TASK_NAME = 'human_pose'


def get_fake_center_params():
  """Returns the fake object center parameter namedtuple."""
  return cnma.ObjectCenterParams(
      classification_loss=losses.WeightedSigmoidClassificationLoss(),
      object_center_loss_weight=1.0,
      min_box_overlap_iou=1.0,
      max_box_predictions=5,
      use_labeled_classes=False)


def get_fake_od_params():
  """Returns the fake object detection parameter namedtuple."""
  return cnma.ObjectDetectionParams(
      localization_loss=losses.L1LocalizationLoss(),
      offset_loss_weight=1.0,
      scale_loss_weight=0.1)


def get_fake_kp_params():
  """Returns the fake keypoint estimation parameter namedtuple."""
  return cnma.KeypointEstimationParams(
      task_name=_TASK_NAME,
      class_id=1,
      keypoint_indices=_KEYPOINT_INDICES,
      keypoint_std_dev=[0.00001] * len(_KEYPOINT_INDICES),
      classification_loss=losses.WeightedSigmoidClassificationLoss(),
      localization_loss=losses.L1LocalizationLoss(),
      keypoint_candidate_score_threshold=0.1)


def get_fake_mask_params():
  """Returns the fake mask estimation parameter namedtuple."""
  return cnma.MaskParams(
      classification_loss=losses.WeightedSoftmaxClassificationLoss(),
      task_loss_weight=1.0,
      mask_height=4,
      mask_width=4)


def get_fake_densepose_params():
  """Returns the fake DensePose estimation parameter namedtuple."""
  return cnma.DensePoseParams(
      class_id=1,
      classification_loss=losses.WeightedSoftmaxClassificationLoss(),
      localization_loss=losses.L1LocalizationLoss(),
      part_loss_weight=1.0,
      coordinate_loss_weight=1.0,
      num_parts=_DENSEPOSE_NUM_PARTS,
      task_loss_weight=1.0,
      upsample_to_input_res=True,
      upsample_method='nearest')


def build_center_net_meta_arch(build_resnet=False):
  """Builds the CenterNet meta architecture."""
  if build_resnet:
    feature_extractor = (
        center_net_resnet_feature_extractor.CenterNetResnetFeatureExtractor(
            'resnet_v2_101'))
  else:
    feature_extractor = DummyFeatureExtractor(
        channel_means=(1.0, 2.0, 3.0),
        channel_stds=(10., 20., 30.),
        bgr_ordering=False,
        num_feature_outputs=2,
        stride=4)
  image_resizer_fn = functools.partial(
      preprocessor.resize_to_range,
      min_dimension=128,
      max_dimension=128,
      pad_to_max_dimesnion=True)
  return cnma.CenterNetMetaArch(
      is_training=True,
      add_summaries=False,
      num_classes=_NUM_CLASSES,
      feature_extractor=feature_extractor,
      image_resizer_fn=image_resizer_fn,
      object_center_params=get_fake_center_params(),
      object_detection_params=get_fake_od_params(),
      keypoint_params_dict={_TASK_NAME: get_fake_kp_params()},
      mask_params=get_fake_mask_params(),
      densepose_params=get_fake_densepose_params())


def _logit(p):
  return np.log(
      (p + np.finfo(np.float32).eps) / (1 - p + np.finfo(np.float32).eps))


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class CenterNetMetaArchLibTest(test_case.TestCase):
  """Test for CenterNet meta architecture related functions."""

  def test_get_keypoint_name(self):
    self.assertEqual('human_pose/keypoint_offset',
                     cnma.get_keypoint_name('human_pose', 'keypoint_offset'))

  def test_get_num_instances_from_weights(self):
    weight1 = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
    weight2 = tf.constant([0.5, 0.9, 0.0], dtype=tf.float32)
    weight3 = tf.constant([0.0, 0.0, 1.0], dtype=tf.float32)

    def graph_fn_1():
      # Total of three elements with non-zero values.
      num_instances = cnma.get_num_instances_from_weights(
          [weight1, weight2, weight3])
      return num_instances
    num_instances = self.execute(graph_fn_1, [])
    self.assertAlmostEqual(3, num_instances)

    # No non-zero value in the weights. Return minimum value: 1.
    def graph_fn_2():
      # Total of three elements with non-zero values.
      num_instances = cnma.get_num_instances_from_weights([weight1, weight1])
      return num_instances
    num_instances = self.execute(graph_fn_2, [])
    self.assertAlmostEqual(1, num_instances)


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class CenterNetMetaArchTest(test_case.TestCase, parameterized.TestCase):
  """Tests for the CenterNet meta architecture."""

  def test_construct_prediction_heads(self):
    model = build_center_net_meta_arch()
    fake_feature_map = np.zeros((4, 128, 128, 8))

    # Check the dictionary contains expected keys and corresponding heads with
    # correct dimensions.
    # "object center" head:
    output = model._prediction_head_dict[cnma.OBJECT_CENTER][-1](
        fake_feature_map)
    self.assertEqual((4, 128, 128, _NUM_CLASSES), output.shape)

    # "object scale" (height/width) head:
    output = model._prediction_head_dict[cnma.BOX_SCALE][-1](fake_feature_map)
    self.assertEqual((4, 128, 128, 2), output.shape)

    # "object offset" head:
    output = model._prediction_head_dict[cnma.BOX_OFFSET][-1](fake_feature_map)
    self.assertEqual((4, 128, 128, 2), output.shape)

    # "keypoint offset" head:
    output = model._prediction_head_dict[
        cnma.get_keypoint_name(_TASK_NAME, cnma.KEYPOINT_OFFSET)][-1](
            fake_feature_map)
    self.assertEqual((4, 128, 128, 2), output.shape)

    # "keypoint heatmap" head:
    output = model._prediction_head_dict[cnma.get_keypoint_name(
        _TASK_NAME, cnma.KEYPOINT_HEATMAP)][-1](
            fake_feature_map)
    self.assertEqual((4, 128, 128, _NUM_KEYPOINTS), output.shape)

    # "keypoint regression" head:
    output = model._prediction_head_dict[cnma.get_keypoint_name(
        _TASK_NAME, cnma.KEYPOINT_REGRESSION)][-1](
            fake_feature_map)
    self.assertEqual((4, 128, 128, 2 * _NUM_KEYPOINTS), output.shape)

    # "mask" head:
    output = model._prediction_head_dict[cnma.SEGMENTATION_HEATMAP][-1](
        fake_feature_map)
    self.assertEqual((4, 128, 128, _NUM_CLASSES), output.shape)

    # "densepose parts" head:
    output = model._prediction_head_dict[cnma.DENSEPOSE_HEATMAP][-1](
        fake_feature_map)
    self.assertEqual((4, 128, 128, _DENSEPOSE_NUM_PARTS), output.shape)

    # "densepose surface coordinates" head:
    output = model._prediction_head_dict[cnma.DENSEPOSE_REGRESSION][-1](
        fake_feature_map)
    self.assertEqual((4, 128, 128, 2 * _DENSEPOSE_NUM_PARTS), output.shape)

  def test_initialize_target_assigners(self):
    model = build_center_net_meta_arch()
    assigner_dict = model._initialize_target_assigners(
        stride=2,
        min_box_overlap_iou=0.7)

    # Check whether the correponding target assigner class is initialized.
    # object center target assigner:
    self.assertIsInstance(assigner_dict[cnma.OBJECT_CENTER],
                          cn_assigner.CenterNetCenterHeatmapTargetAssigner)

    # object detection target assigner:
    self.assertIsInstance(assigner_dict[cnma.DETECTION_TASK],
                          cn_assigner.CenterNetBoxTargetAssigner)

    # keypoint estimation target assigner:
    self.assertIsInstance(assigner_dict[_TASK_NAME],
                          cn_assigner.CenterNetKeypointTargetAssigner)

    # mask estimation target assigner:
    self.assertIsInstance(assigner_dict[cnma.SEGMENTATION_TASK],
                          cn_assigner.CenterNetMaskTargetAssigner)

    # DensePose estimation target assigner:
    self.assertIsInstance(assigner_dict[cnma.DENSEPOSE_TASK],
                          cn_assigner.CenterNetDensePoseTargetAssigner)

  def test_predict(self):
    """Test the predict function."""

    model = build_center_net_meta_arch()
    def graph_fn():
      prediction_dict = model.predict(tf.zeros([2, 128, 128, 3]), None)
      return prediction_dict

    prediction_dict = self.execute(graph_fn, [])

    self.assertEqual(prediction_dict['preprocessed_inputs'].shape,
                     (2, 128, 128, 3))
    self.assertEqual(prediction_dict[cnma.OBJECT_CENTER][0].shape,
                     (2, 32, 32, _NUM_CLASSES))
    self.assertEqual(prediction_dict[cnma.BOX_SCALE][0].shape,
                     (2, 32, 32, 2))
    self.assertEqual(prediction_dict[cnma.BOX_OFFSET][0].shape,
                     (2, 32, 32, 2))
    self.assertEqual(prediction_dict[cnma.SEGMENTATION_HEATMAP][0].shape,
                     (2, 32, 32, _NUM_CLASSES))
    self.assertEqual(prediction_dict[cnma.DENSEPOSE_HEATMAP][0].shape,
                     (2, 32, 32, _DENSEPOSE_NUM_PARTS))
    self.assertEqual(prediction_dict[cnma.DENSEPOSE_REGRESSION][0].shape,
                     (2, 32, 32, 2 * _DENSEPOSE_NUM_PARTS))

  def test_loss(self):
    """Test the loss function."""
    groundtruth_dict = get_fake_groundtruth_dict(16, 32, 4)
    model = build_center_net_meta_arch()
    model.provide_groundtruth(
        groundtruth_boxes_list=groundtruth_dict[fields.BoxListFields.boxes],
        groundtruth_weights_list=groundtruth_dict[fields.BoxListFields.weights],
        groundtruth_classes_list=groundtruth_dict[fields.BoxListFields.classes],
        groundtruth_keypoints_list=groundtruth_dict[
            fields.BoxListFields.keypoints],
        groundtruth_masks_list=groundtruth_dict[
            fields.BoxListFields.masks],
        groundtruth_dp_num_points_list=groundtruth_dict[
            fields.BoxListFields.densepose_num_points],
        groundtruth_dp_part_ids_list=groundtruth_dict[
            fields.BoxListFields.densepose_part_ids],
        groundtruth_dp_surface_coords_list=groundtruth_dict[
            fields.BoxListFields.densepose_surface_coords])

    prediction_dict = get_fake_prediction_dict(
        input_height=16, input_width=32, stride=4)

    def graph_fn():
      loss_dict = model.loss(prediction_dict,
                             tf.constant([[16, 24, 3], [16, 24, 3]]))
      return loss_dict

    loss_dict = self.execute(graph_fn, [])

    # The prediction and groundtruth are curated to produce very low loss.
    self.assertGreater(
        0.01, loss_dict['%s/%s' % (cnma.LOSS_KEY_PREFIX, cnma.OBJECT_CENTER)])
    self.assertGreater(
        0.01, loss_dict['%s/%s' % (cnma.LOSS_KEY_PREFIX, cnma.BOX_SCALE)])
    self.assertGreater(
        0.01, loss_dict['%s/%s' % (cnma.LOSS_KEY_PREFIX, cnma.BOX_OFFSET)])
    self.assertGreater(
        0.01,
        loss_dict['%s/%s' %
                  (cnma.LOSS_KEY_PREFIX,
                   cnma.get_keypoint_name(_TASK_NAME, cnma.KEYPOINT_HEATMAP))])
    self.assertGreater(
        0.01,
        loss_dict['%s/%s' %
                  (cnma.LOSS_KEY_PREFIX,
                   cnma.get_keypoint_name(_TASK_NAME, cnma.KEYPOINT_OFFSET))])
    self.assertGreater(
        0.01, loss_dict['%s/%s' % (cnma.LOSS_KEY_PREFIX,
                                   cnma.get_keypoint_name(
                                       _TASK_NAME, cnma.KEYPOINT_REGRESSION))])
    self.assertGreater(
        0.01, loss_dict['%s/%s' % (cnma.LOSS_KEY_PREFIX,
                                   cnma.SEGMENTATION_HEATMAP)])
    self.assertGreater(
        0.01, loss_dict['%s/%s' % (cnma.LOSS_KEY_PREFIX,
                                   cnma.DENSEPOSE_HEATMAP)])
    self.assertGreater(
        0.01, loss_dict['%s/%s' % (cnma.LOSS_KEY_PREFIX,
                                   cnma.DENSEPOSE_REGRESSION)])

  @parameterized.parameters(
      {'target_class_id': 1},
      {'target_class_id': 2},
  )
  def test_postprocess(self, target_class_id):
    """Test the postprocess function."""
    model = build_center_net_meta_arch()
    max_detection = model._center_params.max_box_predictions
    num_keypoints = len(model._kp_params_dict[_TASK_NAME].keypoint_indices)

    class_center = np.zeros((1, 32, 32, 10), dtype=np.float32)
    height_width = np.zeros((1, 32, 32, 2), dtype=np.float32)
    offset = np.zeros((1, 32, 32, 2), dtype=np.float32)
    keypoint_heatmaps = np.zeros((1, 32, 32, num_keypoints), dtype=np.float32)
    keypoint_offsets = np.zeros((1, 32, 32, 2), dtype=np.float32)
    keypoint_regression = np.random.randn(1, 32, 32, num_keypoints * 2)

    class_probs = np.zeros(10)
    class_probs[target_class_id] = _logit(0.75)
    class_center[0, 16, 16] = class_probs
    height_width[0, 16, 16] = [5, 10]
    offset[0, 16, 16] = [.25, .5]
    keypoint_regression[0, 16, 16] = [
        -1., -1.,
        -1., 1.,
        1., -1.,
        1., 1.]
    keypoint_heatmaps[0, 14, 14, 0] = _logit(0.9)
    keypoint_heatmaps[0, 14, 18, 1] = _logit(0.9)
    keypoint_heatmaps[0, 18, 14, 2] = _logit(0.9)
    keypoint_heatmaps[0, 18, 18, 3] = _logit(0.05)  # Note the low score.

    segmentation_heatmap = np.zeros((1, 32, 32, 10), dtype=np.float32)
    segmentation_heatmap[:, 14:18, 14:18, target_class_id] = 1.0
    segmentation_heatmap = _logit(segmentation_heatmap)

    dp_part_ind = 4
    dp_part_heatmap = np.zeros((1, 32, 32, _DENSEPOSE_NUM_PARTS),
                               dtype=np.float32)
    dp_part_heatmap[0, 14:18, 14:18, dp_part_ind] = 1.0
    dp_part_heatmap = _logit(dp_part_heatmap)

    dp_surf_coords = np.random.randn(1, 32, 32, 2 * _DENSEPOSE_NUM_PARTS)

    class_center = tf.constant(class_center)
    height_width = tf.constant(height_width)
    offset = tf.constant(offset)
    keypoint_heatmaps = tf.constant(keypoint_heatmaps, dtype=tf.float32)
    keypoint_offsets = tf.constant(keypoint_offsets, dtype=tf.float32)
    keypoint_regression = tf.constant(keypoint_regression, dtype=tf.float32)
    segmentation_heatmap = tf.constant(segmentation_heatmap, dtype=tf.float32)
    dp_part_heatmap = tf.constant(dp_part_heatmap, dtype=tf.float32)
    dp_surf_coords = tf.constant(dp_surf_coords, dtype=tf.float32)

    prediction_dict = {
        cnma.OBJECT_CENTER: [class_center],
        cnma.BOX_SCALE: [height_width],
        cnma.BOX_OFFSET: [offset],
        cnma.get_keypoint_name(_TASK_NAME, cnma.KEYPOINT_HEATMAP):
            [keypoint_heatmaps],
        cnma.get_keypoint_name(_TASK_NAME, cnma.KEYPOINT_OFFSET):
            [keypoint_offsets],
        cnma.get_keypoint_name(_TASK_NAME, cnma.KEYPOINT_REGRESSION):
            [keypoint_regression],
        cnma.SEGMENTATION_HEATMAP: [segmentation_heatmap],
        cnma.DENSEPOSE_HEATMAP: [dp_part_heatmap],
        cnma.DENSEPOSE_REGRESSION: [dp_surf_coords]
    }

    def graph_fn():
      detections = model.postprocess(prediction_dict,
                                     tf.constant([[128, 128, 3]]))
      return detections

    detections = self.execute_cpu(graph_fn, [])

    self.assertAllClose(detections['detection_boxes'][0, 0],
                        np.array([55, 46, 75, 86]) / 128.0)
    self.assertAllClose(detections['detection_scores'][0],
                        [.75, .5, .5, .5, .5])
    self.assertEqual(detections['detection_classes'][0, 0], target_class_id)
    self.assertEqual(detections['num_detections'], [5])
    self.assertAllEqual([1, max_detection, num_keypoints, 2],
                        detections['detection_keypoints'].shape)
    self.assertAllEqual([1, max_detection, num_keypoints],
                        detections['detection_keypoint_scores'].shape)
    self.assertAllEqual([1, max_detection, 4, 4],
                        detections['detection_masks'].shape)

    # Masks should be empty for everything but the first detection.
    self.assertAllEqual(
        detections['detection_masks'][0, 1:, :, :],
        np.zeros_like(detections['detection_masks'][0, 1:, :, :]))
    self.assertAllEqual(
        detections['detection_surface_coords'][0, 1:, :, :],
        np.zeros_like(detections['detection_surface_coords'][0, 1:, :, :]))

    if target_class_id == 1:
      expected_kpts_for_obj_0 = np.array(
          [[14., 14.], [14., 18.], [18., 14.], [17., 17.]]) / 32.
      expected_kpt_scores_for_obj_0 = np.array(
          [0.9, 0.9, 0.9, cnma.UNMATCHED_KEYPOINT_SCORE])
      np.testing.assert_allclose(detections['detection_keypoints'][0][0],
                                 expected_kpts_for_obj_0, rtol=1e-6)
      np.testing.assert_allclose(detections['detection_keypoint_scores'][0][0],
                                 expected_kpt_scores_for_obj_0, rtol=1e-6)
      # First detection has DensePose parts.
      self.assertSameElements(
          np.unique(detections['detection_masks'][0, 0, :, :]),
          set([0, dp_part_ind + 1]))
      self.assertGreater(np.sum(np.abs(detections['detection_surface_coords'])),
                         0.0)
    else:
      # All keypoint outputs should be zeros.
      np.testing.assert_allclose(
          detections['detection_keypoints'][0][0],
          np.zeros([num_keypoints, 2], np.float),
          rtol=1e-6)
      np.testing.assert_allclose(
          detections['detection_keypoint_scores'][0][0],
          np.zeros([num_keypoints], np.float),
          rtol=1e-6)
      # Binary segmentation mask.
      self.assertSameElements(
          np.unique(detections['detection_masks'][0, 0, :, :]),
          set([0, 1]))
      # No DensePose surface coordinates.
      np.testing.assert_allclose(
          detections['detection_surface_coords'][0, 0, :, :],
          np.zeros_like(detections['detection_surface_coords'][0, 0, :, :]))

  def test_get_instance_indices(self):
    classes = tf.constant([[0, 1, 2, 0], [2, 1, 2, 2]], dtype=tf.int32)
    num_detections = tf.constant([1, 3], dtype=tf.int32)
    batch_index = 1
    class_id = 2
    model = build_center_net_meta_arch()
    valid_indices = model._get_instance_indices(
        classes, num_detections, batch_index, class_id)
    self.assertAllEqual(valid_indices.numpy(), [0, 2])


def get_fake_prediction_dict(input_height, input_width, stride):
  """Prepares the fake prediction dictionary."""
  output_height = input_height // stride
  output_width = input_width // stride
  object_center = np.zeros((2, output_height, output_width, _NUM_CLASSES),
                           dtype=np.float32)
  # Box center:
  #   y: floor((0.54 + 0.56) / 2 * 4) = 2,
  #   x: floor((0.54 + 0.56) / 2 * 8) = 4
  object_center[0, 2, 4, 1] = 1.0
  object_center = _logit(object_center)

  # Box size:
  #   height: (0.56 - 0.54) * 4 = 0.08
  #   width:  (0.56 - 0.54) * 8 = 0.16
  object_scale = np.zeros((2, output_height, output_width, 2), dtype=np.float32)
  object_scale[0, 2, 4] = 0.08, 0.16

  # Box center offset coordinate (0.55, 0.55):
  #   y-offset: 0.55 * 4 - 2 = 0.2
  #   x-offset: 0.55 * 8 - 4 = 0.4
  object_offset = np.zeros((2, output_height, output_width, 2),
                           dtype=np.float32)
  object_offset[0, 2, 4] = 0.2, 0.4

  keypoint_heatmap = np.zeros((2, output_height, output_width, _NUM_KEYPOINTS),
                              dtype=np.float32)
  keypoint_heatmap[0, 2, 4, 1] = 1.0
  keypoint_heatmap[0, 2, 4, 3] = 1.0
  keypoint_heatmap = _logit(keypoint_heatmap)

  keypoint_offset = np.zeros((2, output_height, output_width, 2),
                             dtype=np.float32)
  keypoint_offset[0, 2, 4] = 0.2, 0.4

  keypoint_regression = np.zeros(
      (2, output_height, output_width, 2 * _NUM_KEYPOINTS), dtype=np.float32)
  keypoint_regression[0, 2, 4] = 0.0, 0.0, 0.2, 0.4, 0.0, 0.0, 0.2, 0.4

  mask_heatmap = np.zeros((2, output_height, output_width, _NUM_CLASSES),
                          dtype=np.float32)
  mask_heatmap[0, 2, 4, 1] = 1.0
  mask_heatmap = _logit(mask_heatmap)

  densepose_heatmap = np.zeros((2, output_height, output_width,
                                _DENSEPOSE_NUM_PARTS), dtype=np.float32)
  densepose_heatmap[0, 2, 4, 5] = 1.0
  densepose_heatmap = _logit(densepose_heatmap)

  densepose_regression = np.zeros((2, output_height, output_width,
                                   2 * _DENSEPOSE_NUM_PARTS), dtype=np.float32)
  # The surface coordinate indices for part index 5 are:
  # (5 * 2, 5 * 2 + 1), or (10, 11).
  densepose_regression[0, 2, 4, 10:12] = 0.4, 0.7

  prediction_dict = {
      'preprocessed_inputs':
          tf.zeros((2, input_height, input_width, 3)),
      cnma.OBJECT_CENTER: [
          tf.constant(object_center),
          tf.constant(object_center)
      ],
      cnma.BOX_SCALE: [
          tf.constant(object_scale),
          tf.constant(object_scale)
      ],
      cnma.BOX_OFFSET: [
          tf.constant(object_offset),
          tf.constant(object_offset)
      ],
      cnma.get_keypoint_name(_TASK_NAME, cnma.KEYPOINT_HEATMAP): [
          tf.constant(keypoint_heatmap),
          tf.constant(keypoint_heatmap)
      ],
      cnma.get_keypoint_name(_TASK_NAME, cnma.KEYPOINT_OFFSET): [
          tf.constant(keypoint_offset),
          tf.constant(keypoint_offset)
      ],
      cnma.get_keypoint_name(_TASK_NAME, cnma.KEYPOINT_REGRESSION): [
          tf.constant(keypoint_regression),
          tf.constant(keypoint_regression)
      ],
      cnma.SEGMENTATION_HEATMAP: [
          tf.constant(mask_heatmap),
          tf.constant(mask_heatmap)
      ],
      cnma.DENSEPOSE_HEATMAP: [
          tf.constant(densepose_heatmap),
          tf.constant(densepose_heatmap),
      ],
      cnma.DENSEPOSE_REGRESSION: [
          tf.constant(densepose_regression),
          tf.constant(densepose_regression),
      ]
  }
  return prediction_dict


def get_fake_groundtruth_dict(input_height, input_width, stride):
  """Prepares the fake groundtruth dictionary."""
  # A small box with center at (0.55, 0.55).
  boxes = [
      tf.constant([[0.54, 0.54, 0.56, 0.56]]),
      tf.constant([[0.0, 0.0, 0.5, 0.5]]),
  ]
  classes = [
      tf.one_hot([1], depth=_NUM_CLASSES),
      tf.one_hot([0], depth=_NUM_CLASSES),
  ]
  weights = [
      tf.constant([1.]),
      tf.constant([0.]),
  ]
  keypoints = [
      tf.tile(
          tf.expand_dims(
              tf.constant([[float('nan'), 0.55,
                            float('nan'), 0.55, 0.55, 0.0]]),
              axis=2),
          multiples=[1, 1, 2]),
      tf.tile(
          tf.expand_dims(
              tf.constant([[float('nan'), 0.55,
                            float('nan'), 0.55, 0.55, 0.0]]),
              axis=2),
          multiples=[1, 1, 2]),
  ]
  labeled_classes = [
      tf.one_hot([1], depth=_NUM_CLASSES) + tf.one_hot([2], depth=_NUM_CLASSES),
      tf.one_hot([0], depth=_NUM_CLASSES) + tf.one_hot([1], depth=_NUM_CLASSES),
  ]
  mask = np.zeros((1, input_height, input_width), dtype=np.float32)
  mask[0, 8:8+stride, 16:16+stride] = 1
  masks = [
      tf.constant(mask),
      tf.zeros_like(mask),
  ]
  densepose_num_points = [
      tf.constant([1], dtype=tf.int32),
      tf.constant([0], dtype=tf.int32),
  ]
  densepose_part_ids = [
      tf.constant([[5, 0, 0]], dtype=tf.int32),
      tf.constant([[0, 0, 0]], dtype=tf.int32),
  ]
  densepose_surface_coords_np = np.zeros((1, 3, 4), dtype=np.float32)
  densepose_surface_coords_np[0, 0, :] = 0.55, 0.55, 0.4, 0.7
  densepose_surface_coords = [
      tf.constant(densepose_surface_coords_np),
      tf.zeros_like(densepose_surface_coords_np)
  ]
  groundtruth_dict = {
      fields.BoxListFields.boxes: boxes,
      fields.BoxListFields.weights: weights,
      fields.BoxListFields.classes: classes,
      fields.BoxListFields.keypoints: keypoints,
      fields.BoxListFields.masks: masks,
      fields.BoxListFields.densepose_num_points: densepose_num_points,
      fields.BoxListFields.densepose_part_ids: densepose_part_ids,
      fields.BoxListFields.densepose_surface_coords:
          densepose_surface_coords,
      fields.InputDataFields.groundtruth_labeled_classes: labeled_classes,
  }
  return groundtruth_dict


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class CenterNetMetaComputeLossTest(test_case.TestCase):
  """Test for CenterNet loss compuation related functions."""

  def setUp(self):
    self.model = build_center_net_meta_arch()
    self.classification_loss_fn = self.model._center_params.classification_loss
    self.localization_loss_fn = self.model._od_params.localization_loss
    self.true_image_shapes = tf.constant([[16, 24, 3], [16, 24, 3]])
    self.input_height = 16
    self.input_width = 32
    self.stride = 4
    self.per_pixel_weights = self.get_per_pixel_weights(self.true_image_shapes,
                                                        self.input_height,
                                                        self.input_width,
                                                        self.stride)
    self.prediction_dict = get_fake_prediction_dict(self.input_height,
                                                    self.input_width,
                                                    self.stride)
    self.model._groundtruth_lists = get_fake_groundtruth_dict(
        self.input_height, self.input_width, self.stride)
    super(CenterNetMetaComputeLossTest, self).setUp()

  def get_per_pixel_weights(self, true_image_shapes, input_height, input_width,
                            stride):
    output_height, output_width = (input_height // stride,
                                   input_width // stride)

    # TODO(vighneshb) Explore whether using floor here is safe.
    output_true_image_shapes = tf.ceil(tf.to_float(true_image_shapes) / stride)
    per_pixel_weights = cnma.get_valid_anchor_weights_in_flattened_image(
        output_true_image_shapes, output_height, output_width)
    per_pixel_weights = tf.expand_dims(per_pixel_weights, 2)
    return per_pixel_weights

  def test_compute_object_center_loss(self):
    def graph_fn():
      loss = self.model._compute_object_center_loss(
          object_center_predictions=self.prediction_dict[cnma.OBJECT_CENTER],
          input_height=self.input_height,
          input_width=self.input_width,
          per_pixel_weights=self.per_pixel_weights)
      return loss

    loss = self.execute(graph_fn, [])

    # The prediction and groundtruth are curated to produce very low loss.
    self.assertGreater(0.01, loss)

    default_value = self.model._center_params.use_only_known_classes
    self.model._center_params = (
        self.model._center_params._replace(use_only_known_classes=True))
    loss = self.model._compute_object_center_loss(
        object_center_predictions=self.prediction_dict[cnma.OBJECT_CENTER],
        input_height=self.input_height,
        input_width=self.input_width,
        per_pixel_weights=self.per_pixel_weights)
    self.model._center_params = (
        self.model._center_params._replace(
            use_only_known_classes=default_value))

    # The prediction and groundtruth are curated to produce very low loss.
    self.assertGreater(0.01, loss)

  def test_compute_box_scale_and_offset_loss(self):
    def graph_fn():
      scale_loss, offset_loss = self.model._compute_box_scale_and_offset_loss(
          scale_predictions=self.prediction_dict[cnma.BOX_SCALE],
          offset_predictions=self.prediction_dict[cnma.BOX_OFFSET],
          input_height=self.input_height,
          input_width=self.input_width)
      return scale_loss, offset_loss

    scale_loss, offset_loss = self.execute(graph_fn, [])

    # The prediction and groundtruth are curated to produce very low loss.
    self.assertGreater(0.01, scale_loss)
    self.assertGreater(0.01, offset_loss)

  def test_compute_kp_heatmap_loss(self):
    def graph_fn():
      loss = self.model._compute_kp_heatmap_loss(
          input_height=self.input_height,
          input_width=self.input_width,
          task_name=_TASK_NAME,
          heatmap_predictions=self.prediction_dict[cnma.get_keypoint_name(
              _TASK_NAME, cnma.KEYPOINT_HEATMAP)],
          classification_loss_fn=self.classification_loss_fn,
          per_pixel_weights=self.per_pixel_weights)
      return loss

    loss = self.execute(graph_fn, [])

    # The prediction and groundtruth are curated to produce very low loss.
    self.assertGreater(0.01, loss)

  def test_compute_kp_offset_loss(self):
    def graph_fn():
      loss = self.model._compute_kp_offset_loss(
          input_height=self.input_height,
          input_width=self.input_width,
          task_name=_TASK_NAME,
          offset_predictions=self.prediction_dict[cnma.get_keypoint_name(
              _TASK_NAME, cnma.KEYPOINT_OFFSET)],
          localization_loss_fn=self.localization_loss_fn)
      return loss

    loss = self.execute(graph_fn, [])

    # The prediction and groundtruth are curated to produce very low loss.
    self.assertGreater(0.01, loss)

  def test_compute_kp_regression_loss(self):
    def graph_fn():
      loss = self.model._compute_kp_regression_loss(
          input_height=self.input_height,
          input_width=self.input_width,
          task_name=_TASK_NAME,
          regression_predictions=self.prediction_dict[cnma.get_keypoint_name(
              _TASK_NAME, cnma.KEYPOINT_REGRESSION,)],
          localization_loss_fn=self.localization_loss_fn)
      return loss

    loss = self.execute(graph_fn, [])

    # The prediction and groundtruth are curated to produce very low loss.
    self.assertGreater(0.01, loss)


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class CenterNetMetaArchRestoreTest(test_case.TestCase):

  def test_restore_map_resnet(self):
    """Test restore map for a resnet backbone."""

    model = build_center_net_meta_arch(build_resnet=True)
    restore_from_objects_map = model.restore_from_objects('classification')
    self.assertIsInstance(restore_from_objects_map['feature_extractor'],
                          tf.keras.Model)

  def test_retore_map_error(self):
    """Test that restoring unsupported checkpoint type raises an error."""

    model = build_center_net_meta_arch(build_resnet=True)
    msg = ("Sub model detection is not defined for ResNet."
           "Supported types are ['classification'].")
    with self.assertRaisesRegex(ValueError, re.escape(msg)):
      model.restore_from_objects('detection')


class DummyFeatureExtractor(cnma.CenterNetFeatureExtractor):

  def __init__(self,
               channel_means,
               channel_stds,
               bgr_ordering,
               num_feature_outputs,
               stride):
    self._num_feature_outputs = num_feature_outputs
    self._stride = stride
    super(DummyFeatureExtractor, self).__init__(
        channel_means=channel_means, channel_stds=channel_stds,
        bgr_ordering=bgr_ordering)

  def predict(self):
    pass

  def loss(self):
    pass

  def postprocess(self):
    pass

  def call(self, inputs):
    batch_size, input_height, input_width, _ = inputs.shape
    fake_output = tf.ones([
        batch_size, input_height // self._stride, input_width // self._stride,
        64
    ], dtype=tf.float32)
    return [fake_output] * self._num_feature_outputs

  @property
  def out_stride(self):
    return self._stride

  @property
  def num_feature_outputs(self):
    return self._num_feature_outputs


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class CenterNetFeatureExtractorTest(test_case.TestCase):
  """Test the base feature extractor class."""

  def test_preprocess(self):
    feature_extractor = DummyFeatureExtractor(
        channel_means=(1.0, 2.0, 3.0),
        channel_stds=(10., 20., 30.), bgr_ordering=False,
        num_feature_outputs=2, stride=4)

    img = np.zeros((2, 32, 32, 3))
    img[:, :, :] = 11, 22, 33

    def graph_fn():
      output = feature_extractor.preprocess(img)
      return output

    output = self.execute(graph_fn, [])
    self.assertAlmostEqual(output.sum(), 2 * 32 * 32 * 3)

  def test_bgr_ordering(self):
    feature_extractor = DummyFeatureExtractor(
        channel_means=(0.0, 0.0, 0.0),
        channel_stds=(1., 1., 1.), bgr_ordering=True,
        num_feature_outputs=2, stride=4)

    img = np.zeros((2, 32, 32, 3), dtype=np.float32)
    img[:, :, :] = 1, 2, 3

    def graph_fn():
      output = feature_extractor.preprocess(img)
      return output

    output = self.execute(graph_fn, [])
    self.assertAllClose(output[..., 2], 1 * np.ones((2, 32, 32)))
    self.assertAllClose(output[..., 1], 2 * np.ones((2, 32, 32)))
    self.assertAllClose(output[..., 0], 3 * np.ones((2, 32, 32)))

  def test_default_ordering(self):
    feature_extractor = DummyFeatureExtractor(
        channel_means=(0.0, 0.0, 0.0),
        channel_stds=(1., 1., 1.), bgr_ordering=False,
        num_feature_outputs=2, stride=4)

    img = np.zeros((2, 32, 32, 3), dtype=np.float32)
    img[:, :, :] = 1, 2, 3

    def graph_fn():
      output = feature_extractor.preprocess(img)
      return output

    output = self.execute(graph_fn, [])
    self.assertAllClose(output[..., 0], 1 * np.ones((2, 32, 32)))
    self.assertAllClose(output[..., 1], 2 * np.ones((2, 32, 32)))
    self.assertAllClose(output[..., 2], 3 * np.ones((2, 32, 32)))


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
