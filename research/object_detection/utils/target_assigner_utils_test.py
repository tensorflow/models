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
"""Tests for utils.target_assigner_utils."""

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.utils import target_assigner_utils as ta_utils
from object_detection.utils import test_case


class TargetUtilTest(parameterized.TestCase, test_case.TestCase):

  def test_image_shape_to_grids(self):
    def graph_fn():
      (y_grid, x_grid) = ta_utils.image_shape_to_grids(height=2, width=3)
      return y_grid, x_grid

    expected_y_grid = np.array([[0, 0, 0], [1, 1, 1]])
    expected_x_grid = np.array([[0, 1, 2], [0, 1, 2]])

    y_grid, x_grid = self.execute(graph_fn, [])

    np.testing.assert_array_equal(y_grid, expected_y_grid)
    np.testing.assert_array_equal(x_grid, expected_x_grid)

  @parameterized.parameters((False,), (True,))
  def test_coordinates_to_heatmap(self, sparse):
    if not hasattr(tf, 'tensor_scatter_nd_max'):
      self.skipTest('Cannot test function due to old TF version.')

    def graph_fn():
      (y_grid, x_grid) = ta_utils.image_shape_to_grids(height=3, width=5)
      y_coordinates = tf.constant([1.5, 0.5], dtype=tf.float32)
      x_coordinates = tf.constant([2.5, 4.5], dtype=tf.float32)
      sigma = tf.constant([0.1, 0.5], dtype=tf.float32)
      channel_onehot = tf.constant([[1, 0, 0], [0, 1, 0]], dtype=tf.float32)
      channel_weights = tf.constant([1, 1], dtype=tf.float32)
      heatmap = ta_utils.coordinates_to_heatmap(y_grid, x_grid, y_coordinates,
                                                x_coordinates, sigma,
                                                channel_onehot,
                                                channel_weights, sparse=sparse)
      return heatmap

    heatmap = self.execute(graph_fn, [])
    # Peak at (1, 2) for the first class.
    self.assertAlmostEqual(1.0, heatmap[1, 2, 0])
    # Peak at (0, 4) for the second class.
    self.assertAlmostEqual(1.0, heatmap[0, 4, 1])

  def test_compute_floor_offsets_with_indices_onlysource(self):

    def graph_fn():
      y_source = tf.constant([1.5, 0.3], dtype=tf.float32)
      x_source = tf.constant([2.5, 4.2], dtype=tf.float32)
      (offsets, indices) = ta_utils.compute_floor_offsets_with_indices(
          y_source, x_source)

      return offsets, indices

    offsets, indices = self.execute(graph_fn, [])

    np.testing.assert_array_almost_equal(offsets,
                                         np.array([[0.5, 0.5], [0.3, 0.2]]))
    np.testing.assert_array_almost_equal(indices,
                                         np.array([[1, 2], [0, 4]]))

  def test_compute_floor_offsets_with_indices_and_targets(self):

    def graph_fn():
      y_source = tf.constant([1.5, 0.3], dtype=tf.float32)
      x_source = tf.constant([2.5, 4.2], dtype=tf.float32)
      y_target = tf.constant([2.1, 0.1], dtype=tf.float32)
      x_target = tf.constant([1.2, 4.5], dtype=tf.float32)
      (offsets, indices) = ta_utils.compute_floor_offsets_with_indices(
          y_source, x_source, y_target, x_target)
      return offsets, indices

    offsets, indices = self.execute(graph_fn, [])

    np.testing.assert_array_almost_equal(offsets,
                                         np.array([[1.1, -0.8], [0.1, 0.5]]))
    np.testing.assert_array_almost_equal(indices, np.array([[1, 2], [0, 4]]))

  def test_compute_floor_offsets_with_indices_multisources(self):

    def graph_fn():
      y_source = tf.constant([[1.0, 0.0], [2.0, 3.0]], dtype=tf.float32)
      x_source = tf.constant([[2.0, 4.0], [3.0, 3.0]], dtype=tf.float32)
      y_target = tf.constant([2.1, 0.1], dtype=tf.float32)
      x_target = tf.constant([1.2, 4.5], dtype=tf.float32)
      (offsets, indices) = ta_utils.compute_floor_offsets_with_indices(
          y_source, x_source, y_target, x_target)
      return offsets, indices

    offsets, indices = self.execute(graph_fn, [])
    # Offset from the first source to target.
    np.testing.assert_array_almost_equal(offsets[:, 0, :],
                                         np.array([[1.1, -0.8], [-1.9, 1.5]]))
    # Offset from the second source to target.
    np.testing.assert_array_almost_equal(offsets[:, 1, :],
                                         np.array([[2.1, -2.8], [-2.9, 1.5]]))
    # Indices from the first source to target.
    np.testing.assert_array_almost_equal(indices[:, 0, :],
                                         np.array([[1, 2], [2, 3]]))
    # Indices from the second source to target.
    np.testing.assert_array_almost_equal(indices[:, 1, :],
                                         np.array([[0, 4], [3, 3]]))

  def test_get_valid_keypoints_mask(self):

    def graph_fn():
      class_onehot = tf.constant(
          [[0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 1, 0, 1]], dtype=tf.float32)
      keypoints = tf.constant(
          [[0.1, float('nan'), 0.2, 0.0],
           [0.0, 0.0, 0.1, 0.9],
           [3.2, 4.3, float('nan'), 0.2]],
          dtype=tf.float32)
      keypoint_coordinates = tf.stack([keypoints, keypoints], axis=2)
      mask, keypoints_nan_to_zeros = ta_utils.get_valid_keypoint_mask_for_class(
          keypoint_coordinates=keypoint_coordinates,
          class_id=2,
          class_onehot=class_onehot,
          keypoint_indices=[1, 2])

      return mask, keypoints_nan_to_zeros

    keypoints = np.array([[0.0, 0.2],
                          [0.0, 0.1],
                          [4.3, 0.0]])
    expected_mask = np.array([[0, 1], [0, 0], [1, 0]])
    expected_keypoints = np.stack([keypoints, keypoints], axis=2)
    mask, keypoints_nan_to_zeros = self.execute(graph_fn, [])

    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_almost_equal(keypoints_nan_to_zeros,
                                         expected_keypoints)

  def test_get_valid_keypoints_with_mask(self):
    def graph_fn():
      class_onehot = tf.constant(
          [[0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 1, 0, 1]], dtype=tf.float32)
      keypoints = tf.constant(
          [[0.1, float('nan'), 0.2, 0.0],
           [0.0, 0.0, 0.1, 0.9],
           [3.2, 4.3, float('nan'), 0.2]],
          dtype=tf.float32)
      keypoint_coordinates = tf.stack([keypoints, keypoints], axis=2)
      weights = tf.constant([0.0, 0.0, 1.0])
      mask, keypoints_nan_to_zeros = ta_utils.get_valid_keypoint_mask_for_class(
          keypoint_coordinates=keypoint_coordinates,
          class_id=2,
          class_onehot=class_onehot,
          class_weights=weights,
          keypoint_indices=[1, 2])
      return mask, keypoints_nan_to_zeros

    expected_mask = np.array([[0, 0], [0, 0], [1, 0]])
    keypoints = np.array([[0.0, 0.2],
                          [0.0, 0.1],
                          [4.3, 0.0]])
    expected_keypoints = np.stack([keypoints, keypoints], axis=2)
    mask, keypoints_nan_to_zeros = self.execute(graph_fn, [])

    np.testing.assert_array_equal(mask, expected_mask)
    np.testing.assert_array_almost_equal(keypoints_nan_to_zeros,
                                         expected_keypoints)

  def test_blackout_pixel_weights_by_box_regions(self):
    def graph_fn():
      boxes = tf.constant(
          [[0.0, 0.0, 5, 5], [0.0, 0.0, 10.0, 20.0], [6.0, 12.0, 8.0, 18.0]],
          dtype=tf.float32)
      blackout = tf.constant([True, False, True], dtype=tf.bool)
      blackout_pixel_weights_by_box_regions = tf.function(
          ta_utils.blackout_pixel_weights_by_box_regions)
      output = blackout_pixel_weights_by_box_regions(10, 20, boxes, blackout)
      return output

    output = self.execute(graph_fn, [])
    # All zeros in region [0:5, 0:5].
    self.assertAlmostEqual(np.sum(output[0:5, 0:5]), 0.0)
    # All zeros in region [12:18, 6:8].
    self.assertAlmostEqual(np.sum(output[6:8, 12:18]), 0.0)
    # All other pixel weights should be 1.0.
    # 20 * 10 - 5 * 5 - 2 * 6 = 163.0
    self.assertAlmostEqual(np.sum(output), 163.0)

  def test_blackout_pixel_weights_by_box_regions_with_weights(self):
    def graph_fn():
      boxes = tf.constant(
          [[0.0, 0.0, 2.0, 2.0],
           [0.0, 0.0, 4.0, 2.0],
           [3.0, 0.0, 4.0, 4.0]],
          dtype=tf.float32)
      blackout = tf.constant([False, False, True], dtype=tf.bool)
      weights = tf.constant([0.4, 0.3, 0.2], tf.float32)
      blackout_pixel_weights_by_box_regions = tf.function(
          ta_utils.blackout_pixel_weights_by_box_regions)
      output = blackout_pixel_weights_by_box_regions(
          4, 4, boxes, blackout, weights)
      return output

    output = self.execute(graph_fn, [])
    expected_weights = [
        [0.4, 0.4, 1.0, 1.0],
        [0.4, 0.4, 1.0, 1.0],
        [0.3, 0.3, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0]]
    np.testing.assert_array_almost_equal(expected_weights, output)

  def test_blackout_pixel_weights_by_box_regions_zero_instance(self):
    def graph_fn():
      boxes = tf.zeros([0, 4], dtype=tf.float32)
      blackout = tf.zeros([0], dtype=tf.bool)
      blackout_pixel_weights_by_box_regions = tf.function(
          ta_utils.blackout_pixel_weights_by_box_regions)
      output = blackout_pixel_weights_by_box_regions(10, 20, boxes, blackout)
      return output

    output = self.execute(graph_fn, [])
    # The output should be all 1s since there's no annotation provided.
    np.testing.assert_array_equal(output, np.ones([10, 20], dtype=np.float32))

  def test_get_surrounding_grids(self):

    def graph_fn():
      y_coordinates = tf.constant([0.5], dtype=tf.float32)
      x_coordinates = tf.constant([4.5], dtype=tf.float32)
      output = ta_utils.get_surrounding_grids(
          height=3,
          width=5,
          y_coordinates=y_coordinates,
          x_coordinates=x_coordinates,
          radius=1)
      return output

    y_indices, x_indices, valid = self.execute(graph_fn, [])

    # Five neighboring indices: [-1, 4] (out of bound), [0, 3], [0, 4],
    # [0, 5] (out of bound), [1, 4].
    np.testing.assert_array_almost_equal(
        y_indices,
        np.array([[0.0, 0.0, 0.0, 0.0, 1.0]]))
    np.testing.assert_array_almost_equal(
        x_indices,
        np.array([[0.0, 3.0, 4.0, 0.0, 4.0]]))
    self.assertAllEqual(valid, [[False, True, True, False, True]])


if __name__ == '__main__':
  tf.test.main()
