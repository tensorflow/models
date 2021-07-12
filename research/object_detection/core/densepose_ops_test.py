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

"""Tests for object_detection.core.densepose_ops."""
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.core import densepose_ops
from object_detection.utils import test_case


class DensePoseOpsTest(test_case.TestCase):
  """Tests for common DensePose operations."""

  def test_scale(self):
    def graph_fn():
      dp_surface_coords = tf.constant([
          [[0.0, 0.0, 0.1, 0.2], [100.0, 200.0, 0.3, 0.4]],
          [[50.0, 120.0, 0.5, 0.6], [100.0, 140.0, 0.7, 0.8]]
      ])
      y_scale = tf.constant(1.0 / 100)
      x_scale = tf.constant(1.0 / 200)

      output = densepose_ops.scale(dp_surface_coords, y_scale, x_scale)
      return output
    output = self.execute(graph_fn, [])

    expected_dp_surface_coords = np.array([
        [[0., 0., 0.1, 0.2], [1.0, 1.0, 0.3, 0.4]],
        [[0.5, 0.6, 0.5, 0.6], [1.0, 0.7, 0.7, 0.8]]
    ])
    self.assertAllClose(output, expected_dp_surface_coords)

  def test_clip_to_window(self):
    def graph_fn():
      dp_surface_coords = tf.constant([
          [[0.25, 0.5, 0.1, 0.2], [0.75, 0.75, 0.3, 0.4]],
          [[0.5, 0.0, 0.5, 0.6], [1.0, 1.0, 0.7, 0.8]]
      ])
      window = tf.constant([0.25, 0.25, 0.75, 0.75])

      output = densepose_ops.clip_to_window(dp_surface_coords, window)
      return output
    output = self.execute(graph_fn, [])

    expected_dp_surface_coords = np.array([
        [[0.25, 0.5, 0.1, 0.2], [0.75, 0.75, 0.3, 0.4]],
        [[0.5, 0.25, 0.5, 0.6], [0.75, 0.75, 0.7, 0.8]]
    ])
    self.assertAllClose(output, expected_dp_surface_coords)

  def test_prune_outside_window(self):
    def graph_fn():
      dp_num_points = tf.constant([2, 0, 1])
      dp_part_ids = tf.constant([[1, 1], [0, 0], [16, 0]])
      dp_surface_coords = tf.constant([
          [[0.9, 0.5, 0.1, 0.2], [0.75, 0.75, 0.3, 0.4]],
          [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
          [[0.8, 0.5, 0.6, 0.6], [0.5, 0.5, 0.7, 0.7]]
      ])
      window = tf.constant([0.25, 0.25, 0.75, 0.75])

      new_dp_num_points, new_dp_part_ids, new_dp_surface_coords = (
          densepose_ops.prune_outside_window(dp_num_points, dp_part_ids,
                                             dp_surface_coords, window))
      return new_dp_num_points, new_dp_part_ids, new_dp_surface_coords
    new_dp_num_points, new_dp_part_ids, new_dp_surface_coords = (
        self.execute_cpu(graph_fn, []))

    expected_dp_num_points = np.array([1, 0, 0])
    expected_dp_part_ids = np.array([[1], [0], [0]])
    expected_dp_surface_coords = np.array([
        [[0.75, 0.75, 0.3, 0.4]],
        [[0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0]]
    ])
    self.assertAllEqual(new_dp_num_points, expected_dp_num_points)
    self.assertAllEqual(new_dp_part_ids, expected_dp_part_ids)
    self.assertAllClose(new_dp_surface_coords, expected_dp_surface_coords)

  def test_change_coordinate_frame(self):
    def graph_fn():
      dp_surface_coords = tf.constant([
          [[0.25, 0.5, 0.1, 0.2], [0.75, 0.75, 0.3, 0.4]],
          [[0.5, 0.0, 0.5, 0.6], [1.0, 1.0, 0.7, 0.8]]
      ])
      window = tf.constant([0.25, 0.25, 0.75, 0.75])

      output = densepose_ops.change_coordinate_frame(dp_surface_coords, window)
      return output
    output = self.execute(graph_fn, [])

    expected_dp_surface_coords = np.array([
        [[0, 0.5, 0.1, 0.2], [1.0, 1.0, 0.3, 0.4]],
        [[0.5, -0.5, 0.5, 0.6], [1.5, 1.5, 0.7, 0.8]]
    ])
    self.assertAllClose(output, expected_dp_surface_coords)

  def test_to_normalized_coordinates(self):
    def graph_fn():
      dp_surface_coords = tf.constant([
          [[10., 30., 0.1, 0.2], [30., 45., 0.3, 0.4]],
          [[20., 0., 0.5, 0.6], [40., 60., 0.7, 0.8]]
      ])
      output = densepose_ops.to_normalized_coordinates(
          dp_surface_coords, 40, 60)
      return output
    output = self.execute(graph_fn, [])

    expected_dp_surface_coords = np.array([
        [[0.25, 0.5, 0.1, 0.2], [0.75, 0.75, 0.3, 0.4]],
        [[0.5, 0.0, 0.5, 0.6], [1.0, 1.0, 0.7, 0.8]]
    ])
    self.assertAllClose(output, expected_dp_surface_coords)

  def test_to_absolute_coordinates(self):
    def graph_fn():
      dp_surface_coords = tf.constant([
          [[0.25, 0.5, 0.1, 0.2], [0.75, 0.75, 0.3, 0.4]],
          [[0.5, 0.0, 0.5, 0.6], [1.0, 1.0, 0.7, 0.8]]
      ])
      output = densepose_ops.to_absolute_coordinates(
          dp_surface_coords, 40, 60)
      return output
    output = self.execute(graph_fn, [])

    expected_dp_surface_coords = np.array([
        [[10., 30., 0.1, 0.2], [30., 45., 0.3, 0.4]],
        [[20., 0., 0.5, 0.6], [40., 60., 0.7, 0.8]]
    ])
    self.assertAllClose(output, expected_dp_surface_coords)

  def test_horizontal_flip(self):
    part_ids_np = np.array([[1, 4], [0, 8]], dtype=np.int32)
    surf_coords_np = np.array([
        [[0.1, 0.7, 0.2, 0.4], [0.3, 0.8, 0.2, 0.4]],
        [[0.0, 0.5, 0.8, 0.7], [0.6, 1.0, 0.7, 0.9]],
    ], dtype=np.float32)
    def graph_fn():
      part_ids = tf.constant(part_ids_np, dtype=tf.int32)
      surf_coords = tf.constant(surf_coords_np, dtype=tf.float32)
      flipped_part_ids, flipped_surf_coords = densepose_ops.flip_horizontal(
          part_ids, surf_coords)
      flipped_twice_part_ids, flipped_twice_surf_coords = (
          densepose_ops.flip_horizontal(flipped_part_ids, flipped_surf_coords))
      return (flipped_part_ids, flipped_surf_coords,
              flipped_twice_part_ids, flipped_twice_surf_coords)
    (flipped_part_ids, flipped_surf_coords, flipped_twice_part_ids,
     flipped_twice_surf_coords) = self.execute(graph_fn, [])

    expected_flipped_part_ids = [[1, 5],  # 1->1, 4->5
                                 [0, 9]]  # 0->0, 8->9
    expected_flipped_surf_coords_yx = np.array([
        [[0.1, 1.0-0.7], [0.3, 1.0-0.8]],
        [[0.0, 1.0-0.5], [0.6, 1.0-1.0]],
    ], dtype=np.float32)
    self.assertAllEqual(expected_flipped_part_ids, flipped_part_ids)
    self.assertAllClose(expected_flipped_surf_coords_yx,
                        flipped_surf_coords[:, :, 0:2])
    self.assertAllEqual(part_ids_np, flipped_twice_part_ids)
    self.assertAllClose(surf_coords_np, flipped_twice_surf_coords, rtol=1e-2,
                        atol=1e-2)


if __name__ == '__main__':
  tf.test.main()
