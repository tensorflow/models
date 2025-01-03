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

"""Tests for object_detection.core.keypoint_ops."""
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.core import keypoint_ops
from object_detection.utils import test_case


class KeypointOpsTest(test_case.TestCase):
  """Tests for common keypoint operations."""

  def test_scale(self):
    def graph_fn():
      keypoints = tf.constant([
          [[0.0, 0.0], [100.0, 200.0]],
          [[50.0, 120.0], [100.0, 140.0]]
      ])
      y_scale = tf.constant(1.0 / 100)
      x_scale = tf.constant(1.0 / 200)

      expected_keypoints = tf.constant([
          [[0., 0.], [1.0, 1.0]],
          [[0.5, 0.6], [1.0, 0.7]]
      ])
      output = keypoint_ops.scale(keypoints, y_scale, x_scale)
      return output, expected_keypoints
    output, expected_keypoints = self.execute(graph_fn, [])
    self.assertAllClose(output, expected_keypoints)

  def test_clip_to_window(self):
    def graph_fn():
      keypoints = tf.constant([
          [[0.25, 0.5], [0.75, 0.75]],
          [[0.5, 0.0], [1.0, 1.0]]
      ])
      window = tf.constant([0.25, 0.25, 0.75, 0.75])

      expected_keypoints = tf.constant([
          [[0.25, 0.5], [0.75, 0.75]],
          [[0.5, 0.25], [0.75, 0.75]]
      ])
      output = keypoint_ops.clip_to_window(keypoints, window)
      return output, expected_keypoints
    output, expected_keypoints = self.execute(graph_fn, [])
    self.assertAllClose(output, expected_keypoints)

  def test_prune_outside_window(self):
    def graph_fn():
      keypoints = tf.constant([
          [[0.25, 0.5], [0.75, 0.75]],
          [[0.5, 0.0], [1.0, 1.0]]
      ])
      window = tf.constant([0.25, 0.25, 0.75, 0.75])

      expected_keypoints = tf.constant([[[0.25, 0.5], [0.75, 0.75]],
                                        [[np.nan, np.nan], [np.nan, np.nan]]])
      output = keypoint_ops.prune_outside_window(keypoints, window)
      return output, expected_keypoints
    output, expected_keypoints = self.execute(graph_fn, [])
    self.assertAllClose(output, expected_keypoints)

  def test_change_coordinate_frame(self):
    def graph_fn():
      keypoints = tf.constant([
          [[0.25, 0.5], [0.75, 0.75]],
          [[0.5, 0.0], [1.0, 1.0]]
      ])
      window = tf.constant([0.25, 0.25, 0.75, 0.75])

      expected_keypoints = tf.constant([
          [[0, 0.5], [1.0, 1.0]],
          [[0.5, -0.5], [1.5, 1.5]]
      ])
      output = keypoint_ops.change_coordinate_frame(keypoints, window)
      return output, expected_keypoints
    output, expected_keypoints = self.execute(graph_fn, [])
    self.assertAllClose(output, expected_keypoints)

  def test_keypoints_to_enclosing_bounding_boxes(self):
    def graph_fn():
      keypoints = tf.constant(
          [
              [  # Instance 0.
                  [5., 10.],
                  [3., 20.],
                  [8., 4.],
              ],
              [  # Instance 1.
                  [2., 12.],
                  [0., 3.],
                  [5., 19.],
              ],
          ], dtype=tf.float32)
      bboxes = keypoint_ops.keypoints_to_enclosing_bounding_boxes(keypoints)
      return bboxes
    output = self.execute(graph_fn, [])
    expected_bboxes = np.array(
        [
            [3., 4., 8., 20.],
            [0., 3., 5., 19.]
        ])
    self.assertAllClose(expected_bboxes, output)

  def test_keypoints_to_enclosing_bounding_boxes_axis2(self):
    def graph_fn():
      keypoints = tf.constant(
          [
              [  # Instance 0.
                  [5., 10.],
                  [3., 20.],
                  [8., 4.],
              ],
              [  # Instance 1.
                  [2., 12.],
                  [0., 3.],
                  [5., 19.],
              ],
          ], dtype=tf.float32)
      keypoints = tf.stack([keypoints, keypoints], axis=0)
      bboxes = keypoint_ops.keypoints_to_enclosing_bounding_boxes(
          keypoints, keypoints_axis=2)
      return bboxes
    output = self.execute(graph_fn, [])

    expected_bboxes = np.array(
        [
            [3., 4., 8., 20.],
            [0., 3., 5., 19.]
        ])
    self.assertAllClose(expected_bboxes, output[0])
    self.assertAllClose(expected_bboxes, output[1])

  def test_to_normalized_coordinates(self):
    def graph_fn():
      keypoints = tf.constant([
          [[10., 30.], [30., 45.]],
          [[20., 0.], [40., 60.]]
      ])
      output = keypoint_ops.to_normalized_coordinates(
          keypoints, 40, 60)
      expected_keypoints = tf.constant([
          [[0.25, 0.5], [0.75, 0.75]],
          [[0.5, 0.0], [1.0, 1.0]]
      ])
      return output, expected_keypoints
    output, expected_keypoints = self.execute(graph_fn, [])
    self.assertAllClose(output, expected_keypoints)

  def test_to_normalized_coordinates_already_normalized(self):
    if self.has_tpu(): return
    def graph_fn():
      keypoints = tf.constant([
          [[0.25, 0.5], [0.75, 0.75]],
          [[0.5, 0.0], [1.0, 1.0]]
      ])
      output = keypoint_ops.to_normalized_coordinates(
          keypoints, 40, 60)
      return output
    with self.assertRaisesOpError('assertion failed'):
      self.execute_cpu(graph_fn, [])

  def test_to_absolute_coordinates(self):
    def graph_fn():
      keypoints = tf.constant([
          [[0.25, 0.5], [0.75, 0.75]],
          [[0.5, 0.0], [1.0, 1.0]]
      ])
      output = keypoint_ops.to_absolute_coordinates(
          keypoints, 40, 60)
      expected_keypoints = tf.constant([
          [[10., 30.], [30., 45.]],
          [[20., 0.], [40., 60.]]
      ])
      return output, expected_keypoints
    output, expected_keypoints = self.execute(graph_fn, [])
    self.assertAllClose(output, expected_keypoints)

  def test_to_absolute_coordinates_already_absolute(self):
    if self.has_tpu(): return
    def graph_fn():
      keypoints = tf.constant([
          [[10., 30.], [30., 45.]],
          [[20., 0.], [40., 60.]]
      ])
      output = keypoint_ops.to_absolute_coordinates(
          keypoints, 40, 60)
      return output
    with self.assertRaisesOpError('assertion failed'):
      self.execute_cpu(graph_fn, [])

  def test_flip_horizontal(self):
    def graph_fn():
      keypoints = tf.constant([
          [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
          [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]]
      ])
      expected_keypoints = tf.constant([
          [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
          [[0.4, 0.6], [0.5, 0.5], [0.6, 0.4]],
      ])
      output = keypoint_ops.flip_horizontal(keypoints, 0.5)
      return output, expected_keypoints

    output, expected_keypoints = self.execute(graph_fn, [])
    self.assertAllClose(output, expected_keypoints)

  def test_flip_horizontal_permutation(self):

    def graph_fn():
      keypoints = tf.constant([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
                               [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]]])
      flip_permutation = [0, 2, 1]

      expected_keypoints = tf.constant([
          [[0.1, 0.9], [0.3, 0.7], [0.2, 0.8]],
          [[0.4, 0.6], [0.6, 0.4], [0.5, 0.5]],
      ])
      output = keypoint_ops.flip_horizontal(keypoints, 0.5, flip_permutation)
      return output, expected_keypoints
    output, expected_keypoints = self.execute(graph_fn, [])
    self.assertAllClose(output, expected_keypoints)

  def test_flip_vertical(self):
    def graph_fn():
      keypoints = tf.constant([
          [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
          [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]]
      ])

      expected_keypoints = tf.constant([
          [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
          [[0.6, 0.4], [0.5, 0.5], [0.4, 0.6]],
      ])
      output = keypoint_ops.flip_vertical(keypoints, 0.5)
      return output, expected_keypoints

    output, expected_keypoints = self.execute(graph_fn, [])
    self.assertAllClose(output, expected_keypoints)

  def test_flip_vertical_permutation(self):

    def graph_fn():
      keypoints = tf.constant([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
                               [[0.4, 0.4], [0.5, 0.5], [0.6, 0.6]]])
      flip_permutation = [0, 2, 1]

      expected_keypoints = tf.constant([
          [[0.9, 0.1], [0.7, 0.3], [0.8, 0.2]],
          [[0.6, 0.4], [0.4, 0.6], [0.5, 0.5]],
      ])
      output = keypoint_ops.flip_vertical(keypoints, 0.5, flip_permutation)
      return output, expected_keypoints
    output, expected_keypoints = self.execute(graph_fn, [])
    self.assertAllClose(output, expected_keypoints)

  def test_rot90(self):
    def graph_fn():
      keypoints = tf.constant([
          [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
          [[0.4, 0.6], [0.5, 0.6], [0.6, 0.7]]
      ])
      expected_keypoints = tf.constant([
          [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
          [[0.4, 0.4], [0.4, 0.5], [0.3, 0.6]],
      ])
      output = keypoint_ops.rot90(keypoints)
      return output, expected_keypoints
    output, expected_keypoints = self.execute(graph_fn, [])
    self.assertAllClose(output, expected_keypoints)

  def test_rot90_permutation(self):

    def graph_fn():
      keypoints = tf.constant([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
                               [[0.4, 0.6], [0.5, 0.6], [0.6, 0.7]]])
      rot_permutation = [0, 2, 1]
      expected_keypoints = tf.constant([
          [[0.9, 0.1], [0.7, 0.3], [0.8, 0.2]],
          [[0.4, 0.4], [0.3, 0.6], [0.4, 0.5]],
      ])
      output = keypoint_ops.rot90(keypoints,
                                  rotation_permutation=rot_permutation)
      return output, expected_keypoints

    output, expected_keypoints = self.execute(graph_fn, [])
    self.assertAllClose(output, expected_keypoints)


  def test_keypoint_weights_from_visibilities(self):
    def graph_fn():
      keypoint_visibilities = tf.constant([
          [True, True, False],
          [False, True, False]
      ])
      per_keypoint_weights = [1.0, 2.0, 3.0]
      keypoint_weights = keypoint_ops.keypoint_weights_from_visibilities(
          keypoint_visibilities, per_keypoint_weights)
      return keypoint_weights
    expected_keypoint_weights = [
        [1.0, 2.0, 0.0],
        [0.0, 2.0, 0.0]
    ]
    output = self.execute(graph_fn, [])
    self.assertAllClose(output, expected_keypoint_weights)

  def test_keypoint_weights_from_visibilities_no_per_kpt_weights(self):
    def graph_fn():
      keypoint_visibilities = tf.constant([
          [True, True, False],
          [False, True, False]
      ])
      keypoint_weights = keypoint_ops.keypoint_weights_from_visibilities(
          keypoint_visibilities)
      return keypoint_weights
    expected_keypoint_weights = [
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ]
    output = self.execute(graph_fn, [])
    self.assertAllClose(expected_keypoint_weights, output)

  def test_set_keypoint_visibilities_no_initial_kpt_vis(self):
    keypoints_np = np.array(
        [
            [[np.nan, 0.2],
             [np.nan, np.nan],
             [-3., 7.]],
            [[0.5, 0.2],
             [4., 1.0],
             [-3., np.nan]],
        ], dtype=np.float32)
    def graph_fn():
      keypoints = tf.constant(keypoints_np, dtype=tf.float32)
      keypoint_visibilities = keypoint_ops.set_keypoint_visibilities(
          keypoints)
      return keypoint_visibilities

    expected_kpt_vis = [
        [False, False, True],
        [True, True, False]
    ]
    output = self.execute(graph_fn, [])
    self.assertAllEqual(expected_kpt_vis, output)

  def test_set_keypoint_visibilities(self):
    keypoints_np = np.array(
        [
            [[np.nan, 0.2],
             [np.nan, np.nan],
             [-3., 7.]],
            [[0.5, 0.2],
             [4., 1.0],
             [-3., np.nan]],
        ], dtype=np.float32)
    initial_keypoint_visibilities_np = np.array(
        [
            [False,
             True,  # Will be overriden by NaN coords.
             False],  # Will be maintained, even though non-NaN coords.
            [True,
             False,  # Will be maintained, even though non-NaN coords.
             False]
        ])
    def graph_fn():
      keypoints = tf.constant(keypoints_np, dtype=tf.float32)
      initial_keypoint_visibilities = tf.constant(
          initial_keypoint_visibilities_np, dtype=tf.bool)
      keypoint_visibilities = keypoint_ops.set_keypoint_visibilities(
          keypoints, initial_keypoint_visibilities)
      return keypoint_visibilities

    expected_kpt_vis = [
        [False, False, False],
        [True, False, False]
    ]
    output = self.execute(graph_fn, [])
    self.assertAllEqual(expected_kpt_vis, output)

if __name__ == '__main__':
  tf.test.main()
