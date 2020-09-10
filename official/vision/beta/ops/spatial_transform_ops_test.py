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
"""Tests for spatial_transform_ops.py."""

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.vision.beta.ops import spatial_transform_ops


class MultiLevelCropAndResizeTest(tf.test.TestCase):

  def test_multilevel_crop_and_resize_square(self):
    """Example test case.

    Input =
    [
      [0, 1, 2, 3],
      [4, 5, 6, 7],
      [8, 9, 10, 11],
      [12, 13, 14, 15],
    ]
    output_size = 2x2
    box =
    [
      [[0, 0, 2, 2]]
    ]
    Gathered data =
    [
      [0, 1, 1, 2],
      [4, 5, 5, 6],
      [4, 5, 5, 6],
      [8, 9, 9, 10],
    ]
    Interpolation kernel =
    [
      [1, 1, 1, 1],
      [1, 1, 1, 1],
      [1, 1, 1, 1],
      [1, 1, 1, 1],
    ]
    Output =
    [
      [2.5, 3.5],
      [6.5, 7.5]
    ]
    """
    input_size = 4
    min_level = 0
    max_level = 0
    batch_size = 1
    output_size = 2
    num_filters = 1
    features = {}
    for level in range(min_level, max_level + 1):
      feat_size = int(input_size / 2**level)

      features[level] = tf.range(
          batch_size * feat_size * feat_size * num_filters, dtype=tf.float32)
      features[level] = tf.reshape(
          features[level], [batch_size, feat_size, feat_size, num_filters])
    boxes = tf.constant([
        [[0, 0, 2, 2]],
    ], dtype=tf.float32)
    tf_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
        features, boxes, output_size)
    roi_features = tf_roi_features.numpy()
    self.assertAllClose(
        roi_features,
        np.array([[2.5, 3.5],
                  [6.5,
                   7.5]]).reshape([batch_size, 1, output_size, output_size, 1]))

  def test_multilevel_crop_and_resize_rectangle(self):
    """Example test case.

    Input =
    [
      [0, 1, 2, 3],
      [4, 5, 6, 7],
      [8, 9, 10, 11],
      [12, 13, 14, 15],
    ]
    output_size = 2x2
    box =
    [
      [[0, 0, 2, 3]]
    ]
    Box vertices =
    [
      [[0.5, 0.75], [0.5, 2.25]],
      [[1.5, 0.75], [1.5, 2.25]],
    ]
    Gathered data =
    [
      [0, 1, 2, 3],
      [4, 5, 6, 7],
      [4, 5, 6, 7],
      [8, 9, 10, 11],
    ]
    Interpolation kernel =
    [
      [0.5 1.5 1.5 0.5],
      [0.5 1.5 1.5 0.5],
      [0.5 1.5 1.5 0.5],
      [0.5 1.5 1.5 0.5],
    ]
    Output =
    [
      [2.75, 4.25],
      [6.75, 8.25]
    ]
    """
    input_size = 4
    min_level = 0
    max_level = 0
    batch_size = 1
    output_size = 2
    num_filters = 1
    features = {}
    for level in range(min_level, max_level + 1):
      feat_size = int(input_size / 2**level)

      features[level] = tf.range(
          batch_size * feat_size * feat_size * num_filters, dtype=tf.float32)
      features[level] = tf.reshape(
          features[level], [batch_size, feat_size, feat_size, num_filters])
    boxes = tf.constant([
        [[0, 0, 2, 3]],
    ], dtype=tf.float32)
    tf_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
        features, boxes, output_size)
    roi_features = tf_roi_features.numpy()
    self.assertAllClose(
        roi_features,
        np.array([[2.75, 4.25],
                  [6.75,
                   8.25]]).reshape([batch_size, 1, output_size, output_size,
                                    1]))

  def test_multilevel_crop_and_resize_two_boxes(self):
    """Test two boxes."""
    input_size = 4
    min_level = 0
    max_level = 0
    batch_size = 1
    output_size = 2
    num_filters = 1
    features = {}
    for level in range(min_level, max_level + 1):
      feat_size = int(input_size / 2**level)

      features[level] = tf.range(
          batch_size * feat_size * feat_size * num_filters, dtype=tf.float32)
      features[level] = tf.reshape(
          features[level], [batch_size, feat_size, feat_size, num_filters])
    boxes = tf.constant([
        [[0, 0, 2, 2], [0, 0, 2, 3]],
    ], dtype=tf.float32)
    tf_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
        features, boxes, output_size)
    roi_features = tf_roi_features.numpy()
    self.assertAllClose(
        roi_features,
        np.array([[[2.5, 3.5], [6.5, 7.5]], [[2.75, 4.25], [6.75, 8.25]]
                 ]).reshape([batch_size, 2, output_size, output_size, 1]))

  def test_multilevel_crop_and_resize_feature_level_assignment(self):
    """Test feature level assignment."""
    input_size = 640
    min_level = 2
    max_level = 5
    batch_size = 1
    output_size = 2
    num_filters = 1
    features = {}
    for level in range(min_level, max_level + 1):
      feat_size = int(input_size / 2**level)

      features[level] = float(level) * tf.ones(
          [batch_size, feat_size, feat_size, num_filters], dtype=tf.float32)
    boxes = tf.constant(
        [
            [
                [0, 0, 111, 111],  # Level 2.
                [0, 0, 113, 113],  # Level 3.
                [0, 0, 223, 223],  # Level 3.
                [0, 0, 225, 225],  # Level 4.
                [0, 0, 449, 449]
            ],  # Level 5.
        ],
        dtype=tf.float32)
    tf_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
        features, boxes, output_size)
    roi_features = tf_roi_features.numpy()
    self.assertAllClose(roi_features[0, 0], 2 * np.ones((2, 2, 1)))
    self.assertAllClose(roi_features[0, 1], 3 * np.ones((2, 2, 1)))
    self.assertAllClose(roi_features[0, 2], 3 * np.ones((2, 2, 1)))
    self.assertAllClose(roi_features[0, 3], 4 * np.ones((2, 2, 1)))
    self.assertAllClose(roi_features[0, 4], 5 * np.ones((2, 2, 1)))

  def test_multilevel_crop_and_resize_large_input(self):
    """Test 512 boxes on TPU."""
    input_size = 1408
    min_level = 2
    max_level = 6
    batch_size = 2
    num_boxes = 512
    num_filters = 256
    output_size = 7
    strategy = tf.distribute.experimental.TPUStrategy()
    with strategy.scope():
      features = {}
      for level in range(min_level, max_level + 1):
        feat_size = int(input_size / 2**level)
        features[level] = tf.constant(
            np.reshape(
                np.arange(
                    batch_size * feat_size * feat_size * num_filters,
                    dtype=np.float32),
                [batch_size, feat_size, feat_size, num_filters]),
            dtype=tf.bfloat16)
      boxes = np.array([
          [[0, 0, 256, 256]]*num_boxes,
      ], dtype=np.float32)
      boxes = np.tile(boxes, [batch_size, 1, 1])
      tf_boxes = tf.constant(boxes, dtype=tf.float32)

      tf_roi_features = spatial_transform_ops.multilevel_crop_and_resize(
          features, tf_boxes)
      roi_features = tf_roi_features.numpy()
      self.assertEqual(
          roi_features.shape,
          (batch_size, num_boxes, output_size, output_size, num_filters))


class CropMaskInTargetBoxTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (False),
      (True),
  )
  def test_crop_mask_in_target_box(self, use_einsum):
    batch_size = 1
    num_masks = 2
    height = 2
    width = 2
    output_size = 2
    strategy = tf.distribute.experimental.TPUStrategy()
    with strategy.scope():
      masks = tf.ones([batch_size, num_masks, height, width])
      boxes = tf.constant(
          [[0., 0., 1., 1.],
           [0., 0., 1., 1.]])
      target_boxes = tf.constant(
          [[0., 0., 1., 1.],
           [-1., -1., 1., 1.]])
      expected_outputs = np.array([
          [[[1., 1.],
            [1., 1.]],
           [[0., 0.],
            [0., 1.]]]])
      boxes = tf.reshape(boxes, [batch_size, num_masks, 4])
      target_boxes = tf.reshape(target_boxes, [batch_size, num_masks, 4])

      tf_cropped_masks = spatial_transform_ops.crop_mask_in_target_box(
          masks, boxes, target_boxes, output_size, use_einsum=use_einsum)
      cropped_masks = tf_cropped_masks.numpy()
      self.assertAllEqual(cropped_masks, expected_outputs)


if __name__ == '__main__':
  tf.test.main()
