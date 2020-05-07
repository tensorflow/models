# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for object_detection.utils.spatial_transform_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
import tensorflow as tf

from object_detection.utils import spatial_transform_ops as spatial_ops
from object_detection.utils import test_case


class BoxGridCoordinateTest(test_case.TestCase):

  def test_4x4_grid(self):
    boxes = np.array([[[0., 0., 6., 6.]]], dtype=np.float32)
    def graph_fn(boxes):
      return spatial_ops.box_grid_coordinate_vectors(boxes, size_y=4, size_x=4)

    grid_y, grid_x = self.execute(graph_fn, [boxes])
    expected_grid_y = np.array([[[0.75, 2.25, 3.75, 5.25]]])
    expected_grid_x = np.array([[[0.75, 2.25, 3.75, 5.25]]])
    self.assertAllClose(expected_grid_y, grid_y)
    self.assertAllClose(expected_grid_x, grid_x)

  def test_2x2_grid(self):
    def graph_fn(boxes):
      return spatial_ops.box_grid_coordinate_vectors(boxes, size_x=2, size_y=2)
    boxes = np.array([[[0., 0., 6., 3.],
                       [0., 0., 3., 6.]]], dtype=np.float32)

    grid_y, grid_x = self.execute(graph_fn, [boxes])
    expected_grid_y = np.array([[[1.5, 4.5],
                                 [0.75, 2.25]]])
    expected_grid_x = np.array([[[0.75, 2.25],
                                 [1.5, 4.5]]])
    self.assertAllClose(expected_grid_y, grid_y)
    self.assertAllClose(expected_grid_x, grid_x)

  def test_2x4_grid(self):
    boxes = np.array([[[0., 0., 6., 6.]]], dtype=np.float32)
    def graph_fn(boxes):
      return spatial_ops.box_grid_coordinate_vectors(boxes, size_y=2, size_x=4)

    grid_y, grid_x = self.execute(graph_fn, [boxes])
    expected_grid_y = np.array([[[1.5, 4.5]]])
    expected_grid_x = np.array([[[0.75, 2.25, 3.75, 5.25]]])
    self.assertAllClose(expected_grid_y, grid_y)
    self.assertAllClose(expected_grid_x, grid_x)

  def test_2x4_grid_with_aligned_corner(self):
    boxes = np.array([[[0., 0., 6., 6.]]], dtype=np.float32)
    def graph_fn(boxes):
      return spatial_ops.box_grid_coordinate_vectors(boxes, size_y=2, size_x=4,
                                                     align_corners=True)

    grid_y, grid_x = self.execute(graph_fn, [boxes])
    expected_grid_y = np.array([[[0, 6]]])
    expected_grid_x = np.array([[[0, 2, 4, 6]]])
    self.assertAllClose(expected_grid_y, grid_y)
    self.assertAllClose(expected_grid_x, grid_x)

  def test_offgrid_boxes(self):
    boxes = np.array([[[1.2, 2.3, 7.2, 8.3]]], dtype=np.float32)
    def graph_fn(boxes):
      return spatial_ops.box_grid_coordinate_vectors(boxes, size_y=4, size_x=4)

    grid_y, grid_x = self.execute(graph_fn, [boxes])
    expected_grid_y = np.array([[[0.75, 2.25, 3.75, 5.25]]]) + 1.2
    expected_grid_x = np.array([[[0.75, 2.25, 3.75, 5.25]]]) + 2.3
    self.assertAllClose(expected_grid_y, grid_y)
    self.assertAllClose(expected_grid_x, grid_x)


class FeatureGridCoordinateTest(test_case.TestCase):

  def test_snap_box_points_to_nearest_4_pixels(self):
    box_grid_y = np.array([[[1.5, 4.6]]], dtype=np.float32)
    box_grid_x = np.array([[[2.4, 5.3]]], dtype=np.float32)

    def graph_fn(box_grid_y, box_grid_x):
      return spatial_ops.feature_grid_coordinate_vectors(box_grid_y, box_grid_x)

    (feature_grid_y0,
     feature_grid_x0, feature_grid_y1, feature_grid_x1) = self.execute(
         graph_fn, [box_grid_y, box_grid_x])
    expected_grid_y0 = np.array([[[1, 4]]])
    expected_grid_y1 = np.array([[[2, 5]]])
    expected_grid_x0 = np.array([[[2, 5]]])
    expected_grid_x1 = np.array([[[3, 6]]])
    self.assertAllEqual(expected_grid_y0, feature_grid_y0)
    self.assertAllEqual(expected_grid_y1, feature_grid_y1)
    self.assertAllEqual(expected_grid_x0, feature_grid_x0)
    self.assertAllEqual(expected_grid_x1, feature_grid_x1)

  def test_snap_box_points_outside_pixel_grid_to_nearest_neighbor(self):
    box_grid_y = np.array([[[0.33, 1., 1.66]]], dtype=np.float32)
    box_grid_x = np.array([[[-0.5, 1., 1.66]]], dtype=np.float32)

    def graph_fn(box_grid_y, box_grid_x):
      return spatial_ops.feature_grid_coordinate_vectors(box_grid_y, box_grid_x)

    (feature_grid_y0,
     feature_grid_x0, feature_grid_y1, feature_grid_x1) = self.execute(
         graph_fn, [box_grid_y, box_grid_x])
    expected_grid_y0 = np.array([[[0, 1, 1]]])
    expected_grid_y1 = np.array([[[1, 2, 2]]])
    expected_grid_x0 = np.array([[[-1, 1, 1]]])
    expected_grid_x1 = np.array([[[0, 2, 2]]])
    self.assertAllEqual(expected_grid_y0, feature_grid_y0)
    self.assertAllEqual(expected_grid_y1, feature_grid_y1)
    self.assertAllEqual(expected_grid_x0, feature_grid_x0)
    self.assertAllEqual(expected_grid_x1, feature_grid_x1)


class RavelIndicesTest(test_case.TestCase):

  def test_feature_point_indices(self):
    feature_grid_y = np.array([[[1, 2, 4, 5],
                                [2, 3, 4, 5]]], dtype=np.int32)
    feature_grid_x = np.array([[[1, 3, 4],
                                [2, 3, 4]]], dtype=np.int32)
    num_feature_levels = 2
    feature_height = 6
    feature_width = 5
    box_levels = np.array([[0, 1]], dtype=np.int32)

    def graph_fn(feature_grid_y, feature_grid_x, box_levels):
      return spatial_ops.ravel_indices(feature_grid_y, feature_grid_x,
                                       num_feature_levels, feature_height,
                                       feature_width, box_levels)

    indices = self.execute(graph_fn,
                           [feature_grid_y, feature_grid_x, box_levels])
    expected_indices = np.array([[[[6, 8, 9],
                                   [11, 13, 14],
                                   [21, 23, 24],
                                   [26, 28, 29]],
                                  [[42, 43, 44],
                                   [47, 48, 49],
                                   [52, 53, 54],
                                   [57, 58, 59]]]])
    self.assertAllEqual(expected_indices.flatten(), indices)


class MultiLevelRoIAlignTest(test_case.TestCase):

  def test_perfectly_aligned_cell_center_and_feature_pixels(self):

    def graph_fn(image, boxes, levels):
      return spatial_ops.multilevel_roi_align([image],
                                              boxes,
                                              levels,
                                              output_size=[2, 2])

    image = np.arange(25).reshape(1, 5, 5, 1).astype(np.float32)
    boxes = np.array([[[0, 0, 1.0, 1.0]]], dtype=np.float32)
    box_levels = np.array([[0]], dtype=np.int32)

    expected_output = [[[[[6], [8]],
                         [[16], [18]]]]]
    crop_output = self.execute(graph_fn, [image, boxes, box_levels])
    self.assertAllClose(crop_output, expected_output)

  def test_interpolation_with_4_points_per_bin(self):

    def graph_fn(image, boxes, levels):
      return spatial_ops.multilevel_roi_align([image],
                                              boxes,
                                              levels,
                                              output_size=[1, 1],
                                              num_samples_per_cell_y=2,
                                              num_samples_per_cell_x=2)

    image = np.array([[[[1], [2], [3], [4]],
                       [[5], [6], [7], [8]],
                       [[9], [10], [11], [12]],
                       [[13], [14], [15], [16]]]],
                     dtype=np.float32)
    boxes = np.array([[[1./3, 1./3, 2./3, 2./3]]], dtype=np.float32)
    box_levels = np.array([[0]], dtype=np.int32)

    expected_output = [[[[[(7.25 + 7.75 + 9.25 + 9.75) / 4]]]]]
    crop_output = self.execute(graph_fn, [image, boxes, box_levels])
    self.assertAllClose(expected_output, crop_output)

  def test_1x1_crop_on_2x2_features(self):

    def graph_fn(image, boxes, levels):
      return spatial_ops.multilevel_roi_align([image],
                                              boxes,
                                              levels,
                                              output_size=[1, 1])

    image = np.array([[[[1], [2]],
                       [[3], [4]]]], dtype=np.float32)
    boxes = np.array([[[0, 0, 1, 1]]], dtype=np.float32)
    box_levels = np.array([[0]], dtype=np.int32)
    expected_output = [[[[[2.5]]]]]
    crop_output = self.execute(graph_fn, [image, boxes, box_levels])
    self.assertAllClose(crop_output, expected_output)

  def test_3x3_crops_on_2x2_features(self):

    def graph_fn(image, boxes, levels):
      return spatial_ops.multilevel_roi_align([image],
                                              boxes,
                                              levels,
                                              output_size=[3, 3])

    image = np.array([[[[1], [2]],
                       [[3], [4]]]], dtype=np.float32)
    boxes = np.array([[[0, 0, 1, 1]]], dtype=np.float32)
    box_levels = np.array([[0]], dtype=np.int32)
    expected_output = [[[[[9./6], [11./6], [13./6]],
                         [[13./6], [15./6], [17./6]],
                         [[17./6], [19./6], [21./6]]]]]
    crop_output = self.execute(graph_fn, [image, boxes, box_levels])
    self.assertAllClose(crop_output, expected_output)

  def test_2x2_crops_on_3x3_features(self):

    def graph_fn(image, boxes, levels):
      return spatial_ops.multilevel_roi_align([image],
                                              boxes,
                                              levels,
                                              output_size=[2, 2])

    image = np.array([[[[1], [2], [3]],
                       [[4], [5], [6]],
                       [[7], [8], [9]]]],
                     dtype=np.float32)
    boxes = np.array([[[0, 0, 1, 1],
                       [0, 0, .5, .5]]],
                     dtype=np.float32)
    box_levels = np.array([[0, 0]], dtype=np.int32)
    expected_output = [[[[[3], [4]],
                         [[6], [7]]],
                        [[[2.], [2.5]],
                         [[3.5], [4.]]]]]
    crop_output = self.execute(graph_fn, [image, boxes, box_levels])
    self.assertAllClose(crop_output, expected_output)

  def test_2x2_crop_on_4x4_features(self):

    def graph_fn(image, boxes, levels):
      return spatial_ops.multilevel_roi_align([image],
                                              boxes,
                                              levels,
                                              output_size=[2, 2])

    image = np.array([[[[0], [1], [2], [3]],
                       [[4], [5], [6], [7]],
                       [[8], [9], [10], [11]],
                       [[12], [13], [14], [15]]]],
                     dtype=np.float32)
    boxes = np.array([[[0, 0, 2./3, 2./3],
                       [0, 0, 2./3, 1.0]]],
                     dtype=np.float32)
    box_levels = np.array([[0, 0]], dtype=np.int32)

    expected_output = np.array([[[[[2.5], [3.5]],
                                  [[6.5], [7.5]]],
                                 [[[2.75], [4.25]],
                                  [[6.75], [8.25]]]]])
    crop_output = self.execute(graph_fn, [image, boxes, box_levels])
    self.assertAllClose(expected_output, crop_output)

  def test_extrapolate_3x3_crop_on_2x2_features(self):
    def graph_fn(image, boxes, levels):
      return spatial_ops.multilevel_roi_align([image],
                                              boxes,
                                              levels,
                                              output_size=[3, 3])
    image = np.array([[[[1], [2]],
                       [[3], [4]]]], dtype=np.float32)
    boxes = np.array([[[-1, -1, 2, 2]]], dtype=np.float32)
    box_levels = np.array([[0]], dtype=np.int32)

    expected_output = np.array([[[[[0.25], [0.75], [0.5]],
                                  [[1.0], [2.5], [1.5]],
                                  [[0.75], [1.75], [1]]]]])
    crop_output = self.execute(graph_fn, [image, boxes, box_levels])
    self.assertAllClose(expected_output, crop_output)

  def test_extrapolate_with_non_zero_value(self):
    def graph_fn(image, boxes, levels):
      return spatial_ops.multilevel_roi_align([image],
                                              boxes,
                                              levels,
                                              output_size=[3, 3],
                                              extrapolation_value=2.0)
    image = np.array([[[[4], [4]],
                       [[4], [4]]]], dtype=np.float32)
    boxes = np.array([[[-1, -1, 2, 2]]], dtype=np.float32)
    box_levels = np.array([[0]], dtype=np.int32)

    expected_output = np.array([[[[[2.5], [3.0], [2.5]],
                                  [[3.0], [4.0], [3.0]],
                                  [[2.5], [3.0], [2.5]]]]])
    crop_output = self.execute(graph_fn, [image, boxes, box_levels])
    self.assertAllClose(expected_output, crop_output)

  def test_multilevel_roi_align(self):
    image_size = 640
    fpn_min_level = 2
    fpn_max_level = 5
    batch_size = 1
    output_size = [2, 2]
    num_filters = 1
    features = []
    for level in range(fpn_min_level, fpn_max_level + 1):
      feat_size = int(image_size / 2**level)
      features.append(
          float(level) *
          np.ones([batch_size, feat_size, feat_size, num_filters],
                  dtype=np.float32))
    boxes = np.array(
        [
            [
                [0, 0, 111, 111],  # Level 2.
                [0, 0, 113, 113],  # Level 3.
                [0, 0, 223, 223],  # Level 3.
                [0, 0, 225, 225],  # Level 4.
                [0, 0, 449, 449]   # Level 5.
            ],
        ],
        dtype=np.float32) / image_size
    levels = np.array([[0, 1, 1, 2, 3]], dtype=np.int32)

    def graph_fn(feature1, feature2, feature3, feature4, boxes, levels):
      roi_features = spatial_ops.multilevel_roi_align(
          [feature1, feature2, feature3, feature4],
          boxes,
          levels,
          output_size)
      return roi_features

    roi_features = self.execute(graph_fn, features + [boxes, levels])
    self.assertAllClose(roi_features[0][0], 2 * np.ones((2, 2, 1)))
    self.assertAllClose(roi_features[0][1], 3 * np.ones((2, 2, 1)))
    self.assertAllClose(roi_features[0][2], 3 * np.ones((2, 2, 1)))
    self.assertAllClose(roi_features[0][3], 4 * np.ones((2, 2, 1)))
    self.assertAllClose(roi_features[0][4], 5 * np.ones((2, 2, 1)))

  def test_large_input(self):
    if self.has_tpu():
      input_size = 1408
      min_level = 2
      max_level = 6
      batch_size = 2
      num_boxes = 512
      num_filters = 256
      output_size = [7, 7]
      features = []
      for level in range(min_level, max_level + 1):
        feat_size = int(input_size / 2**level)
        features.append(
            np.reshape(
                np.arange(
                    batch_size * feat_size * feat_size * num_filters,
                    dtype=np.float32),
                [batch_size, feat_size, feat_size, num_filters]))
      boxes = np.array([
          [[0, 0, 256, 256]]*num_boxes,
      ], dtype=np.float32) / input_size
      boxes = np.tile(boxes, [batch_size, 1, 1])
      levels = np.random.randint(5, size=[batch_size, num_boxes],
                                 dtype=np.int32)
      def crop_and_resize_fn():
        tf_features = [
            tf.constant(feature, dtype=tf.bfloat16) for feature in features
        ]
        return spatial_ops.multilevel_roi_align(
            tf_features, tf.constant(boxes), tf.constant(levels), output_size)
      roi_features = self.execute_tpu(crop_and_resize_fn, [])
      self.assertEqual(roi_features.shape,
                       (batch_size, num_boxes, output_size[0],
                        output_size[1], num_filters))


class MatMulCropAndResizeTest(test_case.TestCase):

  def testMatMulCropAndResize2x2To1x1(self):

    def graph_fn(image, boxes):
      return spatial_ops.matmul_crop_and_resize(image, boxes, crop_size=[1, 1])

    image = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.float32)
    boxes = np.array([[[0, 0, 1, 1]]], dtype=np.float32)
    expected_output = [[[[[2.5]]]]]
    crop_output = self.execute(graph_fn, [image, boxes])
    self.assertAllClose(crop_output, expected_output)

  def testMatMulCropAndResize2x2To1x1Flipped(self):

    def graph_fn(image, boxes):
      return spatial_ops.matmul_crop_and_resize(image, boxes, crop_size=[1, 1])

    image = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.float32)
    boxes = np.array([[[1, 1, 0, 0]]], dtype=np.float32)
    expected_output = [[[[[2.5]]]]]
    crop_output = self.execute(graph_fn, [image, boxes])
    self.assertAllClose(crop_output, expected_output)

  def testMatMulCropAndResize2x2To3x3(self):

    def graph_fn(image, boxes):
      return spatial_ops.matmul_crop_and_resize(image, boxes, crop_size=[3, 3])

    image = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.float32)
    boxes = np.array([[[0, 0, 1, 1]]], dtype=np.float32)
    expected_output = [[[[[1.0], [1.5], [2.0]],
                         [[2.0], [2.5], [3.0]],
                         [[3.0], [3.5], [4.0]]]]]
    crop_output = self.execute(graph_fn, [image, boxes])
    self.assertAllClose(crop_output, expected_output)

  def testMatMulCropAndResize2x2To3x3Flipped(self):

    def graph_fn(image, boxes):
      return spatial_ops.matmul_crop_and_resize(image, boxes, crop_size=[3, 3])

    image = np.array([[[[1], [2]], [[3], [4]]]], dtype=np.float32)
    boxes = np.array([[[1, 1, 0, 0]]], dtype=np.float32)
    expected_output = [[[[[4.0], [3.5], [3.0]],
                         [[3.0], [2.5], [2.0]],
                         [[2.0], [1.5], [1.0]]]]]
    crop_output = self.execute(graph_fn, [image, boxes])
    self.assertAllClose(crop_output, expected_output)

  def testMatMulCropAndResize3x3To2x2(self):

    def graph_fn(image, boxes):
      return spatial_ops.matmul_crop_and_resize(image, boxes, crop_size=[2, 2])

    image = np.array([[[[1], [2], [3]],
                       [[4], [5], [6]],
                       [[7], [8], [9]]]], dtype=np.float32)
    boxes = np.array([[[0, 0, 1, 1],
                       [0, 0, .5, .5]]], dtype=np.float32)
    expected_output = [[[[[1], [3]], [[7], [9]]],
                        [[[1], [2]], [[4], [5]]]]]
    crop_output = self.execute(graph_fn, [image, boxes])
    self.assertAllClose(crop_output, expected_output)

  def testMatMulCropAndResize3x3To2x2_2Channels(self):

    def graph_fn(image, boxes):
      return spatial_ops.matmul_crop_and_resize(image, boxes, crop_size=[2, 2])

    image = np.array([[[[1, 0], [2, 1], [3, 2]],
                       [[4, 3], [5, 4], [6, 5]],
                       [[7, 6], [8, 7], [9, 8]]]], dtype=np.float32)
    boxes = np.array([[[0, 0, 1, 1],
                       [0, 0, .5, .5]]], dtype=np.float32)
    expected_output = [[[[[1, 0], [3, 2]], [[7, 6], [9, 8]]],
                        [[[1, 0], [2, 1]], [[4, 3], [5, 4]]]]]
    crop_output = self.execute(graph_fn, [image, boxes])
    self.assertAllClose(crop_output, expected_output)

  def testBatchMatMulCropAndResize3x3To2x2_2Channels(self):

    def graph_fn(image, boxes):
      return spatial_ops.matmul_crop_and_resize(image, boxes, crop_size=[2, 2])

    image = np.array([[[[1, 0], [2, 1], [3, 2]],
                       [[4, 3], [5, 4], [6, 5]],
                       [[7, 6], [8, 7], [9, 8]]],
                      [[[1, 0], [2, 1], [3, 2]],
                       [[4, 3], [5, 4], [6, 5]],
                       [[7, 6], [8, 7], [9, 8]]]], dtype=np.float32)
    boxes = np.array([[[0, 0, 1, 1],
                       [0, 0, .5, .5]],
                      [[1, 1, 0, 0],
                       [.5, .5, 0, 0]]], dtype=np.float32)
    expected_output = [[[[[1, 0], [3, 2]], [[7, 6], [9, 8]]],
                        [[[1, 0], [2, 1]], [[4, 3], [5, 4]]]],
                       [[[[9, 8], [7, 6]], [[3, 2], [1, 0]]],
                        [[[5, 4], [4, 3]], [[2, 1], [1, 0]]]]]
    crop_output = self.execute(graph_fn, [image, boxes])
    self.assertAllClose(crop_output, expected_output)

  def testMatMulCropAndResize3x3To2x2Flipped(self):

    def graph_fn(image, boxes):
      return spatial_ops.matmul_crop_and_resize(image, boxes, crop_size=[2, 2])

    image = np.array([[[[1], [2], [3]],
                       [[4], [5], [6]],
                       [[7], [8], [9]]]], dtype=np.float32)
    boxes = np.array([[[1, 1, 0, 0],
                       [.5, .5, 0, 0]]], dtype=np.float32)
    expected_output = [[[[[9], [7]], [[3], [1]]],
                        [[[5], [4]], [[2], [1]]]]]
    crop_output = self.execute(graph_fn, [image, boxes])
    self.assertAllClose(crop_output, expected_output)


class NativeCropAndResizeTest(test_case.TestCase):

  def testBatchCropAndResize3x3To2x2_2Channels(self):

    def graph_fn(image, boxes):
      return spatial_ops.native_crop_and_resize(image, boxes, crop_size=[2, 2])

    image = np.array([[[[1, 0], [2, 1], [3, 2]],
                       [[4, 3], [5, 4], [6, 5]],
                       [[7, 6], [8, 7], [9, 8]]],
                      [[[1, 0], [2, 1], [3, 2]],
                       [[4, 3], [5, 4], [6, 5]],
                       [[7, 6], [8, 7], [9, 8]]]], dtype=np.float32)
    boxes = np.array([[[0, 0, 1, 1],
                       [0, 0, .5, .5]],
                      [[1, 1, 0, 0],
                       [.5, .5, 0, 0]]], dtype=np.float32)
    expected_output = [[[[[1, 0], [3, 2]], [[7, 6], [9, 8]]],
                        [[[1, 0], [2, 1]], [[4, 3], [5, 4]]]],
                       [[[[9, 8], [7, 6]], [[3, 2], [1, 0]]],
                        [[[5, 4], [4, 3]], [[2, 1], [1, 0]]]]]
    crop_output = self.execute_cpu(graph_fn, [image, boxes])
    self.assertAllClose(crop_output, expected_output)


if __name__ == '__main__':
  tf.test.main()
