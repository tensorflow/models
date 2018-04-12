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

"""Tests for object_detection.grid_anchor_generator."""
import numpy as np
import tensorflow as tf

from object_detection.anchor_generators import grid_anchor_generator
from object_detection.utils import test_case


class GridAnchorGeneratorTest(test_case.TestCase):

  def test_construct_single_anchor(self):
    """Builds a 1x1 anchor grid to test the size of the output boxes."""
    def graph_fn():
      scales = [0.5, 1.0, 2.0]
      aspect_ratios = [0.25, 1.0, 4.0]
      anchor_offset = [7, -3]
      anchor_generator = grid_anchor_generator.GridAnchorGenerator(
          scales, aspect_ratios, anchor_offset=anchor_offset)
      anchors_list = anchor_generator.generate(feature_map_shape_list=[(1, 1)])
      anchor_corners = anchors_list[0].get()
      return (anchor_corners,)
    exp_anchor_corners = [[-121, -35, 135, 29], [-249, -67, 263, 61],
                          [-505, -131, 519, 125], [-57, -67, 71, 61],
                          [-121, -131, 135, 125], [-249, -259, 263, 253],
                          [-25, -131, 39, 125], [-57, -259, 71, 253],
                          [-121, -515, 135, 509]]
    anchor_corners_out = self.execute(graph_fn, [])
    self.assertAllClose(anchor_corners_out, exp_anchor_corners)

  def test_construct_anchor_grid(self):
    def graph_fn():
      base_anchor_size = [10, 10]
      anchor_stride = [19, 19]
      anchor_offset = [0, 0]
      scales = [0.5, 1.0, 2.0]
      aspect_ratios = [1.0]

      anchor_generator = grid_anchor_generator.GridAnchorGenerator(
          scales,
          aspect_ratios,
          base_anchor_size=base_anchor_size,
          anchor_stride=anchor_stride,
          anchor_offset=anchor_offset)

      anchors_list = anchor_generator.generate(feature_map_shape_list=[(2, 2)])
      anchor_corners = anchors_list[0].get()
      return (anchor_corners,)
    exp_anchor_corners = [[-2.5, -2.5, 2.5, 2.5], [-5., -5., 5., 5.],
                          [-10., -10., 10., 10.], [-2.5, 16.5, 2.5, 21.5],
                          [-5., 14., 5, 24], [-10., 9., 10, 29],
                          [16.5, -2.5, 21.5, 2.5], [14., -5., 24, 5],
                          [9., -10., 29, 10], [16.5, 16.5, 21.5, 21.5],
                          [14., 14., 24, 24], [9., 9., 29, 29]]
    anchor_corners_out = self.execute(graph_fn, [])
    self.assertAllClose(anchor_corners_out, exp_anchor_corners)

  def test_construct_anchor_grid_with_dynamic_feature_map_shapes(self):
    def graph_fn(feature_map_height, feature_map_width):
      base_anchor_size = [10, 10]
      anchor_stride = [19, 19]
      anchor_offset = [0, 0]
      scales = [0.5, 1.0, 2.0]
      aspect_ratios = [1.0]
      anchor_generator = grid_anchor_generator.GridAnchorGenerator(
          scales,
          aspect_ratios,
          base_anchor_size=base_anchor_size,
          anchor_stride=anchor_stride,
          anchor_offset=anchor_offset)

      anchors_list = anchor_generator.generate(
          feature_map_shape_list=[(feature_map_height, feature_map_width)])
      anchor_corners = anchors_list[0].get()
      return (anchor_corners,)

    exp_anchor_corners = [[-2.5, -2.5, 2.5, 2.5], [-5., -5., 5., 5.],
                          [-10., -10., 10., 10.], [-2.5, 16.5, 2.5, 21.5],
                          [-5., 14., 5, 24], [-10., 9., 10, 29],
                          [16.5, -2.5, 21.5, 2.5], [14., -5., 24, 5],
                          [9., -10., 29, 10], [16.5, 16.5, 21.5, 21.5],
                          [14., 14., 24, 24], [9., 9., 29, 29]]
    anchor_corners_out = self.execute_cpu(graph_fn,
                                          [np.array(2, dtype=np.int32),
                                           np.array(2, dtype=np.int32)])
    self.assertAllClose(anchor_corners_out, exp_anchor_corners)


if __name__ == '__main__':
  tf.test.main()
