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

"""Tests for anchor_generators.flexible_grid_anchor_generator_test.py."""
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.anchor_generators import flexible_grid_anchor_generator as fg
from object_detection.utils import test_case


class FlexibleGridAnchorGeneratorTest(test_case.TestCase):

  def test_construct_single_anchor(self):
    def graph_fn():
      anchor_strides = [(32, 32),]
      anchor_offsets = [(16, 16),]
      base_sizes = [(128.0,)]
      aspect_ratios = [(1.0,)]
      im_height = 64
      im_width = 64
      feature_map_shape_list = [(2, 2)]
      anchor_generator = fg.FlexibleGridAnchorGenerator(
          base_sizes, aspect_ratios, anchor_strides, anchor_offsets,
          normalize_coordinates=False)
      anchors_list = anchor_generator.generate(
          feature_map_shape_list, im_height=im_height, im_width=im_width)
      anchor_corners = anchors_list[0].get()
      return anchor_corners
    anchor_corners_out = self.execute(graph_fn, [])
    exp_anchor_corners = [[-48, -48, 80, 80],
                          [-48, -16, 80, 112],
                          [-16, -48, 112, 80],
                          [-16, -16, 112, 112]]
    self.assertAllClose(anchor_corners_out, exp_anchor_corners)

  def test_construct_single_anchor_unit_dimensions(self):
    def graph_fn():
      anchor_strides = [(32, 32),]
      anchor_offsets = [(16, 16),]
      base_sizes = [(32.0,)]
      aspect_ratios = [(1.0,)]
      im_height = 1
      im_width = 1
      feature_map_shape_list = [(2, 2)]
      anchor_generator = fg.FlexibleGridAnchorGenerator(
          base_sizes, aspect_ratios, anchor_strides, anchor_offsets,
          normalize_coordinates=False)
      anchors_list = anchor_generator.generate(
          feature_map_shape_list, im_height=im_height, im_width=im_width)
      anchor_corners = anchors_list[0].get()
      return anchor_corners
    # Positive offsets are produced.
    exp_anchor_corners = [[0, 0, 32, 32],
                          [0, 32, 32, 64],
                          [32, 0, 64, 32],
                          [32, 32, 64, 64]]
    anchor_corners_out = self.execute(graph_fn, [])
    self.assertAllClose(anchor_corners_out, exp_anchor_corners)

  def test_construct_normalized_anchors_fails_with_unit_dimensions(self):
    anchor_generator = fg.FlexibleGridAnchorGenerator(
        [(32.0,)], [(1.0,)], [(32, 32),], [(16, 16),],
        normalize_coordinates=True)
    with self.assertRaisesRegex(ValueError, 'Normalized coordinates'):
      anchor_generator.generate(
          feature_map_shape_list=[(2, 2)], im_height=1, im_width=1)

  def test_construct_single_anchor_in_normalized_coordinates(self):
    def graph_fn():
      anchor_strides = [(32, 32),]
      anchor_offsets = [(16, 16),]
      base_sizes = [(128.0,)]
      aspect_ratios = [(1.0,)]
      im_height = 64
      im_width = 128
      feature_map_shape_list = [(2, 2)]
      anchor_generator = fg.FlexibleGridAnchorGenerator(
          base_sizes, aspect_ratios, anchor_strides, anchor_offsets,
          normalize_coordinates=True)
      anchors_list = anchor_generator.generate(
          feature_map_shape_list, im_height=im_height, im_width=im_width)
      anchor_corners = anchors_list[0].get()
      return anchor_corners
    exp_anchor_corners = [[-48./64, -48./128, 80./64, 80./128],
                          [-48./64, -16./128, 80./64, 112./128],
                          [-16./64, -48./128, 112./64, 80./128],
                          [-16./64, -16./128, 112./64, 112./128]]
    anchor_corners_out = self.execute(graph_fn, [])
    self.assertAllClose(anchor_corners_out, exp_anchor_corners)

  def test_num_anchors_per_location(self):
    anchor_strides = [(32, 32), (64, 64)]
    anchor_offsets = [(16, 16), (32, 32)]
    base_sizes = [(32.0, 64.0, 96.0, 32.0, 64.0, 96.0),
                  (64.0, 128.0, 172.0, 64.0, 128.0, 172.0)]
    aspect_ratios = [(1.0, 1.0, 1.0, 2.0, 2.0, 2.0),
                     (1.0, 1.0, 1.0, 2.0, 2.0, 2.0)]
    anchor_generator = fg.FlexibleGridAnchorGenerator(
        base_sizes, aspect_ratios, anchor_strides, anchor_offsets,
        normalize_coordinates=False)
    self.assertEqual(anchor_generator.num_anchors_per_location(), [6, 6])

  def test_construct_single_anchor_dynamic_size(self):
    def graph_fn():
      anchor_strides = [(32, 32),]
      anchor_offsets = [(0, 0),]
      base_sizes = [(128.0,)]
      aspect_ratios = [(1.0,)]
      im_height = tf.constant(64)
      im_width = tf.constant(64)
      feature_map_shape_list = [(2, 2)]
      anchor_generator = fg.FlexibleGridAnchorGenerator(
          base_sizes, aspect_ratios, anchor_strides, anchor_offsets,
          normalize_coordinates=False)
      anchors_list = anchor_generator.generate(
          feature_map_shape_list, im_height=im_height, im_width=im_width)
      anchor_corners = anchors_list[0].get()
      return anchor_corners
    # Zero offsets are used.
    exp_anchor_corners = [[-64, -64, 64, 64],
                          [-64, -32, 64, 96],
                          [-32, -64, 96, 64],
                          [-32, -32, 96, 96]]
    anchor_corners_out = self.execute_cpu(graph_fn, [])
    self.assertAllClose(anchor_corners_out, exp_anchor_corners)

  def test_construct_single_anchor_with_odd_input_dimension(self):

    def graph_fn():
      anchor_strides = [(32, 32),]
      anchor_offsets = [(0, 0),]
      base_sizes = [(128.0,)]
      aspect_ratios = [(1.0,)]
      im_height = 65
      im_width = 65
      feature_map_shape_list = [(3, 3)]
      anchor_generator = fg.FlexibleGridAnchorGenerator(
          base_sizes, aspect_ratios, anchor_strides, anchor_offsets,
          normalize_coordinates=False)
      anchors_list = anchor_generator.generate(
          feature_map_shape_list, im_height=im_height, im_width=im_width)
      anchor_corners = anchors_list[0].get()
      return (anchor_corners,)
    anchor_corners_out = self.execute(graph_fn, [])
    exp_anchor_corners = [[-64, -64, 64, 64],
                          [-64, -32, 64, 96],
                          [-64, 0, 64, 128],
                          [-32, -64, 96, 64],
                          [-32, -32, 96, 96],
                          [-32, 0, 96, 128],
                          [0, -64, 128, 64],
                          [0, -32, 128, 96],
                          [0, 0, 128, 128]]
    self.assertAllClose(anchor_corners_out, exp_anchor_corners)

  def test_construct_single_anchor_on_two_feature_maps(self):

    def graph_fn():
      anchor_strides = [(32, 32), (64, 64)]
      anchor_offsets = [(16, 16), (32, 32)]
      base_sizes = [(128.0,), (256.0,)]
      aspect_ratios = [(1.0,), (1.0,)]
      im_height = 64
      im_width = 64
      feature_map_shape_list = [(2, 2), (1, 1)]
      anchor_generator = fg.FlexibleGridAnchorGenerator(
          base_sizes, aspect_ratios, anchor_strides, anchor_offsets,
          normalize_coordinates=False)
      anchors_list = anchor_generator.generate(feature_map_shape_list,
                                               im_height=im_height,
                                               im_width=im_width)
      anchor_corners = [anchors.get() for anchors in anchors_list]
      return anchor_corners

    anchor_corners_out = np.concatenate(self.execute(graph_fn, []), axis=0)
    exp_anchor_corners = [[-48, -48, 80, 80],
                          [-48, -16, 80, 112],
                          [-16, -48, 112, 80],
                          [-16, -16, 112, 112],
                          [-96, -96, 160, 160]]
    self.assertAllClose(anchor_corners_out, exp_anchor_corners)

  def test_construct_single_anchor_with_two_scales_per_octave(self):

    def graph_fn():
      anchor_strides = [(64, 64),]
      anchor_offsets = [(32, 32),]
      base_sizes = [(256.0, 362.03867)]
      aspect_ratios = [(1.0, 1.0)]
      im_height = 64
      im_width = 64
      feature_map_shape_list = [(1, 1)]

      anchor_generator = fg.FlexibleGridAnchorGenerator(
          base_sizes, aspect_ratios, anchor_strides, anchor_offsets,
          normalize_coordinates=False)
      anchors_list = anchor_generator.generate(feature_map_shape_list,
                                               im_height=im_height,
                                               im_width=im_width)
      anchor_corners = [anchors.get() for anchors in anchors_list]
      return anchor_corners
    # There are 4 set of anchors in this configuration. The order is:
    # [[2**0.0 intermediate scale + 1.0 aspect],
    #  [2**0.5 intermediate scale + 1.0 aspect]]
    exp_anchor_corners = [[-96., -96., 160., 160.],
                          [-149.0193, -149.0193, 213.0193, 213.0193]]

    anchor_corners_out = self.execute(graph_fn, [])
    self.assertAllClose(anchor_corners_out, exp_anchor_corners)

  def test_construct_single_anchor_with_two_scales_per_octave_and_aspect(self):
    def graph_fn():
      anchor_strides = [(64, 64),]
      anchor_offsets = [(32, 32),]
      base_sizes = [(256.0, 362.03867, 256.0, 362.03867)]
      aspect_ratios = [(1.0, 1.0, 2.0, 2.0)]
      im_height = 64
      im_width = 64
      feature_map_shape_list = [(1, 1)]
      anchor_generator = fg.FlexibleGridAnchorGenerator(
          base_sizes, aspect_ratios, anchor_strides, anchor_offsets,
          normalize_coordinates=False)
      anchors_list = anchor_generator.generate(feature_map_shape_list,
                                               im_height=im_height,
                                               im_width=im_width)
      anchor_corners = [anchors.get() for anchors in anchors_list]
      return anchor_corners
    # There are 4 set of anchors in this configuration. The order is:
    # [[2**0.0 intermediate scale + 1.0 aspect],
    #  [2**0.5 intermediate scale + 1.0 aspect],
    #  [2**0.0 intermediate scale + 2.0 aspect],
    #  [2**0.5 intermediate scale + 2.0 aspect]]

    exp_anchor_corners = [[-96., -96., 160., 160.],
                          [-149.0193, -149.0193, 213.0193, 213.0193],
                          [-58.50967, -149.0193, 122.50967, 213.0193],
                          [-96., -224., 160., 288.]]
    anchor_corners_out = self.execute(graph_fn, [])
    self.assertAllClose(anchor_corners_out, exp_anchor_corners)

  def test_construct_single_anchors_on_feature_maps_with_dynamic_shape(self):

    def graph_fn(feature_map1_height, feature_map1_width, feature_map2_height,
                 feature_map2_width):
      anchor_strides = [(32, 32), (64, 64)]
      anchor_offsets = [(16, 16), (32, 32)]
      base_sizes = [(128.0,), (256.0,)]
      aspect_ratios = [(1.0,), (1.0,)]
      im_height = 64
      im_width = 64
      feature_map_shape_list = [(feature_map1_height, feature_map1_width),
                                (feature_map2_height, feature_map2_width)]
      anchor_generator = fg.FlexibleGridAnchorGenerator(
          base_sizes, aspect_ratios, anchor_strides, anchor_offsets,
          normalize_coordinates=False)
      anchors_list = anchor_generator.generate(feature_map_shape_list,
                                               im_height=im_height,
                                               im_width=im_width)
      anchor_corners = [anchors.get() for anchors in anchors_list]
      return anchor_corners

    anchor_corners_out = np.concatenate(
        self.execute_cpu(graph_fn, [
            np.array(2, dtype=np.int32),
            np.array(2, dtype=np.int32),
            np.array(1, dtype=np.int32),
            np.array(1, dtype=np.int32)
        ]),
        axis=0)
    exp_anchor_corners = [[-48, -48, 80, 80],
                          [-48, -16, 80, 112],
                          [-16, -48, 112, 80],
                          [-16, -16, 112, 112],
                          [-96, -96, 160, 160]]
    self.assertAllClose(anchor_corners_out, exp_anchor_corners)


if __name__ == '__main__':
  tf.test.main()
