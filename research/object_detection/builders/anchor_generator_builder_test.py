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

"""Tests for anchor_generator_builder."""

import math

import tensorflow as tf

from google.protobuf import text_format
from object_detection.anchor_generators import grid_anchor_generator
from object_detection.anchor_generators import multiple_grid_anchor_generator
from object_detection.anchor_generators import multiscale_grid_anchor_generator
from object_detection.builders import anchor_generator_builder
from object_detection.protos import anchor_generator_pb2


class AnchorGeneratorBuilderTest(tf.test.TestCase):

  def assert_almost_list_equal(self, expected_list, actual_list, delta=None):
    self.assertEqual(len(expected_list), len(actual_list))
    for expected_item, actual_item in zip(expected_list, actual_list):
      self.assertAlmostEqual(expected_item, actual_item, delta=delta)

  def test_build_grid_anchor_generator_with_defaults(self):
    anchor_generator_text_proto = """
      grid_anchor_generator {
      }
     """
    anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
    text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
    anchor_generator_object = anchor_generator_builder.build(
        anchor_generator_proto)
    self.assertTrue(isinstance(anchor_generator_object,
                               grid_anchor_generator.GridAnchorGenerator))
    self.assertListEqual(anchor_generator_object._scales, [])
    self.assertListEqual(anchor_generator_object._aspect_ratios, [])
    self.assertAllEqual(anchor_generator_object._anchor_offset, [0, 0])
    self.assertAllEqual(anchor_generator_object._anchor_stride, [16, 16])
    self.assertAllEqual(anchor_generator_object._base_anchor_size, [256, 256])

  def test_build_grid_anchor_generator_with_non_default_parameters(self):
    anchor_generator_text_proto = """
      grid_anchor_generator {
        height: 128
        width: 512
        height_stride: 10
        width_stride: 20
        height_offset: 30
        width_offset: 40
        scales: [0.4, 2.2]
        aspect_ratios: [0.3, 4.5]
      }
     """
    anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
    text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
    anchor_generator_object = anchor_generator_builder.build(
        anchor_generator_proto)
    self.assertTrue(isinstance(anchor_generator_object,
                               grid_anchor_generator.GridAnchorGenerator))
    self.assert_almost_list_equal(anchor_generator_object._scales,
                                  [0.4, 2.2])
    self.assert_almost_list_equal(anchor_generator_object._aspect_ratios,
                                  [0.3, 4.5])
    self.assertAllEqual(anchor_generator_object._anchor_offset, [30, 40])
    self.assertAllEqual(anchor_generator_object._anchor_stride, [10, 20])
    self.assertAllEqual(anchor_generator_object._base_anchor_size, [128, 512])

  def test_build_ssd_anchor_generator_with_defaults(self):
    anchor_generator_text_proto = """
      ssd_anchor_generator {
        aspect_ratios: [1.0]
      }
    """
    anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
    text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
    anchor_generator_object = anchor_generator_builder.build(
        anchor_generator_proto)
    self.assertTrue(isinstance(anchor_generator_object,
                               multiple_grid_anchor_generator.
                               MultipleGridAnchorGenerator))
    for actual_scales, expected_scales in zip(
        list(anchor_generator_object._scales),
        [(0.1, 0.2, 0.2),
         (0.35, 0.418),
         (0.499, 0.570),
         (0.649, 0.721),
         (0.799, 0.871),
         (0.949, 0.974)]):
      self.assert_almost_list_equal(expected_scales, actual_scales, delta=1e-2)
    for actual_aspect_ratio, expected_aspect_ratio in zip(
        list(anchor_generator_object._aspect_ratios),
        [(1.0, 2.0, 0.5)] + 5 * [(1.0, 1.0)]):
      self.assert_almost_list_equal(expected_aspect_ratio, actual_aspect_ratio)
    self.assertAllClose(anchor_generator_object._base_anchor_size, [1.0, 1.0])

  def test_build_ssd_anchor_generator_with_custom_scales(self):
    anchor_generator_text_proto = """
      ssd_anchor_generator {
        aspect_ratios: [1.0]
        scales: [0.1, 0.15, 0.2, 0.4, 0.6, 0.8]
        reduce_boxes_in_lowest_layer: false
      }
    """
    anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
    text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
    anchor_generator_object = anchor_generator_builder.build(
        anchor_generator_proto)
    self.assertTrue(isinstance(anchor_generator_object,
                               multiple_grid_anchor_generator.
                               MultipleGridAnchorGenerator))
    for actual_scales, expected_scales in zip(
        list(anchor_generator_object._scales),
        [(0.1, math.sqrt(0.1 * 0.15)),
         (0.15, math.sqrt(0.15 * 0.2)),
         (0.2, math.sqrt(0.2 * 0.4)),
         (0.4, math.sqrt(0.4 * 0.6)),
         (0.6, math.sqrt(0.6 * 0.8)),
         (0.8, math.sqrt(0.8 * 1.0))]):
      self.assert_almost_list_equal(expected_scales, actual_scales, delta=1e-2)

  def test_build_ssd_anchor_generator_with_custom_interpolated_scale(self):
    anchor_generator_text_proto = """
      ssd_anchor_generator {
        aspect_ratios: [0.5]
        interpolated_scale_aspect_ratio: 0.5
        reduce_boxes_in_lowest_layer: false
      }
    """
    anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
    text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
    anchor_generator_object = anchor_generator_builder.build(
        anchor_generator_proto)
    self.assertTrue(isinstance(anchor_generator_object,
                               multiple_grid_anchor_generator.
                               MultipleGridAnchorGenerator))
    for actual_aspect_ratio, expected_aspect_ratio in zip(
        list(anchor_generator_object._aspect_ratios),
        6 * [(0.5, 0.5)]):
      self.assert_almost_list_equal(expected_aspect_ratio, actual_aspect_ratio)

  def test_build_ssd_anchor_generator_without_reduced_boxes(self):
    anchor_generator_text_proto = """
      ssd_anchor_generator {
        aspect_ratios: [1.0]
        reduce_boxes_in_lowest_layer: false
      }
    """
    anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
    text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
    anchor_generator_object = anchor_generator_builder.build(
        anchor_generator_proto)
    self.assertTrue(isinstance(anchor_generator_object,
                               multiple_grid_anchor_generator.
                               MultipleGridAnchorGenerator))

    for actual_scales, expected_scales in zip(
        list(anchor_generator_object._scales),
        [(0.2, 0.264),
         (0.35, 0.418),
         (0.499, 0.570),
         (0.649, 0.721),
         (0.799, 0.871),
         (0.949, 0.974)]):
      self.assert_almost_list_equal(expected_scales, actual_scales, delta=1e-2)

    for actual_aspect_ratio, expected_aspect_ratio in zip(
        list(anchor_generator_object._aspect_ratios),
        6 * [(1.0, 1.0)]):
      self.assert_almost_list_equal(expected_aspect_ratio, actual_aspect_ratio)

    self.assertAllClose(anchor_generator_object._base_anchor_size, [1.0, 1.0])

  def test_build_ssd_anchor_generator_with_non_default_parameters(self):
    anchor_generator_text_proto = """
      ssd_anchor_generator {
        num_layers: 2
        min_scale: 0.3
        max_scale: 0.8
        aspect_ratios: [2.0]
        height_stride: 16
        height_stride: 32
        width_stride: 20
        width_stride: 30
        height_offset: 8
        height_offset: 16
        width_offset: 0
        width_offset: 10
      }
    """
    anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
    text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
    anchor_generator_object = anchor_generator_builder.build(
        anchor_generator_proto)
    self.assertTrue(isinstance(anchor_generator_object,
                               multiple_grid_anchor_generator.
                               MultipleGridAnchorGenerator))

    for actual_scales, expected_scales in zip(
        list(anchor_generator_object._scales),
        [(0.1, 0.3, 0.3), (0.8, 0.894)]):
      self.assert_almost_list_equal(expected_scales, actual_scales, delta=1e-2)

    for actual_aspect_ratio, expected_aspect_ratio in zip(
        list(anchor_generator_object._aspect_ratios),
        [(1.0, 2.0, 0.5), (2.0, 1.0)]):
      self.assert_almost_list_equal(expected_aspect_ratio, actual_aspect_ratio)

    for actual_strides, expected_strides in zip(
        list(anchor_generator_object._anchor_strides), [(16, 20), (32, 30)]):
      self.assert_almost_list_equal(expected_strides, actual_strides)

    for actual_offsets, expected_offsets in zip(
        list(anchor_generator_object._anchor_offsets), [(8, 0), (16, 10)]):
      self.assert_almost_list_equal(expected_offsets, actual_offsets)

    self.assertAllClose(anchor_generator_object._base_anchor_size, [1.0, 1.0])

  def test_raise_value_error_on_empty_anchor_genertor(self):
    anchor_generator_text_proto = """
    """
    anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
    text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
    with self.assertRaises(ValueError):
      anchor_generator_builder.build(anchor_generator_proto)

  def test_build_multiscale_anchor_generator_custom_aspect_ratios(self):
    anchor_generator_text_proto = """
      multiscale_anchor_generator {
        aspect_ratios: [1.0]
      }
    """
    anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
    text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
    anchor_generator_object = anchor_generator_builder.build(
        anchor_generator_proto)
    self.assertTrue(isinstance(anchor_generator_object,
                               multiscale_grid_anchor_generator.
                               MultiscaleGridAnchorGenerator))
    for level, anchor_grid_info in zip(
        range(3, 8), anchor_generator_object._anchor_grid_info):
      self.assertEqual(set(anchor_grid_info.keys()), set(['level', 'info']))
      self.assertTrue(level, anchor_grid_info['level'])
      self.assertEqual(len(anchor_grid_info['info']), 4)
      self.assertAllClose(anchor_grid_info['info'][0], [2**0, 2**0.5])
      self.assertTrue(anchor_grid_info['info'][1], 1.0)
      self.assertAllClose(anchor_grid_info['info'][2],
                          [4.0 * 2**level, 4.0 * 2**level])
      self.assertAllClose(anchor_grid_info['info'][3], [2**level, 2**level])
      self.assertTrue(anchor_generator_object._normalize_coordinates)

  def test_build_multiscale_anchor_generator_with_anchors_in_pixel_coordinates(
      self):
    anchor_generator_text_proto = """
      multiscale_anchor_generator {
        aspect_ratios: [1.0]
        normalize_coordinates: false
      }
    """
    anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
    text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
    anchor_generator_object = anchor_generator_builder.build(
        anchor_generator_proto)
    self.assertTrue(isinstance(anchor_generator_object,
                               multiscale_grid_anchor_generator.
                               MultiscaleGridAnchorGenerator))
    self.assertFalse(anchor_generator_object._normalize_coordinates)


if __name__ == '__main__':
  tf.test.main()
