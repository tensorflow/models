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

import tensorflow as tf

from google.protobuf import text_format
from object_detection.anchor_generators import grid_anchor_generator
from object_detection.anchor_generators import multiple_grid_anchor_generator
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
    with self.test_session() as sess:
      base_anchor_size, anchor_offset, anchor_stride = sess.run(
          [anchor_generator_object._base_anchor_size,
           anchor_generator_object._anchor_offset,
           anchor_generator_object._anchor_stride])
    self.assertAllEqual(anchor_offset, [0, 0])
    self.assertAllEqual(anchor_stride, [16, 16])
    self.assertAllEqual(base_anchor_size, [256, 256])

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
    with self.test_session() as sess:
      base_anchor_size, anchor_offset, anchor_stride = sess.run(
          [anchor_generator_object._base_anchor_size,
           anchor_generator_object._anchor_offset,
           anchor_generator_object._anchor_stride])
    self.assertAllEqual(anchor_offset, [30, 40])
    self.assertAllEqual(anchor_stride, [10, 20])
    self.assertAllEqual(base_anchor_size, [128, 512])

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

    with self.test_session() as sess:
      base_anchor_size = sess.run(anchor_generator_object._base_anchor_size)
    self.assertAllClose(base_anchor_size, [1.0, 1.0])

  def test_build_ssd_anchor_generator_withoud_reduced_boxes(self):
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

    with self.test_session() as sess:
      base_anchor_size = sess.run(anchor_generator_object._base_anchor_size)
    self.assertAllClose(base_anchor_size, [1.0, 1.0])

  def test_build_ssd_anchor_generator_with_non_default_parameters(self):
    anchor_generator_text_proto = """
      ssd_anchor_generator {
        num_layers: 2
        min_scale: 0.3
        max_scale: 0.8
        aspect_ratios: [2.0]
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
        [(0.1, 0.3, 0.3), (0.8,)]):
      self.assert_almost_list_equal(expected_scales, actual_scales, delta=1e-2)

    for actual_aspect_ratio, expected_aspect_ratio in zip(
        list(anchor_generator_object._aspect_ratios),
        [(1.0, 2.0, 0.5), (2.0,)]):
      self.assert_almost_list_equal(expected_aspect_ratio, actual_aspect_ratio)

    with self.test_session() as sess:
      base_anchor_size = sess.run(anchor_generator_object._base_anchor_size)
    self.assertAllClose(base_anchor_size, [1.0, 1.0])

  def test_raise_value_error_on_empty_anchor_genertor(self):
    anchor_generator_text_proto = """
    """
    anchor_generator_proto = anchor_generator_pb2.AnchorGenerator()
    text_format.Merge(anchor_generator_text_proto, anchor_generator_proto)
    with self.assertRaises(ValueError):
      anchor_generator_builder.build(anchor_generator_proto)


if __name__ == '__main__':
  tf.test.main()
