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

"""Tests for preprocessor_builder."""

import tensorflow as tf

from google.protobuf import text_format

from object_detection.builders import preprocessor_builder
from object_detection.core import preprocessor
from object_detection.protos import preprocessor_pb2


class PreprocessorBuilderTest(tf.test.TestCase):

  def assert_dictionary_close(self, dict1, dict2):
    """Helper to check if two dicts with floatst or integers are close."""
    self.assertEqual(sorted(dict1.keys()), sorted(dict2.keys()))
    for key in dict1:
      value = dict1[key]
      if isinstance(value, float):
        self.assertAlmostEqual(value, dict2[key])
      else:
        self.assertEqual(value, dict2[key])

  def test_build_normalize_image(self):
    preprocessor_text_proto = """
    normalize_image {
      original_minval: 0.0
      original_maxval: 255.0
      target_minval: -1.0
      target_maxval: 1.0
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.normalize_image)
    self.assertEqual(args, {
        'original_minval': 0.0,
        'original_maxval': 255.0,
        'target_minval': -1.0,
        'target_maxval': 1.0,
    })

  def test_build_random_horizontal_flip(self):
    preprocessor_text_proto = """
    random_horizontal_flip {
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_horizontal_flip)
    self.assertEqual(args, {})

  def test_build_random_pixel_value_scale(self):
    preprocessor_text_proto = """
    random_pixel_value_scale {
      minval: 0.8
      maxval: 1.2
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_pixel_value_scale)
    self.assert_dictionary_close(args, {'minval': 0.8, 'maxval': 1.2})

  def test_build_random_image_scale(self):
    preprocessor_text_proto = """
    random_image_scale {
      min_scale_ratio: 0.8
      max_scale_ratio: 2.2
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_image_scale)
    self.assert_dictionary_close(args, {'min_scale_ratio': 0.8,
                                        'max_scale_ratio': 2.2})

  def test_build_random_rgb_to_gray(self):
    preprocessor_text_proto = """
    random_rgb_to_gray {
      probability: 0.8
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_rgb_to_gray)
    self.assert_dictionary_close(args, {'probability': 0.8})

  def test_build_random_adjust_brightness(self):
    preprocessor_text_proto = """
    random_adjust_brightness {
      max_delta: 0.2
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_adjust_brightness)
    self.assert_dictionary_close(args, {'max_delta': 0.2})

  def test_build_random_adjust_contrast(self):
    preprocessor_text_proto = """
    random_adjust_contrast {
      min_delta: 0.7
      max_delta: 1.1
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_adjust_contrast)
    self.assert_dictionary_close(args, {'min_delta': 0.7, 'max_delta': 1.1})

  def test_build_random_adjust_hue(self):
    preprocessor_text_proto = """
    random_adjust_hue {
      max_delta: 0.01
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_adjust_hue)
    self.assert_dictionary_close(args, {'max_delta': 0.01})

  def test_build_random_adjust_saturation(self):
    preprocessor_text_proto = """
    random_adjust_saturation {
      min_delta: 0.75
      max_delta: 1.15
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_adjust_saturation)
    self.assert_dictionary_close(args, {'min_delta': 0.75, 'max_delta': 1.15})

  def test_build_random_distort_color(self):
    preprocessor_text_proto = """
    random_distort_color {
      color_ordering: 1
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_distort_color)
    self.assertEqual(args, {'color_ordering': 1})

  def test_build_random_jitter_boxes(self):
    preprocessor_text_proto = """
    random_jitter_boxes {
      ratio: 0.1
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_jitter_boxes)
    self.assert_dictionary_close(args, {'ratio': 0.1})

  def test_build_random_crop_image(self):
    preprocessor_text_proto = """
    random_crop_image {
      min_object_covered: 0.75
      min_aspect_ratio: 0.75
      max_aspect_ratio: 1.5
      min_area: 0.25
      max_area: 0.875
      overlap_thresh: 0.5
      random_coef: 0.125
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_crop_image)
    self.assertEqual(args, {
        'min_object_covered': 0.75,
        'aspect_ratio_range': (0.75, 1.5),
        'area_range': (0.25, 0.875),
        'overlap_thresh': 0.5,
        'random_coef': 0.125,
    })

  def test_build_random_pad_image(self):
    preprocessor_text_proto = """
    random_pad_image {
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_pad_image)
    self.assertEqual(args, {
        'min_image_size': None,
        'max_image_size': None,
        'pad_color': None,
    })

  def test_build_random_crop_pad_image(self):
    preprocessor_text_proto = """
    random_crop_pad_image {
      min_object_covered: 0.75
      min_aspect_ratio: 0.75
      max_aspect_ratio: 1.5
      min_area: 0.25
      max_area: 0.875
      overlap_thresh: 0.5
      random_coef: 0.125
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_crop_pad_image)
    self.assertEqual(args, {
        'min_object_covered': 0.75,
        'aspect_ratio_range': (0.75, 1.5),
        'area_range': (0.25, 0.875),
        'overlap_thresh': 0.5,
        'random_coef': 0.125,
        'min_padded_size_ratio': None,
        'max_padded_size_ratio': None,
        'pad_color': None,
    })

  def test_build_random_crop_to_aspect_ratio(self):
    preprocessor_text_proto = """
    random_crop_to_aspect_ratio {
      aspect_ratio: 0.85
      overlap_thresh: 0.35
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_crop_to_aspect_ratio)
    self.assert_dictionary_close(args, {'aspect_ratio': 0.85,
                                        'overlap_thresh': 0.35})

  def test_build_random_black_patches(self):
    preprocessor_text_proto = """
    random_black_patches {
      max_black_patches: 20
      probability: 0.95
      size_to_image_ratio: 0.12
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_black_patches)
    self.assert_dictionary_close(args, {'max_black_patches': 20,
                                        'probability': 0.95,
                                        'size_to_image_ratio': 0.12})

  def test_build_random_resize_method(self):
    preprocessor_text_proto = """
    random_resize_method {
      target_height: 75
      target_width: 100
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_resize_method)
    self.assert_dictionary_close(args, {'target_size': [75, 100]})

  def test_build_scale_boxes_to_pixel_coordinates(self):
    preprocessor_text_proto = """
    scale_boxes_to_pixel_coordinates {}
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.scale_boxes_to_pixel_coordinates)
    self.assertEqual(args, {})

  def test_build_resize_image(self):
    preprocessor_text_proto = """
    resize_image {
      new_height: 75
      new_width: 100
      method: BICUBIC
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.resize_image)
    self.assertEqual(args, {'new_height': 75,
                            'new_width': 100,
                            'method': tf.image.ResizeMethod.BICUBIC})

  def test_build_subtract_channel_mean(self):
    preprocessor_text_proto = """
    subtract_channel_mean {
      means: [1.0, 2.0, 3.0]
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.subtract_channel_mean)
    self.assertEqual(args, {'means': [1.0, 2.0, 3.0]})

  def test_build_ssd_random_crop(self):
    preprocessor_text_proto = """
    ssd_random_crop {
      operations {
        min_object_covered: 0.0
        min_aspect_ratio: 0.875
        max_aspect_ratio: 1.125
        min_area: 0.5
        max_area: 1.0
        overlap_thresh: 0.0
        random_coef: 0.375
      }
      operations {
        min_object_covered: 0.25
        min_aspect_ratio: 0.75
        max_aspect_ratio: 1.5
        min_area: 0.5
        max_area: 1.0
        overlap_thresh: 0.25
        random_coef: 0.375
      }
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.ssd_random_crop)
    self.assertEqual(args, {'min_object_covered': [0.0, 0.25],
                            'aspect_ratio_range': [(0.875, 1.125), (0.75, 1.5)],
                            'area_range': [(0.5, 1.0), (0.5, 1.0)],
                            'overlap_thresh': [0.0, 0.25],
                            'random_coef': [0.375, 0.375]})

  def test_build_ssd_random_crop_empty_operations(self):
    preprocessor_text_proto = """
    ssd_random_crop {
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.ssd_random_crop)
    self.assertEqual(args, {})

  def test_build_ssd_random_crop_pad(self):
    preprocessor_text_proto = """
    ssd_random_crop_pad {
      operations {
        min_object_covered: 0.0
        min_aspect_ratio: 0.875
        max_aspect_ratio: 1.125
        min_area: 0.5
        max_area: 1.0
        overlap_thresh: 0.0
        random_coef: 0.375
        min_padded_size_ratio: [0.0, 0.0]
        max_padded_size_ratio: [2.0, 2.0]
        pad_color_r: 0.5
        pad_color_g: 0.5
        pad_color_b: 0.5
      }
      operations {
        min_object_covered: 0.25
        min_aspect_ratio: 0.75
        max_aspect_ratio: 1.5
        min_area: 0.5
        max_area: 1.0
        overlap_thresh: 0.25
        random_coef: 0.375
        min_padded_size_ratio: [0.0, 0.0]
        max_padded_size_ratio: [2.0, 2.0]
        pad_color_r: 0.5
        pad_color_g: 0.5
        pad_color_b: 0.5
      }
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.ssd_random_crop_pad)
    self.assertEqual(args, {'min_object_covered': [0.0, 0.25],
                            'aspect_ratio_range': [(0.875, 1.125), (0.75, 1.5)],
                            'area_range': [(0.5, 1.0), (0.5, 1.0)],
                            'overlap_thresh': [0.0, 0.25],
                            'random_coef': [0.375, 0.375],
                            'min_padded_size_ratio': [(0.0, 0.0), (0.0, 0.0)],
                            'max_padded_size_ratio': [(2.0, 2.0), (2.0, 2.0)],
                            'pad_color': [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]})

  def test_build_ssd_random_crop_fixed_aspect_ratio(self):
    preprocessor_text_proto = """
    ssd_random_crop_fixed_aspect_ratio {
      operations {
        min_object_covered: 0.0
        min_area: 0.5
        max_area: 1.0
        overlap_thresh: 0.0
        random_coef: 0.375
      }
      operations {
        min_object_covered: 0.25
        min_area: 0.5
        max_area: 1.0
        overlap_thresh: 0.25
        random_coef: 0.375
      }
      aspect_ratio: 0.875
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.ssd_random_crop_fixed_aspect_ratio)
    self.assertEqual(args, {'min_object_covered': [0.0, 0.25],
                            'aspect_ratio': 0.875,
                            'area_range': [(0.5, 1.0), (0.5, 1.0)],
                            'overlap_thresh': [0.0, 0.25],
                            'random_coef': [0.375, 0.375]})


if __name__ == '__main__':
  tf.test.main()
