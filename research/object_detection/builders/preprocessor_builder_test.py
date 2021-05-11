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

import tensorflow.compat.v1 as tf

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
      keypoint_flip_permutation: 1
      keypoint_flip_permutation: 0
      keypoint_flip_permutation: 2
      keypoint_flip_permutation: 3
      keypoint_flip_permutation: 5
      keypoint_flip_permutation: 4
      probability: 0.5
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_horizontal_flip)
    self.assertEqual(args, {'keypoint_flip_permutation': (1, 0, 2, 3, 5, 4),
                            'probability': 0.5})

  def test_build_random_vertical_flip(self):
    preprocessor_text_proto = """
    random_vertical_flip {
      keypoint_flip_permutation: 1
      keypoint_flip_permutation: 0
      keypoint_flip_permutation: 2
      keypoint_flip_permutation: 3
      keypoint_flip_permutation: 5
      keypoint_flip_permutation: 4
      probability: 0.5
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_vertical_flip)
    self.assertEqual(args, {'keypoint_flip_permutation': (1, 0, 2, 3, 5, 4),
                            'probability': 0.5})

  def test_build_random_rotation90(self):
    preprocessor_text_proto = """
    random_rotation90 {
      keypoint_rot_permutation: 3
      keypoint_rot_permutation: 0
      keypoint_rot_permutation: 1
      keypoint_rot_permutation: 2
      probability: 0.5
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_rotation90)
    self.assertEqual(args, {'keypoint_rot_permutation': (3, 0, 1, 2),
                            'probability': 0.5})

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
      jitter_mode: SHRINK
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_jitter_boxes)
    self.assert_dictionary_close(args, {'ratio': 0.1, 'jitter_mode': 'shrink'})

  def test_build_random_crop_image(self):
    preprocessor_text_proto = """
    random_crop_image {
      min_object_covered: 0.75
      min_aspect_ratio: 0.75
      max_aspect_ratio: 1.5
      min_area: 0.25
      max_area: 0.875
      overlap_thresh: 0.5
      clip_boxes: False
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
        'clip_boxes': False,
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

  def test_build_random_absolute_pad_image(self):
    preprocessor_text_proto = """
    random_absolute_pad_image {
      max_height_padding: 50
      max_width_padding: 100
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_absolute_pad_image)
    self.assertEqual(args, {
        'max_height_padding': 50,
        'max_width_padding': 100,
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
      clip_boxes: False
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
        'clip_boxes': False,
        'random_coef': 0.125,
        'pad_color': None,
    })

  def test_build_random_crop_pad_image_with_optional_parameters(self):
    preprocessor_text_proto = """
    random_crop_pad_image {
      min_object_covered: 0.75
      min_aspect_ratio: 0.75
      max_aspect_ratio: 1.5
      min_area: 0.25
      max_area: 0.875
      overlap_thresh: 0.5
      clip_boxes: False
      random_coef: 0.125
      min_padded_size_ratio: 0.5
      min_padded_size_ratio: 0.75
      max_padded_size_ratio: 0.5
      max_padded_size_ratio: 0.75
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
        'clip_boxes': False,
        'random_coef': 0.125,
        'min_padded_size_ratio': (0.5, 0.75),
        'max_padded_size_ratio': (0.5, 0.75),
        'pad_color': None,
    })

  def test_build_random_crop_to_aspect_ratio(self):
    preprocessor_text_proto = """
    random_crop_to_aspect_ratio {
      aspect_ratio: 0.85
      overlap_thresh: 0.35
      clip_boxes: False
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_crop_to_aspect_ratio)
    self.assert_dictionary_close(args, {'aspect_ratio': 0.85,
                                        'overlap_thresh': 0.35,
                                        'clip_boxes': False})

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

  def test_build_random_jpeg_quality(self):
    preprocessor_text_proto = """
    random_jpeg_quality {
      random_coef: 0.5
      min_jpeg_quality: 40
      max_jpeg_quality: 90
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Parse(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_jpeg_quality)
    self.assert_dictionary_close(args, {'random_coef': 0.5,
                                        'min_jpeg_quality': 40,
                                        'max_jpeg_quality': 90})

  def test_build_random_downscale_to_target_pixels(self):
    preprocessor_text_proto = """
    random_downscale_to_target_pixels {
      random_coef: 0.5
      min_target_pixels: 200
      max_target_pixels: 900
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Parse(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_downscale_to_target_pixels)
    self.assert_dictionary_close(args, {
        'random_coef': 0.5,
        'min_target_pixels': 200,
        'max_target_pixels': 900
    })

  def test_build_random_patch_gaussian(self):
    preprocessor_text_proto = """
    random_patch_gaussian {
      random_coef: 0.5
      min_patch_size: 10
      max_patch_size: 300
      min_gaussian_stddev: 0.2
      max_gaussian_stddev: 1.5
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Parse(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_patch_gaussian)
    self.assert_dictionary_close(args, {
        'random_coef': 0.5,
        'min_patch_size': 10,
        'max_patch_size': 300,
        'min_gaussian_stddev': 0.2,
        'max_gaussian_stddev': 1.5
    })

  def test_auto_augment_image(self):
    preprocessor_text_proto = """
    autoaugment_image {
      policy_name: 'v0'
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.autoaugment_image)
    self.assert_dictionary_close(args, {'policy_name': 'v0'})

  def test_drop_label_probabilistically(self):
    preprocessor_text_proto = """
    drop_label_probabilistically{
      label: 2
      drop_probability: 0.5
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.drop_label_probabilistically)
    self.assert_dictionary_close(args, {
        'dropped_label': 2,
        'drop_probability': 0.5
    })

  def test_remap_labels(self):
    preprocessor_text_proto = """
    remap_labels{
      original_labels: 1
      original_labels: 2
      new_label: 3
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.remap_labels)
    self.assert_dictionary_close(args, {
        'original_labels': [1, 2],
        'new_label': 3
    })

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

  def test_build_rgb_to_gray(self):
    preprocessor_text_proto = """
    rgb_to_gray {}
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.rgb_to_gray)
    self.assertEqual(args, {})

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

  def test_random_self_concat_image(self):
    preprocessor_text_proto = """
    random_self_concat_image {
      concat_vertical_probability: 0.5
      concat_horizontal_probability: 0.25
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_self_concat_image)
    self.assertEqual(args, {'concat_vertical_probability': 0.5,
                            'concat_horizontal_probability': 0.25})

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
        clip_boxes: False
        random_coef: 0.375
      }
      operations {
        min_object_covered: 0.25
        min_aspect_ratio: 0.75
        max_aspect_ratio: 1.5
        min_area: 0.5
        max_area: 1.0
        overlap_thresh: 0.25
        clip_boxes: True
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
                            'clip_boxes': [False, True],
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
        clip_boxes: False
        random_coef: 0.375
        min_padded_size_ratio: [1.0, 1.0]
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
        clip_boxes: True
        random_coef: 0.375
        min_padded_size_ratio: [1.0, 1.0]
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
                            'clip_boxes': [False, True],
                            'random_coef': [0.375, 0.375],
                            'min_padded_size_ratio': [(1.0, 1.0), (1.0, 1.0)],
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
        clip_boxes: False
        random_coef: 0.375
      }
      operations {
        min_object_covered: 0.25
        min_area: 0.5
        max_area: 1.0
        overlap_thresh: 0.25
        clip_boxes: True
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
                            'clip_boxes': [False, True],
                            'random_coef': [0.375, 0.375]})

  def test_build_ssd_random_crop_pad_fixed_aspect_ratio(self):
    preprocessor_text_proto = """
    ssd_random_crop_pad_fixed_aspect_ratio {
      operations {
        min_object_covered: 0.0
        min_aspect_ratio: 0.875
        max_aspect_ratio: 1.125
        min_area: 0.5
        max_area: 1.0
        overlap_thresh: 0.0
        clip_boxes: False
        random_coef: 0.375
      }
      operations {
        min_object_covered: 0.25
        min_aspect_ratio: 0.75
        max_aspect_ratio: 1.5
        min_area: 0.5
        max_area: 1.0
        overlap_thresh: 0.25
        clip_boxes: True
        random_coef: 0.375
      }
      aspect_ratio: 0.875
      min_padded_size_ratio: [1.0, 1.0]
      max_padded_size_ratio: [2.0, 2.0]
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function,
                     preprocessor.ssd_random_crop_pad_fixed_aspect_ratio)
    self.assertEqual(args, {'min_object_covered': [0.0, 0.25],
                            'aspect_ratio': 0.875,
                            'aspect_ratio_range': [(0.875, 1.125), (0.75, 1.5)],
                            'area_range': [(0.5, 1.0), (0.5, 1.0)],
                            'overlap_thresh': [0.0, 0.25],
                            'clip_boxes': [False, True],
                            'random_coef': [0.375, 0.375],
                            'min_padded_size_ratio': (1.0, 1.0),
                            'max_padded_size_ratio': (2.0, 2.0)})

  def test_build_normalize_image_convert_class_logits_to_softmax(self):
    preprocessor_text_proto = """
    convert_class_logits_to_softmax {
        temperature: 2
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.convert_class_logits_to_softmax)
    self.assertEqual(args, {'temperature': 2})

  def test_random_crop_by_scale(self):
    preprocessor_text_proto = """
    random_square_crop_by_scale {
      scale_min: 0.25
      scale_max: 2.0
      num_scales: 8
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Merge(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.random_square_crop_by_scale)
    self.assertEqual(args, {
        'scale_min': 0.25,
        'scale_max': 2.0,
        'num_scales': 8,
        'max_border': 128
    })

  def test_adjust_gamma(self):
    preprocessor_text_proto = """
    adjust_gamma {
      gamma: 2.2
      gain: 2.0
    }
    """
    preprocessor_proto = preprocessor_pb2.PreprocessingStep()
    text_format.Parse(preprocessor_text_proto, preprocessor_proto)
    function, args = preprocessor_builder.build(preprocessor_proto)
    self.assertEqual(function, preprocessor.adjust_gamma)
    self.assert_dictionary_close(args, {'gamma': 2.2, 'gain': 2.0})



if __name__ == '__main__':
  tf.test.main()
