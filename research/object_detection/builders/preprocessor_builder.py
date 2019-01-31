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

"""Builder for preprocessing steps."""

import tensorflow as tf

from object_detection.core import preprocessor
from object_detection.protos import preprocessor_pb2


def _get_step_config_from_proto(preprocessor_step_config, step_name):
  """Returns the value of a field named step_name from proto.

  Args:
    preprocessor_step_config: A preprocessor_pb2.PreprocessingStep object.
    step_name: Name of the field to get value from.

  Returns:
    result_dict: a sub proto message from preprocessor_step_config which will be
                 later converted to a dictionary.

  Raises:
    ValueError: If field does not exist in proto.
  """
  for field, value in preprocessor_step_config.ListFields():
    if field.name == step_name:
      return value

  raise ValueError('Could not get field %s from proto!', step_name)


def _get_dict_from_proto(config):
  """Helper function to put all proto fields into a dictionary.

  For many preprocessing steps, there's an trivial 1-1 mapping from proto fields
  to function arguments. This function automatically populates a dictionary with
  the arguments from the proto.

  Protos that CANNOT be trivially populated include:
  * nested messages.
  * steps that check if an optional field is set (ie. where None != 0).
  * protos that don't map 1-1 to arguments (ie. list should be reshaped).
  * fields requiring additional validation (ie. repeated field has n elements).

  Args:
    config: A protobuf object that does not violate the conditions above.

  Returns:
    result_dict: |config| converted into a python dictionary.
  """
  result_dict = {}
  for field, value in config.ListFields():
    result_dict[field.name] = value
  return result_dict


# A map from a PreprocessingStep proto config field name to the preprocessing
# function that should be used. The PreprocessingStep proto should be parsable
# with _get_dict_from_proto.
PREPROCESSING_FUNCTION_MAP = {
    'normalize_image':
        preprocessor.normalize_image,
    'random_pixel_value_scale':
        preprocessor.random_pixel_value_scale,
    'random_image_scale':
        preprocessor.random_image_scale,
    'random_rgb_to_gray':
        preprocessor.random_rgb_to_gray,
    'random_adjust_brightness':
        preprocessor.random_adjust_brightness,
    'random_adjust_contrast':
        preprocessor.random_adjust_contrast,
    'random_adjust_hue':
        preprocessor.random_adjust_hue,
    'random_adjust_saturation':
        preprocessor.random_adjust_saturation,
    'random_distort_color':
        preprocessor.random_distort_color,
    'random_jitter_boxes':
        preprocessor.random_jitter_boxes,
    'random_crop_to_aspect_ratio':
        preprocessor.random_crop_to_aspect_ratio,
    'random_black_patches':
        preprocessor.random_black_patches,
    'rgb_to_gray':
        preprocessor.rgb_to_gray,
    'scale_boxes_to_pixel_coordinates': (
        preprocessor.scale_boxes_to_pixel_coordinates),
    'subtract_channel_mean':
        preprocessor.subtract_channel_mean,
    'convert_class_logits_to_softmax':
        preprocessor.convert_class_logits_to_softmax,
}


# A map to convert from preprocessor_pb2.ResizeImage.Method enum to
# tf.image.ResizeMethod.
RESIZE_METHOD_MAP = {
    preprocessor_pb2.ResizeImage.AREA: tf.image.ResizeMethod.AREA,
    preprocessor_pb2.ResizeImage.BICUBIC: tf.image.ResizeMethod.BICUBIC,
    preprocessor_pb2.ResizeImage.BILINEAR: tf.image.ResizeMethod.BILINEAR,
    preprocessor_pb2.ResizeImage.NEAREST_NEIGHBOR: (
        tf.image.ResizeMethod.NEAREST_NEIGHBOR),
}


def build(preprocessor_step_config):
  """Builds preprocessing step based on the configuration.

  Args:
    preprocessor_step_config: PreprocessingStep configuration proto.

  Returns:
    function, argmap: A callable function and an argument map to call function
                      with.

  Raises:
    ValueError: On invalid configuration.
  """
  step_type = preprocessor_step_config.WhichOneof('preprocessing_step')

  if step_type in PREPROCESSING_FUNCTION_MAP:
    preprocessing_function = PREPROCESSING_FUNCTION_MAP[step_type]
    step_config = _get_step_config_from_proto(preprocessor_step_config,
                                              step_type)
    function_args = _get_dict_from_proto(step_config)
    return (preprocessing_function, function_args)

  if step_type == 'random_horizontal_flip':
    config = preprocessor_step_config.random_horizontal_flip
    return (preprocessor.random_horizontal_flip,
            {
                'keypoint_flip_permutation': tuple(
                    config.keypoint_flip_permutation),
            })

  if step_type == 'random_vertical_flip':
    config = preprocessor_step_config.random_vertical_flip
    return (preprocessor.random_vertical_flip,
            {
                'keypoint_flip_permutation': tuple(
                    config.keypoint_flip_permutation),
            })

  if step_type == 'random_rotation90':
    return (preprocessor.random_rotation90, {})

  if step_type == 'random_crop_image':
    config = preprocessor_step_config.random_crop_image
    return (preprocessor.random_crop_image,
            {
                'min_object_covered': config.min_object_covered,
                'aspect_ratio_range': (config.min_aspect_ratio,
                                       config.max_aspect_ratio),
                'area_range': (config.min_area, config.max_area),
                'overlap_thresh': config.overlap_thresh,
                'clip_boxes': config.clip_boxes,
                'random_coef': config.random_coef,
            })

  if step_type == 'random_pad_image':
    config = preprocessor_step_config.random_pad_image
    min_image_size = None
    if (config.HasField('min_image_height') !=
        config.HasField('min_image_width')):
      raise ValueError('min_image_height and min_image_width should be either '
                       'both set or both unset.')
    if config.HasField('min_image_height'):
      min_image_size = (config.min_image_height, config.min_image_width)

    max_image_size = None
    if (config.HasField('max_image_height') !=
        config.HasField('max_image_width')):
      raise ValueError('max_image_height and max_image_width should be either '
                       'both set or both unset.')
    if config.HasField('max_image_height'):
      max_image_size = (config.max_image_height, config.max_image_width)

    pad_color = config.pad_color or None
    if pad_color:
      if len(pad_color) == 3:
        pad_color = tf.to_float([x for x in config.pad_color])
      else:
        raise ValueError('pad_color should have 3 elements (RGB) if set!')
    return (preprocessor.random_pad_image,
            {
                'min_image_size': min_image_size,
                'max_image_size': max_image_size,
                'pad_color': pad_color,
            })

  if step_type == 'random_crop_pad_image':
    config = preprocessor_step_config.random_crop_pad_image
    min_padded_size_ratio = config.min_padded_size_ratio
    if min_padded_size_ratio and len(min_padded_size_ratio) != 2:
      raise ValueError('min_padded_size_ratio should have 2 elements if set!')
    max_padded_size_ratio = config.max_padded_size_ratio
    if max_padded_size_ratio and len(max_padded_size_ratio) != 2:
      raise ValueError('max_padded_size_ratio should have 2 elements if set!')
    pad_color = config.pad_color
    if pad_color and len(pad_color) != 3:
      raise ValueError('pad_color should have 3 elements if set!')
    kwargs = {
        'min_object_covered': config.min_object_covered,
        'aspect_ratio_range': (config.min_aspect_ratio,
                               config.max_aspect_ratio),
        'area_range': (config.min_area, config.max_area),
        'overlap_thresh': config.overlap_thresh,
        'clip_boxes': config.clip_boxes,
        'random_coef': config.random_coef,
    }
    if min_padded_size_ratio:
      kwargs['min_padded_size_ratio'] = tuple(min_padded_size_ratio)
    if max_padded_size_ratio:
      kwargs['max_padded_size_ratio'] = tuple(max_padded_size_ratio)
    if pad_color:
      kwargs['pad_color'] = tuple(pad_color)
    return (preprocessor.random_crop_pad_image, kwargs)

  if step_type == 'random_resize_method':
    config = preprocessor_step_config.random_resize_method
    return (preprocessor.random_resize_method,
            {
                'target_size': [config.target_height, config.target_width],
            })

  if step_type == 'resize_image':
    config = preprocessor_step_config.resize_image
    method = RESIZE_METHOD_MAP[config.method]
    return (preprocessor.resize_image,
            {
                'new_height': config.new_height,
                'new_width': config.new_width,
                'method': method
            })

  if step_type == 'ssd_random_crop':
    config = preprocessor_step_config.ssd_random_crop
    if config.operations:
      min_object_covered = [op.min_object_covered for op in config.operations]
      aspect_ratio_range = [(op.min_aspect_ratio, op.max_aspect_ratio)
                            for op in config.operations]
      area_range = [(op.min_area, op.max_area) for op in config.operations]
      overlap_thresh = [op.overlap_thresh for op in config.operations]
      clip_boxes = [op.clip_boxes for op in config.operations]
      random_coef = [op.random_coef for op in config.operations]
      return (preprocessor.ssd_random_crop,
              {
                  'min_object_covered': min_object_covered,
                  'aspect_ratio_range': aspect_ratio_range,
                  'area_range': area_range,
                  'overlap_thresh': overlap_thresh,
                  'clip_boxes': clip_boxes,
                  'random_coef': random_coef,
              })
    return (preprocessor.ssd_random_crop, {})

  if step_type == 'ssd_random_crop_pad':
    config = preprocessor_step_config.ssd_random_crop_pad
    if config.operations:
      min_object_covered = [op.min_object_covered for op in config.operations]
      aspect_ratio_range = [(op.min_aspect_ratio, op.max_aspect_ratio)
                            for op in config.operations]
      area_range = [(op.min_area, op.max_area) for op in config.operations]
      overlap_thresh = [op.overlap_thresh for op in config.operations]
      clip_boxes = [op.clip_boxes for op in config.operations]
      random_coef = [op.random_coef for op in config.operations]
      min_padded_size_ratio = [tuple(op.min_padded_size_ratio)
                               for op in config.operations]
      max_padded_size_ratio = [tuple(op.max_padded_size_ratio)
                               for op in config.operations]
      pad_color = [(op.pad_color_r, op.pad_color_g, op.pad_color_b)
                   for op in config.operations]
      return (preprocessor.ssd_random_crop_pad,
              {
                  'min_object_covered': min_object_covered,
                  'aspect_ratio_range': aspect_ratio_range,
                  'area_range': area_range,
                  'overlap_thresh': overlap_thresh,
                  'clip_boxes': clip_boxes,
                  'random_coef': random_coef,
                  'min_padded_size_ratio': min_padded_size_ratio,
                  'max_padded_size_ratio': max_padded_size_ratio,
                  'pad_color': pad_color,
              })
    return (preprocessor.ssd_random_crop_pad, {})

  if step_type == 'ssd_random_crop_fixed_aspect_ratio':
    config = preprocessor_step_config.ssd_random_crop_fixed_aspect_ratio
    if config.operations:
      min_object_covered = [op.min_object_covered for op in config.operations]
      area_range = [(op.min_area, op.max_area) for op in config.operations]
      overlap_thresh = [op.overlap_thresh for op in config.operations]
      clip_boxes = [op.clip_boxes for op in config.operations]
      random_coef = [op.random_coef for op in config.operations]
      return (preprocessor.ssd_random_crop_fixed_aspect_ratio,
              {
                  'min_object_covered': min_object_covered,
                  'aspect_ratio': config.aspect_ratio,
                  'area_range': area_range,
                  'overlap_thresh': overlap_thresh,
                  'clip_boxes': clip_boxes,
                  'random_coef': random_coef,
              })
    return (preprocessor.ssd_random_crop_fixed_aspect_ratio, {})

  if step_type == 'ssd_random_crop_pad_fixed_aspect_ratio':
    config = preprocessor_step_config.ssd_random_crop_pad_fixed_aspect_ratio
    kwargs = {}
    aspect_ratio = config.aspect_ratio
    if aspect_ratio:
      kwargs['aspect_ratio'] = aspect_ratio
    min_padded_size_ratio = config.min_padded_size_ratio
    if min_padded_size_ratio:
      if len(min_padded_size_ratio) != 2:
        raise ValueError('min_padded_size_ratio should have 2 elements if set!')
      kwargs['min_padded_size_ratio'] = tuple(min_padded_size_ratio)
    max_padded_size_ratio = config.max_padded_size_ratio
    if max_padded_size_ratio:
      if len(max_padded_size_ratio) != 2:
        raise ValueError('max_padded_size_ratio should have 2 elements if set!')
      kwargs['max_padded_size_ratio'] = tuple(max_padded_size_ratio)
    if config.operations:
      kwargs['min_object_covered'] = [op.min_object_covered
                                      for op in config.operations]
      kwargs['aspect_ratio_range'] = [(op.min_aspect_ratio, op.max_aspect_ratio)
                                      for op in config.operations]
      kwargs['area_range'] = [(op.min_area, op.max_area)
                              for op in config.operations]
      kwargs['overlap_thresh'] = [op.overlap_thresh for op in config.operations]
      kwargs['clip_boxes'] = [op.clip_boxes for op in config.operations]
      kwargs['random_coef'] = [op.random_coef for op in config.operations]
    return (preprocessor.ssd_random_crop_pad_fixed_aspect_ratio, kwargs)

  raise ValueError('Unknown preprocessing step.')
