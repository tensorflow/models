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
"""Records previous preprocessing operations and allows them to be repeated.

Used with object_detection.core.preprocessor. Passing a PreprocessorCache
into individual data augmentation functions or the general preprocess() function
will store all randomly generated variables in the PreprocessorCache. When
a preprocessor function is called multiple times with the same
PreprocessorCache object, that function will perform the same augmentation
on all calls.
"""

from collections import defaultdict


class PreprocessorCache(object):
  """Dictionary wrapper storing random variables generated during preprocessing.
  """

  # Constant keys representing different preprocessing functions
  ROTATION90 = 'rotation90'
  HORIZONTAL_FLIP = 'horizontal_flip'
  VERTICAL_FLIP = 'vertical_flip'
  PIXEL_VALUE_SCALE = 'pixel_value_scale'
  IMAGE_SCALE = 'image_scale'
  RGB_TO_GRAY = 'rgb_to_gray'
  ADJUST_BRIGHTNESS = 'adjust_brightness'
  ADJUST_CONTRAST = 'adjust_contrast'
  ADJUST_HUE = 'adjust_hue'
  ADJUST_SATURATION = 'adjust_saturation'
  DISTORT_COLOR = 'distort_color'
  STRICT_CROP_IMAGE = 'strict_crop_image'
  CROP_IMAGE = 'crop_image'
  PAD_IMAGE = 'pad_image'
  CROP_TO_ASPECT_RATIO = 'crop_to_aspect_ratio'
  RESIZE_METHOD = 'resize_method'
  PAD_TO_ASPECT_RATIO = 'pad_to_aspect_ratio'
  BLACK_PATCHES = 'black_patches'
  ADD_BLACK_PATCH = 'add_black_patch'
  SELECTOR = 'selector'
  SELECTOR_TUPLES = 'selector_tuples'
  SELF_CONCAT_IMAGE = 'self_concat_image'
  SSD_CROP_SELECTOR_ID = 'ssd_crop_selector_id'
  SSD_CROP_PAD_SELECTOR_ID = 'ssd_crop_pad_selector_id'
  JPEG_QUALITY = 'jpeg_quality'
  DOWNSCALE_TO_TARGET_PIXELS = 'downscale_to_target_pixels'
  PATCH_GAUSSIAN = 'patch_gaussian'

  # 27 permitted function ids
  _VALID_FNS = [ROTATION90, HORIZONTAL_FLIP, VERTICAL_FLIP, PIXEL_VALUE_SCALE,
                IMAGE_SCALE, RGB_TO_GRAY, ADJUST_BRIGHTNESS, ADJUST_CONTRAST,
                ADJUST_HUE, ADJUST_SATURATION, DISTORT_COLOR, STRICT_CROP_IMAGE,
                CROP_IMAGE, PAD_IMAGE, CROP_TO_ASPECT_RATIO, RESIZE_METHOD,
                PAD_TO_ASPECT_RATIO, BLACK_PATCHES, ADD_BLACK_PATCH, SELECTOR,
                SELECTOR_TUPLES, SELF_CONCAT_IMAGE, SSD_CROP_SELECTOR_ID,
                SSD_CROP_PAD_SELECTOR_ID, JPEG_QUALITY,
                DOWNSCALE_TO_TARGET_PIXELS, PATCH_GAUSSIAN]

  def __init__(self):
    self._history = defaultdict(dict)

  def clear(self):
    """Resets cache."""
    self._history = defaultdict(dict)

  def get(self, function_id, key):
    """Gets stored value given a function id and key.

    Args:
      function_id: identifier for the preprocessing function used.
      key: identifier for the variable stored.
    Returns:
      value: the corresponding value, expected to be a tensor or
             nested structure of tensors.
    Raises:
      ValueError: if function_id is not one of the 23 valid function ids.
    """
    if function_id not in self._VALID_FNS:
      raise ValueError('Function id not recognized: %s.' % str(function_id))
    return self._history[function_id].get(key)

  def update(self, function_id, key, value):
    """Adds a value to the dictionary.

    Args:
      function_id: identifier for the preprocessing function used.
      key: identifier for the variable stored.
      value: the value to store, expected to be a tensor or nested structure
             of tensors.
    Raises:
      ValueError: if function_id is not one of the 23 valid function ids.
    """
    if function_id not in self._VALID_FNS:
      raise ValueError('Function id not recognized: %s.' % str(function_id))
    self._history[function_id][key] = value
