# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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
"""Module to construct DELF feature extractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
import tensorflow as tf

from delf import feature_extractor

# Minimum dimensions below which DELF features are not extracted (empty
# features are returned). This applies after any resizing is performed.
_MIN_HEIGHT = 10
_MIN_WIDTH = 10


def ResizeImage(image, config):
  """Resizes image according to config.

  Args:
    image: Uint8 array with shape (height, width, 3).
    config: DelfConfig proto containing the model configuration.

  Returns:
    resized_image: Uint8 array with resized image.
    scale_factor: Float with factor used for resizing (If upscaling, larger than
      1; if downscaling, smaller than 1).

  Raises:
    ValueError: If `image` has incorrect number of dimensions/channels.
  """
  if image.ndim != 3:
    raise ValueError('image has incorrect number of dimensions: %d' %
                     image.ndims)
  height, width, channels = image.shape

  if channels != 3:
    raise ValueError('image has incorrect number of channels: %d' % channels)

  if config.max_image_size != -1 and (width > config.max_image_size or
                                      height > config.max_image_size):
    scale_factor = config.max_image_size / max(width, height)
  elif config.min_image_size != -1 and (width < config.min_image_size and
                                        height < config.min_image_size):
    scale_factor = config.min_image_size / max(width, height)
  else:
    # No resizing needed, early return.
    return image, 1.0

  new_shape = (int(width * scale_factor), int(height * scale_factor))
  pil_image = Image.fromarray(image)
  resized_image = np.array(pil_image.resize(new_shape, resample=Image.BILINEAR))

  return resized_image, scale_factor


def MakeExtractor(sess, config, import_scope=None):
  """Creates a function to extract features from an image.

  Args:
    sess: TensorFlow session to use.
    config: DelfConfig proto containing the model configuration.
    import_scope: Optional scope to use for model.

  Returns:
    Function that receives an image and returns features.
  """
  tf.saved_model.loader.load(
      sess, [tf.saved_model.tag_constants.SERVING],
      config.model_path,
      import_scope=import_scope)
  import_scope_prefix = import_scope + '/' if import_scope is not None else ''
  input_image = sess.graph.get_tensor_by_name('%sinput_image:0' %
                                              import_scope_prefix)
  input_score_threshold = sess.graph.get_tensor_by_name('%sinput_abs_thres:0' %
                                                        import_scope_prefix)
  input_image_scales = sess.graph.get_tensor_by_name('%sinput_scales:0' %
                                                     import_scope_prefix)
  input_max_feature_num = sess.graph.get_tensor_by_name(
      '%sinput_max_feature_num:0' % import_scope_prefix)
  boxes = sess.graph.get_tensor_by_name('%sboxes:0' % import_scope_prefix)
  raw_descriptors = sess.graph.get_tensor_by_name('%sfeatures:0' %
                                                  import_scope_prefix)
  feature_scales = sess.graph.get_tensor_by_name('%sscales:0' %
                                                 import_scope_prefix)
  attention_with_extra_dim = sess.graph.get_tensor_by_name('%sscores:0' %
                                                           import_scope_prefix)
  attention = tf.reshape(attention_with_extra_dim,
                         [tf.shape(attention_with_extra_dim)[0]])

  locations, descriptors = feature_extractor.DelfFeaturePostProcessing(
      boxes, raw_descriptors, config)

  def ExtractorFn(image):
    """Receives an image and returns DELF features.

    If image is too small, returns empty set of features.

    Args:
      image: Uint8 array with shape (height, width, 3) containing the RGB image.

    Returns:
      Tuple (locations, descriptors, feature_scales, attention)
    """
    resized_image, scale_factor = ResizeImage(image, config)

    # If the image is too small, returns empty features.
    if resized_image.shape[0] < _MIN_HEIGHT or resized_image.shape[
        1] < _MIN_WIDTH:
      return np.array([]), np.array([]), np.array([]), np.array([])

    (locations_out, descriptors_out, feature_scales_out,
     attention_out) = sess.run(
         [locations, descriptors, feature_scales, attention],
         feed_dict={
             input_image: resized_image,
             input_score_threshold: config.delf_local_config.score_threshold,
             input_image_scales: list(config.image_scales),
             input_max_feature_num: config.delf_local_config.max_feature_num
         })
    rescaled_locations_out = locations_out / scale_factor

    return (rescaled_locations_out, descriptors_out, feature_scales_out,
            attention_out)

  return ExtractorFn
