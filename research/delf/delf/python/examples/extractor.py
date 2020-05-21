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


def ResizeImage(image, config, resize_factor=1.0):
  """Resizes image according to config.

  Args:
    image: Uint8 array with shape (height, width, 3).
    config: DelfConfig proto containing the model configuration.
    resize_factor: Optional float resize factor for the input image. If given,
      the maximum and minimum allowed image sizes in `config` are scaled by this
      factor. Must be non-negative.

  Returns:
    resized_image: Uint8 array with resized image.
    scale_factors: 2D float array, with factors used for resizing along height
      and width (If upscaling, larger than 1; if downscaling, smaller than 1).

  Raises:
    ValueError: If `image` has incorrect number of dimensions/channels.
  """
  if resize_factor < 0.0:
    raise ValueError('negative resize_factor is not allowed: %f' %
                     resize_factor)
  if image.ndim != 3:
    raise ValueError('image has incorrect number of dimensions: %d' %
                     image.ndims)
  height, width, channels = image.shape

  # Take into account resize factor.
  max_image_size = resize_factor * config.max_image_size
  min_image_size = resize_factor * config.min_image_size

  if channels != 3:
    raise ValueError('image has incorrect number of channels: %d' % channels)

  largest_side = max(width, height)

  if max_image_size >= 0 and largest_side > max_image_size:
    scale_factor = max_image_size / largest_side
  elif min_image_size >= 0 and largest_side < min_image_size:
    scale_factor = min_image_size / largest_side
  elif config.use_square_images and (height != width):
    scale_factor = 1.0
  else:
    # No resizing needed, early return.
    return image, np.ones(2, dtype=float)

  # Note that new_shape is in (width, height) format (PIL convention), while
  # scale_factors are in (height, width) convention (NumPy convention).
  if config.use_square_images:
    new_shape = (int(round(largest_side * scale_factor)),
                 int(round(largest_side * scale_factor)))
  else:
    new_shape = (int(round(width * scale_factor)),
                 int(round(height * scale_factor)))

  scale_factors = np.array([new_shape[1] / height, new_shape[0] / width],
                           dtype=float)

  pil_image = Image.fromarray(image)
  resized_image = np.array(pil_image.resize(new_shape, resample=Image.BILINEAR))

  return resized_image, scale_factors


def MakeExtractor(sess, config, import_scope=None):
  """Creates a function to extract global and/or local features from an image.

  Args:
    sess: TensorFlow session to use.
    config: DelfConfig proto containing the model configuration.
    import_scope: Optional scope to use for model.

  Returns:
    Function that receives an image and returns features.
  """
  # Load model.
  tf.compat.v1.saved_model.loader.load(
      sess, [tf.compat.v1.saved_model.tag_constants.SERVING],
      config.model_path,
      import_scope=import_scope)
  import_scope_prefix = import_scope + '/' if import_scope is not None else ''

  # Input tensors.
  input_image = sess.graph.get_tensor_by_name('%sinput_image:0' %
                                              import_scope_prefix)
  input_image_scales = sess.graph.get_tensor_by_name('%sinput_scales:0' %
                                                     import_scope_prefix)
  if config.use_local_features:
    input_score_threshold = sess.graph.get_tensor_by_name(
        '%sinput_abs_thres:0' % import_scope_prefix)
    input_max_feature_num = sess.graph.get_tensor_by_name(
        '%sinput_max_feature_num:0' % import_scope_prefix)

  # Output tensors.
  if config.use_global_features:
    raw_global_descriptors = sess.graph.get_tensor_by_name(
        '%sglobal_descriptors:0' % import_scope_prefix)
  if config.use_local_features:
    boxes = sess.graph.get_tensor_by_name('%sboxes:0' % import_scope_prefix)
    raw_local_descriptors = sess.graph.get_tensor_by_name('%sfeatures:0' %
                                                          import_scope_prefix)
    feature_scales = sess.graph.get_tensor_by_name('%sscales:0' %
                                                   import_scope_prefix)
    attention_with_extra_dim = sess.graph.get_tensor_by_name(
        '%sscores:0' % import_scope_prefix)

  # Post-process extracted features: normalize, PCA (optional), pooling.
  if config.use_global_features:
    if config.delf_global_config.image_scales_ind:
      raw_global_descriptors_selected_scales = tf.gather(
          raw_global_descriptors,
          list(config.delf_global_config.image_scales_ind))
    else:
      raw_global_descriptors_selected_scales = raw_global_descriptors
    global_descriptors_per_scale = feature_extractor.PostProcessDescriptors(
        raw_global_descriptors_selected_scales,
        config.delf_global_config.use_pca,
        config.delf_global_config.pca_parameters)
    unnormalized_global_descriptor = tf.reduce_sum(
        global_descriptors_per_scale, axis=0, name='sum_pooling')
    global_descriptor = tf.nn.l2_normalize(
        unnormalized_global_descriptor, axis=0, name='final_l2_normalization')

  if config.use_local_features:
    attention = tf.reshape(attention_with_extra_dim,
                           [tf.shape(attention_with_extra_dim)[0]])
    locations, local_descriptors = feature_extractor.DelfFeaturePostProcessing(
        boxes, raw_local_descriptors, config)

  def ExtractorFn(image, resize_factor=1.0):
    """Receives an image and returns DELF global and/or local features.

    If image is too small, returns empty features.

    Args:
      image: Uint8 array with shape (height, width, 3) containing the RGB image.
      resize_factor: Optional float resize factor for the input image. If given,
        the maximum and minimum allowed image sizes in the config are scaled by
        this factor.

    Returns:
      extracted_features: A dict containing the extracted global descriptors
        (key 'global_descriptor' mapping to a [D] float array), and/or local
        features (key 'local_features' mapping to a dict with keys 'locations',
        'descriptors', 'scales', 'attention').
    """

    resized_image, scale_factors = ResizeImage(
        image, config, resize_factor=resize_factor)

    # If the image is too small, returns empty features.
    if resized_image.shape[0] < _MIN_HEIGHT or resized_image.shape[
        1] < _MIN_WIDTH:
      extracted_features = {'global_descriptor': np.array([])}
      if config.use_local_features:
        extracted_features.update({
            'local_features': {
                'locations': np.array([]),
                'descriptors': np.array([]),
                'scales': np.array([]),
                'attention': np.array([]),
            }
        })
      return extracted_features

    feed_dict = {
        input_image: resized_image,
        input_image_scales: list(config.image_scales),
    }
    fetches = {}
    if config.use_global_features:
      fetches.update({
          'global_descriptor': global_descriptor,
      })
    if config.use_local_features:
      feed_dict.update({
          input_score_threshold: config.delf_local_config.score_threshold,
          input_max_feature_num: config.delf_local_config.max_feature_num,
      })
      fetches.update({
          'local_features': {
              'locations': locations,
              'descriptors': local_descriptors,
              'scales': feature_scales,
              'attention': attention,
          }
      })

    extracted_features = sess.run(fetches, feed_dict=feed_dict)

    # Adjust local feature positions due to rescaling.
    if config.use_local_features:
      extracted_features['local_features']['locations'] /= scale_factors

    return extracted_features

  return ExtractorFn
