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

from delf import datum_io
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


def MakeExtractor(config):
  """Creates a function to extract global and/or local features from an image.

  Args:
    config: DelfConfig proto containing the model configuration.

  Returns:
    Function that receives an image and returns features.

  Raises:
    ValueError: if config is invalid.
  """
  # Assert the configuration
  if config.use_global_features and hasattr(
      config, 'is_tf2_exported') and config.is_tf2_exported:
    raise ValueError('use_global_features is incompatible with is_tf2_exported')

  # Load model.
  model = tf.saved_model.load(config.model_path)

  # Input/output end-points/tensors.
  feeds = ['input_image:0', 'input_scales:0']
  fetches = []
  image_scales_tensor = tf.convert_to_tensor(list(config.image_scales))

  # Custom configuration needed when local features are used.
  if config.use_local_features:
    # Extra input/output end-points/tensors.
    feeds.append('input_abs_thres:0')
    feeds.append('input_max_feature_num:0')
    fetches.append('boxes:0')
    fetches.append('features:0')
    fetches.append('scales:0')
    fetches.append('scores:0')
    score_threshold_tensor = tf.constant(
        config.delf_local_config.score_threshold)
    max_feature_num_tensor = tf.constant(
        config.delf_local_config.max_feature_num)

    # If using PCA, pre-load required parameters.
    local_pca_parameters = {}
    if config.delf_local_config.use_pca:
      local_pca_parameters['mean'] = tf.constant(
          datum_io.ReadFromFile(
              config.delf_local_config.pca_parameters.mean_path),
          dtype=tf.float32)
      local_pca_parameters['matrix'] = tf.constant(
          datum_io.ReadFromFile(
              config.delf_local_config.pca_parameters.projection_matrix_path),
          dtype=tf.float32)
      local_pca_parameters[
          'dim'] = config.delf_local_config.pca_parameters.pca_dim
      local_pca_parameters['use_whitening'] = (
          config.delf_local_config.pca_parameters.use_whitening)
      if config.delf_local_config.pca_parameters.use_whitening:
        local_pca_parameters['variances'] = tf.squeeze(
            tf.constant(
                datum_io.ReadFromFile(
                    config.delf_local_config.pca_parameters.pca_variances_path),
                dtype=tf.float32))
      else:
        local_pca_parameters['variances'] = None

  # Custom configuration needed when global features are used.
  if config.use_global_features:
    # Extra output end-point.
    fetches.append('global_descriptors:0')

    # If using PCA, pre-load required parameters.
    global_pca_parameters = {}
    if config.delf_global_config.use_pca:
      global_pca_parameters['mean'] = tf.constant(
          datum_io.ReadFromFile(
              config.delf_global_config.pca_parameters.mean_path),
          dtype=tf.float32)
      global_pca_parameters['matrix'] = tf.constant(
          datum_io.ReadFromFile(
              config.delf_global_config.pca_parameters.projection_matrix_path),
          dtype=tf.float32)
      global_pca_parameters[
          'dim'] = config.delf_global_config.pca_parameters.pca_dim
      global_pca_parameters['use_whitening'] = (
          config.delf_global_config.pca_parameters.use_whitening)
      if config.delf_global_config.pca_parameters.use_whitening:
        global_pca_parameters['variances'] = tf.squeeze(
            tf.constant(
                datum_io.ReadFromFile(config.delf_global_config.pca_parameters
                                      .pca_variances_path),
                dtype=tf.float32))
      else:
        global_pca_parameters['variances'] = None

  if not hasattr(config, 'is_tf2_exported') or not config.is_tf2_exported:
    model = model.prune(feeds=feeds, fetches=fetches)

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

    # Input tensors.
    image_tensor = tf.convert_to_tensor(resized_image)

    # Extracted features.
    extracted_features = {}
    output = None

    if config.use_local_features:
      if hasattr(config, 'is_tf2_exported') and config.is_tf2_exported:
        predict = model.signatures['serving_default']
        output_dict = predict(
            input_image=image_tensor,
            input_scales=image_scales_tensor,
            input_max_feature_num=max_feature_num_tensor,
            input_abs_thres=score_threshold_tensor)
        output = [
            output_dict['boxes'], output_dict['features'],
            output_dict['scales'], output_dict['scores']
        ]
      else:
        output = model(image_tensor, image_scales_tensor,
                       score_threshold_tensor, max_feature_num_tensor)
    else:
      output = model(image_tensor, image_scales_tensor)

    # Post-process extracted features: normalize, PCA (optional), pooling.
    if config.use_global_features:
      raw_global_descriptors = output[-1]
      if config.delf_global_config.image_scales_ind:
        raw_global_descriptors_selected_scales = tf.gather(
            raw_global_descriptors,
            list(config.delf_global_config.image_scales_ind))
      else:
        raw_global_descriptors_selected_scales = raw_global_descriptors
      global_descriptors_per_scale = feature_extractor.PostProcessDescriptors(
          raw_global_descriptors_selected_scales,
          config.delf_global_config.use_pca, global_pca_parameters)
      unnormalized_global_descriptor = tf.reduce_sum(
          global_descriptors_per_scale, axis=0, name='sum_pooling')
      global_descriptor = tf.nn.l2_normalize(
          unnormalized_global_descriptor, axis=0, name='final_l2_normalization')
      extracted_features.update({
          'global_descriptor': global_descriptor.numpy(),
      })

    if config.use_local_features:
      boxes = output[0]
      raw_local_descriptors = output[1]
      feature_scales = output[2]
      attention_with_extra_dim = output[3]

      attention = tf.reshape(attention_with_extra_dim,
                             [tf.shape(attention_with_extra_dim)[0]])
      locations, local_descriptors = (
          feature_extractor.DelfFeaturePostProcessing(
              boxes, raw_local_descriptors, config.delf_local_config.use_pca,
              local_pca_parameters))
      locations /= scale_factors

      extracted_features.update({
          'local_features': {
              'locations': locations.numpy(),
              'descriptors': local_descriptors.numpy(),
              'scales': feature_scales.numpy(),
              'attention': attention.numpy(),
          }
      })

    return extracted_features

  return ExtractorFn
