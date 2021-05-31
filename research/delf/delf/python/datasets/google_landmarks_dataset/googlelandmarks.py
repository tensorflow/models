# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Google Landmarks Dataset(GLD).

Placeholder for Google Landmarks dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf


class _GoogleLandmarksInfo(object):
  """Metadata about the Google Landmarks dataset."""
  num_classes = {'gld_v1': 14951, 'gld_v2': 203094, 'gld_v2_clean': 81313}


class _DataAugmentationParams(object):
  """Default parameters for augmentation."""
  # The following are used for training.
  min_object_covered = 0.1
  aspect_ratio_range_min = 3. / 4
  aspect_ratio_range_max = 4. / 3
  area_range_min = 0.08
  area_range_max = 1.0
  max_attempts = 100
  update_labels = False
  # 'central_fraction' is used for central crop in inference.
  central_fraction = 0.875

  random_reflection = False


def NormalizeImages(images, pixel_value_scale=0.5, pixel_value_offset=0.5):
  """Normalize pixel values in image.

  Output is computed as
  normalized_images = (images - pixel_value_offset) / pixel_value_scale.

  Args:
    images: `Tensor`, images to normalize.
    pixel_value_scale: float, scale.
    pixel_value_offset: float, offset.

  Returns:
    normalized_images: `Tensor`, normalized images.
  """
  images = tf.cast(images, tf.float32)
  normalized_images = tf.math.divide(
      tf.subtract(images, pixel_value_offset), pixel_value_scale)
  return normalized_images


def _ImageNetCrop(image, image_size):
  """Imagenet-style crop with random bbox and aspect ratio.

  Args:
    image: a `Tensor`, image to crop.
    image_size: an `int`. The image size for the decoded image, on each side.

  Returns:
    cropped_image: `Tensor`, cropped image.
  """

  params = _DataAugmentationParams()
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  (bbox_begin, bbox_size, _) = tf.image.sample_distorted_bounding_box(
      tf.shape(image),
      bounding_boxes=bbox,
      min_object_covered=params.min_object_covered,
      aspect_ratio_range=(params.aspect_ratio_range_min,
                          params.aspect_ratio_range_max),
      area_range=(params.area_range_min, params.area_range_max),
      max_attempts=params.max_attempts,
      use_image_if_no_bounding_boxes=True)
  cropped_image = tf.slice(image, bbox_begin, bbox_size)
  cropped_image.set_shape([None, None, 3])

  cropped_image = tf.image.resize(
      cropped_image, [image_size, image_size], method='area')
  if params.random_reflection:
    cropped_image = tf.image.random_flip_left_right(cropped_image)

  return cropped_image


def _ParseFunction(example, name_to_features, image_size, augmentation):
  """Parse a single TFExample to get the image and label and process the image.

  Args:
    example: a `TFExample`.
    name_to_features: a `dict`. The mapping from feature names to its type.
    image_size: an `int`. The image size for the decoded image, on each side.
    augmentation: a `boolean`. True if the image will be augmented.

  Returns:
    image: a `Tensor`. The processed image.
    label: a `Tensor`. The ground-truth label.
  """
  parsed_example = tf.io.parse_single_example(example, name_to_features)
  # Parse to get image.
  image = parsed_example['image/encoded']
  image = tf.io.decode_jpeg(image)
  image = NormalizeImages(
      image, pixel_value_scale=128.0, pixel_value_offset=128.0)
  if augmentation:
    image = _ImageNetCrop(image, image_size)
  else:
    image = tf.image.resize(image, [image_size, image_size])
    image.set_shape([image_size, image_size, 3])
  # Parse to get label.
  label = parsed_example['image/class/label']

  return image, label


def CreateDataset(file_pattern,
                  image_size=321,
                  batch_size=32,
                  augmentation=False,
                  seed=0):
  """Creates a dataset.

  Args:
    file_pattern: str, file pattern of the dataset files.
    image_size: int, image size.
    batch_size: int, batch size.
    augmentation: bool, whether to apply augmentation.
    seed: int, seed for shuffling the dataset.

  Returns:
     tf.data.TFRecordDataset.
  """

  filenames = tf.io.gfile.glob(file_pattern)

  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.repeat().shuffle(buffer_size=100, seed=seed)

  # Create a description of the features.
  feature_description = {
      'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
      'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
      'image/channels': tf.io.FixedLenFeature([], tf.int64, default_value=0),
      'image/format': tf.io.FixedLenFeature([], tf.string, default_value=''),
      'image/id': tf.io.FixedLenFeature([], tf.string, default_value=''),
      'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
      'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
      'image/class/label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
  }

  customized_parse_func = functools.partial(
      _ParseFunction,
      name_to_features=feature_description,
      image_size=image_size,
      augmentation=augmentation)
  dataset = dataset.map(customized_parse_func)
  dataset = dataset.batch(batch_size)

  return dataset


def GoogleLandmarksInfo():
  """Returns metadata information on the Google Landmarks dataset.

  Returns:
     object _GoogleLandmarksInfo containing metadata about the GLD dataset.
  """
  return _GoogleLandmarksInfo()
