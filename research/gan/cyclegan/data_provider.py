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
"""Contains code for loading and preprocessing image data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf


def normalize_image(image):
  """Rescale from range [0, 255] to [-1, 1]."""
  return (tf.to_float(image) - 127.5) / 127.5


def undo_normalize_image(normalized_image):
  """Convert to a numpy array that can be read by PIL."""
  # Convert from NHWC to HWC.
  normalized_image = np.squeeze(normalized_image, axis=0)
  return np.uint8(normalized_image * 127.5 + 127.5)


def _sample_patch(image, patch_size):
  """Crop image to square shape and resize it to `patch_size`.

  Args:
    image: A 3D `Tensor` of HWC format.
    patch_size: A Python scalar.  The output image size.

  Returns:
    A 3D `Tensor` of HWC format which has the shape of
    [patch_size, patch_size, 3].
  """
  image_shape = tf.shape(image)
  height, width = image_shape[0], image_shape[1]
  target_size = tf.minimum(height, width)
  image = tf.image.resize_image_with_crop_or_pad(image, target_size,
                                                 target_size)
  # tf.image.resize_area only accepts 4D tensor, so expand dims first.
  image = tf.expand_dims(image, axis=0)
  image = tf.image.resize_images(image, [patch_size, patch_size])
  image = tf.squeeze(image, axis=0)
  # Force image num_channels = 3
  image = tf.tile(image, [1, 1, tf.maximum(1, 4 - tf.shape(image)[2])])
  image = tf.slice(image, [0, 0, 0], [patch_size, patch_size, 3])
  return image


def full_image_to_patch(image, patch_size):
  image = normalize_image(image)
  # Sample a patch of fixed size.
  image_patch = _sample_patch(image, patch_size)
  image_patch.shape.assert_is_compatible_with([patch_size, patch_size, 3])
  return image_patch


def _provide_custom_dataset(image_file_pattern,
                            batch_size,
                            shuffle=True,
                            num_threads=1,
                            patch_size=128):
  """Provides batches of custom image data.

  Args:
    image_file_pattern: A string of glob pattern of image files.
    batch_size: The number of images in each batch.
    shuffle: Whether to shuffle the read images.  Defaults to True.
    num_threads: Number of prefetching threads.  Defaults to 1.
    patch_size: Size of the path to extract from the image.  Defaults to 128.

  Returns:
    A float `Tensor` of shape [batch_size, patch_size, patch_size, 3]
    representing a batch of images.
  """
  filename_queue = tf.train.string_input_producer(
      tf.train.match_filenames_once(image_file_pattern),
      shuffle=shuffle,
      capacity=5 * batch_size)
  image_reader = tf.WholeFileReader()

  _, image_bytes = image_reader.read(filename_queue)
  image = tf.image.decode_image(image_bytes)
  image_patch = full_image_to_patch(image, patch_size)

  if shuffle:
    return tf.train.shuffle_batch(
        [image_patch],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=5 * batch_size,
        min_after_dequeue=batch_size)
  else:
    return tf.train.batch(
        [image_patch],
        batch_size=batch_size,
        num_threads=1,  # no threads so it's deterministic
        capacity=5 * batch_size)


def provide_custom_datasets(image_file_patterns,
                            batch_size,
                            shuffle=True,
                            num_threads=1,
                            patch_size=128):
  """Provides multiple batches of custom image data.

  Args:
    image_file_patterns: A list of glob patterns of image files.
    batch_size: The number of images in each batch.
    shuffle: Whether to shuffle the read images.  Defaults to True.
    num_threads: Number of prefetching threads.  Defaults to 1.
    patch_size: Size of the patch to extract from the image.  Defaults to 128.

  Returns:
    A list of float `Tensor`s with the same size of `image_file_patterns`.
    Each of the `Tensor` in the list has a shape of
    [batch_size, patch_size, patch_size, 3] representing a batch of images.

  Raises:
    ValueError: If image_file_patterns is not a list or tuple.
  """
  if not isinstance(image_file_patterns, (list, tuple)):
    raise ValueError(
        '`image_file_patterns` should be either list or tuple, but was {}.'.
        format(type(image_file_patterns)))
  custom_datasets = []
  for pattern in image_file_patterns:
    custom_datasets.append(
        _provide_custom_dataset(
            pattern,
            batch_size=batch_size,
            shuffle=shuffle,
            num_threads=num_threads,
            patch_size=patch_size))
  return custom_datasets
