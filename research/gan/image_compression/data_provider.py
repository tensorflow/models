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
"""Contains code for loading and preprocessing the compression image data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from slim.datasets import dataset_factory as datasets

slim = tf.contrib.slim


def provide_data(split_name, batch_size, dataset_dir,
                 dataset_name='imagenet', num_readers=1, num_threads=1,
                 patch_size=128):
  """Provides batches of image data for compression.

  Args:
    split_name: Either 'train' or 'validation'.
    batch_size: The number of images in each batch.
    dataset_dir: The directory where the data can be found. If `None`, use
      default.
    dataset_name: Name of the dataset.
    num_readers: Number of dataset readers.
    num_threads: Number of prefetching threads.
    patch_size: Size of the path to extract from the image.

  Returns:
    images: A `Tensor` of size [batch_size, patch_size, patch_size, channels]
  """
  randomize = split_name == 'train'
  dataset = datasets.get_dataset(
      dataset_name, split_name, dataset_dir=dataset_dir)
  provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=num_readers,
      common_queue_capacity=5 * batch_size,
      common_queue_min=batch_size,
      shuffle=randomize)
  [image] = provider.get(['image'])

  # Sample a patch of fixed size.
  patch = tf.image.resize_image_with_crop_or_pad(image, patch_size, patch_size)
  patch.shape.assert_is_compatible_with([patch_size, patch_size, 3])

  # Preprocess the images. Make the range lie in a strictly smaller range than
  # [-1, 1], so that network outputs aren't forced to the extreme ranges.
  patch = (tf.to_float(patch) - 128.0) / 142.0

  if randomize:
    image_batch = tf.train.shuffle_batch(
        [patch],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=5 * batch_size,
        min_after_dequeue=batch_size)
  else:
    image_batch = tf.train.batch(
        [patch],
        batch_size=batch_size,
        num_threads=1,  # no threads so it's deterministic
        capacity=5 * batch_size)

  return image_batch


def float_image_to_uint8(image):
  """Convert float image in ~[-0.9, 0.9) to [0, 255] uint8.

  Args:
    image: An image tensor. Values should be in [-0.9, 0.9).

  Returns:
    Input image cast to uint8 and with integer values in [0, 255].
  """
  image = (image * 142.0) + 128.0
  return tf.cast(image, tf.uint8)
