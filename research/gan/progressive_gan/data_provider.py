# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Loading and preprocessing image data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from slim.datasets import dataset_factory as datasets


def normalize_image(image):
  """Rescales image from range [0, 255] to [-1, 1]."""
  return (tf.to_float(image) - 127.5) / 127.5


def sample_patch(image, patch_height, patch_width, colors):
  """Crops image to the desired aspect ratio shape and resizes it.

  If the image has shape H x W, crops a square in the center of
  shape min(H,W) x min(H,W).

  Args:
    image: A 3D `Tensor` of HWC format.
    patch_height: A Python integer. The output images height.
    patch_width: A Python integer. The output images width.
    colors: Number of output image channels. Defaults to 3.

  Returns:
    A 3D `Tensor` of HWC format with shape [patch_height, patch_width, colors].
  """
  image_shape = tf.shape(image)
  h, w = image_shape[0], image_shape[1]

  h_major_target_h = h
  h_major_target_w = tf.maximum(1, tf.to_int32(
      (h * patch_width) / patch_height))
  w_major_target_h = tf.maximum(1, tf.to_int32(
      (w * patch_height) / patch_width))
  w_major_target_w = w
  target_hw = tf.cond(
      h_major_target_w <= w,
      lambda: tf.convert_to_tensor([h_major_target_h, h_major_target_w]),
      lambda: tf.convert_to_tensor([w_major_target_h, w_major_target_w]))
  # Cut a patch of shape (target_h, target_w).
  image = tf.image.resize_image_with_crop_or_pad(image, target_hw[0],
                                                 target_hw[1])
  # Resize the cropped image to (patch_h, patch_w).
  image = tf.image.resize_images([image], [patch_height, patch_width])[0]
  # Force number of channels: repeat the channel dimension enough
  # number of times and then slice the first `colors` channels.
  num_repeats = tf.to_int32(tf.ceil(colors / image_shape[2]))
  image = tf.tile(image, [1, 1, num_repeats])
  image = tf.slice(image, [0, 0, 0], [-1, -1, colors])
  image.set_shape([patch_height, patch_width, colors])
  return image


def batch_images(image, patch_height, patch_width, colors, batch_size, shuffle,
                 num_threads):
  """Creates a batch of images.

  Args:
    image: A 3D `Tensor` of HWC format.
    patch_height: A Python integer. The output images height.
    patch_width: A Python integer. The output images width.
    colors: Number of channels.
    batch_size: The number of images in each minibatch. Defaults to 32.
    shuffle: Whether to shuffle the read images.
    num_threads: Number of prefetching threads.

  Returns:
    A float `Tensor`s with shape [batch_size, patch_height, patch_width, colors]
    representing a batch of images.
  """
  image = sample_patch(image, patch_height, patch_width, colors)
  images = None
  if shuffle:
    images = tf.train.shuffle_batch(
        [image],
        batch_size=batch_size,
        num_threads=num_threads,
        capacity=5 * batch_size,
        min_after_dequeue=batch_size)
  else:
    images = tf.train.batch(
        [image],
        batch_size=batch_size,
        num_threads=1,  # no threads so it's deterministic
        capacity=5 * batch_size)
  images.set_shape([batch_size, patch_height, patch_width, colors])
  return images


def provide_data(dataset_name='cifar10',
                 split_name='train',
                 dataset_dir,
                 batch_size=32,
                 shuffle=True,
                 num_threads=1,
                 patch_height=32,
                 patch_width=32,
                 colors=3):
  """Provides a batch of image data from predefined dataset.

  Args:
    dataset_name: A string of dataset name. Defaults to 'cifar10'.
    split_name: Either 'train' or 'validation'. Defaults to 'train'.
    dataset_dir: The directory where the data can be found. If `None`, use
      default.
    batch_size: The number of images in each minibatch. Defaults to 32.
    shuffle: Whether to shuffle the read images. Defaults to True.
    num_threads: Number of prefetching threads. Defaults to 1.
    patch_height: A Python integer. The read images height. Defaults to 32.
    patch_width: A Python integer. The read images width. Defaults to 32.
    colors: Number of channels. Defaults to 3.

  Returns:
    A float `Tensor`s with shape [batch_size, patch_height, patch_width, colors]
    representing a batch of images.
  """
  dataset = datasets.get_dataset(
      dataset_name, split_name, dataset_dir=dataset_dir)
  provider = tf.contrib.slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      num_readers=1,
      common_queue_capacity=5 * batch_size,
      common_queue_min=batch_size,
      shuffle=shuffle)
  return batch_images(
      image=normalize_image(provider.get(['image'])[0]),
      patch_height=patch_height,
      patch_width=patch_width,
      colors=colors,
      batch_size=batch_size,
      shuffle=shuffle,
      num_threads=num_threads)


def provide_data_from_image_files(file_pattern,
                                  batch_size=32,
                                  shuffle=True,
                                  num_threads=1,
                                  patch_height=32,
                                  patch_width=32,
                                  colors=3):
  """Provides a batch of image data from image files.

  Args:
    file_pattern: A file pattern (glob), or 1D `Tensor` of file patterns.
    batch_size: The number of images in each minibatch.  Defaults to 32.
    shuffle: Whether to shuffle the read images. Defaults to True.
    num_threads: Number of prefetching threads. Defaults to 1.
    patch_height: A Python integer. The read images height. Defaults to 32.
    patch_width: A Python integer. The read images width. Defaults to 32.
    colors: Number of channels. Defaults to 3.

  Returns:
    A float `Tensor` of shape [batch_size, patch_height, patch_width, 3]
    representing a batch of images.
  """
  filename_queue = tf.train.string_input_producer(
      tf.train.match_filenames_once(file_pattern),
      shuffle=shuffle,
      capacity=5 * batch_size)
  _, image_bytes = tf.WholeFileReader().read(filename_queue)
  return batch_images(
      image=normalize_image(tf.image.decode_image(image_bytes)),
      patch_height=patch_height,
      patch_width=patch_width,
      colors=colors,
      batch_size=batch_size,
      shuffle=shuffle,
      num_threads=num_threads)
