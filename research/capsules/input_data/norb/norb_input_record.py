# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Input utility functions for reading smallNorb dataset.

Handles reading from smallNorb dataset saved in tfrecord format. Scales and
normalizes the images as the preprocessing step. It can distort the images by
random cropping and contrast adjusting.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

SMALLNORB_SIZE = 96


def _read_and_decode(filename_queue, image_dim=48, distort=False,
                     split='train'):
  """Reads a single record and converts it to a tensor.

  Args:
    filename_queue: Tensor Queue, list of input files.
    image_dim: Scalar, the height (and width) of the image in pixels.
    distort: Boolean, whether to distort the input or not.
    split: String, the split of the data (test or train) to read from.

  Returns:
    Dictionary of the (Image, label) and the image height.

  Raises:
    ValueError: If image_dim is larger than original smallNORB size (96).
  """
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
          'height': tf.FixedLenFeature([], tf.int64),
          'width': tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64),
          'meta': tf.FixedLenFeature([4], tf.int64),
      })

  image = tf.decode_raw(features['image_raw'], tf.uint8)
  height = tf.cast(features['height'], tf.int32)
  depth = tf.cast(features['depth'], tf.int32)
  image = tf.reshape(image, tf.stack([depth, height, height]))
  image = tf.transpose(image, [1, 2, 0])
  image = tf.cast(image, tf.float32)
  if image_dim < SMALLNORB_SIZE:
    tf.logging.info('image resizing to {}'.format(image_dim))
    image = tf.image.resize_images(image, [image_dim, image_dim])
  elif image_dim > SMALLNORB_SIZE:
    raise ValueError(
        'Image dim must be <= {}, got {}'.format(SMALLNORB_SIZE, image_dim))

  if image_dim == 48:
    distorted_dim = 32
  elif image_dim == 32:
    distorted_dim = 22
  if distort and split == 'train':
    image = tf.image.random_brightness(image, max_delta=63)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.random_crop(image,
                           tf.stack([distorted_dim, distorted_dim, depth]))
    image_dim = distorted_dim
  elif distort:
    image = tf.image.resize_image_with_crop_or_pad(image, distorted_dim,
                                                   distorted_dim)
    image_dim = distorted_dim

  image = tf.image.per_image_standardization(image)
  image.set_shape([image_dim, image_dim, 2])
  image = tf.transpose(image, [2, 0, 1])
  label = tf.cast(features['label'], tf.int32)

  features = {
      'images': image,
      'labels': tf.one_hot(label, 5),
      'recons_image': image,
      'recons_label': label,
  }
  return features, image_dim


def inputs(data_dir,
           batch_size,
           split,
           height=48,
           distort=True,
           batch_capacity=5000,
           ):
  """Reads input data.

  Args:
    data_dir: Directory of the data.
    batch_size: Number of examples per returned batch.
    split: train or test
    height: image height.
    distort: whether to distort the input image.
    batch_capacity: the number of elements to prefetch in a batch.

  Returns:
    Dictionary of Batched features and labels.
  """
  filenames = [os.path.join(data_dir, '{}duo.tfrecords'.format(split))]

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(filenames)

    features, image_dim = _read_and_decode(
        filename_queue, image_dim=height, distort=distort, split=split)
    if split == 'train':
      batched_features = tf.train.shuffle_batch(
          features,
          batch_size=batch_size,
          num_threads=2,
          capacity=batch_capacity + 3 * batch_size,
          # Ensures a minimum amount of shuffling of examples.
          min_after_dequeue=batch_capacity)
    else:
      batched_features = tf.train.batch(
          features,
          batch_size=batch_size,
          num_threads=1,
          capacity=batch_capacity + 3 * batch_size)
    batched_features['height'] = image_dim
    batched_features['depth'] = 2
    batched_features['num_targets'] = 1
    batched_features['num_classes'] = 5
    return batched_features
