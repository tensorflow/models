# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_datasets as tfds

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def _get_images_labels(batch_size, split, distords=False):
  """Returns Dataset for given split."""
  dataset = tfds.load(name='cifar10', split=split)
  scope = 'data_augmentation' if distords else 'input'
  with tf.name_scope(scope):
    dataset = dataset.map(DataPreprocessor(distords), num_parallel_calls=10)
  # Dataset is small enough to be fully loaded on memory:
  dataset = dataset.prefetch(-1)
  dataset = dataset.repeat().batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  images_labels = iterator.get_next()
  images, labels = images_labels['input'], images_labels['target']
  tf.summary.image('images', images)
  return images, labels


class DataPreprocessor(object):
  """Applies transformations to dataset record."""

  def __init__(self, distords):
    self._distords = distords

  def __call__(self, record):
    """Process img for training or eval."""
    img = record['image']
    img = tf.cast(img, tf.float32)
    if self._distords:  # training
      # Randomly crop a [height, width] section of the image.
      img = tf.random_crop(img, [IMAGE_SIZE, IMAGE_SIZE, 3])
      # Randomly flip the image horizontally.
      img = tf.image.random_flip_left_right(img)
      # Because these operations are not commutative, consider randomizing
      # the order their operation.
      # NOTE: since per_image_standardization zeros the mean and makes
      # the stddev unit, this likely has no effect see tensorflow#1458.
      img = tf.image.random_brightness(img, max_delta=63)
      img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
    else:  # Image processing for evaluation.
      # Crop the central [height, width] of the image.
      img = tf.image.resize_image_with_crop_or_pad(img, IMAGE_SIZE, IMAGE_SIZE)
    # Subtract off the mean and divide by the variance of the pixels.
    img = tf.image.per_image_standardization(img)
    return dict(input=img, target=record['label'])


def distorted_inputs(batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  return _get_images_labels(batch_size, tfds.Split.TRAIN, distords=True)


def inputs(eval_data, batch_size):
  """Construct input for CIFAR evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  split = tfds.Split.TEST if eval_data == 'test' else tfds.Split.TRAIN
  return _get_images_labels(batch_size, split)
