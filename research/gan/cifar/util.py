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
"""Convenience functions for training and evaluating a TFGAN CIFAR example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf
tfgan = tf.contrib.gan


def get_generator_conditioning(batch_size, num_classes):
  """Generates TFGAN conditioning inputs for evaluation.

  Args:
    batch_size: A Python integer. The desired batch size.
    num_classes: A Python integer. The number of classes.

  Returns:
    A Tensor of one-hot vectors corresponding to an even distribution over
    classes.

  Raises:
    ValueError: If `batch_size` isn't evenly divisible by `num_classes`.
  """
  if batch_size % num_classes != 0:
    raise ValueError('`batch_size` %i must be evenly divisible by '
                     '`num_classes` %i.' % (batch_size, num_classes))
  labels = [lbl for lbl in xrange(num_classes)
            for _ in xrange(batch_size // num_classes)]
  return tf.one_hot(tf.constant(labels), num_classes)


def get_image_grid(images, batch_size, num_classes, num_images_per_class):
  """Combines images from each class in a single summary image.

  Args:
    images: Tensor of images that are arranged by class. The first
      `batch_size / num_classes` images belong to the first class, the second
      group belong to the second class, etc. Shape is
      [batch, width, height, channels].
    batch_size: Python integer. Batch dimension.
    num_classes: Number of classes to show.
    num_images_per_class: Number of image examples per class to show.

  Raises:
    ValueError: If the batch dimension of `images` is known at graph
      construction, and it isn't `batch_size`.
    ValueError: If there aren't enough images to show
      `num_classes * num_images_per_class` images.
    ValueError: If `batch_size` isn't divisible by `num_classes`.

  Returns:
    A single image.
  """
  # Validate inputs.
  images.shape[0:1].assert_is_compatible_with([batch_size])
  if batch_size < num_classes * num_images_per_class:
    raise ValueError('Not enough images in batch to show the desired number of '
                     'images.')
  if batch_size % num_classes != 0:
    raise ValueError('`batch_size` must be divisible by `num_classes`.')

  # Only get a certain number of images per class.
  num_batches = batch_size // num_classes
  indices = [i * num_batches + j for i in xrange(num_classes)
             for j in xrange(num_images_per_class)]
  sampled_images = tf.gather(images, indices)
  return tfgan.eval.image_reshaper(
      sampled_images, num_cols=num_images_per_class)


def get_inception_scores(images, batch_size, num_inception_images):
  """Get Inception score for some images.

  Args:
    images: Image minibatch. Shape [batch size, width, height, channels]. Values
      are in [-1, 1].
    batch_size: Python integer. Batch dimension.
    num_inception_images: Number of images to run through Inception at once.

  Returns:
    Inception scores. Tensor shape is [batch size].

  Raises:
    ValueError: If `batch_size` is incompatible with the first dimension of
      `images`.
    ValueError: If `batch_size` isn't divisible by `num_inception_images`.
  """
  # Validate inputs.
  images.shape[0:1].assert_is_compatible_with([batch_size])
  if batch_size % num_inception_images != 0:
    raise ValueError(
        '`batch_size` must be divisible by `num_inception_images`.')

  # Resize images.
  size = 299
  resized_images = tf.image.resize_bilinear(images, [size, size])

  # Run images through Inception.
  num_batches = batch_size // num_inception_images
  inc_score = tfgan.eval.inception_score(
      resized_images, num_batches=num_batches)

  return inc_score


def get_frechet_inception_distance(real_images, generated_images, batch_size,
                                   num_inception_images):
  """Get Frechet Inception Distance between real and generated images.

  Args:
    real_images: Real images minibatch. Shape [batch size, width, height,
      channels. Values are in [-1, 1].
    generated_images: Generated images minibatch. Shape [batch size, width,
      height, channels]. Values are in [-1, 1].
    batch_size: Python integer. Batch dimension.
    num_inception_images: Number of images to run through Inception at once.

  Returns:
    Frechet Inception distance. A floating-point scalar.

  Raises:
    ValueError: If the minibatch size is known at graph construction time, and
      doesn't batch `batch_size`.
  """
  # Validate input dimensions.
  real_images.shape[0:1].assert_is_compatible_with([batch_size])
  generated_images.shape[0:1].assert_is_compatible_with([batch_size])

  # Resize input images.
  size = 299
  resized_real_images = tf.image.resize_bilinear(real_images, [size, size])
  resized_generated_images = tf.image.resize_bilinear(
      generated_images, [size, size])

  # Compute Frechet Inception Distance.
  num_batches = batch_size // num_inception_images
  fid = tfgan.eval.frechet_inception_distance(
      resized_real_images, resized_generated_images, num_batches=num_batches)

  return fid
