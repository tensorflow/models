# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities to preprocess images in CIFAR-10.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_PADDING = 4

slim = tf.contrib.slim


def preprocess_for_train(image,
                         output_height,
                         output_width,
                         padding=_PADDING,
                         add_image_summaries=True):
  """Preprocesses the given image for training.

  Note that the actual resizing scale is sampled from
    [`resize_size_min`, `resize_size_max`].

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    padding: The amound of padding before and after each dimension of the image.
    add_image_summaries: Enable image summaries.

  Returns:
    A preprocessed image.
  """
  if add_image_summaries:
    tf.summary.image('image', tf.expand_dims(image, 0))

  # Transform the image to floats.
  image = tf.to_float(image)
  if padding > 0:
    image = tf.pad(image, [[padding, padding], [padding, padding], [0, 0]])
  # Randomly crop a [height, width] section of the image.
  distorted_image = tf.random_crop(image,
                                   [output_height, output_width, 3])

  # Randomly flip the image horizontally.
  distorted_image = tf.image.random_flip_left_right(distorted_image)

  if add_image_summaries:
    tf.summary.image('distorted_image', tf.expand_dims(distorted_image, 0))

  # Because these operations are not commutative, consider randomizing
  # the order their operation.
  distorted_image = tf.image.random_brightness(distorted_image,
                                               max_delta=63)
  distorted_image = tf.image.random_contrast(distorted_image,
                                             lower=0.2, upper=1.8)
  # Subtract off the mean and divide by the variance of the pixels.
  return tf.image.per_image_standardization(distorted_image)


def preprocess_for_eval(image, output_height, output_width,
                        add_image_summaries=True):
  """Preprocesses the given image for evaluation.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    add_image_summaries: Enable image summaries.

  Returns:
    A preprocessed image.
  """
  if add_image_summaries:
    tf.summary.image('image', tf.expand_dims(image, 0))
  # Transform the image to floats.
  image = tf.to_float(image)

  # Resize and crop if needed.
  resized_image = tf.image.resize_image_with_crop_or_pad(image,
                                                         output_width,
                                                         output_height)
  if add_image_summaries:
    tf.summary.image('resized_image', tf.expand_dims(resized_image, 0))

  # Subtract off the mean and divide by the variance of the pixels.
  return tf.image.per_image_standardization(resized_image)


def preprocess_image(image, output_height, output_width, is_training=False,
                     add_image_summaries=True):
  """Preprocesses the given image.

  Args:
    image: A `Tensor` representing an image of arbitrary size.
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
    add_image_summaries: Enable image summaries.

  Returns:
    A preprocessed image.
  """
  if is_training:
    return preprocess_for_train(
        image, output_height, output_width,
        add_image_summaries=add_image_summaries)
  else:
    return preprocess_for_eval(
        image, output_height, output_width,
        add_image_summaries=add_image_summaries)
