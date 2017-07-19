# Copyright 2017 Google Inc.
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

"""Contains functions for preprocessing the inputs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow as tf


def preprocess_classification(image, labels, is_training=False):
  """Preprocesses the image and labels for classification purposes.

  Preprocessing includes shifting the images to be 0-centered between -1 and 1.
  This is not only a popular method of preprocessing (inception) but is also
  the mechanism used by DSNs.

  Args:
    image: A `Tensor` of size [height, width, 3].
    labels: A dictionary of labels.
    is_training: Whether or not we're training the model.

  Returns:
    The preprocessed image and labels.
  """
  # If the image is uint8, this will scale it to 0-1.
  image = tf.image.convert_image_dtype(image, tf.float32)
  image -= 0.5
  image *= 2

  return image, labels


def preprocess_style_transfer(image,
                              labels,
                              augment=False,
                              size=None,
                              is_training=False):
  """Preprocesses the image and labels for style transfer purposes.

  Args:
    image: A `Tensor` of size [height, width, 3].
    labels: A dictionary of labels.
    augment: Whether to apply data augmentation to inputs
    size: The height and width to which images should be resized. If left as
      `None`, then no resizing is performed
    is_training: Whether or not we're training the model

  Returns:
    The preprocessed image and labels. Scaled to [-1, 1]
  """
  # If the image is uint8, this will scale it to 0-1.
  image = tf.image.convert_image_dtype(image, tf.float32)
  if augment and is_training:
    image = image_augmentation(image)

  if size:
    image = resize_image(image, size)

  image -= 0.5
  image *= 2

  return image, labels


def image_augmentation(image):
  """Performs data augmentation by randomly permuting the inputs.

  Args:
    image: A float `Tensor` of size [height, width, channels] with values
      in range[0,1].

  Returns:
    The mutated batch of images
  """
  # Apply photometric data augmentation (contrast etc.)
  num_channels = image.shape_as_list()[-1]
  if num_channels == 4:
    # Only augment image part
    image, depth = image[:, :, 0:3], image[:, :, 3:4]
  elif num_channels == 1:
    image = tf.image.grayscale_to_rgb(image)
  image = tf.image.random_brightness(image, max_delta=0.1)
  image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
  image = tf.image.random_hue(image, max_delta=0.032)
  image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
  image = tf.clip_by_value(image, 0, 1.0)
  if num_channels == 4:
    image = tf.concat(2, [image, depth])
  elif num_channels == 1:
    image = tf.image.rgb_to_grayscale(image)
  return image


def resize_image(image, size=None):
  """Resize image to target size.

  Args:
    image: A `Tensor` of size [height, width, 3].
    size: (height, width) to resize image to.

  Returns:
    resized image
  """
  if size is None:
    raise ValueError('Must specify size')

  if image.shape_as_list()[:2] == size:
    # Don't resize if not necessary
    return image
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_images(image, size)
  image = tf.squeeze(image, 0)
  return image
