# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""This module provides utilities to normalize image tensors.
"""
from typing import Sequence
import tensorflow as tf, tf_keras

MEAN_NORM = (0.485, 0.456, 0.406)
STDDEV_NORM = (0.229, 0.224, 0.225)


def normalize_image(
    image: tf.Tensor,
    offset: Sequence[float] = MEAN_NORM,
    scale: Sequence[float] = STDDEV_NORM,
) -> tf.Tensor:
  """Normalizes the image to zero mean and unit variance.

  If the input image dtype is float, it is expected to either have values in
  [0, 1) and offset is MEAN_NORM, or have values in [0, 255] and offset is
  MEAN_RGB.

  Args:
    image: A tf.Tensor in either (1) float dtype with values in range [0, 1) or
      [0, 255], or (2) int type with values in range [0, 255].
    offset: A tuple of mean values to be subtracted from the image.
    scale: A tuple of normalization factors.

  Returns:
    A normalized image tensor.
  """
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return normalize_scaled_float_image(image, offset, scale)


def normalize_scaled_float_image(
    image: tf.Tensor,
    offset: Sequence[float] = MEAN_NORM,
    scale: Sequence[float] = STDDEV_NORM,
):
  """Normalizes a scaled float image to zero mean and unit variance.

  It assumes the input image is float dtype with values in [0, 1) if offset is
  MEAN_NORM, values in [0, 255] if offset is MEAN_RGB.

  Args:
    image: A tf.Tensor in float32 dtype with values in range [0, 1) or [0, 255].
    offset: A tuple of mean values to be subtracted from the image.
    scale: A tuple of normalization factors.

  Returns:
    A normalized image tensor.
  """
  offset = tf.constant(offset)
  offset = tf.expand_dims(offset, axis=0)
  offset = tf.expand_dims(offset, axis=0)
  image -= offset

  scale = tf.constant(scale)
  scale = tf.expand_dims(scale, axis=0)
  scale = tf.expand_dims(scale, axis=0)
  image /= scale
  return image
