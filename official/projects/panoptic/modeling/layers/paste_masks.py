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

"""Contains definition for bilinear grid sampling and mask pasting layers."""

from typing import List

import tensorflow as tf, tf_keras


class BilinearGridSampler(tf_keras.layers.Layer):
  """Bilinear Grid Sampling layer."""

  def __init__(self, align_corners: bool = False, **kwargs):
    """Generates panoptic segmentation masks.

    Args:
      align_corners: A `bool` bool, if True, the centers of the 4 corner
        pixels of the input and output tensors are aligned, preserving the
        values at the corner pixels.
      **kwargs: Additional kwargs arguments.
    """
    super(BilinearGridSampler, self).__init__(**kwargs)
    self.align_corners = align_corners

    self._config = {
        'align_corners': align_corners
    }

  def build(self, input_shape):
    features_shape, _, _ = input_shape
    _, height, width, channels = features_shape.as_list()

    self._height = height
    self._width = width
    self._channels = channels

  def _valid_coordinates(self, x_coord, y_coord):
    return tf.logical_and(
        tf.logical_and(
            tf.greater_equal(x_coord, 0),
            tf.greater_equal(y_coord, 0)),
        tf.logical_and(
            tf.less(x_coord, self._width),
            tf.less(y_coord, self._height)))

  def _get_pixel(self, features, x_coord, y_coord):
    x_coord = tf.cast(x_coord, dtype=tf.int32)
    y_coord = tf.cast(y_coord, dtype=tf.int32)

    clipped_x = tf.clip_by_value(x_coord, 0, self._width - 1)
    clipped_y = tf.clip_by_value(y_coord, 0, self._height - 1)

    batch_size, _, _, _ = features.shape.as_list()
    if batch_size is None:
      batch_size = tf.shape(features)[0]

    batch_indices = tf.reshape(
        tf.range(batch_size, dtype=tf.int32),
        shape=[batch_size, 1, 1])
    batch_indices = tf.tile(
        batch_indices,
        multiples=[1, x_coord.shape[1], x_coord.shape[2]])
    indices = tf.cast(
        tf.stack([batch_indices, clipped_y, clipped_x], axis=-1),
        dtype=tf.int32)
    gathered_pixels = tf.gather_nd(features, indices)

    return tf.where(
        tf.expand_dims(self._valid_coordinates(x_coord, y_coord), axis=-1),
        gathered_pixels,
        tf.zeros_like(gathered_pixels))

  def call(self, inputs):
    features, x_coord, y_coord = inputs

    x_coord += 1
    y_coord += 1

    if self.align_corners:
      x_coord = (x_coord * 0.5) * (self._width - 1)
      y_coord = (y_coord * 0.5) * (self._height - 1)
    else:
      x_coord = (x_coord * self._width - 1) * 0.5
      y_coord = (y_coord * self._height - 1) * 0.5

    left = tf.floor(x_coord)
    top = tf.floor(y_coord)
    right = left + 1
    bottom = top + 1

    top_left = (right - x_coord) * (bottom - y_coord)
    top_right = (x_coord - left) * (bottom - y_coord)
    bottom_left = (right - x_coord) * (y_coord - top)
    bottom_right = (x_coord - left) * (y_coord - top)

    i_top_left = self._get_pixel(features, left, top)
    i_top_right = self._get_pixel(features, right, top)
    i_bottom_left = self._get_pixel(features, left, bottom)
    i_bottom_right = self._get_pixel(features, right, bottom)

    i_top_left *= tf.expand_dims(top_left, axis=-1)
    i_top_right *= tf.expand_dims(top_right, axis=-1)
    i_bottom_left *= tf.expand_dims(bottom_left, axis=-1)
    i_bottom_right *= tf.expand_dims(bottom_right, axis=-1)

    interpolated_features = tf.math.add_n(
        [i_top_left, i_top_right, i_bottom_left, i_bottom_right])
    return interpolated_features

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)


class PasteMasks(tf_keras.layers.Layer):
  """Layer to paste instance masks."""

  def __init__(self, output_size: List[int],
               grid_sampler, **kwargs):
    """Resizes and pastes instance masks to match image size.

    Args:
      output_size: A `List` of integers that represent the height and width of
        the output mask.
      grid_sampler: A grid sampling layer. Currently only `BilinearGridSampler`
        is supported.
      **kwargs: Additional kwargs arguments.
    """
    super(PasteMasks, self).__init__(**kwargs)
    self._output_size = output_size
    self._grid_sampler = grid_sampler

    self._config = {
        'output_size': output_size,
        'grid_sampler': grid_sampler
    }

  def build(self, input_shape):
    self._x_coords = tf.range(0, self._output_size[1], dtype=tf.float32)
    self._y_coords = tf.range(0, self._output_size[0], dtype=tf.float32)

  def call(self, inputs):
    masks, boxes = inputs
    y0, x0, y1, x1 = tf.split(boxes, 4, axis=1)

    x_coords = tf.cast(self._x_coords, dtype=boxes.dtype)
    y_coords = tf.cast(self._y_coords, dtype=boxes.dtype)
    x_coords = (x_coords - x0) / (x1 - x0) * 2 - 1
    y_coords = (y_coords - y0) / (y1 - y0) * 2 - 1

    x_coords = tf.tile(
        tf.expand_dims(x_coords, axis=1),
        multiples=[1, self._output_size[0], 1])
    y_coords = tf.tile(
        tf.expand_dims(y_coords, axis=2),
        multiples=[1, 1, self._output_size[1]])
    pasted_masks = self._grid_sampler((masks, x_coords, y_coords))
    return pasted_masks

  def get_config(self):
    return self._config_dict

  @classmethod
  def from_config(cls, config):
    return cls(**config)
