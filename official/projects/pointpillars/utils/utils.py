# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Contains utility functions for pointpillars."""

import collections
from typing import Any, List, Mapping, Tuple

import numpy as np
import tensorflow as tf


CLASSES = {'vehicle': 1, 'pedestrian': 2, 'cyclist': 3}


def assert_shape(x: np.ndarray, shape: List[int]):
  if tuple(x.shape) != tuple(shape):
    raise ValueError('Shape of array should be {}, but {} found'.format(
        shape, x.shape))


def assert_channels_last():
  if tf.keras.backend.image_data_format() != 'channels_last':
    raise ValueError('Only "channels_last" mode is supported')


def pad_or_trim_to_shape(x: np.ndarray, shape: List[int]) -> np.ndarray:
  """Pad and trim x to the specified shape, x should have same rank as shape.

  Args:
    x: An np array.
    shape: A list of int indicating a array shape.

  Returns:
    y: An np array with padded/trimmed shape.
  """
  shape = np.array(shape)

  # Try to pad from end
  pad_end = shape - np.minimum(x.shape, shape)
  pad_begin = np.zeros_like(pad_end)
  padder = np.stack([pad_begin, pad_end], axis=1)
  x = np.pad(x, padder)

  # Try to trim from end.
  slice_end = shape
  slice_begin = np.zeros_like(slice_end)
  slicer = tuple(map(slice, slice_begin, slice_end))
  y = x[slicer].reshape(shape)
  return y


def clip_boxes(boxes: np.ndarray, image_height: int,
               image_width: int) -> np.ndarray:
  """Clip boxes to image boundaries.

  Args:
    boxes: An np array of boxes, [y0, x0, y1, y1].
    image_height: An int of image height.
    image_width: An int of image width.
  Returns:
    clipped_boxes: An np array of boxes, [y0, x0, y1, y1].
  """
  max_length = [image_height, image_width, image_height, image_width]
  clipped_boxes = np.maximum(np.minimum(boxes, max_length), 0.0)
  return clipped_boxes


def get_vehicle_xy(image_height: int, image_width: int,
                   x_range: Tuple[float, float],
                   y_range: Tuple[float, float]) -> Tuple[int, int]:
  """Get vehicle x/y in image coordinate.

  Args:
    image_height: A float of image height.
    image_width: A float of image width.
    x_range: A float tuple of (-x, +x).
    y_range: A float tuple of (-y, +x).
  Returns:
    vehicle_xy: An int tuple of (col, row).
  """
  vehicle_col = (image_width * (-x_range[0] / (-x_range[0] + x_range[1])))
  vehicle_row = (image_height * (-y_range[0] / (-y_range[0] + y_range[1])))
  vehicle_xy = (int(vehicle_col), int(vehicle_row))
  return vehicle_xy


def frame_to_image_coord(frame_xy: np.ndarray, vehicle_xy: Tuple[int, int],
                         one_over_resolution: float) -> np.ndarray:
  """Convert float frame (x, y) to int image (x, y).

  Args:
    frame_xy: An np array of frame xy coordinates.
    vehicle_xy: An int tuple of (vehicle_x, vehicle_y) in image.
    one_over_resolution: A float of one over image resolution.
  Returns:
    image_xy: An np array of image xy cooridnates.
  """
  image_xy = np.floor(frame_xy * one_over_resolution).astype(np.int32)
  image_xy[..., 0] += vehicle_xy[0]
  image_xy[..., 1] = vehicle_xy[1] - 1 - image_xy[..., 1]
  return image_xy


def image_to_frame_coord(image_xy: np.ndarray, vehicle_xy: Tuple[int, int],
                         resolution: float) -> np.ndarray:
  """Convert int image (x, y) to float frame (x, y).

  Args:
    image_xy: An np array of image xy cooridnates.
    vehicle_xy: An int tuple of (vehicle_x, vehicle_y) in image.
    resolution: A float of image resolution.
  Returns:
    frame_xy: An np array of frame xy coordinates.
  """
  frame_xy = image_xy.astype(np.float32)
  frame_xy[..., 0] = (frame_xy[..., 0] - vehicle_xy[0]) * resolution
  frame_xy[..., 1] = (vehicle_xy[1] - 1 - frame_xy[..., 1]) * resolution
  return frame_xy


def frame_to_image_boxes(frame_boxes: Any, vehicle_xy: Tuple[int, int],
                         one_over_resolution: float) -> Any:
  """Convert boxes from frame coordinate to image coordinate.

  Args:
    frame_boxes: A [N, 4] array or tensor, [center_x, center_y, length, width]
      in frame coordinate.
    vehicle_xy: An int tuple of (vehicle_x, vehicle_y) in image.
    one_over_resolution: A float number, 1.0 / resolution.

  Returns:
    image_boxes: A [N, 4] array or tensor, [ymin, xmin, ymax, xmax] in image
      coordinate.
  """
  center_x = frame_boxes[..., 0]
  center_y = frame_boxes[..., 1]
  box_length = frame_boxes[..., 2]
  box_width = frame_boxes[..., 3]

  image_box_length = box_length * one_over_resolution
  image_box_width = box_width * one_over_resolution
  image_box_center_x = (center_x * one_over_resolution + vehicle_xy[0])
  image_box_center_y = (vehicle_xy[1] - 1 - center_y * one_over_resolution)

  ymin = image_box_center_y - image_box_width * 0.5
  xmin = image_box_center_x - image_box_length * 0.5
  ymax = image_box_center_y + image_box_width * 0.5
  xmax = image_box_center_x + image_box_length * 0.5

  image_boxes = np.stack([ymin, xmin, ymax, xmax], axis=-1)
  return image_boxes


def image_to_frame_boxes(image_boxes: Any, vehicle_xy: Tuple[float],
                         resolution: float) -> Any:
  """Convert boxes from image coordinate to frame coordinate.

  Args:
    image_boxes: A [N, 4] array or tensor, [ymin, xmin, ymax, xmax] in image
      coordinate.
    vehicle_xy: A float tuple of (vehicle_x, vehicle_y) in image.
    resolution: A float number representing pillar grid resolution.

  Returns:
    frame_boxes: A [N, 4] array or tensor, [center_x, center_y, length, width]
      in frame coordinate.
  """
  ymin = image_boxes[..., 0]
  xmin = image_boxes[..., 1]
  ymax = image_boxes[..., 2]
  xmax = image_boxes[..., 3]

  image_box_length = xmax - xmin
  image_box_width = ymax - ymin
  image_box_center_x = xmin + image_box_length * 0.5
  image_box_center_y = ymin + image_box_width * 0.5

  center_x = (image_box_center_x - vehicle_xy[0]) * resolution
  center_y = (vehicle_xy[1] - 1 - image_box_center_y) * resolution
  box_length = image_box_length * resolution
  box_width = image_box_width * resolution

  frame_boxes = np.stack([center_x, center_y, box_length, box_width], axis=-1)
  return frame_boxes


def clip_heading(heading: Any) -> Any:
  """Clip heading to the range [-pi, pi]."""
  heading = tf.nest.map_structure(lambda x: np.pi * tf.tanh(x), heading)
  return heading


def wrap_angle_rad(angles_rad: Any,
                   min_val: float = -np.pi,
                   max_val: float = np.pi) -> Any:
  """Wrap the value of `angles_rad` to the range [min_val, max_val]."""
  max_min_diff = max_val - min_val
  return min_val + tf.math.floormod(angles_rad + max_val, max_min_diff)


def generate_anchors(min_level: int, max_level: int, image_size: Tuple[int],
                     anchor_sizes: List[Tuple[float]]) -> Mapping[str, Any]:
  """Generate anchor boxes without scale to level stride.

  Args:
    min_level: integer number of minimum level of the output.
    max_level: integer number of maximum level of the output.
    image_size: a tuple (image_height, image_width).
    anchor_sizes: a list of tuples, each tuple is (anchor_length, anchor_width).

  Returns:
    boxes_all: a {level: boxes_i} dict, each boxes_i is a [h_i, w_i, 4] tensor
      for boxes at level i, each box is (ymin, xmin, ymax, xmax).

  Notations:
    k: length of anchor_sizes, the number of indicated anchors.
    w: the image width at a specific level.
    h: the image height at a specifc level.
  """
  # Prepare k anchors' lengths and widths
  k = len(anchor_sizes)
  # (k,)
  anchor_lengths = []
  anchor_widths = []
  for anchor_size in anchor_sizes:
    anchor_lengths.append(anchor_size[0])
    anchor_widths.append(anchor_size[1])
  anchor_lengths = tf.convert_to_tensor(anchor_lengths, dtype=tf.float32)
  anchor_widths = tf.convert_to_tensor(anchor_widths, dtype=tf.float32)
  # (1, 1, k)
  half_anchor_lengths = tf.reshape(0.5 * anchor_lengths, [1, 1, k])
  half_anchor_widths = tf.reshape(0.5 * anchor_widths, [1, 1, k])

  boxes_all = collections.OrderedDict()
  for level in range(min_level, max_level + 1):
    # Generate anchor boxes for this level with stride.
    boxes_i = []
    stride = 2 ** level
    # (w,)
    x = tf.range(stride / 2, image_size[1], stride, dtype=tf.float32)
    # (h,)
    y = tf.range(stride / 2, image_size[0], stride, dtype=tf.float32)
    # (h, w)
    xv, yv = tf.meshgrid(x, y)
    # (h, w, 1)
    xv = tf.expand_dims(xv, axis=-1)
    yv = tf.expand_dims(yv, axis=-1)
    # (h, w, k, 1)
    y_min = tf.expand_dims(yv - half_anchor_widths, axis=-1)
    y_max = tf.expand_dims(yv + half_anchor_widths, axis=-1)
    x_min = tf.expand_dims(xv - half_anchor_lengths, axis=-1)
    x_max = tf.expand_dims(xv + half_anchor_lengths, axis=-1)
    # (h, w, k, 4)
    boxes_i = tf.concat([y_min, x_min, y_max, x_max], axis=-1)
    # [h, w, k * 4]
    shape = boxes_i.shape.as_list()
    boxes_i = tf.reshape(boxes_i, [shape[0], shape[1], shape[2] * shape[3]])

    boxes_all[str(level)] = boxes_i
  return boxes_all
