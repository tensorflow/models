# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions used by target assigner."""

import tensorflow.compat.v1 as tf

from object_detection.utils import shape_utils


def image_shape_to_grids(height, width):
  """Computes xy-grids given the shape of the image.

  Args:
    height: The height of the image.
    width: The width of the image.

  Returns:
    A tuple of two tensors:
      y_grid: A float tensor with shape [height, width] representing the
        y-coordinate of each pixel grid.
      x_grid: A float tensor with shape [height, width] representing the
        x-coordinate of each pixel grid.
  """
  out_height = tf.cast(height, tf.float32)
  out_width = tf.cast(width, tf.float32)
  x_range = tf.range(out_width, dtype=tf.float32)
  y_range = tf.range(out_height, dtype=tf.float32)
  x_grid, y_grid = tf.meshgrid(x_range, y_range, indexing='xy')
  return (y_grid, x_grid)


def coordinates_to_heatmap(y_grid,
                           x_grid,
                           y_coordinates,
                           x_coordinates,
                           sigma,
                           channel_onehot,
                           channel_weights=None):
  """Returns the heatmap targets from a set of point coordinates.

  This function maps a set of point coordinates to the output heatmap image
  applied using a Gaussian kernel. Note that this function be can used by both
  object detection and keypoint estimation tasks. For object detection, the
  "channel" refers to the object class. For keypoint estimation, the "channel"
  refers to the number of keypoint types.

  Args:
    y_grid: A 2D tensor with shape [height, width] which contains the grid
      y-coordinates given in the (output) image dimensions.
    x_grid: A 2D tensor with shape [height, width] which contains the grid
      x-coordinates given in the (output) image dimensions.
    y_coordinates: A 1D tensor with shape [num_instances] representing the
      y-coordinates of the instances in the output space coordinates.
    x_coordinates: A 1D tensor with shape [num_instances] representing the
      x-coordinates of the instances in the output space coordinates.
    sigma: A 1D tensor with shape [num_instances] representing the standard
      deviation of the Gaussian kernel to be applied to the point.
    channel_onehot: A 2D tensor with shape [num_instances, num_channels]
      representing the one-hot encoded channel labels for each point.
    channel_weights: A 1D tensor with shape [num_instances] corresponding to the
      weight of each instance.

  Returns:
    heatmap: A tensor of size [height, width, num_channels] representing the
      heatmap. Output (height, width) match the dimensions of the input grids.
  """
  num_instances, num_channels = (
      shape_utils.combined_static_and_dynamic_shape(channel_onehot))

  x_grid = tf.expand_dims(x_grid, 2)
  y_grid = tf.expand_dims(y_grid, 2)
  # The raw center coordinates in the output space.
  x_diff = x_grid - tf.math.floor(x_coordinates)
  y_diff = y_grid - tf.math.floor(y_coordinates)
  squared_distance = x_diff**2 + y_diff**2

  gaussian_map = tf.exp(-squared_distance / (2 * sigma * sigma))

  reshaped_gaussian_map = tf.expand_dims(gaussian_map, axis=-1)
  reshaped_channel_onehot = tf.reshape(channel_onehot,
                                       (1, 1, num_instances, num_channels))
  gaussian_per_box_per_class_map = (
      reshaped_gaussian_map * reshaped_channel_onehot)

  if channel_weights is not None:
    reshaped_weights = tf.reshape(channel_weights, (1, 1, num_instances, 1))
    gaussian_per_box_per_class_map *= reshaped_weights

  # Take maximum along the "instance" dimension so that all per-instance
  # heatmaps of the same class are merged together.
  heatmap = tf.reduce_max(gaussian_per_box_per_class_map, axis=2)

  # Maximum of an empty tensor is -inf, the following is to avoid that.
  heatmap = tf.maximum(heatmap, 0)

  return heatmap


def compute_floor_offsets_with_indices(y_source,
                                       x_source,
                                       y_target=None,
                                       x_target=None):
  """Computes offsets from floored source(floored) to target coordinates.

  This function computes the offsets from source coordinates ("floored" as if
  they were put on the grids) to target coordinates. Note that the input
  coordinates should be the "absolute" coordinates in terms of the output image
  dimensions as opposed to the normalized coordinates (i.e. values in [0, 1]).
  If the input y and x source have the second dimension (representing the
  neighboring pixels), then the offsets are computed from each of the
  neighboring pixels to their corresponding target (first dimension).

  Args:
    y_source: A tensor with shape [num_points] (or [num_points, num_neighbors])
      representing the absolute y-coordinates (in the output image space) of the
      source points.
    x_source: A tensor with shape [num_points] (or [num_points, num_neighbors])
      representing the absolute x-coordinates (in the output image space) of the
      source points.
    y_target: A tensor with shape [num_points] representing the absolute
      y-coordinates (in the output image space) of the target points. If not
      provided, then y_source is used as the targets.
    x_target: A tensor with shape [num_points] representing the absolute
      x-coordinates (in the output image space) of the target points. If not
      provided, then x_source is used as the targets.

  Returns:
    A tuple of two tensors:
      offsets: A tensor with shape [num_points, 2] (or
        [num_points, num_neighbors, 2]) representing the offsets of each input
        point.
      indices: A tensor with shape [num_points, 2] (or
        [num_points, num_neighbors, 2]) representing the indices of where the
        offsets should be retrieved in the output image dimension space.

  Raise:
    ValueError: source and target shapes have unexpected values.
  """
  y_source_floored = tf.floor(y_source)
  x_source_floored = tf.floor(x_source)

  source_shape = shape_utils.combined_static_and_dynamic_shape(y_source)
  if y_target is None and x_target is None:
    y_target = y_source
    x_target = x_source
  else:
    target_shape = shape_utils.combined_static_and_dynamic_shape(y_target)
    if len(source_shape) == 2 and len(target_shape) == 1:
      _, num_neighbors = source_shape
      y_target = tf.tile(
          tf.expand_dims(y_target, -1), multiples=[1, num_neighbors])
      x_target = tf.tile(
          tf.expand_dims(x_target, -1), multiples=[1, num_neighbors])
    elif source_shape != target_shape:
      raise ValueError('Inconsistent source and target shape.')

  y_offset = y_target - y_source_floored
  x_offset = x_target - x_source_floored

  y_source_indices = tf.cast(y_source_floored, tf.int32)
  x_source_indices = tf.cast(x_source_floored, tf.int32)

  indices = tf.stack([y_source_indices, x_source_indices], axis=-1)
  offsets = tf.stack([y_offset, x_offset], axis=-1)
  return offsets, indices


def get_valid_keypoint_mask_for_class(keypoint_coordinates,
                                      class_id,
                                      class_onehot,
                                      class_weights=None,
                                      keypoint_indices=None):
  """Mask keypoints by their class ids and indices.

  For a given task, we may want to only consider a subset of instances or
  keypoints. This function is used to provide the mask (in terms of weights) to
  mark those elements which should be considered based on the classes of the
  instances and optionally, their keypoint indices. Note that the NaN values
  in the keypoints will also be masked out.

  Args:
    keypoint_coordinates: A float tensor with shape [num_instances,
      num_keypoints, 2] which contains the coordinates of each keypoint.
    class_id: An integer representing the target class id to be selected.
    class_onehot: A 2D tensor of shape [num_instances, num_classes] repesents
      the onehot (or k-hot) encoding of the class for each instance.
    class_weights: A 1D tensor of shape [num_instances] repesents the weight of
      each instance. If not provided, all instances are weighted equally.
    keypoint_indices: A list of integers representing the keypoint indices used
      to select the values on the keypoint dimension. If provided, the output
      dimension will be [num_instances, len(keypoint_indices)]

  Returns:
    A tuple of tensors:
      mask: A float tensor of shape [num_instances, K], where K is num_keypoints
        or len(keypoint_indices) if provided. The tensor has values either 0 or
        1 indicating whether an element in the input keypoints should be used.
      keypoints_nan_to_zeros: Same as input keypoints with the NaN values
        replaced by zeros and selected columns corresponding to the
        keypoint_indices (if provided). The shape of this tensor will always be
        the same as the output mask.
  """
  num_keypoints = tf.shape(keypoint_coordinates)[1]
  class_mask = class_onehot[:, class_id]
  reshaped_class_mask = tf.tile(
      tf.expand_dims(class_mask, axis=-1), multiples=[1, num_keypoints])
  not_nan = tf.math.logical_not(tf.math.is_nan(keypoint_coordinates))
  mask = reshaped_class_mask * tf.cast(not_nan[:, :, 0], dtype=tf.float32)
  keypoints_nan_to_zeros = tf.where(not_nan, keypoint_coordinates,
                                    tf.zeros_like(keypoint_coordinates))
  if class_weights is not None:
    reshaped_class_weight = tf.tile(
        tf.expand_dims(class_weights, axis=-1), multiples=[1, num_keypoints])
    mask = mask * reshaped_class_weight

  if keypoint_indices is not None:
    mask = tf.gather(mask, indices=keypoint_indices, axis=1)
    keypoints_nan_to_zeros = tf.gather(
        keypoints_nan_to_zeros, indices=keypoint_indices, axis=1)
  return mask, keypoints_nan_to_zeros


def blackout_pixel_weights_by_box_regions(height, width, boxes, blackout):
  """Blackout the pixel weights in the target box regions.

  This function is used to generate the pixel weight mask (usually in the output
  image dimension). The mask is to ignore some regions when computing loss.

  Args:
    height: int, height of the (output) image.
    width: int, width of the (output) image.
    boxes: A float tensor with shape [num_instances, 4] indicating the
      coordinates of the four corners of the boxes.
    blackout: A boolean tensor with shape [num_instances] indicating whether to
      blackout (zero-out) the weights within the box regions.

  Returns:
    A float tensor with shape [height, width] where all values within the
    regions of the blackout boxes are 0.0 and 1.0 else where.
  """
  num_instances, _ = shape_utils.combined_static_and_dynamic_shape(boxes)
  # If no annotation instance is provided, return all ones (instead of
  # unexpected values) to avoid NaN loss value.
  if num_instances == 0:
    return tf.ones([height, width], dtype=tf.float32)

  (y_grid, x_grid) = image_shape_to_grids(height, width)
  y_grid = tf.expand_dims(y_grid, axis=0)
  x_grid = tf.expand_dims(x_grid, axis=0)
  y_min = tf.expand_dims(boxes[:, 0:1], axis=-1)
  x_min = tf.expand_dims(boxes[:, 1:2], axis=-1)
  y_max = tf.expand_dims(boxes[:, 2:3], axis=-1)
  x_max = tf.expand_dims(boxes[:, 3:], axis=-1)

  # Make the mask with all 1.0 in the box regions.
  # Shape: [num_instances, height, width]
  in_boxes = tf.cast(
      tf.logical_and(
          tf.logical_and(y_grid >= y_min, y_grid <= y_max),
          tf.logical_and(x_grid >= x_min, x_grid <= x_max)),
      dtype=tf.float32)

  # Shape: [num_instances, height, width]
  blackout = tf.tile(
      tf.expand_dims(tf.expand_dims(blackout, axis=-1), axis=-1),
      [1, height, width])

  # Select only the boxes specified by blackout.
  selected_in_boxes = tf.where(blackout, in_boxes, tf.zeros_like(in_boxes))
  out_boxes = tf.reduce_max(selected_in_boxes, axis=0)
  out_boxes = tf.ones_like(out_boxes) - out_boxes
  return out_boxes


def _get_yx_indices_offset_by_radius(radius):
  """Gets the y and x index offsets that are within the radius."""
  y_offsets = []
  x_offsets = []
  for y_offset in range(-radius, radius + 1, 1):
    for x_offset in range(-radius, radius + 1, 1):
      if x_offset ** 2 + y_offset ** 2 <= radius ** 2:
        y_offsets.append(y_offset)
        x_offsets.append(x_offset)
  return (tf.constant(y_offsets, dtype=tf.float32),
          tf.constant(x_offsets, dtype=tf.float32))


def get_surrounding_grids(height, width, y_coordinates, x_coordinates, radius):
  """Gets the indices of the surrounding pixels of the input y, x coordinates.

  This function returns the pixel indices corresponding to the (floor of the)
  input coordinates and their surrounding pixels within the radius. If the
  radius is set to 0, then only the pixels that correspond to the floor of the
  coordinates will be returned. If the radius is larger than 0, then all of the
  pixels within the radius of the "floor pixels" will also be returned. For
  example, if the input coorindate is [2.1, 3.5] and radius is 1, then the five
  pixel indices will be returned: [2, 3], [1, 3], [2, 2], [2, 4], [3, 3]. Also,
  if the surrounding pixels are outside of valid image region, then the returned
  pixel indices will be [0, 0] and its corresponding "valid" value will be
  False.

  Args:
    height: int, the height of the output image.
    width: int, the width of the output image.
    y_coordinates: A tensor with shape [num_points] representing the absolute
      y-coordinates (in the output image space) of the points.
    x_coordinates: A tensor with shape [num_points] representing the absolute
      x-coordinates (in the output image space) of the points.
    radius: int, the radius of the neighboring pixels to be considered and
      returned. If set to 0, then only the pixel indices corresponding to the
      floor of the input coordinates will be returned.

  Returns:
    A tuple of three tensors:
      y_indices: A [num_points, num_neighbors] float tensor representing the
        pixel y indices corresponding to the input points within radius. The
        "num_neighbors" is determined by the size of the radius.
      x_indices: A [num_points, num_neighbors] float tensor representing the
        pixel x indices corresponding to the input points within radius. The
        "num_neighbors" is determined by the size of the radius.
      valid: A [num_points, num_neighbors] boolean tensor representing whether
        each returned index is in valid image region or not.
  """
  # Floored y, x: [num_points, 1].
  y_center = tf.expand_dims(tf.math.floor(y_coordinates), axis=-1)
  x_center = tf.expand_dims(tf.math.floor(x_coordinates), axis=-1)
  y_offsets, x_offsets = _get_yx_indices_offset_by_radius(radius)
  # Indices offsets: [1, num_neighbors].
  y_offsets = tf.expand_dims(y_offsets, axis=0)
  x_offsets = tf.expand_dims(x_offsets, axis=0)

  # Floor + offsets: [num_points, num_neighbors].
  y_output = y_center + y_offsets
  x_output = x_center + x_offsets
  default_output = tf.zeros_like(y_output)
  valid = tf.logical_and(
      tf.logical_and(x_output >= 0, x_output < width),
      tf.logical_and(y_output >= 0, y_output < height))
  y_output = tf.where(valid, y_output, default_output)
  x_output = tf.where(valid, x_output, default_output)
  return (y_output, x_output, valid)
