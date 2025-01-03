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

from object_detection.core import box_list
from object_detection.core import box_list_ops
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


def _coordinates_to_heatmap_dense(y_grid, x_grid, y_coordinates, x_coordinates,
                                  sigma, channel_onehot, channel_weights=None):
  """Dense version of coordinates to heatmap that uses an outer product."""
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

  return tf.stop_gradient(heatmap)


def _coordinates_to_heatmap_sparse(y_grid, x_grid, y_coordinates, x_coordinates,
                                   sigma, channel_onehot, channel_weights=None):
  """Sparse version of coordinates to heatmap using tf.scatter."""

  if not hasattr(tf, 'tensor_scatter_nd_max'):
    raise RuntimeError(
        ('Please upgrade tensowflow to use `tensor_scatter_nd_max` or set '
         'compute_heatmap_sparse=False'))
  _, num_channels = (
      shape_utils.combined_static_and_dynamic_shape(channel_onehot))

  height, width = shape_utils.combined_static_and_dynamic_shape(y_grid)
  x_grid = tf.expand_dims(x_grid, 2)
  y_grid = tf.expand_dims(y_grid, 2)
  # The raw center coordinates in the output space.
  x_diff = x_grid - tf.math.floor(x_coordinates)
  y_diff = y_grid - tf.math.floor(y_coordinates)
  squared_distance = x_diff**2 + y_diff**2

  gaussian_map = tf.exp(-squared_distance / (2 * sigma * sigma))

  if channel_weights is not None:
    gaussian_map = gaussian_map * channel_weights[tf.newaxis, tf.newaxis, :]

  channel_indices = tf.argmax(channel_onehot, axis=1)

  channel_indices = channel_indices[:, tf.newaxis]
  heatmap_init = tf.zeros((num_channels, height, width))

  gaussian_map = tf.transpose(gaussian_map, (2, 0, 1))
  heatmap = tf.tensor_scatter_nd_max(
      heatmap_init, channel_indices, gaussian_map)

  # Maximum of an empty tensor is -inf, the following is to avoid that.
  heatmap = tf.maximum(heatmap, 0)

  return tf.stop_gradient(tf.transpose(heatmap, (1, 2, 0)))


def coordinates_to_heatmap(y_grid,
                           x_grid,
                           y_coordinates,
                           x_coordinates,
                           sigma,
                           channel_onehot,
                           channel_weights=None,
                           sparse=False):
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
    sparse: bool, indicating whether or not to use the sparse implementation
      of the function. The sparse version scales better with number of channels,
      but in some cases is known to cause OOM error. See (b/170989061).

  Returns:
    heatmap: A tensor of size [height, width, num_channels] representing the
      heatmap. Output (height, width) match the dimensions of the input grids.
  """

  if sparse:
    return _coordinates_to_heatmap_sparse(
        y_grid, x_grid, y_coordinates, x_coordinates, sigma, channel_onehot,
        channel_weights)
  else:
    return _coordinates_to_heatmap_dense(
        y_grid, x_grid, y_coordinates, x_coordinates, sigma, channel_onehot,
        channel_weights)


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


def coordinates_to_iou(y_grid, x_grid, blist,
                       channels_onehot, weights=None):
  """Computes a per-pixel IoU with groundtruth boxes.

  At each pixel, we return the IoU assuming that we predicted the
  ideal height and width for the box at that location.

  Args:
   y_grid: A 2D tensor with shape [height, width] which contains the grid
      y-coordinates given in the (output) image dimensions.
    x_grid: A 2D tensor with shape [height, width] which contains the grid
      x-coordinates given in the (output) image dimensions.
    blist: A BoxList object with `num_instances` number of boxes.
    channels_onehot: A 2D tensor with shape [num_instances, num_channels]
      representing the one-hot encoded channel labels for each point.
    weights: A 1D tensor with shape [num_instances] corresponding to the
      weight of each instance.

  Returns:
    iou_heatmap: A [height, width, num_channels] shapes float tensor denoting
      the IoU based heatmap.
  """

  image_height, image_width = tf.shape(y_grid)[0], tf.shape(y_grid)[1]
  num_pixels = image_height * image_width
  _, _, height, width = blist.get_center_coordinates_and_sizes()
  num_boxes = tf.shape(height)[0]

  per_pixel_ymin = (y_grid[tf.newaxis, :, :] -
                    (height[:, tf.newaxis, tf.newaxis] / 2.0))
  per_pixel_xmin = (x_grid[tf.newaxis, :, :] -
                    (width[:, tf.newaxis, tf.newaxis] / 2.0))
  per_pixel_ymax = (y_grid[tf.newaxis, :, :] +
                    (height[:, tf.newaxis, tf.newaxis] / 2.0))
  per_pixel_xmax = (x_grid[tf.newaxis, :, :] +
                    (width[:, tf.newaxis, tf.newaxis] / 2.0))

  # [num_boxes, height, width] -> [num_boxes * height * width]
  per_pixel_ymin = tf.reshape(
      per_pixel_ymin, [num_pixels * num_boxes])
  per_pixel_xmin = tf.reshape(
      per_pixel_xmin, [num_pixels * num_boxes])
  per_pixel_ymax = tf.reshape(
      per_pixel_ymax, [num_pixels * num_boxes])
  per_pixel_xmax = tf.reshape(
      per_pixel_xmax, [num_pixels * num_boxes])
  per_pixel_blist = box_list.BoxList(
      tf.stack([per_pixel_ymin, per_pixel_xmin,
                per_pixel_ymax, per_pixel_xmax], axis=1))

  target_boxes = tf.tile(
      blist.get()[:, tf.newaxis, :], [1, num_pixels, 1])
  # [num_boxes, height * width, 4] -> [num_boxes * height * wdith, 4]
  target_boxes = tf.reshape(target_boxes,
                            [num_pixels * num_boxes, 4])
  target_blist = box_list.BoxList(target_boxes)

  ious = box_list_ops.matched_iou(target_blist, per_pixel_blist)
  ious = tf.reshape(ious, [num_boxes, image_height, image_width])
  per_class_iou = (
      ious[:, :, :, tf.newaxis] *
      channels_onehot[:, tf.newaxis, tf.newaxis, :])

  if weights is not None:
    per_class_iou = (
        per_class_iou * weights[:, tf.newaxis, tf.newaxis, tf.newaxis])

  per_class_iou = tf.maximum(per_class_iou, 0.0)
  return tf.reduce_max(per_class_iou, axis=0)


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


def blackout_pixel_weights_by_box_regions(height, width, boxes, blackout,
                                          weights=None,
                                          boxes_scale=1.0):
  """Apply weights at pixel locations.

  This function is used to generate the pixel weight mask (usually in the output
  image dimension). The mask is to ignore some regions when computing loss.

  Weights are applied as follows:
  - Any region outside of a box gets the default weight 1.0
  - Any box for which an explicit weight is specifed gets that weight. If
    multiple boxes overlap, the maximum of the weights is applied.
  - Any box for which blackout=True is specified will get a weight of 0.0,
    regardless of whether an equivalent non-zero weight is specified. Also, the
    blackout region takes precedence over other boxes which may overlap with
    non-zero weight.

    Example:
    height = 4
    width = 4
    boxes = [[0., 0., 2., 2.],
             [0., 0., 4., 2.],
             [3., 0., 4., 4.]]
    blackout = [False, False, True]
    weights = [4.0, 3.0, 2.0]
    blackout_pixel_weights_by_box_regions(height, width, boxes, blackout,
                                          weights)
    >> [[4.0, 4.0, 1.0, 1.0],
        [4.0, 4.0, 1.0, 1.0],
        [3.0, 3.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0]]


  Args:
    height: int, height of the (output) image.
    width: int, width of the (output) image.
    boxes: A float tensor with shape [num_instances, 4] indicating the
      coordinates of the four corners of the boxes.
    blackout: A boolean tensor with shape [num_instances] indicating whether to
      blackout (zero-out) the weights within the box regions.
    weights: An optional float32 tensor with shape [num_instances] indicating
      a value to apply in each box region. Note that if blackout=True for a
      given box, the weight will be zero. If None, all weights are assumed to be
      1.
    boxes_scale: The amount to scale the height/width of the boxes before
      constructing the blackout regions. This is often useful to guarantee that
      the proper weight fully covers the object boxes/masks during supervision,
      as shifting might occur during image resizing, network stride, etc.

  Returns:
    A float tensor with shape [height, width] where all values within the
    regions of the blackout boxes are 0.0 and 1.0 (or weights if supplied)
    elsewhere.
  """
  num_instances, _ = shape_utils.combined_static_and_dynamic_shape(boxes)
  # If no annotation instance is provided, return all ones (instead of
  # unexpected values) to avoid NaN loss value.
  if num_instances == 0:
    return tf.ones([height, width], dtype=tf.float32)

  (y_grid, x_grid) = image_shape_to_grids(height, width)
  y_grid = tf.expand_dims(y_grid, axis=0)
  x_grid = tf.expand_dims(x_grid, axis=0)
  boxlist = box_list.BoxList(boxes)
  boxlist = box_list_ops.scale_height_width(
      boxlist, y_scale=boxes_scale, x_scale=boxes_scale)
  boxes = boxlist.get()
  y_min = tf.expand_dims(boxes[:, 0:1], axis=-1)
  x_min = tf.expand_dims(boxes[:, 1:2], axis=-1)
  y_max = tf.expand_dims(boxes[:, 2:3], axis=-1)
  x_max = tf.expand_dims(boxes[:, 3:], axis=-1)

  # Make the mask with all 1.0 in the box regions.
  # Shape: [num_instances, height, width]
  in_boxes = tf.math.logical_and(
      tf.math.logical_and(y_grid >= y_min, y_grid < y_max),
      tf.math.logical_and(x_grid >= x_min, x_grid < x_max))

  if weights is None:
    weights = tf.ones_like(blackout, dtype=tf.float32)

  # Compute a [height, width] tensor with the maximum weight in each box, and
  # 0.0 elsewhere.
  weights_tiled = tf.tile(
      weights[:, tf.newaxis, tf.newaxis], [1, height, width])
  weights_3d = tf.where(in_boxes, weights_tiled,
                        tf.zeros_like(weights_tiled))
  weights_2d = tf.math.maximum(
      tf.math.reduce_max(weights_3d, axis=0), 0.0)

  # Add 1.0 to all regions outside a box.
  weights_2d = tf.where(
      tf.math.reduce_any(in_boxes, axis=0),
      weights_2d,
      tf.ones_like(weights_2d))

  # Now enforce that blackout regions all have zero weights.
  keep_region = tf.cast(tf.math.logical_not(blackout), tf.float32)
  keep_region_tiled = tf.tile(
      keep_region[:, tf.newaxis, tf.newaxis], [1, height, width])
  keep_region_3d = tf.where(in_boxes, keep_region_tiled,
                            tf.ones_like(keep_region_tiled))
  keep_region_2d = tf.math.reduce_min(keep_region_3d, axis=0)
  return weights_2d * keep_region_2d


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
