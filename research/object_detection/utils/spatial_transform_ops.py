# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Spatial transformation ops like RoIAlign, CropAndResize."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from object_detection.utils import shape_utils


def _coordinate_vector_1d(start, end, size, align_endpoints):
  """Generates uniformly spaced coordinate vector.

  Args:
    start: A float tensor of shape [batch, num_boxes] indicating start values.
    end: A float tensor of shape [batch, num_boxes] indicating end values.
    size: Number of points in coordinate vector.
    align_endpoints: Whether to align first and last points exactly to
      endpoints.

  Returns:
    A 3D float tensor of shape [batch, num_boxes, size] containing grid
    coordinates.
  """
  start = tf.expand_dims(start, -1)
  end = tf.expand_dims(end, -1)
  length = end - start
  if align_endpoints:
    relative_grid_spacing = tf.linspace(0.0, 1.0, size)
    offset = 0 if size > 1 else length / 2
  else:
    relative_grid_spacing = tf.linspace(0.0, 1.0, size + 1)[:-1]
    offset = length / (2 * size)
  relative_grid_spacing = tf.reshape(relative_grid_spacing, [1, 1, size])
  relative_grid_spacing = tf.cast(relative_grid_spacing, dtype=start.dtype)
  absolute_grid = start + offset + relative_grid_spacing * length
  return absolute_grid


def box_grid_coordinate_vectors(boxes, size_y, size_x, align_corners=False):
  """Generates coordinate vectors for a `size x size` grid in boxes.

  Each box is subdivided uniformly into a grid consisting of size x size
  rectangular cells. This function returns coordinate vectors describing
  the center of each cell.

  If `align_corners` is true, grid points are uniformly spread such that the
  corner points on the grid exactly overlap corners of the boxes.

  Note that output coordinates are expressed in the same coordinate frame as
  input boxes.

  Args:
    boxes: A float tensor of shape [batch, num_boxes, 4] containing boxes of the
      form [ymin, xmin, ymax, xmax].
    size_y: Size of the grid in y axis.
    size_x: Size of the grid in x axis.
    align_corners: Whether to align the corner grid points exactly with box
      corners.

  Returns:
    box_grid_y: A float tensor of shape [batch, num_boxes, size_y] containing y
      coordinates for grid points.
    box_grid_x: A float tensor of shape [batch, num_boxes, size_x] containing x
      coordinates for grid points.
  """
  ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=-1)
  box_grid_y = _coordinate_vector_1d(ymin, ymax, size_y, align_corners)
  box_grid_x = _coordinate_vector_1d(xmin, xmax, size_x, align_corners)
  return box_grid_y, box_grid_x


def feature_grid_coordinate_vectors(box_grid_y, box_grid_x):
  """Returns feature grid point coordinate vectors for bilinear interpolation.

  Box grid is specified in absolute coordinate system with origin at left top
  (0, 0). The returned coordinate vectors contain 0-based feature point indices.

  This function snaps each point in the box grid to nearest 4 points on the
  feature map.

  In this function we also follow the convention of treating feature pixels as
  point objects with no spatial extent.

  Args:
    box_grid_y: A float tensor of shape [batch, num_boxes, size] containing y
      coordinate vector of the box grid.
    box_grid_x: A float tensor of shape [batch, num_boxes, size] containing x
      coordinate vector of the box grid.

  Returns:
    feature_grid_y0: An int32 tensor of shape [batch, num_boxes, size]
      containing y coordinate vector for the top neighbors.
    feature_grid_x0: A int32 tensor of shape [batch, num_boxes, size]
      containing x coordinate vector for the left neighbors.
    feature_grid_y1: A int32 tensor of shape [batch, num_boxes, size]
      containing y coordinate vector for the bottom neighbors.
    feature_grid_x1: A int32 tensor of shape [batch, num_boxes, size]
      containing x coordinate vector for the right neighbors.
  """
  feature_grid_y0 = tf.floor(box_grid_y)
  feature_grid_x0 = tf.floor(box_grid_x)
  feature_grid_y1 = tf.floor(box_grid_y + 1)
  feature_grid_x1 = tf.floor(box_grid_x + 1)
  feature_grid_y0 = tf.cast(feature_grid_y0, dtype=tf.int32)
  feature_grid_y1 = tf.cast(feature_grid_y1, dtype=tf.int32)
  feature_grid_x0 = tf.cast(feature_grid_x0, dtype=tf.int32)
  feature_grid_x1 = tf.cast(feature_grid_x1, dtype=tf.int32)
  return (feature_grid_y0, feature_grid_x0, feature_grid_y1, feature_grid_x1)


def _valid_indicator(feature_grid_y, feature_grid_x, true_feature_shapes):
  """Computes a indicator vector for valid indices.

  Computes an indicator vector which is true for points on feature map and
  false for points off feature map.

  Args:
    feature_grid_y: An int32 tensor of shape [batch, num_boxes, size_y]
      containing y coordinate vector.
    feature_grid_x: An int32 tensor of shape [batch, num_boxes, size_x]
      containing x coordinate vector.
    true_feature_shapes: A int32 tensor of shape [batch, num_boxes, 2]
      containing valid height and width of feature maps. Feature maps are
      assumed to be aligned to the left top corner.

  Returns:
    indices: A 1D bool tensor indicating valid feature indices.
  """
  height = tf.cast(true_feature_shapes[:, :, 0:1], dtype=feature_grid_y.dtype)
  width = tf.cast(true_feature_shapes[:, :, 1:2], dtype=feature_grid_x.dtype)
  valid_indicator = tf.logical_and(
      tf.expand_dims(
          tf.logical_and(feature_grid_y >= 0, tf.less(feature_grid_y, height)),
          3),
      tf.expand_dims(
          tf.logical_and(feature_grid_x >= 0, tf.less(feature_grid_x, width)),
          2))
  return tf.reshape(valid_indicator, [-1])


def ravel_indices(feature_grid_y, feature_grid_x, num_levels, height, width,
                  box_levels):
  """Returns grid indices in a flattened feature map of shape [-1, channels].

  The returned 1-D array can be used to gather feature grid points from a
  feature map that has been flattened from [batch, num_levels, max_height,
  max_width, channels] to [batch * num_levels * max_height * max_width,
  channels].

  Args:
    feature_grid_y: An int32 tensor of shape [batch, num_boxes, size_y]
      containing y coordinate vector.
    feature_grid_x: An int32 tensor of shape [batch, num_boxes, size_x]
      containing x coordinate vector.
    num_levels: Number of feature levels.
    height: An integer indicating the padded height of feature maps.
    width: An integer indicating the padded width of feature maps.
    box_levels: An int32 tensor of shape [batch, num_boxes] indicating
      feature level assigned to each box.

  Returns:
    indices: A 1D int32 tensor containing feature point indices in a flattened
      feature grid.
  """
  num_boxes = tf.shape(feature_grid_y)[1]
  batch_size = tf.shape(feature_grid_y)[0]
  size_y = tf.shape(feature_grid_y)[2]
  size_x = tf.shape(feature_grid_x)[2]
  height_dim_offset = width
  level_dim_offset = height * height_dim_offset
  batch_dim_offset = num_levels * level_dim_offset

  batch_dim_indices = (
      tf.reshape(
          tf.range(batch_size) * batch_dim_offset, [batch_size, 1, 1, 1]) *
      tf.ones([1, num_boxes, size_y, size_x], dtype=tf.int32))
  box_level_indices = (
      tf.reshape(box_levels * level_dim_offset, [batch_size, num_boxes, 1, 1]) *
      tf.ones([1, 1, size_y, size_x], dtype=tf.int32))
  height_indices = (
      tf.reshape(feature_grid_y * height_dim_offset,
                 [batch_size, num_boxes, size_y, 1]) *
      tf.ones([1, 1, 1, size_x], dtype=tf.int32))
  width_indices = (
      tf.reshape(feature_grid_x, [batch_size, num_boxes, 1, size_x])
      * tf.ones([1, 1, size_y, 1], dtype=tf.int32))
  indices = (
      batch_dim_indices + box_level_indices + height_indices + width_indices)
  flattened_indices = tf.reshape(indices, [-1])
  return flattened_indices


def pad_to_max_size(features):
  """Pads features to max height and max width and stacks them up.

  Args:
    features: A list of num_levels 4D float tensors of shape [batch, height_i,
      width_i, channels] containing feature maps.

  Returns:
    stacked_features: A 5D float tensor of shape [batch, num_levels, max_height,
      max_width, channels] containing stacked features.
    true_feature_shapes: A 2D int32 tensor of shape [num_levels, 2] containing
      height and width of the feature maps before padding.
  """
  if len(features) == 1:
    return tf.expand_dims(features[0],
                          1), tf.expand_dims(tf.shape(features[0])[1:3], 0)

  if all([feature.shape.is_fully_defined() for feature in features]):
    heights = [feature.shape[1] for feature in features]
    widths = [feature.shape[2] for feature in features]
    max_height = max(heights)
    max_width = max(widths)
  else:
    heights = [tf.shape(feature)[1] for feature in features]
    widths = [tf.shape(feature)[2] for feature in features]
    max_height = tf.reduce_max(heights)
    max_width = tf.reduce_max(widths)
  features_all = [
      tf.image.pad_to_bounding_box(feature, 0, 0, max_height,
                                   max_width) for feature in features
  ]
  features_all = tf.stack(features_all, axis=1)
  true_feature_shapes = tf.stack([tf.shape(feature)[1:3]
                                  for feature in features])
  return features_all, true_feature_shapes


def _gather_valid_indices(tensor, indices, padding_value=0.0):
  """Gather values for valid indices.

  TODO(rathodv): We can't use ops.gather_with_padding_values due to cyclic
  dependency. Start using it after migrating all users of spatial ops to import
  this module directly rather than util/ops.py

  Args:
    tensor: A tensor to gather valid values from.
    indices: A 1-D int32 tensor containing indices along axis 0 of `tensor`.
      Invalid indices must be marked with -1.
    padding_value: Value to return for invalid indices.

  Returns:
    A tensor sliced based on indices. For indices that are equal to -1, returns
    rows of padding value.
  """
  padded_tensor = tf.concat(
      [
          padding_value *
          tf.ones([1, tf.shape(tensor)[-1]], dtype=tensor.dtype), tensor
      ],
      axis=0,
  )
  # tf.concat gradient op uses tf.where(condition) (which is not
  # supported on TPU) when the inputs to it are tf.IndexedSlices instead of
  # tf.Tensor. Since gradient op for tf.gather returns tf.IndexedSlices,
  # we add a dummy op inbetween tf.concat and tf.gather to ensure tf.concat
  # gradient function gets tf.Tensor inputs and not tf.IndexedSlices.
  padded_tensor *= 1.0
  return tf.gather(padded_tensor, indices + 1)


def multilevel_roi_align(features, boxes, box_levels, output_size,
                         num_samples_per_cell_y=1, num_samples_per_cell_x=1,
                         align_corners=False, extrapolation_value=0.0,
                         scope=None):
  """Applies RoI Align op and returns feature for boxes.

  Given multiple features maps indexed by different levels, and a set of boxes
  where each box is mapped to a certain level, this function selectively crops
  and resizes boxes from the corresponding feature maps.

  We follow the RoI Align technique in https://arxiv.org/pdf/1703.06870.pdf
  figure 3. Specifically, each box is subdivided uniformly into a grid
  consisting of output_size[0] x output_size[1] rectangular cells. Within each
  cell we select `num_points` points uniformly and compute feature values using
  bilinear interpolation. Finally, we average pool the interpolated values in
  each cell to obtain a [output_size[0], output_size[1], channels] feature.

  If `align_corners` is true, sampling points are uniformly spread such that
  corner points exactly overlap corners of the boxes.

  In this function we also follow the convention of treating feature pixels as
  point objects with no spatial extent.

  Args:
    features: A list of 4D float tensors of shape [batch_size, max_height,
      max_width, channels] containing features. Note that each feature map must
      have the same number of channels.
    boxes: A 3D float tensor of shape [batch_size, num_boxes, 4] containing
      boxes of the form [ymin, xmin, ymax, xmax] in normalized coordinates.
    box_levels: A 3D int32 tensor of shape [batch_size, num_boxes]
      representing the feature level index for each box.
    output_size: An list of two integers [size_y, size_x] indicating the output
      feature size for each box.
    num_samples_per_cell_y: Number of grid points to sample along y axis in each
      cell.
    num_samples_per_cell_x: Number of grid points to sample along x axis in each
      cell.
    align_corners: Whether to align the corner grid points exactly with box
      corners.
    extrapolation_value: a float value to use for extrapolation.
    scope: Scope name to use for this op.

  Returns:
    A 5D float tensor of shape [batch_size, num_boxes, output_size[0],
    output_size[1], channels] representing the cropped features.
  """
  with tf.name_scope(scope, 'MultiLevelRoIAlign'):
    features, true_feature_shapes = pad_to_max_size(features)
    batch_size = shape_utils.combined_static_and_dynamic_shape(features)[0]
    num_levels = features.get_shape().as_list()[1]
    max_feature_height = tf.shape(features)[2]
    max_feature_width = tf.shape(features)[3]
    num_filters = features.get_shape().as_list()[4]
    num_boxes = tf.shape(boxes)[1]

    # Convert boxes to absolute co-ordinates.
    true_feature_shapes = tf.cast(true_feature_shapes, dtype=boxes.dtype)
    true_feature_shapes = tf.gather(true_feature_shapes, box_levels)
    boxes *= tf.concat([true_feature_shapes - 1] * 2, axis=-1)

    size_y = output_size[0] * num_samples_per_cell_y
    size_x = output_size[1] * num_samples_per_cell_x
    box_grid_y, box_grid_x = box_grid_coordinate_vectors(
        boxes, size_y=size_y, size_x=size_x, align_corners=align_corners)
    (feature_grid_y0, feature_grid_x0, feature_grid_y1,
     feature_grid_x1) = feature_grid_coordinate_vectors(box_grid_y, box_grid_x)
    feature_grid_y = tf.reshape(
        tf.stack([feature_grid_y0, feature_grid_y1], axis=3),
        [batch_size, num_boxes, -1])
    feature_grid_x = tf.reshape(
        tf.stack([feature_grid_x0, feature_grid_x1], axis=3),
        [batch_size, num_boxes, -1])
    feature_coordinates = ravel_indices(feature_grid_y, feature_grid_x,
                                        num_levels, max_feature_height,
                                        max_feature_width, box_levels)
    valid_indices = _valid_indicator(feature_grid_y, feature_grid_x,
                                     true_feature_shapes)
    feature_coordinates = tf.where(valid_indices, feature_coordinates,
                                   -1 * tf.ones_like(feature_coordinates))
    flattened_features = tf.reshape(features, [-1, num_filters])
    flattened_feature_values = _gather_valid_indices(flattened_features,
                                                     feature_coordinates,
                                                     extrapolation_value)
    features_per_box = tf.reshape(
        flattened_feature_values,
        [batch_size, num_boxes, size_y * 2, size_x * 2, num_filters])

    # Cast tensors into dtype of features.
    box_grid_y = tf.cast(box_grid_y, dtype=features_per_box.dtype)
    box_grid_x = tf.cast(box_grid_x, dtype=features_per_box.dtype)
    feature_grid_y0 = tf.cast(feature_grid_y0, dtype=features_per_box.dtype)
    feature_grid_x0 = tf.cast(feature_grid_x0, dtype=features_per_box.dtype)

    # RoI Align operation is a bilinear interpolation of four
    # neighboring feature points f0, f1, f2, and f3 onto point y, x given by
    # f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
    #                       [f10, f11]]
    #
    # Unrolling the matrix multiplies gives us:
    # f(y, x) = (hy * hx) f00 + (hy * lx) f01 + (ly * hx) f10 + (lx * ly) f11
    # f(y, x) = w00 * f00 + w01 * f01 + w10 * f10 + w11 * f11
    #
    # This can be computed by applying pointwise multiplication and sum_pool in
    # a 2x2 window.
    ly = box_grid_y - feature_grid_y0
    lx = box_grid_x - feature_grid_x0
    hy = 1.0 - ly
    hx = 1.0 - lx

    kernel_y = tf.reshape(
        tf.stack([hy, ly], axis=3), [batch_size, num_boxes, size_y * 2, 1])

    kernel_x = tf.reshape(
        tf.stack([hx, lx], axis=3), [batch_size, num_boxes, 1, size_x * 2])

    # Multiplier 4 is to make tf.nn.avg_pool behave like sum_pool.
    interpolation_kernel = kernel_y * kernel_x * 4

    # Interpolate the gathered features with computed interpolation kernels.
    features_per_box *= tf.expand_dims(interpolation_kernel, axis=4),
    features_per_box = tf.reshape(
        features_per_box,
        [batch_size * num_boxes, size_y * 2, size_x * 2, num_filters])

    # This combines the two pooling operations - sum_pool to perform bilinear
    # interpolation and avg_pool to pool the values in each bin.
    features_per_box = tf.nn.avg_pool(
        features_per_box,
        [1, num_samples_per_cell_y * 2, num_samples_per_cell_x * 2, 1],
        [1, num_samples_per_cell_y * 2, num_samples_per_cell_x * 2, 1], 'VALID')
    features_per_box = tf.reshape(
        features_per_box,
        [batch_size, num_boxes, output_size[0], output_size[1], num_filters])

    return features_per_box


def multilevel_native_crop_and_resize(images, boxes, box_levels,
                                      crop_size, scope=None):
  """Multilevel native crop and resize.

  Same as `multilevel_matmul_crop_and_resize` but uses tf.image.crop_and_resize.

  Args:
    images: A list of 4-D tensor of shape
      [batch, image_height, image_width, depth] representing features of
      different size.
    boxes: A `Tensor` of type `float32`.
      A 3-D tensor of shape `[batch, num_boxes, 4]`. The boxes are specified in
      normalized coordinates and are of the form `[y1, x1, y2, x2]`. A
      normalized coordinate value of `y` is mapped to the image coordinate at
      `y * (image_height - 1)`, so as the `[0, 1]` interval of normalized image
      height is mapped to `[0, image_height - 1] in image height coordinates.
      We do allow y1 > y2, in which case the sampled crop is an up-down flipped
      version of the original image. The width dimension is treated similarly.
      Normalized coordinates outside the `[0, 1]` range are allowed, in which
      case we use `extrapolation_value` to extrapolate the input image values.
    box_levels: A 2-D tensor of shape [batch, num_boxes] representing the level
      of the box.
    crop_size: A list of two integers `[crop_height, crop_width]`. All
      cropped image patches are resized to this size. The aspect ratio of the
      image content is not preserved. Both `crop_height` and `crop_width` need
      to be positive.
    scope: A name for the operation (optional).

  Returns:
    A 5-D float tensor of shape `[batch, num_boxes, crop_height, crop_width,
    depth]`
  """
  if box_levels is None:
    return native_crop_and_resize(images[0], boxes, crop_size, scope)
  with tf.name_scope('MultiLevelNativeCropAndResize'):
    cropped_feature_list = []
    for level, image in enumerate(images):
      # For each level, crop the feature according to all boxes
      # set the cropped feature not at this level to 0 tensor.
      # Consider more efficient way of computing cropped features.
      cropped = native_crop_and_resize(image, boxes, crop_size, scope)
      cond = tf.tile(
          tf.equal(box_levels, level)[:, :, tf.newaxis],
          [1, 1] + [tf.math.reduce_prod(cropped.shape.as_list()[2:])])
      cond = tf.reshape(cond, cropped.shape)
      cropped_final = tf.where(cond, cropped, tf.zeros_like(cropped))
      cropped_feature_list.append(cropped_final)
    return tf.math.reduce_sum(cropped_feature_list, axis=0)


def native_crop_and_resize(image, boxes, crop_size, scope=None):
  """Same as `matmul_crop_and_resize` but uses tf.image.crop_and_resize."""
  def get_box_inds(proposals):
    proposals_shape = proposals.shape.as_list()
    if any(dim is None for dim in proposals_shape):
      proposals_shape = tf.shape(proposals)
    ones_mat = tf.ones(proposals_shape[:2], dtype=tf.int32)
    multiplier = tf.expand_dims(
        tf.range(start=0, limit=proposals_shape[0]), 1)
    return tf.reshape(ones_mat * multiplier, [-1])

  with tf.name_scope(scope, 'CropAndResize'):
    cropped_regions = tf.image.crop_and_resize(
        image, tf.reshape(boxes, [-1] + boxes.shape.as_list()[2:]),
        get_box_inds(boxes), crop_size)
    final_shape = tf.concat([tf.shape(boxes)[:2],
                             tf.shape(cropped_regions)[1:]], axis=0)
    return tf.reshape(cropped_regions, final_shape)


def multilevel_matmul_crop_and_resize(images, boxes, box_levels, crop_size,
                                      extrapolation_value=0.0, scope=None):
  """Multilevel matmul crop and resize.

  Same as `matmul_crop_and_resize` but crop images according to box levels.

  Args:
    images: A list of 4-D tensor of shape
      [batch, image_height, image_width, depth] representing features of
      different size.
    boxes: A `Tensor` of type `float32` or 'bfloat16'.
      A 3-D tensor of shape `[batch, num_boxes, 4]`. The boxes are specified in
      normalized coordinates and are of the form `[y1, x1, y2, x2]`. A
      normalized coordinate value of `y` is mapped to the image coordinate at
      `y * (image_height - 1)`, so as the `[0, 1]` interval of normalized image
      height is mapped to `[0, image_height - 1] in image height coordinates.
      We do allow y1 > y2, in which case the sampled crop is an up-down flipped
      version of the original image. The width dimension is treated similarly.
      Normalized coordinates outside the `[0, 1]` range are allowed, in which
      case we use `extrapolation_value` to extrapolate the input image values.
    box_levels: A 2-D tensor of shape [batch, num_boxes] representing the level
      of the box.
    crop_size: A list of two integers `[crop_height, crop_width]`. All
      cropped image patches are resized to this size. The aspect ratio of the
      image content is not preserved. Both `crop_height` and `crop_width` need
      to be positive.
    extrapolation_value: A float value to use for extrapolation.
    scope: A name for the operation (optional).

  Returns:
    A 5-D float tensor of shape `[batch, num_boxes, crop_height, crop_width,
    depth]`
  """
  with tf.name_scope(scope, 'MultiLevelMatMulCropAndResize'):
    if box_levels is None:
      box_levels = tf.zeros(tf.shape(boxes)[:2], dtype=tf.int32)
    return multilevel_roi_align(images,
                                boxes,
                                box_levels,
                                crop_size,
                                align_corners=True,
                                extrapolation_value=extrapolation_value)


def matmul_crop_and_resize(image, boxes, crop_size, extrapolation_value=0.0,
                           scope=None):
  """Matrix multiplication based implementation of the crop and resize op.

  Extracts crops from the input image tensor and bilinearly resizes them
  (possibly with aspect ratio change) to a common output size specified by
  crop_size. This is more general than the crop_to_bounding_box op which
  extracts a fixed size slice from the input image and does not allow
  resizing or aspect ratio change.

  Returns a tensor with crops from the input image at positions defined at
  the bounding box locations in boxes. The cropped boxes are all resized
  (with bilinear interpolation) to a fixed size = `[crop_height, crop_width]`.
  The result is a 5-D tensor `[batch, num_boxes, crop_height, crop_width,
  depth]`.

  Note that this operation is meant to replicate the behavior of the standard
  tf.image.crop_and_resize operation but there are a few differences.
  Specifically:
    1) There is no `box_indices` argument --- to run this op on multiple images,
      one must currently call this op independently on each image.
    2) The `crop_size` parameter is assumed to be statically defined.
      Moreover, the number of boxes must be strictly nonzero.

  Args:
    image: A `Tensor`. Must be one of the following types: `uint8`, `int8`,
      `int16`, `int32`, `int64`, `half`, 'bfloat16', `float32`, `float64`.
      A 4-D tensor of shape `[batch, image_height, image_width, depth]`.
      Both `image_height` and `image_width` need to be positive.
    boxes: A `Tensor` of type `float32` or 'bfloat16'.
      A 3-D tensor of shape `[batch, num_boxes, 4]`. The boxes are specified in
      normalized coordinates and are of the form `[y1, x1, y2, x2]`. A
      normalized coordinate value of `y` is mapped to the image coordinate at
      `y * (image_height - 1)`, so as the `[0, 1]` interval of normalized image
      height is mapped to `[0, image_height - 1] in image height coordinates.
      We do allow y1 > y2, in which case the sampled crop is an up-down flipped
      version of the original image. The width dimension is treated similarly.
      Normalized coordinates outside the `[0, 1]` range are allowed, in which
      case we use `extrapolation_value` to extrapolate the input image values.
    crop_size: A list of two integers `[crop_height, crop_width]`. All
      cropped image patches are resized to this size. The aspect ratio of the
      image content is not preserved. Both `crop_height` and `crop_width` need
      to be positive.
    extrapolation_value: a float value to use for extrapolation.
    scope: A name for the operation (optional).

  Returns:
    A 5-D tensor of shape `[batch, num_boxes, crop_height, crop_width, depth]`
  """
  with tf.name_scope(scope, 'MatMulCropAndResize'):
    box_levels = tf.zeros(tf.shape(boxes)[:2], dtype=tf.int32)
    return multilevel_roi_align([image],
                                boxes,
                                box_levels,
                                crop_size,
                                align_corners=True,
                                extrapolation_value=extrapolation_value)
