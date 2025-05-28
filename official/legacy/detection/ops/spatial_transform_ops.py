# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Functions to performa spatial transformation for Tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf, tf_keras

_EPSILON = 1e-8


def nearest_upsampling(data, scale):
  """Nearest neighbor upsampling implementation.

  Args:
    data: A tensor with a shape of [batch, height_in, width_in, channels].
    scale: An integer multiple to scale resolution of input data.

  Returns:
    data_up: A tensor with a shape of
      [batch, height_in*scale, width_in*scale, channels]. Same dtype as input
      data.
  """
  with tf.name_scope('nearest_upsampling'):
    bs, _, _, c = data.get_shape().as_list()
    shape = tf.shape(input=data)
    h = shape[1]
    w = shape[2]
    bs = -1 if bs is None else bs
    # Uses reshape to quickly upsample the input.  The nearest pixel is selected
    # implicitly via broadcasting.
    data = tf.reshape(data, [bs, h, 1, w, 1, c]) * tf.ones(
        [1, 1, scale, 1, scale, 1], dtype=data.dtype)
    return tf.reshape(data, [bs, h * scale, w * scale, c])


def feature_bilinear_interpolation(features, kernel_y, kernel_x):
  """Feature bilinear interpolation.

  The RoIAlign feature f can be computed by bilinear interpolation
  of four neighboring feature points f0, f1, f2, and f3.

  f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
                        [f10, f11]]
  f(y, x) = (hy*hx)f00 + (hy*lx)f01 + (ly*hx)f10 + (lx*ly)f11
  f(y, x) = w00*f00 + w01*f01 + w10*f10 + w11*f11
  kernel_y = [hy, ly]
  kernel_x = [hx, lx]

  Args:
    features: The features are in shape of [batch_size, num_boxes, output_size *
      2, output_size * 2, num_filters].
    kernel_y: Tensor of size [batch_size, boxes, output_size, 2, 1].
    kernel_x: Tensor of size [batch_size, boxes, output_size, 2, 1].

  Returns:
    A 5-D tensor representing feature crop of shape
    [batch_size, num_boxes, output_size, output_size, num_filters].

  """
  (batch_size, num_boxes, output_size, _,
   num_filters) = features.get_shape().as_list()
  output_size = output_size // 2
  kernel_y = tf.reshape(kernel_y, [batch_size, num_boxes, output_size * 2, 1])
  kernel_x = tf.reshape(kernel_x, [batch_size, num_boxes, 1, output_size * 2])
  # Use implicit broadcast to generate the interpolation kernel. The
  # multiplier `4` is for avg pooling.
  interpolation_kernel = kernel_y * kernel_x * 4

  # Interpolate the gathered features with computed interpolation kernels.
  features *= tf.cast(
      tf.expand_dims(interpolation_kernel, axis=-1), dtype=features.dtype)
  features = tf.reshape(
      features,
      [batch_size * num_boxes, output_size * 2, output_size * 2, num_filters])
  features = tf.nn.avg_pool(features, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')
  features = tf.reshape(
      features, [batch_size, num_boxes, output_size, output_size, num_filters])
  return features


def compute_grid_positions(boxes, boundaries, output_size, sample_offset):
  """Compute the grid position w.r.t.

  the corresponding feature map.

  Args:
    boxes: a 3-D tensor of shape [batch_size, num_boxes, 4] encoding the
      information of each box w.r.t. the corresponding feature map.
      boxes[:, :, 0:2] are the grid position in (y, x) (float) of the top-left
      corner of each box. boxes[:, :, 2:4] are the box sizes in (h, w) (float)
        in terms of the number of pixels of the corresponding feature map size.
    boundaries: a 3-D tensor of shape [batch_size, num_boxes, 2] representing
      the boundary (in (y, x)) of the corresponding feature map for each box.
      Any resampled grid points that go beyond the bounary will be clipped.
    output_size: a scalar indicating the output crop size.
    sample_offset: a float number in [0, 1] indicates the subpixel sample offset
      from grid point.

  Returns:
    kernel_y: Tensor of size [batch_size, boxes, output_size, 2, 1].
    kernel_x: Tensor of size [batch_size, boxes, output_size, 2, 1].
    box_grid_y0y1: Tensor of size [batch_size, boxes, output_size, 2]
    box_grid_x0x1: Tensor of size [batch_size, boxes, output_size, 2]
  """
  batch_size, num_boxes, _ = boxes.get_shape().as_list()
  box_grid_x = []
  box_grid_y = []
  for i in range(output_size):
    box_grid_x.append(boxes[:, :, 1] +
                      (i + sample_offset) * boxes[:, :, 3] / output_size)
    box_grid_y.append(boxes[:, :, 0] +
                      (i + sample_offset) * boxes[:, :, 2] / output_size)
  box_grid_x = tf.stack(box_grid_x, axis=2)
  box_grid_y = tf.stack(box_grid_y, axis=2)

  box_grid_y0 = tf.floor(box_grid_y)
  box_grid_x0 = tf.floor(box_grid_x)
  box_grid_x0 = tf.maximum(0., box_grid_x0)
  box_grid_y0 = tf.maximum(0., box_grid_y0)

  box_grid_x0 = tf.minimum(box_grid_x0, tf.expand_dims(boundaries[:, :, 1], -1))
  box_grid_x1 = tf.minimum(box_grid_x0 + 1,
                           tf.expand_dims(boundaries[:, :, 1], -1))
  box_grid_y0 = tf.minimum(box_grid_y0, tf.expand_dims(boundaries[:, :, 0], -1))
  box_grid_y1 = tf.minimum(box_grid_y0 + 1,
                           tf.expand_dims(boundaries[:, :, 0], -1))

  box_gridx0x1 = tf.stack([box_grid_x0, box_grid_x1], axis=-1)
  box_gridy0y1 = tf.stack([box_grid_y0, box_grid_y1], axis=-1)

  # The RoIAlign feature f can be computed by bilinear interpolation of four
  # neighboring feature points f0, f1, f2, and f3.
  # f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
  #                       [f10, f11]]
  # f(y, x) = (hy*hx)f00 + (hy*lx)f01 + (ly*hx)f10 + (lx*ly)f11
  # f(y, x) = w00*f00 + w01*f01 + w10*f10 + w11*f11
  ly = box_grid_y - box_grid_y0
  lx = box_grid_x - box_grid_x0
  hy = 1.0 - ly
  hx = 1.0 - lx
  kernel_y = tf.reshape(
      tf.stack([hy, ly], axis=3), [batch_size, num_boxes, output_size, 2, 1])
  kernel_x = tf.reshape(
      tf.stack([hx, lx], axis=3), [batch_size, num_boxes, output_size, 2, 1])
  return kernel_y, kernel_x, box_gridy0y1, box_gridx0x1


def get_grid_one_hot(box_gridy0y1, box_gridx0x1, feature_height, feature_width):
  """Get grid_one_hot from indices and feature_size."""
  (batch_size, num_boxes, output_size, _) = box_gridx0x1.get_shape().as_list()
  y_indices = tf.cast(
      tf.reshape(box_gridy0y1, [batch_size, num_boxes, output_size, 2]),
      dtype=tf.int32)
  x_indices = tf.cast(
      tf.reshape(box_gridx0x1, [batch_size, num_boxes, output_size, 2]),
      dtype=tf.int32)

  # shape is [batch_size, num_boxes, output_size, 2, height]
  grid_y_one_hot = tf.one_hot(tf.cast(y_indices, tf.int32), feature_height)
  # shape is [batch_size, num_boxes, output_size, 2, width]
  grid_x_one_hot = tf.one_hot(tf.cast(x_indices, tf.int32), feature_width)

  return grid_y_one_hot, grid_x_one_hot


def selective_crop_and_resize(features,
                              boxes,
                              box_levels,
                              boundaries,
                              output_size=7,
                              sample_offset=0.5,
                              use_einsum_gather=False):
  """Crop and resize boxes on a set of feature maps.

  Given multiple features maps indexed by different levels, and a set of boxes
  where each box is mapped to a certain level, it selectively crops and resizes
  boxes from the corresponding feature maps to generate the box features.

  We follow the ROIAlign technique (see https://arxiv.org/pdf/1703.06870.pdf,
  figure 3 for reference). Specifically, for each feature map, we select an
  (output_size, output_size) set of pixels corresponding to the box location,
  and then use bilinear interpolation to select the feature value for each
  pixel.

  For performance, we perform the gather and interpolation on all layers as a
  single operation. In this op the multi-level features are first stacked and
  gathered into [2*output_size, 2*output_size] feature points. Then bilinear
  interpolation is performed on the gathered feature points to generate
  [output_size, output_size] RoIAlign feature map.

  Here is the step-by-step algorithm:
    1. The multi-level features are gathered into a
       [batch_size, num_boxes, output_size*2, output_size*2, num_filters]
       Tensor. The Tensor contains four neighboring feature points for each
       vertice in the output grid.
    2. Compute the interpolation kernel of shape
       [batch_size, num_boxes, output_size*2, output_size*2]. The last 2 axis
       can be seen as stacking 2x2 interpolation kernels for all vertices in the
       output grid.
    3. Element-wise multiply the gathered features and interpolation kernel.
       Then apply 2x2 average pooling to reduce spatial dimension to
       output_size.

  Args:
    features: a 5-D tensor of shape [batch_size, num_levels, max_height,
      max_width, num_filters] where cropping and resizing are based.
    boxes: a 3-D tensor of shape [batch_size, num_boxes, 4] encoding the
      information of each box w.r.t. the corresponding feature map.
      boxes[:, :, 0:2] are the grid position in (y, x) (float) of the top-left
      corner of each box. boxes[:, :, 2:4] are the box sizes in (h, w) (float)
        in terms of the number of pixels of the corresponding feature map size.
    box_levels: a 3-D tensor of shape [batch_size, num_boxes, 1] representing
      the 0-based corresponding feature level index of each box.
    boundaries: a 3-D tensor of shape [batch_size, num_boxes, 2] representing
      the boundary (in (y, x)) of the corresponding feature map for each box.
      Any resampled grid points that go beyond the bounary will be clipped.
    output_size: a scalar indicating the output crop size.
    sample_offset: a float number in [0, 1] indicates the subpixel sample offset
      from grid point.
    use_einsum_gather: use einsum to replace gather or not. Replacing einsum
      with gather can improve performance when feature size is not large, einsum
      is friendly with model partition as well. Gather's performance is better
      when feature size is very large and there are multiple box levels.

  Returns:
    features_per_box: a 5-D tensor of shape
      [batch_size, num_boxes, output_size, output_size, num_filters]
      representing the cropped features.
  """
  (batch_size, num_levels, max_feature_height, max_feature_width,
   num_filters) = features.get_shape().as_list()
  _, num_boxes, _ = boxes.get_shape().as_list()

  kernel_y, kernel_x, box_gridy0y1, box_gridx0x1 = compute_grid_positions(
      boxes, boundaries, output_size, sample_offset)
  x_indices = tf.cast(
      tf.reshape(box_gridx0x1, [batch_size, num_boxes, output_size * 2]),
      dtype=tf.int32)
  y_indices = tf.cast(
      tf.reshape(box_gridy0y1, [batch_size, num_boxes, output_size * 2]),
      dtype=tf.int32)

  if use_einsum_gather:
    # Blinear interpolation is done during the last two gathers:
    #        f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
    #                              [f10, f11]]
    #        [[f00, f01],
    #         [f10, f11]] = tf.einsum(tf.einsum(features, y_one_hot), x_one_hot)
    #       where [hy, ly] and [hx, lx] are the bilinear interpolation kernel.

    # shape is [batch_size, boxes, output_size, 2, 1]
    grid_y_one_hot, grid_x_one_hot = get_grid_one_hot(box_gridy0y1,
                                                      box_gridx0x1,
                                                      max_feature_height,
                                                      max_feature_width)

    # shape is [batch_size, num_boxes, output_size, height]
    grid_y_weight = tf.reduce_sum(
        tf.multiply(grid_y_one_hot, kernel_y), axis=-2)
    # shape is [batch_size, num_boxes, output_size, width]
    grid_x_weight = tf.reduce_sum(
        tf.multiply(grid_x_one_hot, kernel_x), axis=-2)

    # Gather for y_axis.
    # shape is [batch_size, num_boxes, output_size, width, features]
    features_per_box = tf.einsum('bmhwf,bmoh->bmowf', features,
                                 tf.cast(grid_y_weight, features.dtype))
    # Gather for x_axis.
    # shape is [batch_size, num_boxes, output_size, output_size, features]
    features_per_box = tf.einsum('bmhwf,bmow->bmhof', features_per_box,
                                 tf.cast(grid_x_weight, features.dtype))
  else:
    height_dim_offset = max_feature_width
    level_dim_offset = max_feature_height * height_dim_offset
    batch_dim_offset = num_levels * level_dim_offset

    batch_size_offset = tf.tile(
        tf.reshape(
            tf.range(batch_size) * batch_dim_offset, [batch_size, 1, 1, 1]),
        [1, num_boxes, output_size * 2, output_size * 2])
    box_levels_offset = tf.tile(
        tf.reshape(box_levels * level_dim_offset,
                   [batch_size, num_boxes, 1, 1]),
        [1, 1, output_size * 2, output_size * 2])
    y_indices_offset = tf.tile(
        tf.reshape(y_indices * height_dim_offset,
                   [batch_size, num_boxes, output_size * 2, 1]),
        [1, 1, 1, output_size * 2])
    x_indices_offset = tf.tile(
        tf.reshape(x_indices, [batch_size, num_boxes, 1, output_size * 2]),
        [1, 1, output_size * 2, 1])

    indices = tf.reshape(
        batch_size_offset + box_levels_offset + y_indices_offset +
        x_indices_offset, [-1])

    features = tf.reshape(features, [-1, num_filters])
    # TODO(wangtao): replace tf.gather with tf.gather_nd and try to get similar
    # performance.
    features_per_box = tf.reshape(
        tf.gather(features, indices),
        [batch_size, num_boxes, output_size * 2, output_size * 2, num_filters])
    features_per_box = feature_bilinear_interpolation(features_per_box,
                                                      kernel_y, kernel_x)

  return features_per_box


def multilevel_crop_and_resize(features, boxes, output_size=7):
  """Crop and resize on multilevel feature pyramid.

  Generate the (output_size, output_size) set of pixels for each input box
  by first locating the box into the correct feature level, and then cropping
  and resizing it using the correspoding feature map of that level.

  Args:
    features: A dictionary with key as pyramid level and value as features. The
      features are in shape of [batch_size, height_l, width_l, num_filters].
    boxes: A 3-D Tensor of shape [batch_size, num_boxes, 4]. Each row represents
      a box with [y1, x1, y2, x2] in un-normalized coordinates.
    output_size: A scalar to indicate the output crop size.

  Returns:
    A 5-D tensor representing feature crop of shape
    [batch_size, num_boxes, output_size, output_size, num_filters].
  """

  with tf.name_scope('multilevel_crop_and_resize'):
    levels = list(features.keys())
    min_level = min(levels)
    max_level = max(levels)
    batch_size, max_feature_height, max_feature_width, num_filters = (
        features[min_level].get_shape().as_list())
    _, num_boxes, _ = boxes.get_shape().as_list()

    # Stack feature pyramid into a features_all of shape
    # [batch_size, levels, height, width, num_filters].
    features_all = []
    feature_heights = []
    feature_widths = []
    for level in range(min_level, max_level + 1):
      shape = features[level].get_shape().as_list()
      feature_heights.append(shape[1])
      feature_widths.append(shape[2])
      # Concat tensor of [batch_size, height_l * width_l, num_filters] for each
      # levels.
      features_all.append(
          tf.reshape(features[level], [batch_size, -1, num_filters]))
      features_r2 = tf.reshape(tf.concat(features_all, 1), [-1, num_filters])

    # Calculate height_l * width_l for each level.
    level_dim_sizes = [
        feature_widths[i] * feature_heights[i]
        for i in range(len(feature_widths))
    ]
    # level_dim_offsets is accumulated sum of level_dim_size.
    level_dim_offsets = [0]
    for i in range(len(feature_widths) - 1):
      level_dim_offsets.append(level_dim_offsets[i] + level_dim_sizes[i])
    batch_dim_size = level_dim_offsets[-1] + level_dim_sizes[-1]
    level_dim_offsets = tf.constant(level_dim_offsets, tf.int32)
    height_dim_sizes = tf.constant(feature_widths, tf.int32)

    # Assigns boxes to the right level.
    box_width = boxes[:, :, 3] - boxes[:, :, 1]
    box_height = boxes[:, :, 2] - boxes[:, :, 0]
    areas_sqrt = tf.sqrt(box_height * box_width)
    levels = tf.cast(
        tf.math.floordiv(
            tf.math.log(tf.divide(areas_sqrt, 224.0)), tf.math.log(2.0)) + 4.0,
        dtype=tf.int32)
    # Maps levels between [min_level, max_level].
    levels = tf.minimum(max_level, tf.maximum(levels, min_level))

    # Projects box location and sizes to corresponding feature levels.
    scale_to_level = tf.cast(
        tf.pow(tf.constant(2.0), tf.cast(levels, tf.float32)),
        dtype=boxes.dtype)
    boxes /= tf.expand_dims(scale_to_level, axis=2)
    box_width /= scale_to_level
    box_height /= scale_to_level
    boxes = tf.concat([
        boxes[:, :, 0:2],
        tf.expand_dims(box_height, -1),
        tf.expand_dims(box_width, -1)
    ],
                      axis=-1)

    # Maps levels to [0, max_level-min_level].
    levels -= min_level
    level_strides = tf.pow([[2.0]], tf.cast(levels, tf.float32))
    boundary = tf.cast(
        tf.concat([
            tf.expand_dims(
                [[tf.cast(max_feature_height, tf.float32)]] / level_strides - 1,
                axis=-1),
            tf.expand_dims(
                [[tf.cast(max_feature_width, tf.float32)]] / level_strides - 1,
                axis=-1),
        ],
                  axis=-1), boxes.dtype)

    # Compute grid positions.
    kernel_y, kernel_x, box_gridy0y1, box_gridx0x1 = compute_grid_positions(
        boxes, boundary, output_size, sample_offset=0.5)

    x_indices = tf.cast(
        tf.reshape(box_gridx0x1, [batch_size, num_boxes, output_size * 2]),
        dtype=tf.int32)
    y_indices = tf.cast(
        tf.reshape(box_gridy0y1, [batch_size, num_boxes, output_size * 2]),
        dtype=tf.int32)

    batch_size_offset = tf.tile(
        tf.reshape(
            tf.range(batch_size) * batch_dim_size, [batch_size, 1, 1, 1]),
        [1, num_boxes, output_size * 2, output_size * 2])
    # Get level offset for each box. Each box belongs to one level.
    levels_offset = tf.tile(
        tf.reshape(
            tf.gather(level_dim_offsets, levels),
            [batch_size, num_boxes, 1, 1]),
        [1, 1, output_size * 2, output_size * 2])
    y_indices_offset = tf.tile(
        tf.reshape(
            y_indices * tf.expand_dims(tf.gather(height_dim_sizes, levels), -1),
            [batch_size, num_boxes, output_size * 2, 1]),
        [1, 1, 1, output_size * 2])
    x_indices_offset = tf.tile(
        tf.reshape(x_indices, [batch_size, num_boxes, 1, output_size * 2]),
        [1, 1, output_size * 2, 1])
    indices = tf.reshape(
        batch_size_offset + levels_offset + y_indices_offset + x_indices_offset,
        [-1])

    # TODO(wangtao): replace tf.gather with tf.gather_nd and try to get similar
    # performance.
    features_per_box = tf.reshape(
        tf.gather(features_r2, indices),
        [batch_size, num_boxes, output_size * 2, output_size * 2, num_filters])

    # Bilinear interpolation.
    features_per_box = feature_bilinear_interpolation(features_per_box,
                                                      kernel_y, kernel_x)
    return features_per_box


def single_level_feature_crop(features, level_boxes, detection_prior_levels,
                              min_mask_level, mask_crop_size):
  """Crop the FPN features at the appropriate levels for each detection.


  Args:
    features: a float tensor of shape [batch_size, num_levels, max_feature_size,
      max_feature_size, num_downsample_channels].
    level_boxes: a float Tensor of the level boxes to crop from. [batch_size,
      num_instances, 4].
    detection_prior_levels: an int Tensor of instance assigned level of shape
      [batch_size, num_instances].
    min_mask_level: minimum FPN level to crop mask feature from.
    mask_crop_size: an int of mask crop size.

  Returns:
    crop_features: a float Tensor of shape [batch_size * num_instances,
        mask_crop_size, mask_crop_size, num_downsample_channels]. This is the
        instance feature crop.
  """
  (batch_size, num_levels, max_feature_size, _,
   num_downsample_channels) = features.get_shape().as_list()
  _, num_of_instances, _ = level_boxes.get_shape().as_list()
  level_boxes = tf.cast(level_boxes, tf.int32)
  assert num_of_instances == detection_prior_levels.get_shape().as_list()[1]

  x_start_indices = level_boxes[:, :, 1]
  y_start_indices = level_boxes[:, :, 0]
  # generate the full indices (not just the starting index)
  x_idx_list = []
  y_idx_list = []
  for i in range(mask_crop_size):
    x_idx_list.append(x_start_indices + i)
    y_idx_list.append(y_start_indices + i)

  x_indices = tf.stack(x_idx_list, axis=2)
  y_indices = tf.stack(y_idx_list, axis=2)
  levels = detection_prior_levels - min_mask_level
  height_dim_size = max_feature_size
  level_dim_size = max_feature_size * height_dim_size
  batch_dim_size = num_levels * level_dim_size
  # TODO(weicheng) change this to gather_nd for better readability.
  indices = tf.reshape(
      tf.tile(
          tf.reshape(
              tf.range(batch_size) * batch_dim_size, [batch_size, 1, 1, 1]),
          [1, num_of_instances, mask_crop_size, mask_crop_size]) + tf.tile(
              tf.reshape(levels * level_dim_size,
                         [batch_size, num_of_instances, 1, 1]),
              [1, 1, mask_crop_size, mask_crop_size]) + tf.tile(
                  tf.reshape(y_indices * height_dim_size,
                             [batch_size, num_of_instances, mask_crop_size, 1]),
                  [1, 1, 1, mask_crop_size]) +
      tf.tile(
          tf.reshape(x_indices,
                     [batch_size, num_of_instances, 1, mask_crop_size]),
          [1, 1, mask_crop_size, 1]), [-1])

  features_r2 = tf.reshape(features, [-1, num_downsample_channels])
  crop_features = tf.reshape(
      tf.gather(features_r2, indices), [
          batch_size * num_of_instances, mask_crop_size, mask_crop_size,
          num_downsample_channels
      ])

  return crop_features


def crop_mask_in_target_box(masks,
                            boxes,
                            target_boxes,
                            output_size,
                            sample_offset=0,
                            use_einsum=True):
  """Crop masks in target boxes.

  Args:
    masks: A tensor with a shape of [batch_size, num_masks, height, width].
    boxes: a float tensor representing box cooridnates that tightly enclose
      masks with a shape of [batch_size, num_masks, 4] in un-normalized
      coordinates. A box is represented by [ymin, xmin, ymax, xmax].
    target_boxes: a float tensor representing target box cooridnates for masks
      with a shape of [batch_size, num_masks, 4] in un-normalized coordinates. A
      box is represented by [ymin, xmin, ymax, xmax].
    output_size: A scalar to indicate the output crop size. It currently only
      supports to output a square shape outputs.
    sample_offset: a float number in [0, 1] indicates the subpixel sample offset
      from grid point.
    use_einsum: Use einsum to replace gather in selective_crop_and_resize.

  Returns:
    A 4-D tensor representing feature crop of shape
    [batch_size, num_boxes, output_size, output_size].
  """
  with tf.name_scope('crop_mask_in_target_box'):
    batch_size, num_masks, height, width = masks.get_shape().as_list()
    masks = tf.reshape(masks, [batch_size * num_masks, height, width, 1])
    # Pad zeros on the boundary of masks.
    masks = tf.image.pad_to_bounding_box(masks, 2, 2, height + 4, width + 4)
    masks = tf.reshape(masks, [batch_size, num_masks, height + 4, width + 4, 1])

    # Projects target box locations and sizes to corresponding cropped
    # mask coordinates.
    gt_y_min, gt_x_min, gt_y_max, gt_x_max = tf.split(
        value=boxes, num_or_size_splits=4, axis=2)
    bb_y_min, bb_x_min, bb_y_max, bb_x_max = tf.split(
        value=target_boxes, num_or_size_splits=4, axis=2)
    y_transform = (bb_y_min - gt_y_min) * height / (gt_y_max - gt_y_min +
                                                    _EPSILON) + 2
    x_transform = (bb_x_min - gt_x_min) * height / (gt_x_max - gt_x_min +
                                                    _EPSILON) + 2
    h_transform = (bb_y_max - bb_y_min) * width / (
        gt_y_max - gt_y_min + _EPSILON)
    w_transform = (bb_x_max - bb_x_min) * width / (
        gt_x_max - gt_x_min + _EPSILON)

    boundaries = tf.concat([
        tf.cast(
            tf.ones_like(y_transform) * ((height + 4) - 1), dtype=tf.float32),
        tf.cast(
            tf.ones_like(x_transform) * ((width + 4) - 1), dtype=tf.float32)
    ],
                           axis=-1)

    # Reshape tensors to have the right shape for selective_crop_and_resize.
    trasnformed_boxes = tf.concat(
        [y_transform, x_transform, h_transform, w_transform], -1)
    levels = tf.tile(
        tf.reshape(tf.range(num_masks), [1, num_masks]), [batch_size, 1])

    cropped_masks = selective_crop_and_resize(
        masks,
        trasnformed_boxes,
        levels,
        boundaries,
        output_size,
        sample_offset=sample_offset,
        use_einsum_gather=use_einsum)
    cropped_masks = tf.squeeze(cropped_masks, axis=-1)

  return cropped_masks
