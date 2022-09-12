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

"""Spatial transform ops."""

from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from official.vision.ops.box_ops import bbox2mask

_EPSILON = 1e-8


def _feature_bilinear_interpolation(features: tf.Tensor, kernel_y: tf.Tensor,
                                    kernel_x: tf.Tensor) -> tf.Tensor:
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
  features_shape = tf.shape(features)
  batch_size, num_boxes, output_size, num_filters = (
      features_shape[0], features_shape[1], features_shape[2],
      features_shape[4])

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


def _compute_grid_positions(
    boxes: tf.Tensor, boundaries: tf.Tensor, output_size: int,
    sample_offset: float) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
  """Computes the grid position w.r.t.

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
  boxes_shape = tf.shape(boxes)
  batch_size, num_boxes = boxes_shape[0], boxes_shape[1]
  if batch_size is None:
    batch_size = tf.shape(boxes)[0]
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
  box_grid_x0 = tf.maximum(tf.cast(0., dtype=box_grid_x0.dtype), box_grid_x0)
  box_grid_y0 = tf.maximum(tf.cast(0., dtype=box_grid_y0.dtype), box_grid_y0)

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


def multilevel_crop_and_resize(features: Dict[str, tf.Tensor],
                               boxes: tf.Tensor,
                               output_size: int = 7,
                               sample_offset: float = 0.5) -> tf.Tensor:
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
    sample_offset: a float number in [0, 1] indicates the subpixel sample offset
      from grid point.

  Returns:
    A 5-D tensor representing feature crop of shape
    [batch_size, num_boxes, output_size, output_size, num_filters].
  """

  with tf.name_scope('multilevel_crop_and_resize'):
    levels = list(features.keys())
    min_level = int(min(levels))
    max_level = int(max(levels))
    features_shape = tf.shape(features[str(min_level)])
    batch_size, max_feature_height, max_feature_width, num_filters = (
        features_shape[0], features_shape[1], features_shape[2],
        features_shape[3])

    num_boxes = tf.shape(boxes)[1]

    # Stack feature pyramid into a features_all of shape
    # [batch_size, levels, height, width, num_filters].
    features_all = []
    feature_heights = []
    feature_widths = []
    for level in range(min_level, max_level + 1):
      shape = features[str(level)].get_shape().as_list()
      feature_heights.append(shape[1])
      feature_widths.append(shape[2])
      # Concat tensor of [batch_size, height_l * width_l, num_filters] for each
      # levels.
      features_all.append(
          tf.reshape(features[str(level)], [batch_size, -1, num_filters]))
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
    areas_sqrt = tf.sqrt(
        tf.cast(box_height, tf.float32) * tf.cast(box_width, tf.float32))

    levels = tf.cast(
        tf.math.floordiv(
            tf.math.log(tf.math.divide_no_nan(areas_sqrt, 224.0)),
            tf.math.log(2.0)) + 4.0,
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
    boxes = tf.concat([boxes[:, :, 0:2],
                       tf.expand_dims(box_height, -1),
                       tf.expand_dims(box_width, -1)], axis=-1)

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
    kernel_y, kernel_x, box_gridy0y1, box_gridx0x1 = _compute_grid_positions(
        boxes, boundary, output_size, sample_offset)

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
    features_per_box = _feature_bilinear_interpolation(
        features_per_box, kernel_y, kernel_x)
    return features_per_box


def _selective_crop_and_resize(features: tf.Tensor,
                               boxes: tf.Tensor,
                               box_levels: tf.Tensor,
                               boundaries: tf.Tensor,
                               output_size: int = 7,
                               sample_offset: float = 0.5,
                               use_einsum_gather: bool = False) -> tf.Tensor:
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
       vertex in the output grid.
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
  if batch_size is None:
    batch_size = tf.shape(features)[0]
  _, num_boxes, _ = boxes.get_shape().as_list()

  kernel_y, kernel_x, box_gridy0y1, box_gridx0x1 = _compute_grid_positions(
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
    y_indices = tf.cast(
        tf.reshape(box_gridy0y1, [batch_size, num_boxes, output_size, 2]),
        dtype=tf.int32)
    x_indices = tf.cast(
        tf.reshape(box_gridx0x1, [batch_size, num_boxes, output_size, 2]),
        dtype=tf.int32)

    # shape is [batch_size, num_boxes, output_size, 2, height]
    grid_y_one_hot = tf.one_hot(
        tf.cast(y_indices, tf.int32), max_feature_height, dtype=kernel_y.dtype)
    # shape is [batch_size, num_boxes, output_size, 2, width]
    grid_x_one_hot = tf.one_hot(
        tf.cast(x_indices, tf.int32), max_feature_width, dtype=kernel_x.dtype)

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
    features_per_box = _feature_bilinear_interpolation(
        features_per_box, kernel_y, kernel_x)

  return features_per_box


def crop_mask_in_target_box(masks: tf.Tensor,
                            boxes: tf.Tensor,
                            target_boxes: tf.Tensor,
                            output_size: int,
                            sample_offset: float = 0.0,
                            use_einsum: bool = True) -> tf.Tensor:
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
    # Cast to float32, as the y_transform and other transform variables may
    # overflow in float16
    masks = tf.cast(masks, tf.float32)
    boxes = tf.cast(boxes, tf.float32)
    target_boxes = tf.cast(target_boxes, tf.float32)

    batch_size, num_masks, height, width = masks.get_shape().as_list()
    if batch_size is None:
      batch_size = tf.shape(masks)[0]
    masks = tf.reshape(masks, [batch_size * num_masks, height, width, 1])
    # Pad zeros on the boundary of masks.
    masks = tf.image.pad_to_bounding_box(masks, 2, 2, height + 4, width + 4)
    masks = tf.reshape(masks, [batch_size, num_masks, height+4, width+4, 1])

    # Projects target box locations and sizes to corresponding cropped
    # mask coordinates.
    gt_y_min, gt_x_min, gt_y_max, gt_x_max = tf.split(
        value=boxes, num_or_size_splits=4, axis=2)
    bb_y_min, bb_x_min, bb_y_max, bb_x_max = tf.split(
        value=target_boxes, num_or_size_splits=4, axis=2)
    y_transform = (bb_y_min - gt_y_min) * height / (
        gt_y_max - gt_y_min + _EPSILON) + 2
    x_transform = (bb_x_min - gt_x_min) * height / (
        gt_x_max - gt_x_min + _EPSILON) + 2
    h_transform = (bb_y_max - bb_y_min) * width / (
        gt_y_max - gt_y_min + _EPSILON)
    w_transform = (bb_x_max - bb_x_min) * width / (
        gt_x_max - gt_x_min + _EPSILON)

    boundaries = tf.concat(
        [tf.ones_like(y_transform) * ((height + 4) - 1),
         tf.ones_like(x_transform) * ((width + 4) - 1)],
        axis=-1)
    boundaries = tf.cast(boundaries, dtype=y_transform.dtype)

    # Reshape tensors to have the right shape for selective_crop_and_resize.
    trasnformed_boxes = tf.concat(
        [y_transform, x_transform, h_transform, w_transform], -1)
    levels = tf.tile(tf.reshape(tf.range(num_masks), [1, num_masks]),
                     [batch_size, 1])

    cropped_masks = _selective_crop_and_resize(
        masks,
        trasnformed_boxes,
        levels,
        boundaries,
        output_size,
        sample_offset=sample_offset,
        use_einsum_gather=use_einsum)
    cropped_masks = tf.squeeze(cropped_masks, axis=-1)

  return cropped_masks


def nearest_upsampling(data: tf.Tensor,
                       scale: int,
                       use_keras_layer: bool = False) -> tf.Tensor:
  """Nearest neighbor upsampling implementation.

  Args:
    data: A tensor with a shape of [batch, height_in, width_in, channels].
    scale: An integer multiple to scale resolution of input data.
    use_keras_layer: If True, use keras Upsampling2D layer.

  Returns:
    data_up: A tensor with a shape of
      [batch, height_in*scale, width_in*scale, channels]. Same dtype as input
      data.
  """
  if use_keras_layer:
    return tf.keras.layers.UpSampling2D(size=(scale, scale),
                                        interpolation='nearest')(data)
  with tf.name_scope('nearest_upsampling'):
    bs, _, _, c = data.get_shape().as_list()
    shape = tf.shape(input=data)
    h = shape[1]
    w = shape[2]
    bs = -1 if bs is None else bs
    # Uses reshape to quickly upsample the input.  The nearest pixel is selected
    # via tiling.
    data = tf.tile(
        tf.reshape(data, [bs, h, 1, w, 1, c]), [1, 1, scale, 1, scale, 1])
    return tf.reshape(data, [bs, h * scale, w * scale, c])


def _gather_rows_from_matrix(input_matrix: tf.Tensor,
                             row_indices: tf.Tensor) -> tf.Tensor:
  """Gather rows from the input matrix (2-D tensor).

  This operation is equivalent to tf.gather(input_matrix, row_indices), but is
  implemented in sparse matrix multiplication.

  Args:
    input_matrix: A 2-D tensor in shape (input_h, input_w) from which to gather
      values. The shape must be 2-D, since sparse matrix multiplication is
      currently only supported on 2-D matrices.
    row_indices: A 1-D int tensor in shape (output_h) which stored the row
      indices of the input.

  Returns:
    A tensor in shape (output_h, input_w) which stores the gathered rows.
  """
  input_matrix_shape = input_matrix.get_shape().as_list()
  if len(input_matrix_shape) != 2:
    raise ValueError(
        'Expected the input_matrix tensor (input_h, input_w) has rank == 2, '
        'was: %s' % input_matrix_shape)
  row_indices_shape = row_indices.get_shape().as_list()
  if len(row_indices_shape) != 1:
    raise ValueError(
        'Expected the row_indices tensor (output_h) has rank == 1, was: %s' %
        row_indices_shape)

  # (output_h, input_h)
  indices_one_hot = tf.one_hot(
      row_indices, depth=input_matrix_shape[0], dtype=input_matrix.dtype)
  # Matrix multiplication: (output_h, input_h) x (input_h, input_w)
  # (output_h, input_w)
  return tf.linalg.matmul(indices_one_hot, input_matrix, a_is_sparse=True)


def bilinear_resize_to_bbox(images: tf.Tensor, bbox: tf.Tensor,
                            output_size: tf.Tensor) -> tf.Tensor:
  """Bilinear resizes the images to fit into the bounding boxes in the output.

  Args:
    images: A tensor in shape (batch_size, input_h, input_w, ...) with arbitrary
      numbers of channel dimensions.
    bbox: A tensor in shape (batch_size, 4), representing the absolute
      coordinates (ymin, xmin, ymax, xmax) for each bounding box.
    output_size: The size of the output images in (output_h, output_w).

  Returns:
    A tensor in shape (batch_size, output_h, output_w, ...).
  """
  images_shape = images.get_shape().as_list()
  images_rank = len(images_shape)
  if images_rank < 3:
    raise ValueError(
        'Expected the input images (batch_size, height, width, ...) '
        'has rank >= 3, was: %s' % images_shape)
  bbox_shape = bbox.get_shape().as_list()
  if bbox_shape[-1] != 4:
    raise ValueError(
        'Expected the last dimension of `bbox` has size == 4, but the shape '
        'of `bbox` was: %s' % bbox_shape)

  rank_range = list(range(images_rank))
  extra_dims = images_shape[3:]
  extra_dims_perm = rank_range[3:]
  extra_dims_product = 1
  for d in extra_dims:
    extra_dims_product *= d

  input_h = tf.cast(tf.shape(images)[1], tf.float32)
  input_w = tf.cast(tf.shape(images)[2], tf.float32)
  output_h = output_size[0]
  output_w = output_size[1]

  bbox = tf.cast(bbox, tf.float32)
  # (batch_size, 1)
  bbox_ymin = bbox[:, 0:1]
  bbox_xmin = bbox[:, 1:2]
  bbox_ymax = bbox[:, 2:3]
  bbox_xmax = bbox[:, 3:4]
  bbox_h = bbox_ymax - bbox_ymin
  bbox_w = bbox_xmax - bbox_xmin
  scale_h = tf.math.divide_no_nan(input_h, bbox_h)
  scale_w = tf.math.divide_no_nan(input_w, bbox_w)

  # Generates the output grids.
  # (output_h)
  output_y_grid = tf.range(output_h, dtype=bbox_ymin.dtype)
  # (output_w)
  output_x_grid = tf.range(output_w, dtype=bbox_xmin.dtype)

  # Computes the input source positions (float) which map to the output grids
  # (integer).
  # Applies half pixel offset here to ensure the output is center-aligned to the
  # input.
  # TODO(b/245614786): support align_corners=True.
  # (batch_size, output_h)
  input_y_pos = tf.clip_by_value(
      (output_y_grid - bbox_ymin + 0.5) * scale_h - 0.5, 0.0, input_h - 1.0)
  # (batch_size, output_w)
  input_x_pos = tf.clip_by_value(
      (output_x_grid - bbox_xmin + 0.5) * scale_w - 0.5, 0.0, input_w - 1.0)

  # Gets the positions (integer) of the four nearest neighbors of the input
  # source position (float).
  # (y0, x0): left-top
  # (y0, x1): right-top
  # (y1, x0): left-bottom
  # (y1, x1): right-bottom
  # (batch_size, output_h)
  input_y0 = tf.cast(
      tf.clip_by_value(tf.floor(input_y_pos), 0.0, input_h - 2.0), tf.int32)
  input_y1 = input_y0 + 1
  # (batch_size, output_w)
  input_x0 = tf.cast(
      tf.clip_by_value(tf.floor(input_x_pos), 0.0, input_w - 2.0), tf.int32)
  input_x1 = input_x0 + 1

  # (batch_size, output_h)
  output_y_mask = (bbox_ymin <= output_y_grid) & (output_y_grid < bbox_ymax)
  # (batch_size, output_w)
  output_x_mask = (bbox_xmin <= output_x_grid) & (output_x_grid < bbox_xmax)

  # Masks the output pixels outside the bounding box by setting their input
  # neighbors to -1. This makes `tf.one_hot` operation produce all zeros at
  # these pixels, so as to accelerate the sparse matrix multiplication in
  # `_gather_rows_from_matrix`.
  # (batch_size, output_h)
  input_y0 = tf.where(output_y_mask, input_y0, -tf.ones_like(input_y0))
  input_y1 = tf.where(output_y_mask, input_y1, -tf.ones_like(input_y1))
  # (batch_size, output_w)
  input_x0 = tf.where(output_x_mask, input_x0, -tf.ones_like(input_x0))
  input_x1 = tf.where(output_x_mask, input_x1, -tf.ones_like(input_x1))

  input_h = tf.cast(input_h, tf.int32)
  input_w = tf.cast(input_w, tf.int32)
  images = tf.cast(images, tf.float32)
  if images_rank > 3:
    # Reshapes the images since _gather_rows_from_matrix only takes 2-D tensor.
    # (batch_size, input_h, input_w * extra_dims_product)
    images = tf.reshape(images, [-1, input_h, input_w * extra_dims_product])

  # Fetches the rows from the input source images.
  # (batch_size, output_h, input_w * extra_dims_product)
  val_y0 = tf.map_fn(
      lambda x: _gather_rows_from_matrix(x[0], x[1]),
      elems=(images, input_y0),
      fn_output_signature=tf.float32,
      parallel_iterations=32)
  val_y1 = tf.map_fn(
      lambda x: _gather_rows_from_matrix(x[0], x[1]),
      elems=(images, input_y1),
      fn_output_signature=tf.float32,
      parallel_iterations=32)

  if images_rank > 3:
    new_shape = [-1, output_h, input_w] + extra_dims
    # (batch_size, output_h, input_w, ...)
    val_y0 = tf.reshape(val_y0, new_shape)
    val_y1 = tf.reshape(val_y1, new_shape)

  # Transposes the tensors for reusing _gather_rows_from_matrix later.
  new_perm = [0, 2, 1] + extra_dims_perm
  # (batch_size, input_w, output_h, ...)
  val_y0 = tf.transpose(val_y0, new_perm)
  val_y1 = tf.transpose(val_y1, new_perm)

  if images_rank > 3:
    new_shape = [-1, input_w, output_h * extra_dims_product]
    # (batch_size, input_w, output_h * extra_dims_product)
    val_y0 = tf.reshape(val_y0, new_shape)
    val_y1 = tf.reshape(val_y1, new_shape)

  # Fetches the pixels from the rows using the column indices.
  # val_00, val_01, val_10, val_11 store the pixels of the four nearest
  # neighbors of the input source position.
  # (batch_size, output_w, output_h * extra_dims_product)
  val_00 = tf.map_fn(
      lambda x: _gather_rows_from_matrix(x[0], x[1]),
      elems=(val_y0, input_x0),
      fn_output_signature=tf.float32,
      parallel_iterations=32)
  val_01 = tf.map_fn(
      lambda x: _gather_rows_from_matrix(x[0], x[1]),
      elems=(val_y0, input_x1),
      fn_output_signature=tf.float32,
      parallel_iterations=32)
  val_10 = tf.map_fn(
      lambda x: _gather_rows_from_matrix(x[0], x[1]),
      elems=(val_y1, input_x0),
      fn_output_signature=tf.float32,
      parallel_iterations=32)
  val_11 = tf.map_fn(
      lambda x: _gather_rows_from_matrix(x[0], x[1]),
      elems=(val_y1, input_x1),
      fn_output_signature=tf.float32,
      parallel_iterations=32)

  if images_rank > 3:
    new_shape = [-1, output_w, output_h] + extra_dims
    # (batch_size, output_w, output_h, ...)
    val_00 = tf.reshape(val_00, new_shape)
    val_01 = tf.reshape(val_01, new_shape)
    val_10 = tf.reshape(val_10, new_shape)
    val_11 = tf.reshape(val_11, new_shape)

  # (..., batch_size, output_h, output_w)
  new_perm = extra_dims_perm + [0, 2, 1]
  val_00 = tf.transpose(val_00, new_perm)
  val_01 = tf.transpose(val_01, new_perm)
  val_10 = tf.transpose(val_10, new_perm)
  val_11 = tf.transpose(val_11, new_perm)

  # (batch_size, output_height, 1)
  input_y_pos = input_y_pos[:, :, tf.newaxis]
  input_y0 = tf.cast(input_y0[:, :, tf.newaxis], input_y_pos.dtype)
  input_y1 = tf.cast(input_y1[:, :, tf.newaxis], input_y_pos.dtype)
  # (batch_size, 1, output_width)
  input_x_pos = input_x_pos[:, tf.newaxis, :]
  input_x0 = tf.cast(input_x0[:, tf.newaxis, :], input_x_pos.dtype)
  input_x1 = tf.cast(input_x1[:, tf.newaxis, :], input_x_pos.dtype)

  # Compute the weights of the four nearest neighbors for interpolation.
  # (batch_size, output_height, output_width)
  weight_00 = (input_y1 - input_y_pos) * (input_x1 - input_x_pos)
  weight_01 = (input_y1 - input_y_pos) * (input_x_pos - input_x0)
  weight_10 = (input_y_pos - input_y0) * (input_x1 - input_x_pos)
  weight_11 = (input_y_pos - input_y0) * (input_x_pos - input_x0)

  # (..., batch_size, output_height, output_width)
  output_images = (
      val_00 * weight_00 + val_01 * weight_01 + val_10 * weight_10 +
      val_11 * weight_11)

  # (batch_size, output_height, output_width, ...)
  return tf.transpose(output_images, np.roll(rank_range, -len(extra_dims)))


def bilinear_resize_with_crop_and_pad(images: tf.Tensor,
                                      rescale_size: tf.Tensor,
                                      crop_offset: tf.Tensor,
                                      crop_size: tf.Tensor,
                                      output_size: tf.Tensor) -> tf.Tensor:
  """Bilinear resizes the images, then crops and finally pads to output size.

  Args:
    images: A tensor in shape (batch_size, input_h, input_w, ...) with arbitrary
      numbers of channel dimensions.
    rescale_size: An int tensor in shape (batch_size, 2), representing the sizes
      of the rescaled images.
    crop_offset: An int tensor in shape (batch_size, 2), representing the
      left-top offset of the crop box. Applying negative offsets means adding
      extra margins at the left-top.
    crop_size: An int tensor in shape (batch_size, 2), representing the sizes of
      the cropped images.
    output_size: The size of the output image in (output_h, output_w).

  Returns:
    A tensor in shape (batch_size, output_h, output_w, ...).
  """
  images_shape = images.get_shape().as_list()
  images_rank = len(images_shape)
  if images_rank < 3:
    raise ValueError(
        'Expected the input images (batch_size, height, width, ...) '
        'has rank >= 3, was: %s' % images_shape)
  num_extra_dims = images_rank - 3

  # Rescales the images, applies the offset and pastes to the output canvas.

  # (batch_size, 2)
  ymin_xmin = -crop_offset
  # (batch_size, 2)
  ymax_xmax = ymin_xmin + tf.cast(rescale_size, ymin_xmin.dtype)
  # (batch_size, 4)
  rescale_bbox = tf.concat([ymin_xmin, ymax_xmax], axis=1)
  # (batch_size, output_height, output_width, ...)
  rescaled_padded_images = bilinear_resize_to_bbox(images, rescale_bbox,
                                                   output_size)

  # Masks out the pixels outside of the crop box.
  # (batch_size, 2)
  y0_x0 = tf.broadcast_to(
      tf.constant([[0, 0]], dtype=crop_size.dtype), tf.shape(crop_size))
  # (batch_size, 4)
  crop_bbox = tf.concat([y0_x0, crop_size], axis=1)
  # (batch_size, output_height, output_width, ...)
  crop_bbox_mask = bbox2mask(
      crop_bbox,
      image_height=output_size[0],
      image_width=output_size[1],
      dtype=rescaled_padded_images.dtype)[[...] + [tf.newaxis] * num_extra_dims]
  # (batch_size, output_height, output_width, ...)
  return rescaled_padded_images * crop_bbox_mask
