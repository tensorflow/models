# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Generating target centers for centernet based on ODAPI"""

import tensorflow as tf

from official.vision.beta.projects.centernet.ops import loss_ops
from official.vision.beta.projects.centernet.ops import target_assigner


def image_shape_to_grids(height: int, width: int):
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


def max_distance_for_overlap(height, width, min_iou):
  """Computes how far apart bbox corners can lie while maintaining the iou.

  Given a bounding box size, this function returns a lower bound on how far
  apart the corners of another box can lie while still maintaining the given
  IoU. The implementation is based on the `gaussian_radius` function in the
  Objects as Points github repo: https://github.com/xingyizhou/CenterNet

  Args:
    height: A 1-D float Tensor representing height of the ground truth boxes.
    width: A 1-D float Tensor representing width of the ground truth boxes.
    min_iou: A float representing the minimum IoU desired.

  Returns:
   distance: A 1-D Tensor of distances, of the same length as the input
     height and width tensors.
  """
  
  # Given that the detected box is displaced at a distance `d`, the exact
  # IoU value will depend on the angle at which each corner is displaced.
  # We simplify our computation by assuming that each corner is displaced by
  # a distance `d` in both x and y direction. This gives us a lower IoU than
  # what is actually realizable and ensures that any box with corners less
  # than `d` distance apart will always have an IoU greater than or equal
  # to `min_iou`
  
  # The following 3 cases can be worked on geometrically and come down to
  # solving a quadratic inequality. In each case, to ensure `min_iou` we use
  # the smallest positive root of the equation.
  
  # Case where detected box is offset from ground truth and no box completely
  # contains the other.
  
  distance_detection_offset = target_assigner.smallest_positive_root(
      a=1, b=-(height + width),
      c=width * height * ((1 - min_iou) / (1 + min_iou))
  )
  
  # Case where detection is smaller than ground truth and completely contained
  # in it.
  distance_detection_in_gt = target_assigner.smallest_positive_root(
      a=4, b=-2 * (height + width),
      c=(1 - min_iou) * width * height
  )
  
  # Case where ground truth is smaller than detection and completely contained
  # in it.
  distance_gt_in_detection = target_assigner.smallest_positive_root(
      a=4 * min_iou, b=(2 * min_iou) * (width + height),
      c=(min_iou - 1) * width * height
  )
  
  return tf.reduce_min([distance_detection_offset,
                        distance_gt_in_detection,
                        distance_detection_in_gt], axis=0)


def compute_std_dev_from_box_size(boxes_height, boxes_width, min_overlap):
  """Computes the standard deviation of the Gaussian kernel from box size.

  Args:
    boxes_height: A 1D tensor with shape [num_instances] representing the height
      of each box.
    boxes_width: A 1D tensor with shape [num_instances] representing the width
      of each box.
    min_overlap: The minimum IOU overlap that boxes need to have to not be
      penalized.

  Returns:
    A 1D tensor with shape [num_instances] representing the computed Gaussian
    sigma for each of the box.
  """
  # We are dividing by 3 so that points closer than the computed
  # distance have a >99% CDF.
  sigma = max_distance_for_overlap(boxes_height, boxes_width, min_overlap)
  sigma = (2 * tf.math.maximum(tf.math.floor(sigma), 0.0) + 1) / 6.0
  return sigma


@tf.function
def assign_center_targets_odapi(out_height: int,
                                out_width: int,
                                y_center: tf.Tensor,
                                x_center: tf.Tensor,
                                boxes_height: tf.Tensor,
                                boxes_width: tf.Tensor,
                                channel_onehot: tf.Tensor,
                                gaussian_iou: float):
  """Computes the object center heatmap target based on ODAPI implementation.

  Args:
    out_height: int, height of output to the model. This is used to
      determine the height of the output.
    out_width: int, width of the output to the model. This is used to
      determine the width of the output.
    y_center: A 1D tensor with shape [num_instances] representing the
      y-coordinates of the instances in the output space coordinates.
    x_center: A 1D tensor with shape [num_instances] representing the
      x-coordinates of the instances in the output space coordinates.
    boxes_height: A 1D tensor with shape [num_instances] representing the height
      of each box.
    boxes_width: A 1D tensor with shape [num_instances] representing the width
      of each box.
    channel_onehot: A 2D tensor with shape [num_instances, num_channels]
      representing the one-hot encoded channel labels for each point.
    gaussian_iou: The minimum IOU overlap that boxes need to have to not be
      penalized.

  Returns:
    heatmap: A Tensor of size [output_height, output_width,
      num_classes] representing the per class center heatmap. output_height
      and output_width are computed by dividing the input height and width by
      the stride specified during initialization.
  """
  (y_grid, x_grid) = image_shape_to_grids(out_height, out_width)
  
  sigma = compute_std_dev_from_box_size(boxes_height, boxes_width,
                                        gaussian_iou)
  
  num_instances, num_channels = (
      loss_ops.combined_static_and_dynamic_shape(channel_onehot))
  
  x_grid = tf.expand_dims(x_grid, 2)
  y_grid = tf.expand_dims(y_grid, 2)
  # The raw center coordinates in the output space.
  x_diff = x_grid - tf.math.floor(x_center)
  y_diff = y_grid - tf.math.floor(y_center)
  squared_distance = x_diff ** 2 + y_diff ** 2
  
  gaussian_map = tf.exp(-squared_distance / (2 * sigma * sigma))
  
  reshaped_gaussian_map = tf.expand_dims(gaussian_map, axis=-1)
  reshaped_channel_onehot = tf.reshape(channel_onehot,
                                       (1, 1, num_instances, num_channels))
  gaussian_per_box_per_class_map = (
      reshaped_gaussian_map * reshaped_channel_onehot)
  
  # Take maximum along the "instance" dimension so that all per-instance
  # heatmaps of the same class are merged together.
  heatmap = tf.reduce_max(gaussian_per_box_per_class_map, axis=2)
  
  # Maximum of an empty tensor is -inf, the following is to avoid that.
  heatmap = tf.maximum(heatmap, 0)
  return heatmap
