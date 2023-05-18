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

"""Generate targets (center, scale, offsets,...) for centernet."""

from typing import Dict, List

import tensorflow as tf

from official.vision.ops import sampling_ops


def smallest_positive_root(a, b, c):
  """Returns the smallest positive root of a quadratic equation."""

  discriminant = tf.sqrt(b ** 2 - 4 * a * c)

  return (-b + discriminant) / (2.0)


@tf.function
def cartesian_product(*tensors, repeat: int = 1) -> tf.Tensor:
  """Equivalent of itertools.product except for TensorFlow tensors.

  Example:
    cartesian_product(tf.range(3), tf.range(4))

    array([[0, 0],
       [0, 1],
       [0, 2],
       [0, 3],
       [1, 0],
       [1, 1],
       [1, 2],
       [1, 3],
       [2, 0],
       [2, 1],
       [2, 2],
       [2, 3]], dtype=int32)>

  Args:
    *tensors: a list of 1D tensors to compute the product of
    repeat: an `int` number of times to repeat the tensors

  Returns:
    An nD tensor where n is the number of tensors
  """
  tensors = tensors * repeat
  return tf.reshape(tf.transpose(tf.stack(tf.meshgrid(*tensors, indexing='ij')),
                                 [*[i + 1 for i in range(len(tensors))], 0]),
                    (-1, len(tensors)))


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

  distance_detection_offset = smallest_positive_root(
      a=1, b=-(height + width),
      c=width * height * ((1 - min_iou) / (1 + min_iou))
  )

  # Case where detection is smaller than ground truth and completely contained
  # in it.
  distance_detection_in_gt = smallest_positive_root(
      a=4, b=-2 * (height + width),
      c=(1 - min_iou) * width * height
  )

  # Case where ground truth is smaller than detection and completely contained
  # in it.
  distance_gt_in_detection = smallest_positive_root(
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
def assign_center_targets(out_height: int,
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
      sampling_ops.combined_static_and_dynamic_shape(channel_onehot))

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
  return tf.stop_gradient(heatmap)


def assign_centernet_targets(labels: Dict[str, tf.Tensor],
                             output_size: List[int],
                             input_size: List[int],
                             num_classes: int = 90,
                             max_num_instances: int = 128,
                             gaussian_iou: float = 0.7,
                             class_offset: int = 0,
                             dtype='float32'):
  """Generates the ground truth labels for centernet.

  Ground truth labels are generated by splatting gaussians on heatmaps for
  corners and centers. Regressed features (offsets and sizes) are also
  generated.

  Args:
    labels: A dictionary of COCO ground truth labels with at minimum the
      following fields:
      "bbox" A `Tensor` of shape [max_num_instances, 4], where the
        last dimension corresponds to the top left x, top left y,
        bottom right x, and bottom left y coordinates of the bounding box
      "classes" A `Tensor` of shape [max_num_instances] that contains
        the class of each box, given in the same order as the boxes
      "num_detections" A `Tensor` or int that gives the number of objects
    output_size: A `list` of length 2 containing the desired output height
      and width of the heatmaps
    input_size: A `list` of length 2 the expected input height and width of
      the image
    num_classes: A `Tensor` or `int` for the number of classes.
    max_num_instances: An `int` for maximum number of instances in an image.
    gaussian_iou: A `float` number for the minimum desired IOU used when
      determining the gaussian radius of center locations in the heatmap.
    class_offset: A `int` for subtracting a value from the ground truth classes
    dtype: `str`, data type. One of {`bfloat16`, `float32`, `float16`}.

  Returns:
    Dictionary of labels with the following fields:
      'ct_heatmaps': Tensor of shape [output_h, output_w, num_classes],
        heatmap with splatted gaussians centered at the positions and channels
        corresponding to the center location and class of the object
      'ct_offset': `Tensor` of shape [max_num_instances, 2], where the first
        num_boxes entries contain the x-offset and y-offset of the center of
        an object. All other entires are 0
      'size': `Tensor` of shape [max_num_instances, 2], where the first
        num_boxes entries contain the width and height of an object. All
        other entires are 0
      'box_mask': `Tensor` of shape [max_num_instances], where the first
        num_boxes entries are 1. All other entires are 0
      'box_indices': `Tensor` of shape [max_num_instances, 2], where the first
        num_boxes entries contain the y-center and x-center of a valid box.
        These are used to extract the regressed box features from the
        prediction when computing the loss

  Raises:
    Exception: if datatype is not supported.
  """
  if dtype == 'float16':
    dtype = tf.float16
  elif dtype == 'bfloat16':
    dtype = tf.bfloat16
  elif dtype == 'float32':
    dtype = tf.float32
  else:
    raise Exception(
        'Unsupported datatype used in ground truth builder only '
        '{float16, bfloat16, or float32}')

  # Get relevant bounding box and class information from labels
  # only keep the first num_objects boxes and classes
  num_objects = labels['groundtruths']['num_detections']
  # shape of labels['boxes'] is [max_num_instances, 4]
  # [ymin, xmin, ymax, xmax]
  boxes = tf.cast(labels['boxes'], dtype)
  # shape of labels['classes'] is [max_num_instances, ]
  classes = tf.cast(labels['classes'] - class_offset, dtype)

  # Compute scaling factors for center/corner positions on heatmap
  # input_size = tf.cast(input_size, dtype)
  # output_size = tf.cast(output_size, dtype)
  input_h, input_w = input_size[0], input_size[1]
  output_h, output_w = output_size[0], output_size[1]

  width_ratio = output_w / input_w
  height_ratio = output_h / input_h

  # Original box coordinates
  # [max_num_instances, ]
  ytl, ybr = boxes[..., 0], boxes[..., 2]
  xtl, xbr = boxes[..., 1], boxes[..., 3]
  yct = (ytl + ybr) / 2
  xct = (xtl + xbr) / 2

  # Scaled box coordinates (could be floating point)
  # [max_num_instances, ]
  scale_xct = xct * width_ratio
  scale_yct = yct * height_ratio

  # Floor the scaled box coordinates to be placed on heatmaps
  # [max_num_instances, ]
  scale_xct_floor = tf.math.floor(scale_xct)
  scale_yct_floor = tf.math.floor(scale_yct)

  # Offset computations to make up for discretization error
  # used for offset maps
  # [max_num_instances, 2]
  ct_offset_values = tf.stack([scale_yct - scale_yct_floor,
                               scale_xct - scale_xct_floor], axis=-1)

  # Get the scaled box dimensions for computing the gaussian radius
  # [max_num_instances, ]
  box_widths = boxes[..., 3] - boxes[..., 1]
  box_heights = boxes[..., 2] - boxes[..., 0]

  box_widths = box_widths * width_ratio
  box_heights = box_heights * height_ratio

  # Used for size map
  # [max_num_instances, 2]
  box_heights_widths = tf.stack([box_heights, box_widths], axis=-1)

  # Center/corner heatmaps
  # [output_h, output_w, num_classes]
  ct_heatmap = tf.zeros((output_h, output_w, num_classes), dtype)

  # Maps for offset and size features for each instance of a box
  # [max_num_instances, 2]
  ct_offset = tf.zeros((max_num_instances, 2), dtype)
  # [max_num_instances, 2]
  size = tf.zeros((max_num_instances, 2), dtype)

  # Mask for valid box instances and their center indices in the heatmap
  # [max_num_instances, ]
  box_mask = tf.zeros((max_num_instances,), tf.int32)
  # [max_num_instances, 2]
  box_indices = tf.zeros((max_num_instances, 2), tf.int32)

  if num_objects > 0:
    # Need to gaussians around the centers and corners of the objects
    ct_heatmap = assign_center_targets(
        out_height=output_h,
        out_width=output_w,
        y_center=scale_yct_floor[:num_objects],
        x_center=scale_xct_floor[:num_objects],
        boxes_height=box_heights[:num_objects],
        boxes_width=box_widths[:num_objects],
        channel_onehot=tf.one_hot(tf.cast(classes[:num_objects], tf.int32),
                                  num_classes, off_value=0.),
        gaussian_iou=gaussian_iou)

    # Indices used to update offsets and sizes for valid box instances
    update_indices = cartesian_product(
        tf.range(max_num_instances), tf.range(2))
    # [max_num_instances, 2, 2]
    update_indices = tf.reshape(update_indices, shape=[max_num_instances, 2, 2])

    # Write the offsets of each box instance
    ct_offset = tf.tensor_scatter_nd_update(
        ct_offset, update_indices, ct_offset_values)

    # Write the size of each bounding box
    size = tf.tensor_scatter_nd_update(
        size, update_indices, box_heights_widths)

    # Initially the mask is zeros, so now we unmask each valid box instance
    box_mask = tf.where(tf.range(max_num_instances) < num_objects, 1, 0)

    # Write the y and x coordinate of each box center in the heatmap
    box_index_values = tf.cast(
        tf.stack([scale_yct_floor, scale_xct_floor], axis=-1),
        dtype=tf.int32)
    box_indices = tf.tensor_scatter_nd_update(
        box_indices, update_indices, box_index_values)

  ct_labels = {
      # [output_h, output_w, num_classes]
      'ct_heatmaps': ct_heatmap,
      # [max_num_instances, 2]
      'ct_offset': ct_offset,
      # [max_num_instances, 2]
      'size': size,
      # [max_num_instances, ]
      'box_mask': box_mask,
      # [max_num_instances, 2]
      'box_indices': box_indices
  }
  return ct_labels
