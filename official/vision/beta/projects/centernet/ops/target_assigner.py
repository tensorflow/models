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

"""Generate targets (center, scale, offsets,...) for centernet"""

from typing import List, Dict, Tuple

import tensorflow as tf
from official.vision.beta.projects.centernet.ops import target_assigner_odapi


def smallest_positive_root(a, b, c):
  """Returns the smallest positive root of a quadratic equation."""
  
  discriminant = tf.sqrt(b ** 2 - 4 * a * c)
  
  # TODO(vighneshb) We are currently using the slightly incorrect
  # CenterNet implementation. The commented lines implement the fixed version
  # in https://github.com/princeton-vl/CornerNet. Change the implementation
  # after verifying it has no negative impact.
  # root1 = (-b - discriminant) / (2 * a)
  # root2 = (-b + discriminant) / (2 * a)
  
  # return tf.where(tf.less(root1, 0), root2, root1)
  
  return (-b + discriminant) / (2.0)


@tf.function
def cartesian_product(*tensors, repeat=1):
  """
  Equivalent of itertools.product except for TensorFlow tensors.

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

  Params:
    tensors (list[tf.Tensor]): a list of 1D tensors to compute the product of
    repeat (int): number of times to repeat the tensors
      (https://docs.python.org/3/library/itertools.html#itertools.product)

  Returns:
    An nD tensor where n is the number of tensors
  """
  tensors = tensors * repeat
  return tf.reshape(tf.transpose(tf.stack(tf.meshgrid(*tensors, indexing='ij')),
                                 [*[i + 1 for i in range(len(tensors))], 0]),
                    (-1, len(tensors)))


def gaussian_radius(det_size, min_overlap=0.7) -> int:
  """
    Given a bounding box size, returns a lower bound on how far apart the
    corners of another bounding box can lie while still maintaining the given
    minimum overlap, or IoU. Modified from implementation found in
    https://github.com/tensorflow/models/blob/master/research/object_detection/core/target_assigner.py.

    Params:
        det_size (tuple): tuple of integers representing height and width
        min_overlap (tf.float32): minimum IoU desired
    Returns:
        int representing desired gaussian radius
    """
  height, width = det_size[0], det_size[1]
  
  # Case where detected box is offset from ground truth and no box completely
  # contains the other.
  
  a1 = 1
  b1 = -(height + width)
  c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
  r1 = smallest_positive_root(a1, b1, c1)
  
  # Case where detection is smaller than ground truth and completely contained
  # in it.
  
  a2 = 4
  b2 = -2 * (height + width)
  c2 = (1 - min_overlap) * width * height
  r2 = smallest_positive_root(a2, b2, c2)
  
  # Case where ground truth is smaller than detection and completely contained
  # in it.
  
  a3 = 4 * min_overlap
  b3 = 2 * min_overlap * (height + width)
  c3 = (min_overlap - 1) * width * height
  r3 = smallest_positive_root(a3, b3, c3)
  # TODO discuss whether to return scalar or tensor
  
  return tf.reduce_min([r1, r2, r3], axis=0)


def gaussian_penalty(radius: int, dtype=tf.float32) -> tf.Tensor:
  """
  This represents the penalty reduction around a point.
  Params:
      radius (int): integer for radius of penalty reduction
      type (tf.dtypes.DType): datatype of returned tensor
  Returns:
      tf.Tensor of shape (2 * radius + 1, 2 * radius + 1).
  """
  width = 2 * radius + 1
  sigma = tf.cast(radius / 3, dtype=dtype)
  
  range_width = tf.range(width)
  range_width = tf.cast(range_width - tf.expand_dims(radius, axis=-1),
                        dtype=dtype)
  
  x = tf.expand_dims(range_width, axis=-1)
  y = tf.expand_dims(range_width, axis=-2)
  
  exponent = ((-1 * (x ** 2) - (y ** 2)) / (2 * sigma ** 2))
  return tf.math.exp(exponent)


@tf.function
def draw_gaussian(hm_shape, blob, dtype, scaling_factor=1):
  """ Draws an instance of a 2D gaussian on a heatmap.

  A heatmap with shape hm_shape and of type dtype is generated with
  a gaussian with a given center, radius, and scaling factor

  Args:
    hm_shape: A `list` of `Tensor` of shape [3] that gives the height, width,
      and number of channels in the heatmap
    blob: A `Tensor` of shape [4] that gives the channel number, y, x, and
      radius for the desired gaussian to be drawn onto
    dtype: The desired type of the heatmap
    scaling_factor: A `int` that can be used to scale the magnitude of the
      gaussian
  Returns:
    A `Tensor` with shape hm_shape and type dtype with a 2D gaussian
  """
  gaussian_heatmap = tf.zeros(shape=hm_shape, dtype=dtype)
  
  blob = tf.cast(blob, tf.int32)
  obj_class, y, x, radius = blob[0], blob[1], blob[2], blob[3]
  
  height, width = hm_shape[0], hm_shape[1]
  
  left = tf.math.minimum(x, radius)
  right = tf.math.minimum(width - x, radius + 1)
  top = tf.math.minimum(y, radius)
  bottom = tf.math.minimum(height - y, radius + 1)
  
  gaussian = gaussian_penalty(radius=radius, dtype=dtype)
  gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  gaussian = tf.reshape(gaussian, [-1])
  
  heatmap_indices = cartesian_product(
      tf.range(y - top, y + bottom), tf.range(x - left, x + right), [obj_class])
  gaussian_heatmap = tf.tensor_scatter_nd_update(
      gaussian_heatmap, heatmap_indices, gaussian * scaling_factor)
  
  return gaussian_heatmap


def assign_center_targets(y_center: tf.Tensor,
                          x_center: tf.Tensor,
                          box_heights_widths: tf.Tensor,
                          classes: tf.Tensor,
                          gaussian_rad: int,
                          gaussian_iou: float,
                          output_shape: Tuple[int, int, int]):
  """
  
  Args:
    y_center: A 1D tensor with shape [num_instances] representing the
      y-coordinates of the instances in the output space coordinates.
    x_center: A 1D tensor with shape [num_instances] representing the
      x-coordinates of the instances in the output space coordinates.
    box_heights_widths: A 1D tensor with shape [num_instances， 2] representing
    the height and width of each box.
    classes: A 1D tensor with shape [num_instances] representing the class
      labels for each point
    gaussian_rad: A `int` for the desired radius of the gaussian. If this
      value is set to -1, then the radius is computed using gaussian_iou.
    gaussian_iou: A `float` number for the minimum desired IOU used when
      determining the gaussian radius of center locations in the heatmap.
    output_shape: A `tuple` indicating the output shape of [output_height,
      output_width, num_classes]

  Returns:
    heatmap: A Tensor of size [output_height, output_width, num_classes]
      representing the per class center heatmap. output_height and output_width
      are computed by dividing the input height and width by the stride
      specified during initialization.
  """
  num_objects = classes.get_shape()[0]
  # First compute the desired gaussian radius
  if gaussian_rad == -1:
    radius = tf.map_fn(
        fn=lambda x: gaussian_radius(x, gaussian_iou),
        elems=tf.math.ceil(box_heights_widths))
    radius = tf.math.maximum(tf.math.floor(radius),
                             tf.cast(1.0, radius.dtype))
  else:
    radius = tf.constant([gaussian_rad] * num_objects, box_heights_widths.dtype)
  # These blobs contain information needed to draw the gaussian
  ct_blobs = tf.stack([classes, y_center, x_center, radius],
                      axis=-1)
  
  # Get individual gaussian contributions from each bounding box
  ct_gaussians = tf.map_fn(
      fn=lambda x: draw_gaussian(
          output_shape, x, box_heights_widths.dtype),
      elems=ct_blobs)
  
  # Combine contributions into single heatmaps
  ct_heatmap = tf.math.reduce_max(ct_gaussians, axis=0)
  return ct_heatmap


def assign_centernet_targets(labels: Dict,
                             output_size: List[int],
                             input_size: List[int],
                             num_classes: int = 90,
                             max_num_instances: int = 128,
                             use_gaussian_bump: bool = True,
                             use_odapi_gaussian: bool = False,
                             gaussian_rad: int = -1,
                             gaussian_iou: float = 0.7,
                             class_offset: int = 0,
                             dtype='float32'):
  """ Generates the ground truth labels for centernet.
  
  Ground truth labels are generated by splatting gaussians on heatmaps for
  corners and centers. Regressed features (offsets and sizes) are also
  generated.

  Args:
    labels: A dictionary of COCO ground truth labels with at minimum the
      following fields:
      bbox: A `Tensor` of shape [max_num_instances, 4], where the
        last dimension corresponds to the top left x, top left y, bottom right x,
        and bottom left y coordinates of the bounding box
      classes: A `Tensor` of shape [max_num_instances] that contains
        the class of each box, given in the same order as the boxes
      num_detections: A `Tensor` or int that gives the number of objects
    output_size: A `list` of length 2 containing the desired output height
      and width of the heatmaps
    input_size: A `list` of length 2 the expected input height and width of
      the image
    num_classes: A `Tensor` or `int` for the number of classes.
    max_num_instances: An `int` number of maximum number of instances in an image.
    use_gaussian_bump: A `boolean` indicating whether or not to splat a
      gaussian onto the heatmaps. If set to False, a value of 1 is placed at
      the would-be center of the gaussian.
    use_odapi_gaussian: a `boolean` indicating whether ot not to use center
      target generating logic from ODAPI.
    gaussian_rad: A `int` for the desired radius of the gaussian. If this
      value is set to -1, then the radius is computed using gaussian_iou.
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
  num_objects = labels['num_detections']
  # shape of labels['bbox'] is [max_num_instances, 4]
  # [ymin, xmin, ymax, xmax]
  boxes = tf.cast(labels['bbox'], dtype)[:num_objects]
  # shape of labels['classes'] is [max_num_instances, ]
  classes = tf.cast(labels['classes'] - class_offset, dtype)[:num_objects]
  
  # Compute scaling factors for center/corner positions on heatmap
  # input_size = tf.cast(input_size, dtype)
  # output_size = tf.cast(output_size, dtype)
  input_h, input_w = input_size[0], input_size[1]
  output_h, output_w = output_size[0], output_size[1]
  
  width_ratio = output_w / input_w
  height_ratio = output_h / input_h
  
  # Original box coordinates
  # [num_objects, ]
  ytl, ybr = boxes[..., 0], boxes[..., 2]
  xtl, xbr = boxes[..., 1], boxes[..., 3]
  yct = (ytl + ybr) / 2
  xct = (xtl + xbr) / 2
  
  # Scaled box coordinates (could be floating point)
  # [num_objects, ]
  scale_xct = xct * width_ratio
  scale_yct = yct * height_ratio
  
  # Floor the scaled box coordinates to be placed on heatmaps
  # [num_objects, ]
  scale_xct_floor = tf.math.floor(scale_xct)
  scale_yct_floor = tf.math.floor(scale_yct)
  
  # Offset computations to make up for discretization error
  # used for offset maps
  # [num_objects, 2]
  ct_offset_values = tf.stack([scale_yct - scale_yct_floor,
                               scale_xct - scale_xct_floor], axis=-1)
  
  # Get the scaled box dimensions for computing the gaussian radius
  # [num_objects, ]
  box_widths = boxes[..., 3] - boxes[..., 1]
  box_heights = boxes[..., 2] - boxes[..., 0]
  
  box_widths = box_widths * width_ratio
  box_heights = box_heights * height_ratio
  
  # Used for size map
  # [num_objects, 2]
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
    if use_gaussian_bump:
      # Need to gaussians around the centers and corners of the objects
      if not use_odapi_gaussian:
        ct_heatmap = assign_center_targets(
            box_heights_widths=box_heights_widths,
            classes=classes,
            y_center=scale_yct_floor,
            x_center=scale_xct_floor,
            gaussian_rad=gaussian_rad,
            gaussian_iou=gaussian_iou,
            output_shape=(output_h, output_w, num_classes))
      else:
        ct_heatmap = target_assigner_odapi.assign_center_targets_odapi(
            out_height=output_h,
            out_width=output_w,
            y_center=scale_yct_floor,
            x_center=scale_xct_floor,
            boxes_height=box_heights,
            boxes_width=box_widths,
            channel_onehot=tf.one_hot(tf.cast(classes, tf.int32), num_classes),
            gaussian_iou=gaussian_iou)
    else:
      # Instead of a gaussian, insert 1s in the center and corner heatmaps
      # [num_objects, 3]
      ct_hm_update_indices = tf.cast(
          tf.stack([scale_yct_floor, scale_xct_floor, classes], axis=-1),
          tf.int32)
      
      ct_heatmap = tf.tensor_scatter_nd_update(ct_heatmap,
                                               ct_hm_update_indices,
                                               [1] * num_objects)
    
    # Indices used to update offsets and sizes for valid box instances
    update_indices = cartesian_product(
        tf.range(num_objects), tf.range(2))
    # [num_objects, 2, 2]
    update_indices = tf.reshape(update_indices, shape=[num_objects, 2, 2])
    
    # Write the offsets of each box instance
    ct_offset = tf.tensor_scatter_nd_update(
        ct_offset, update_indices, ct_offset_values)
    
    # Write the size of each bounding box
    size = tf.tensor_scatter_nd_update(
        size, update_indices, box_heights_widths)
    
    # Initially the mask is zeros, so now we unmask each valid box instance
    mask_indices = tf.expand_dims(tf.range(num_objects), -1)
    mask_values = tf.repeat(1, num_objects)
    box_mask = tf.tensor_scatter_nd_update(box_mask, mask_indices, mask_values)
    
    # Write the y and x coordinate of each box center in the heatmap
    box_index_values = tf.cast(
        tf.stack([scale_yct_floor, scale_xct_floor], axis=-1),
        dtype=tf.int32)
    box_indices = tf.tensor_scatter_nd_update(
        box_indices, update_indices, box_index_values)
  
  labels = {
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
  return labels
