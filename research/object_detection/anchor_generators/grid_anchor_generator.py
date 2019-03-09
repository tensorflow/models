# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Generates grid anchors on the fly as used in Faster RCNN.

Generates grid anchors on the fly as described in:
"Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks"
Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
"""

import tensorflow as tf

from object_detection.core import anchor_generator
from object_detection.core import box_list
from object_detection.utils import ops


class GridAnchorGenerator(anchor_generator.AnchorGenerator):
  """Generates a grid of anchors at given scales and aspect ratios."""

  def __init__(self,
               scales=(0.5, 1.0, 2.0),
               aspect_ratios=(0.5, 1.0, 2.0),
               base_anchor_size=None,
               anchor_stride=None,
               anchor_offset=None):
    """Constructs a GridAnchorGenerator.

    Args:
      scales: a list of (float) scales, default=(0.5, 1.0, 2.0)
      aspect_ratios: a list of (float) aspect ratios, default=(0.5, 1.0, 2.0)
      base_anchor_size: base anchor size as height, width (
                        (length-2 float32 list or tensor, default=[256, 256])
      anchor_stride: difference in centers between base anchors for adjacent
                     grid positions (length-2 float32 list or tensor,
                     default=[16, 16])
      anchor_offset: center of the anchor with scale and aspect ratio 1 for the
                     upper left element of the grid, this should be zero for
                     feature networks with only VALID padding and even receptive
                     field size, but may need additional calculation if other
                     padding is used (length-2 float32 list or tensor,
                     default=[0, 0])
    """
    # Handle argument defaults
    if base_anchor_size is None:
      base_anchor_size = [256, 256]
    if anchor_stride is None:
      anchor_stride = [16, 16]
    if anchor_offset is None:
      anchor_offset = [0, 0]

    self._scales = scales
    self._aspect_ratios = aspect_ratios
    self._base_anchor_size = base_anchor_size
    self._anchor_stride = anchor_stride
    self._anchor_offset = anchor_offset

  def name_scope(self):
    return 'GridAnchorGenerator'

  def num_anchors_per_location(self):
    """Returns the number of anchors per spatial location.

    Returns:
      a list of integers, one for each expected feature map to be passed to
      the `generate` function.
    """
    return [len(self._scales) * len(self._aspect_ratios)]

  def _generate(self, feature_map_shape_list):
    """Generates a collection of bounding boxes to be used as anchors.

    Args:
      feature_map_shape_list: list of pairs of convnet layer resolutions in the
        format [(height_0, width_0)].  For example, setting
        feature_map_shape_list=[(8, 8)] asks for anchors that correspond
        to an 8x8 layer.  For this anchor generator, only lists of length 1 are
        allowed.

    Returns:
      boxes_list: a list of BoxLists each holding anchor boxes corresponding to
        the input feature map shapes.

    Raises:
      ValueError: if feature_map_shape_list, box_specs_list do not have the same
        length.
      ValueError: if feature_map_shape_list does not consist of pairs of
        integers
    """
    if not (isinstance(feature_map_shape_list, list)
            and len(feature_map_shape_list) == 1):
      raise ValueError('feature_map_shape_list must be a list of length 1.')
    if not all([isinstance(list_item, tuple) and len(list_item) == 2
                for list_item in feature_map_shape_list]):
      raise ValueError('feature_map_shape_list must be a list of pairs.')
    self._base_anchor_size = tf.to_float(tf.convert_to_tensor(
        self._base_anchor_size))
    self._anchor_stride = tf.to_float(tf.convert_to_tensor(
        self._anchor_stride))
    self._anchor_offset = tf.to_float(tf.convert_to_tensor(
        self._anchor_offset))

    grid_height, grid_width = feature_map_shape_list[0]
    scales_grid, aspect_ratios_grid = ops.meshgrid(self._scales,
                                                   self._aspect_ratios)
    scales_grid = tf.reshape(scales_grid, [-1])
    aspect_ratios_grid = tf.reshape(aspect_ratios_grid, [-1])
    anchors = tile_anchors(grid_height,
                           grid_width,
                           scales_grid,
                           aspect_ratios_grid,
                           self._base_anchor_size,
                           self._anchor_stride,
                           self._anchor_offset)

    num_anchors = anchors.num_boxes_static()
    if num_anchors is None:
      num_anchors = anchors.num_boxes()
    anchor_indices = tf.zeros([num_anchors])
    anchors.add_field('feature_map_index', anchor_indices)
    return [anchors]


def tile_anchors(grid_height,
                 grid_width,
                 scales,
                 aspect_ratios,
                 base_anchor_size,
                 anchor_stride,
                 anchor_offset):
  """Create a tiled set of anchors strided along a grid in image space.

  This op creates a set of anchor boxes by placing a "basis" collection of
  boxes with user-specified scales and aspect ratios centered at evenly
  distributed points along a grid.  The basis collection is specified via the
  scale and aspect_ratios arguments.  For example, setting scales=[.1, .2, .2]
  and aspect ratios = [2,2,1/2] means that we create three boxes: one with scale
  .1, aspect ratio 2, one with scale .2, aspect ratio 2, and one with scale .2
  and aspect ratio 1/2.  Each box is multiplied by "base_anchor_size" before
  placing it over its respective center.

  Grid points are specified via grid_height, grid_width parameters as well as
  the anchor_stride and anchor_offset parameters.

  Args:
    grid_height: size of the grid in the y direction (int or int scalar tensor)
    grid_width: size of the grid in the x direction (int or int scalar tensor)
    scales: a 1-d  (float) tensor representing the scale of each box in the
      basis set.
    aspect_ratios: a 1-d (float) tensor representing the aspect ratio of each
      box in the basis set.  The length of the scales and aspect_ratios tensors
      must be equal.
    base_anchor_size: base anchor size as [height, width]
      (float tensor of shape [2])
    anchor_stride: difference in centers between base anchors for adjacent grid
                   positions (float tensor of shape [2])
    anchor_offset: center of the anchor with scale and aspect ratio 1 for the
                   upper left element of the grid, this should be zero for
                   feature networks with only VALID padding and even receptive
                   field size, but may need some additional calculation if other
                   padding is used (float tensor of shape [2])
  Returns:
    a BoxList holding a collection of N anchor boxes
  """
  ratio_sqrts = tf.sqrt(aspect_ratios)
  heights = scales / ratio_sqrts * base_anchor_size[0]
  widths = scales * ratio_sqrts * base_anchor_size[1]

  # Get a grid of box centers
  y_centers = tf.to_float(tf.range(grid_height))
  y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
  x_centers = tf.to_float(tf.range(grid_width))
  x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
  x_centers, y_centers = ops.meshgrid(x_centers, y_centers)

  widths_grid, x_centers_grid = ops.meshgrid(widths, x_centers)
  heights_grid, y_centers_grid = ops.meshgrid(heights, y_centers)
  bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=3)
  bbox_sizes = tf.stack([heights_grid, widths_grid], axis=3)
  bbox_centers = tf.reshape(bbox_centers, [-1, 2])
  bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
  bbox_corners = _center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)
  return box_list.BoxList(bbox_corners)


def _center_size_bbox_to_corners_bbox(centers, sizes):
  """Converts bbox center-size representation to corners representation.

  Args:
    centers: a tensor with shape [N, 2] representing bounding box centers
    sizes: a tensor with shape [N, 2] representing bounding boxes

  Returns:
    corners: tensor with shape [N, 4] representing bounding boxes in corners
      representation
  """
  return tf.concat([centers - .5 * sizes, centers + .5 * sizes], 1)
