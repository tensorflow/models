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

"""Generates grid anchors on the fly corresponding to multiple CNN layers.

Generates grid anchors on the fly corresponding to multiple CNN layers as
described in:
"SSD: Single Shot MultiBox Detector"
Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg
(see Section 2.2: Choosing scales and aspect ratios for default boxes)
"""

import numpy as np

import tensorflow as tf

from object_detection.anchor_generators import grid_anchor_generator
from object_detection.core import anchor_generator
from object_detection.core import box_list_ops


class MultipleGridAnchorGenerator(anchor_generator.AnchorGenerator):
  """Generate a grid of anchors for multiple CNN layers."""

  def __init__(self,
               box_specs_list,
               base_anchor_size=None,
               anchor_strides=None,
               anchor_offsets=None,
               clip_window=None):
    """Constructs a MultipleGridAnchorGenerator.

    To construct anchors, at multiple grid resolutions, one must provide a
    list of feature_map_shape_list (e.g., [(8, 8), (4, 4)]), and for each grid
    size, a corresponding list of (scale, aspect ratio) box specifications.

    For example:
    box_specs_list = [[(.1, 1.0), (.1, 2.0)],  # for 8x8 grid
                      [(.2, 1.0), (.3, 1.0), (.2, 2.0)]]  # for 4x4 grid

    To support the fully convolutional setting, we pass grid sizes in at
    generation time, while scale and aspect ratios are fixed at construction
    time.

    Args:
      box_specs_list: list of list of (scale, aspect ratio) pairs with the
        outside list having the same number of entries as feature_map_shape_list
        (which is passed in at generation time).
      base_anchor_size: base anchor size as [height, width]
                        (length-2 float tensor, default=[1.0, 1.0]).
                        The height and width values are normalized to the
                        minimum dimension of the input height and width, so that
                        when the base anchor height equals the base anchor
                        width, the resulting anchor is square even if the input
                        image is not square.
      anchor_strides: list of pairs of strides in pixels (in y and x directions
        respectively). For example, setting anchor_strides=[(25, 25), (50, 50)]
        means that we want the anchors corresponding to the first layer to be
        strided by 25 pixels and those in the second layer to be strided by 50
        pixels in both y and x directions. If anchor_strides=None, they are set
        to be the reciprocal of the corresponding feature map shapes.
      anchor_offsets: list of pairs of offsets in pixels (in y and x directions
        respectively). The offset specifies where we want the center of the
        (0, 0)-th anchor to lie for each layer. For example, setting
        anchor_offsets=[(10, 10), (20, 20)]) means that we want the
        (0, 0)-th anchor of the first layer to lie at (10, 10) in pixel space
        and likewise that we want the (0, 0)-th anchor of the second layer to
        lie at (25, 25) in pixel space. If anchor_offsets=None, then they are
        set to be half of the corresponding anchor stride.
      clip_window: a tensor of shape [4] specifying a window to which all
        anchors should be clipped. If clip_window is None, then no clipping
        is performed.

    Raises:
      ValueError: if box_specs_list is not a list of list of pairs
      ValueError: if clip_window is not either None or a tensor of shape [4]
    """
    if isinstance(box_specs_list, list) and all(
        [isinstance(list_item, list) for list_item in box_specs_list]):
      self._box_specs = box_specs_list
    else:
      raise ValueError('box_specs_list is expected to be a '
                       'list of lists of pairs')
    if base_anchor_size is None:
      base_anchor_size = tf.constant([256, 256], dtype=tf.float32)
    self._base_anchor_size = base_anchor_size
    self._anchor_strides = anchor_strides
    self._anchor_offsets = anchor_offsets
    if clip_window is not None and clip_window.get_shape().as_list() != [4]:
      raise ValueError('clip_window must either be None or a shape [4] tensor')
    self._clip_window = clip_window
    self._scales = []
    self._aspect_ratios = []
    for box_spec in self._box_specs:
      if not all([isinstance(entry, tuple) and len(entry) == 2
                  for entry in box_spec]):
        raise ValueError('box_specs_list is expected to be a '
                         'list of lists of pairs')
      scales, aspect_ratios = zip(*box_spec)
      self._scales.append(scales)
      self._aspect_ratios.append(aspect_ratios)

    for arg, arg_name in zip([self._anchor_strides, self._anchor_offsets],
                             ['anchor_strides', 'anchor_offsets']):
      if arg and not (isinstance(arg, list) and
                      len(arg) == len(self._box_specs)):
        raise ValueError('%s must be a list with the same length '
                         'as self._box_specs' % arg_name)
      if arg and not all([
          isinstance(list_item, tuple) and len(list_item) == 2
          for list_item in arg
      ]):
        raise ValueError('%s must be a list of pairs.' % arg_name)

  def name_scope(self):
    return 'MultipleGridAnchorGenerator'

  def num_anchors_per_location(self):
    """Returns the number of anchors per spatial location.

    Returns:
      a list of integers, one for each expected feature map to be passed to
      the Generate function.
    """
    return [len(box_specs) for box_specs in self._box_specs]

  def _generate(self, feature_map_shape_list, im_height=1, im_width=1):
    """Generates a collection of bounding boxes to be used as anchors.

    The number of anchors generated for a single grid with shape MxM where we
    place k boxes over each grid center is k*M^2 and thus the total number of
    anchors is the sum over all grids. In our box_specs_list example
    (see the constructor docstring), we would place two boxes over each grid
    point on an 8x8 grid and three boxes over each grid point on a 4x4 grid and
    thus end up with 2*8^2 + 3*4^2 = 176 anchors in total. The layout of the
    output anchors follows the order of how the grid sizes and box_specs are
    specified (with box_spec index varying the fastest, followed by width
    index, then height index, then grid index).

    Args:
      feature_map_shape_list: list of pairs of convnet layer resolutions in the
        format [(height_0, width_0), (height_1, width_1), ...]. For example,
        setting feature_map_shape_list=[(8, 8), (7, 7)] asks for anchors that
        correspond to an 8x8 layer followed by a 7x7 layer.
      im_height: the height of the image to generate the grid for. If both
        im_height and im_width are 1, the generated anchors default to
        absolute coordinates, otherwise normalized coordinates are produced.
      im_width: the width of the image to generate the grid for. If both
        im_height and im_width are 1, the generated anchors default to
        absolute coordinates, otherwise normalized coordinates are produced.

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
            and len(feature_map_shape_list) == len(self._box_specs)):
      raise ValueError('feature_map_shape_list must be a list with the same '
                       'length as self._box_specs')
    if not all([isinstance(list_item, tuple) and len(list_item) == 2
                for list_item in feature_map_shape_list]):
      raise ValueError('feature_map_shape_list must be a list of pairs.')

    im_height = tf.to_float(im_height)
    im_width = tf.to_float(im_width)

    if not self._anchor_strides:
      anchor_strides = [(1.0 / tf.to_float(pair[0]), 1.0 / tf.to_float(pair[1]))
                        for pair in feature_map_shape_list]
    else:
      anchor_strides = [(tf.to_float(stride[0]) / im_height,
                         tf.to_float(stride[1]) / im_width)
                        for stride in self._anchor_strides]
    if not self._anchor_offsets:
      anchor_offsets = [(0.5 * stride[0], 0.5 * stride[1])
                        for stride in anchor_strides]
    else:
      anchor_offsets = [(tf.to_float(offset[0]) / im_height,
                         tf.to_float(offset[1]) / im_width)
                        for offset in self._anchor_offsets]

    for arg, arg_name in zip([anchor_strides, anchor_offsets],
                             ['anchor_strides', 'anchor_offsets']):
      if not (isinstance(arg, list) and len(arg) == len(self._box_specs)):
        raise ValueError('%s must be a list with the same length '
                         'as self._box_specs' % arg_name)
      if not all([isinstance(list_item, tuple) and len(list_item) == 2
                  for list_item in arg]):
        raise ValueError('%s must be a list of pairs.' % arg_name)

    anchor_grid_list = []
    min_im_shape = tf.minimum(im_height, im_width)
    scale_height = min_im_shape / im_height
    scale_width = min_im_shape / im_width
    base_anchor_size = [
        scale_height * self._base_anchor_size[0],
        scale_width * self._base_anchor_size[1]
    ]
    for feature_map_index, (grid_size, scales, aspect_ratios, stride,
                            offset) in enumerate(
                                zip(feature_map_shape_list, self._scales,
                                    self._aspect_ratios, anchor_strides,
                                    anchor_offsets)):
      tiled_anchors = grid_anchor_generator.tile_anchors(
          grid_height=grid_size[0],
          grid_width=grid_size[1],
          scales=scales,
          aspect_ratios=aspect_ratios,
          base_anchor_size=base_anchor_size,
          anchor_stride=stride,
          anchor_offset=offset)
      if self._clip_window is not None:
        tiled_anchors = box_list_ops.clip_to_window(
            tiled_anchors, self._clip_window, filter_nonoverlapping=False)
      num_anchors_in_layer = tiled_anchors.num_boxes_static()
      if num_anchors_in_layer is None:
        num_anchors_in_layer = tiled_anchors.num_boxes()
      anchor_indices = feature_map_index * tf.ones([num_anchors_in_layer])
      tiled_anchors.add_field('feature_map_index', anchor_indices)
      anchor_grid_list.append(tiled_anchors)

    return anchor_grid_list


def create_ssd_anchors(num_layers=6,
                       min_scale=0.2,
                       max_scale=0.95,
                       scales=None,
                       aspect_ratios=(1.0, 2.0, 3.0, 1.0 / 2, 1.0 / 3),
                       interpolated_scale_aspect_ratio=1.0,
                       base_anchor_size=None,
                       anchor_strides=None,
                       anchor_offsets=None,
                       reduce_boxes_in_lowest_layer=True):
  """Creates MultipleGridAnchorGenerator for SSD anchors.

  This function instantiates a MultipleGridAnchorGenerator that reproduces
  ``default box`` construction proposed by Liu et al in the SSD paper.
  See Section 2.2 for details. Grid sizes are assumed to be passed in
  at generation time from finest resolution to coarsest resolution --- this is
  used to (linearly) interpolate scales of anchor boxes corresponding to the
  intermediate grid sizes.

  Anchors that are returned by calling the `generate` method on the returned
  MultipleGridAnchorGenerator object are always in normalized coordinates
  and clipped to the unit square: (i.e. all coordinates lie in [0, 1]x[0, 1]).

  Args:
    num_layers: integer number of grid layers to create anchors for (actual
      grid sizes passed in at generation time)
    min_scale: scale of anchors corresponding to finest resolution (float)
    max_scale: scale of anchors corresponding to coarsest resolution (float)
    scales: As list of anchor scales to use. When not None and not empty,
      min_scale and max_scale are not used.
    aspect_ratios: list or tuple of (float) aspect ratios to place on each
      grid point.
    interpolated_scale_aspect_ratio: An additional anchor is added with this
      aspect ratio and a scale interpolated between the scale for a layer
      and the scale for the next layer (1.0 for the last layer).
      This anchor is not included if this value is 0.
    base_anchor_size: base anchor size as [height, width].
      The height and width values are normalized to the minimum dimension of the
      input height and width, so that when the base anchor height equals the
      base anchor width, the resulting anchor is square even if the input image
      is not square.
    anchor_strides: list of pairs of strides in pixels (in y and x directions
      respectively). For example, setting anchor_strides=[(25, 25), (50, 50)]
      means that we want the anchors corresponding to the first layer to be
      strided by 25 pixels and those in the second layer to be strided by 50
      pixels in both y and x directions. If anchor_strides=None, they are set to
      be the reciprocal of the corresponding feature map shapes.
    anchor_offsets: list of pairs of offsets in pixels (in y and x directions
      respectively). The offset specifies where we want the center of the
      (0, 0)-th anchor to lie for each layer. For example, setting
      anchor_offsets=[(10, 10), (20, 20)]) means that we want the
      (0, 0)-th anchor of the first layer to lie at (10, 10) in pixel space
      and likewise that we want the (0, 0)-th anchor of the second layer to lie
      at (25, 25) in pixel space. If anchor_offsets=None, then they are set to
      be half of the corresponding anchor stride.
    reduce_boxes_in_lowest_layer: a boolean to indicate whether the fixed 3
      boxes per location is used in the lowest layer.

  Returns:
    a MultipleGridAnchorGenerator
  """
  if base_anchor_size is None:
    base_anchor_size = [1.0, 1.0]
  base_anchor_size = tf.constant(base_anchor_size, dtype=tf.float32)
  box_specs_list = []
  if scales is None or not scales:
    scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1)
              for i in range(num_layers)] + [1.0]
  else:
    # Add 1.0 to the end, which will only be used in scale_next below and used
    # for computing an interpolated scale for the largest scale in the list.
    scales += [1.0]

  for layer, scale, scale_next in zip(
      range(num_layers), scales[:-1], scales[1:]):
    layer_box_specs = []
    if layer == 0 and reduce_boxes_in_lowest_layer:
      layer_box_specs = [(0.1, 1.0), (scale, 2.0), (scale, 0.5)]
    else:
      for aspect_ratio in aspect_ratios:
        layer_box_specs.append((scale, aspect_ratio))
      # Add one more anchor, with a scale between the current scale, and the
      # scale for the next layer, with a specified aspect ratio (1.0 by
      # default).
      if interpolated_scale_aspect_ratio > 0.0:
        layer_box_specs.append((np.sqrt(scale*scale_next),
                                interpolated_scale_aspect_ratio))
    box_specs_list.append(layer_box_specs)

  return MultipleGridAnchorGenerator(box_specs_list, base_anchor_size,
                                     anchor_strides, anchor_offsets)
