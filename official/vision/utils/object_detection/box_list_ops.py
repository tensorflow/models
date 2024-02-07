# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Bounding Box List operations.

Example box operations that are supported:
  * areas: compute bounding box areas
  * iou: pairwise intersection-over-union scores
  * sq_dist: pairwise distances between bounding boxes

Whenever box_list_ops functions output a BoxList, the fields of the incoming
BoxList are retained unless documented otherwise.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import tensorflow as tf, tf_keras

from official.vision.utils.object_detection import box_list
from official.vision.utils.object_detection import ops


class SortOrder(object):
  """Enum class for sort order.

  Attributes:
    ascend: ascend order.
    descend: descend order.
  """
  ascend = 1
  descend = 2


def area(boxlist, scope=None):
  """Computes area of boxes.

  Args:
    boxlist: BoxList holding N boxes
    scope: name scope.

  Returns:
    a tensor with shape [N] representing box areas.
  """
  with tf.name_scope(scope or 'Area'):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist.get(), num_or_size_splits=4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


def height_width(boxlist, scope=None):
  """Computes height and width of boxes in boxlist.

  Args:
    boxlist: BoxList holding N boxes
    scope: name scope.

  Returns:
    Height: A tensor with shape [N] representing box heights.
    Width: A tensor with shape [N] representing box widths.
  """
  with tf.name_scope(scope or 'HeightWidth'):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist.get(), num_or_size_splits=4, axis=1)
    return tf.squeeze(y_max - y_min, [1]), tf.squeeze(x_max - x_min, [1])


def scale(boxlist, y_scale, x_scale, scope=None):
  """scale box coordinates in x and y dimensions.

  Args:
    boxlist: BoxList holding N boxes
    y_scale: (float) scalar tensor
    x_scale: (float) scalar tensor
    scope: name scope.

  Returns:
    boxlist: BoxList holding N boxes
  """
  with tf.name_scope(scope or 'Scale'):
    y_scale = tf.cast(y_scale, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist.get(), num_or_size_splits=4, axis=1)
    y_min = y_scale * y_min
    y_max = y_scale * y_max
    x_min = x_scale * x_min
    x_max = x_scale * x_max
    scaled_boxlist = box_list.BoxList(
        tf.concat([y_min, x_min, y_max, x_max], 1))
    return _copy_extra_fields(scaled_boxlist, boxlist)


def clip_to_window(boxlist, window, filter_nonoverlapping=True, scope=None):
  """Clip bounding boxes to a window.

  This op clips any input bounding boxes (represented by bounding box
  corners) to a window, optionally filtering out boxes that do not
  overlap at all with the window.

  Args:
    boxlist: BoxList holding M_in boxes
    window: a tensor of shape [4] representing the [y_min, x_min, y_max, x_max]
      window to which the op should clip boxes.
    filter_nonoverlapping: whether to filter out boxes that do not overlap at
      all with the window.
    scope: name scope.

  Returns:
    a BoxList holding M_out boxes where M_out <= M_in
  """
  with tf.name_scope(scope or 'ClipToWindow'):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist.get(), num_or_size_splits=4, axis=1)
    win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
    y_min_clipped = tf.maximum(tf.minimum(y_min, win_y_max), win_y_min)
    y_max_clipped = tf.maximum(tf.minimum(y_max, win_y_max), win_y_min)
    x_min_clipped = tf.maximum(tf.minimum(x_min, win_x_max), win_x_min)
    x_max_clipped = tf.maximum(tf.minimum(x_max, win_x_max), win_x_min)
    clipped = box_list.BoxList(
        tf.concat([y_min_clipped, x_min_clipped, y_max_clipped, x_max_clipped],
                  1))
    clipped = _copy_extra_fields(clipped, boxlist)
    if filter_nonoverlapping:
      areas = area(clipped)
      nonzero_area_indices = tf.cast(
          tf.reshape(tf.where(tf.greater(areas, 0.0)), [-1]), tf.int32)
      clipped = gather(clipped, nonzero_area_indices)
    return clipped


def prune_outside_window(boxlist, window, scope=None):
  """Prunes bounding boxes that fall outside a given window.

  This function prunes bounding boxes that even partially fall outside the given
  window. See also clip_to_window which only prunes bounding boxes that fall
  completely outside the window, and clips any bounding boxes that partially
  overflow.

  Args:
    boxlist: a BoxList holding M_in boxes.
    window: a float tensor of shape [4] representing [ymin, xmin, ymax, xmax] of
      the window
    scope: name scope.

  Returns:
    pruned_corners: a tensor with shape [M_out, 4] where M_out <= M_in
    valid_indices: a tensor with shape [M_out] indexing the valid bounding boxes
     in the input tensor.
  """
  with tf.name_scope(scope or 'PruneOutsideWindow'):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist.get(), num_or_size_splits=4, axis=1)
    win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
    coordinate_violations = tf.concat([
        tf.less(y_min, win_y_min),
        tf.less(x_min, win_x_min),
        tf.greater(y_max, win_y_max),
        tf.greater(x_max, win_x_max)
    ], 1)
    valid_indices = tf.reshape(
        tf.where(tf.logical_not(tf.reduce_any(coordinate_violations, 1))), [-1])
    return gather(boxlist, valid_indices), valid_indices


def prune_completely_outside_window(boxlist, window, scope=None):
  """Prunes bounding boxes that fall completely outside of the given window.

  The function clip_to_window prunes bounding boxes that fall
  completely outside the window, but also clips any bounding boxes that
  partially overflow. This function does not clip partially overflowing boxes.

  Args:
    boxlist: a BoxList holding M_in boxes.
    window: a float tensor of shape [4] representing [ymin, xmin, ymax, xmax] of
      the window
    scope: name scope.

  Returns:
    pruned_boxlist: a new BoxList with all bounding boxes partially or fully in
      the window.
    valid_indices: a tensor with shape [M_out] indexing the valid bounding boxes
     in the input tensor.
  """
  with tf.name_scope(scope or 'PruneCompleteleyOutsideWindow'):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist.get(), num_or_size_splits=4, axis=1)
    win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
    coordinate_violations = tf.concat([
        tf.greater_equal(y_min, win_y_max),
        tf.greater_equal(x_min, win_x_max),
        tf.less_equal(y_max, win_y_min),
        tf.less_equal(x_max, win_x_min)
    ], 1)
    valid_indices = tf.reshape(
        tf.where(tf.logical_not(tf.reduce_any(coordinate_violations, 1))), [-1])
    return gather(boxlist, valid_indices), valid_indices


def intersection(boxlist1, boxlist2, scope=None):
  """Compute pairwise intersection areas between boxes.

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.

  Returns:
    a tensor with shape [N, M] representing pairwise intersections
  """
  with tf.name_scope(scope or 'Intersection'):
    y_min1, x_min1, y_max1, x_max1 = tf.split(
        value=boxlist1.get(), num_or_size_splits=4, axis=1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(
        value=boxlist2.get(), num_or_size_splits=4, axis=1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def matched_intersection(boxlist1, boxlist2, scope=None):
  """Compute intersection areas between corresponding boxes in two boxlists.

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding N boxes
    scope: name scope.

  Returns:
    a tensor with shape [N] representing pairwise intersections
  """
  with tf.name_scope(scope or 'MatchedIntersection'):
    y_min1, x_min1, y_max1, x_max1 = tf.split(
        value=boxlist1.get(), num_or_size_splits=4, axis=1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(
        value=boxlist2.get(), num_or_size_splits=4, axis=1)
    min_ymax = tf.minimum(y_max1, y_max2)
    max_ymin = tf.maximum(y_min1, y_min2)
    intersect_heights = tf.maximum(0.0, min_ymax - max_ymin)
    min_xmax = tf.minimum(x_max1, x_max2)
    max_xmin = tf.maximum(x_min1, x_min2)
    intersect_widths = tf.maximum(0.0, min_xmax - max_xmin)
    return tf.reshape(intersect_heights * intersect_widths, [-1])


def iou(boxlist1, boxlist2, scope=None):
  """Computes pairwise intersection-over-union between box collections.

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.

  Returns:
    a tensor with shape [N, M] representing pairwise iou scores.
  """
  with tf.name_scope(scope or 'IOU'):
    intersections = intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0), tf.zeros_like(intersections),
        tf.truediv(intersections, unions))


def matched_iou(boxlist1, boxlist2, scope=None):
  """Compute intersection-over-union between corresponding boxes in boxlists.

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding N boxes
    scope: name scope.

  Returns:
    a tensor with shape [N] representing pairwise iou scores.
  """
  with tf.name_scope(scope or 'MatchedIOU'):
    intersections = matched_intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = areas1 + areas2 - intersections
    return tf.where(
        tf.equal(intersections, 0.0), tf.zeros_like(intersections),
        tf.truediv(intersections, unions))


def ioa(boxlist1, boxlist2, scope=None):
  """Computes pairwise intersection-over-area between box collections.

  intersection-over-area (IOA) between two boxes box1 and box2 is defined as
  their intersection area over box2's area. Note that ioa is not symmetric,
  that is, ioa(box1, box2) != ioa(box2, box1).

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.

  Returns:
    a tensor with shape [N, M] representing pairwise ioa scores.
  """
  with tf.name_scope(scope or 'IOA'):
    intersections = intersection(boxlist1, boxlist2)
    areas = tf.expand_dims(area(boxlist2), 0)
    return tf.truediv(intersections, areas)


def prune_non_overlapping_boxes(boxlist1,
                                boxlist2,
                                min_overlap=0.0,
                                scope=None):
  """Prunes the boxes in boxlist1 that overlap less than thresh with boxlist2.

  For each box in boxlist1, we want its IOA to be more than minoverlap with
  at least one of the boxes in boxlist2. If it does not, we remove it.

  Args:
    boxlist1: BoxList holding N boxes.
    boxlist2: BoxList holding M boxes.
    min_overlap: Minimum required overlap between boxes, to count them as
      overlapping.
    scope: name scope.

  Returns:
    new_boxlist1: A pruned boxlist with size [N', 4].
    keep_inds: A tensor with shape [N'] indexing kept bounding boxes in the
      first input BoxList `boxlist1`.
  """
  with tf.name_scope(scope or 'PruneNonOverlappingBoxes'):
    ioa_ = ioa(boxlist2, boxlist1)  # [M, N] tensor
    ioa_ = tf.reduce_max(ioa_, reduction_indices=[0])  # [N] tensor
    keep_bool = tf.greater_equal(ioa_, tf.constant(min_overlap))
    keep_inds = tf.squeeze(tf.where(keep_bool), axis=[1])
    new_boxlist1 = gather(boxlist1, keep_inds)
    return new_boxlist1, keep_inds


def prune_small_boxes(boxlist, min_side, scope=None):
  """Prunes small boxes in the boxlist which have a side smaller than min_side.

  Args:
    boxlist: BoxList holding N boxes.
    min_side: Minimum width AND height of box to survive pruning.
    scope: name scope.

  Returns:
    A pruned boxlist.
  """
  with tf.name_scope(scope or 'PruneSmallBoxes'):
    height, width = height_width(boxlist)
    is_valid = tf.logical_and(
        tf.greater_equal(width, min_side), tf.greater_equal(height, min_side))
    return gather(boxlist, tf.reshape(tf.where(is_valid), [-1]))


def change_coordinate_frame(boxlist, window, scope=None):
  """Change coordinate frame of the boxlist to be relative to window's frame.

  Given a window of the form [ymin, xmin, ymax, xmax],
  changes bounding box coordinates from boxlist to be relative to this window
  (e.g., the min corner maps to (0,0) and the max corner maps to (1,1)).

  An example use case is data augmentation: where we are given groundtruth
  boxes (boxlist) and would like to randomly crop the image to some
  window (window). In this case we need to change the coordinate frame of
  each groundtruth box to be relative to this new window.

  Args:
    boxlist: A BoxList object holding N boxes.
    window: A rank 1 tensor [4].
    scope: name scope.

  Returns:
    Returns a BoxList object with N boxes.
  """
  with tf.name_scope(scope or 'ChangeCoordinateFrame'):
    win_height = window[2] - window[0]
    win_width = window[3] - window[1]
    boxlist_new = scale(
        box_list.BoxList(boxlist.get() -
                         [window[0], window[1], window[0], window[1]]),
        1.0 / win_height, 1.0 / win_width)
    boxlist_new = _copy_extra_fields(boxlist_new, boxlist)
    return boxlist_new


def sq_dist(boxlist1, boxlist2, scope=None):
  """Computes the pairwise squared distances between box corners.

  This op treats each box as if it were a point in a 4d Euclidean space and
  computes pairwise squared distances.

  Mathematically, we are given two matrices of box coordinates X and Y,
  where X(i,:) is the i'th row of X, containing the 4 numbers defining the
  corners of the i'th box in boxlist1. Similarly Y(j,:) corresponds to
  boxlist2.  We compute
  Z(i,j) = ||X(i,:) - Y(j,:)||^2
         = ||X(i,:)||^2 + ||Y(j,:)||^2 - 2 X(i,:)' * Y(j,:),

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.

  Returns:
    a tensor with shape [N, M] representing pairwise distances
  """
  with tf.name_scope(scope or 'SqDist'):
    sqnorm1 = tf.reduce_sum(tf.square(boxlist1.get()), 1, keep_dims=True)
    sqnorm2 = tf.reduce_sum(tf.square(boxlist2.get()), 1, keep_dims=True)
    innerprod = tf.matmul(
        boxlist1.get(), boxlist2.get(), transpose_a=False, transpose_b=True)
    return sqnorm1 + tf.transpose(sqnorm2) - 2.0 * innerprod


def boolean_mask(boxlist,
                 indicator,
                 fields=None,
                 scope=None,
                 use_static_shapes=False,
                 indicator_sum=None):
  """Select boxes from BoxList according to indicator and return new BoxList.

  `boolean_mask` returns the subset of boxes that are marked as "True" by the
  indicator tensor. By default, `boolean_mask` returns boxes corresponding to
  the input index list, as well as all additional fields stored in the boxlist
  (indexing into the first dimension).  However one can optionally only draw
  from a subset of fields.

  Args:
    boxlist: BoxList holding N boxes
    indicator: a rank-1 boolean tensor
    fields: (optional) list of fields to also gather from.  If None (default),
      all fields are gathered from.  Pass an empty fields list to only gather
      the box coordinates.
    scope: name scope.
    use_static_shapes: Whether to use an implementation with static shape
      gurantees.
    indicator_sum: An integer containing the sum of `indicator` vector. Only
      required if `use_static_shape` is True.

  Returns:
    subboxlist: a BoxList corresponding to the subset of the input BoxList
      specified by indicator
  Raises:
    ValueError: if `indicator` is not a rank-1 boolean tensor.
  """
  with tf.name_scope(scope or 'BooleanMask'):
    if indicator.shape.ndims != 1:
      raise ValueError('indicator should have rank 1')
    if indicator.dtype != tf.bool:
      raise ValueError('indicator should be a boolean tensor')
    if use_static_shapes:
      if not (indicator_sum and isinstance(indicator_sum, int)):
        raise ValueError('`indicator_sum` must be a of type int')
      selected_positions = tf.cast(indicator, dtype=tf.float32)
      indexed_positions = tf.cast(
          tf.multiply(tf.cumsum(selected_positions), selected_positions),
          dtype=tf.int32)
      one_hot_selector = tf.one_hot(
          indexed_positions - 1, indicator_sum, dtype=tf.float32)
      sampled_indices = tf.cast(
          tf.tensordot(
              tf.cast(tf.range(tf.shape(indicator)[0]), dtype=tf.float32),
              one_hot_selector,
              axes=[0, 0]),
          dtype=tf.int32)
      return gather(boxlist, sampled_indices, use_static_shapes=True)
    else:
      subboxlist = box_list.BoxList(tf.boolean_mask(boxlist.get(), indicator))
      if fields is None:
        fields = boxlist.get_extra_fields()
      for field in fields:
        if not boxlist.has_field(field):
          raise ValueError('boxlist must contain all specified fields')
        subfieldlist = tf.boolean_mask(boxlist.get_field(field), indicator)
        subboxlist.add_field(field, subfieldlist)
      return subboxlist


def gather(boxlist, indices, fields=None, scope=None, use_static_shapes=False):
  """Gather boxes from BoxList according to indices and return new BoxList.

  By default, `gather` returns boxes corresponding to the input index list, as
  well as all additional fields stored in the boxlist (indexing into the
  first dimension).  However one can optionally only gather from a
  subset of fields.

  Args:
    boxlist: BoxList holding N boxes
    indices: a rank-1 tensor of type int32 / int64
    fields: (optional) list of fields to also gather from.  If None (default),
      all fields are gathered from.  Pass an empty fields list to only gather
      the box coordinates.
    scope: name scope.
    use_static_shapes: Whether to use an implementation with static shape
      gurantees.

  Returns:
    subboxlist: a BoxList corresponding to the subset of the input BoxList
    specified by indices
  Raises:
    ValueError: if specified field is not contained in boxlist or if the
      indices are not of type int32
  """
  with tf.name_scope(scope or 'Gather'):
    if len(indices.shape.as_list()) != 1:
      raise ValueError('indices should have rank 1')
    if indices.dtype != tf.int32 and indices.dtype != tf.int64:
      raise ValueError('indices should be an int32 / int64 tensor')
    gather_op = tf.gather
    if use_static_shapes:
      gather_op = ops.matmul_gather_on_zeroth_axis
    subboxlist = box_list.BoxList(gather_op(boxlist.get(), indices))
    if fields is None:
      fields = boxlist.get_extra_fields()
    fields += ['boxes']
    for field in fields:
      if not boxlist.has_field(field):
        raise ValueError('boxlist must contain all specified fields')
      subfieldlist = gather_op(boxlist.get_field(field), indices)
      subboxlist.add_field(field, subfieldlist)
    return subboxlist


def concatenate(boxlists, fields=None, scope=None):
  """Concatenate list of BoxLists.

  This op concatenates a list of input BoxLists into a larger BoxList.  It also
  handles concatenation of BoxList fields as long as the field tensor shapes
  are equal except for the first dimension.

  Args:
    boxlists: list of BoxList objects
    fields: optional list of fields to also concatenate.  By default, all fields
      from the first BoxList in the list are included in the concatenation.
    scope: name scope.

  Returns:
    a BoxList with number of boxes equal to
      sum([boxlist.num_boxes() for boxlist in BoxList])
  Raises:
    ValueError: if boxlists is invalid (i.e., is not a list, is empty, or
      contains non BoxList objects), or if requested fields are not contained in
      all boxlists
  """
  with tf.name_scope(scope or 'Concatenate'):
    if not isinstance(boxlists, list):
      raise ValueError('boxlists should be a list')
    if not boxlists:
      raise ValueError('boxlists should have nonzero length')
    for boxlist in boxlists:
      if not isinstance(boxlist, box_list.BoxList):
        raise ValueError('all elements of boxlists should be BoxList objects')
    concatenated = box_list.BoxList(
        tf.concat([boxlist.get() for boxlist in boxlists], 0))
    if fields is None:
      fields = boxlists[0].get_extra_fields()
    for field in fields:
      first_field_shape = boxlists[0].get_field(field).get_shape().as_list()
      first_field_shape[0] = -1
      if None in first_field_shape:
        raise ValueError('field %s must have fully defined shape except for the'
                         ' 0th dimension.' % field)
      for boxlist in boxlists:
        if not boxlist.has_field(field):
          raise ValueError('boxlist must contain all requested fields')
        field_shape = boxlist.get_field(field).get_shape().as_list()
        field_shape[0] = -1
        if field_shape != first_field_shape:
          raise ValueError('field %s must have same shape for all boxlists '
                           'except for the 0th dimension.' % field)
      concatenated_field = tf.concat(
          [boxlist.get_field(field) for boxlist in boxlists], 0)
      concatenated.add_field(field, concatenated_field)
    return concatenated


def sort_by_field(boxlist, field, order=SortOrder.descend, scope=None):
  """Sort boxes and associated fields according to a scalar field.

  A common use case is reordering the boxes according to descending scores.

  Args:
    boxlist: BoxList holding N boxes.
    field: A BoxList field for sorting and reordering the BoxList.
    order: (Optional) descend or ascend. Default is descend.
    scope: name scope.

  Returns:
    sorted_boxlist: A sorted BoxList with the field in the specified order.

  Raises:
    ValueError: if specified field does not exist
    ValueError: if the order is not either descend or ascend
  """
  with tf.name_scope(scope or 'SortByField'):
    if order != SortOrder.descend and order != SortOrder.ascend:
      raise ValueError('Invalid sort order')

    field_to_sort = boxlist.get_field(field)
    if len(field_to_sort.shape.as_list()) != 1:
      raise ValueError('Field should have rank 1')

    num_boxes = boxlist.num_boxes()
    num_entries = tf.size(field_to_sort)
    length_assert = tf.Assert(
        tf.equal(num_boxes, num_entries),
        ['Incorrect field size: actual vs expected.', num_entries, num_boxes])

    with tf.control_dependencies([length_assert]):
      _, sorted_indices = tf.nn.top_k(field_to_sort, num_boxes, sorted=True)

    if order == SortOrder.ascend:
      sorted_indices = tf.reverse_v2(sorted_indices, [0])

    return gather(boxlist, sorted_indices)


def visualize_boxes_in_image(image, boxlist, normalized=False, scope=None):
  """Overlay bounding box list on image.

  Currently this visualization plots a 1 pixel thick red bounding box on top
  of the image.  Note that tf.image.draw_bounding_boxes essentially is
  1 indexed.

  Args:
    image: an image tensor with shape [height, width, 3]
    boxlist: a BoxList
    normalized: (boolean) specify whether corners are to be interpreted as
      absolute coordinates in image space or normalized with respect to the
      image size.
    scope: name scope.

  Returns:
    image_and_boxes: an image tensor with shape [height, width, 3]
  """
  with tf.name_scope(scope or 'VisualizeBoxesInImage'):
    if not normalized:
      height, width, _ = tf.unstack(tf.shape(image))
      boxlist = scale(boxlist, 1.0 / tf.cast(height, tf.float32),
                      1.0 / tf.cast(width, tf.float32))
    corners = tf.expand_dims(boxlist.get(), 0)
    image = tf.expand_dims(image, 0)
    return tf.squeeze(tf.image.draw_bounding_boxes(image, corners), [0])


def filter_field_value_equals(boxlist, field, value, scope=None):
  """Filter to keep only boxes with field entries equal to the given value.

  Args:
    boxlist: BoxList holding N boxes.
    field: field name for filtering.
    value: scalar value.
    scope: name scope.

  Returns:
    a BoxList holding M boxes where M <= N

  Raises:
    ValueError: if boxlist not a BoxList object or if it does not have
      the specified field.
  """
  with tf.name_scope(scope or 'FilterFieldValueEquals'):
    if not isinstance(boxlist, box_list.BoxList):
      raise ValueError('boxlist must be a BoxList')
    if not boxlist.has_field(field):
      raise ValueError('boxlist must contain the specified field')
    filter_field = boxlist.get_field(field)
    gather_index = tf.reshape(tf.where(tf.equal(filter_field, value)), [-1])
    return gather(boxlist, gather_index)


def filter_greater_than(boxlist, thresh, scope=None):
  """Filter to keep only boxes with score exceeding a given threshold.

  This op keeps the collection of boxes whose corresponding scores are
  greater than the input threshold.

  TODO(jonathanhuang): Change function name to filter_scores_greater_than

  Args:
    boxlist: BoxList holding N boxes.  Must contain a 'scores' field
      representing detection scores.
    thresh: scalar threshold
    scope: name scope.

  Returns:
    a BoxList holding M boxes where M <= N

  Raises:
    ValueError: if boxlist not a BoxList object or if it does not
      have a scores field
  """
  with tf.name_scope(scope or 'FilterGreaterThan'):
    if not isinstance(boxlist, box_list.BoxList):
      raise ValueError('boxlist must be a BoxList')
    if not boxlist.has_field('scores'):
      raise ValueError('input boxlist must have \'scores\' field')
    scores = boxlist.get_field('scores')
    if len(scores.shape.as_list()) > 2:
      raise ValueError('Scores should have rank 1 or 2')
    if len(scores.shape.as_list()) == 2 and scores.shape.as_list()[1] != 1:
      raise ValueError('Scores should have rank 1 or have shape '
                       'consistent with [None, 1]')
    high_score_indices = tf.cast(
        tf.reshape(tf.where(tf.greater(scores, thresh)), [-1]), tf.int32)
    return gather(boxlist, high_score_indices)


def non_max_suppression(boxlist, thresh, max_output_size, scope=None):
  """Non maximum suppression.

  This op greedily selects a subset of detection bounding boxes, pruning
  away boxes that have high IOU (intersection over union) overlap (> thresh)
  with already selected boxes.  Note that this only works for a single class ---
  to apply NMS to multi-class predictions, use MultiClassNonMaxSuppression.

  Args:
    boxlist: BoxList holding N boxes.  Must contain a 'scores' field
      representing detection scores.
    thresh: scalar threshold
    max_output_size: maximum number of retained boxes
    scope: name scope.

  Returns:
    a BoxList holding M boxes where M <= max_output_size
  Raises:
    ValueError: if thresh is not in [0, 1]
  """
  with tf.name_scope(scope or 'NonMaxSuppression'):
    if not 0 <= thresh <= 1.0:
      raise ValueError('thresh must be between 0 and 1')
    if not isinstance(boxlist, box_list.BoxList):
      raise ValueError('boxlist must be a BoxList')
    if not boxlist.has_field('scores'):
      raise ValueError('input boxlist must have \'scores\' field')
    selected_indices = tf.image.non_max_suppression(
        boxlist.get(),
        boxlist.get_field('scores'),
        max_output_size,
        iou_threshold=thresh)
    return gather(boxlist, selected_indices)


def _copy_extra_fields(boxlist_to_copy_to, boxlist_to_copy_from):
  """Copies the extra fields of boxlist_to_copy_from to boxlist_to_copy_to.

  Args:
    boxlist_to_copy_to: BoxList to which extra fields are copied.
    boxlist_to_copy_from: BoxList from which fields are copied.

  Returns:
    boxlist_to_copy_to with extra fields.
  """
  for field in boxlist_to_copy_from.get_extra_fields():
    boxlist_to_copy_to.add_field(field, boxlist_to_copy_from.get_field(field))
  return boxlist_to_copy_to


def to_normalized_coordinates(boxlist,
                              height,
                              width,
                              check_range=True,
                              scope=None):
  """Converts absolute box coordinates to normalized coordinates in [0, 1].

  Usually one uses the dynamic shape of the image or conv-layer tensor:
    boxlist = box_list_ops.to_normalized_coordinates(boxlist,
                                                     tf.shape(images)[1],
                                                     tf.shape(images)[2]),

  This function raises an assertion failed error at graph execution time when
  the maximum coordinate is smaller than 1.01 (which means that coordinates are
  already normalized). The value 1.01 is to deal with small rounding errors.

  Args:
    boxlist: BoxList with coordinates in terms of pixel-locations.
    height: Maximum value for height of absolute box coordinates.
    width: Maximum value for width of absolute box coordinates.
    check_range: If True, checks if the coordinates are normalized or not.
    scope: name scope.

  Returns:
    boxlist with normalized coordinates in [0, 1].
  """
  with tf.name_scope(scope or 'ToNormalizedCoordinates'):
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)

    if check_range:
      max_val = tf.reduce_max(boxlist.get())
      max_assert = tf.Assert(
          tf.greater(max_val, 1.01),
          ['max value is lower than 1.01: ', max_val])
      with tf.control_dependencies([max_assert]):
        width = tf.identity(width)

    return scale(boxlist, 1 / height, 1 / width)


def to_absolute_coordinates(boxlist,
                            height,
                            width,
                            check_range=True,
                            maximum_normalized_coordinate=1.1,
                            scope=None):
  """Converts normalized box coordinates to absolute pixel coordinates.

  This function raises an assertion failed error when the maximum box coordinate
  value is larger than maximum_normalized_coordinate (in which case coordinates
  are already absolute).

  Args:
    boxlist: BoxList with coordinates in range [0, 1].
    height: Maximum value for height of absolute box coordinates.
    width: Maximum value for width of absolute box coordinates.
    check_range: If True, checks if the coordinates are normalized or not.
    maximum_normalized_coordinate: Maximum coordinate value to be considered as
      normalized, default to 1.1.
    scope: name scope.

  Returns:
    boxlist with absolute coordinates in terms of the image size.

  """
  with tf.name_scope(scope or 'ToAbsoluteCoordinates'):
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)

    # Ensure range of input boxes is correct.
    if check_range:
      box_maximum = tf.reduce_max(boxlist.get())
      max_assert = tf.Assert(
          tf.greater_equal(maximum_normalized_coordinate, box_maximum), [
              'maximum box coordinate value is larger '
              'than %f: ' % maximum_normalized_coordinate, box_maximum
          ])
      with tf.control_dependencies([max_assert]):
        width = tf.identity(width)

    return scale(boxlist, height, width)


def refine_boxes_multi_class(pool_boxes,
                             num_classes,
                             nms_iou_thresh,
                             nms_max_detections,
                             voting_iou_thresh=0.5):
  """Refines a pool of boxes using non max suppression and box voting.

  Box refinement is done independently for each class.

  Args:
    pool_boxes: (BoxList) A collection of boxes to be refined. pool_boxes must
      have a rank 1 'scores' field and a rank 1 'classes' field.
    num_classes: (int scalar) Number of classes.
    nms_iou_thresh: (float scalar) iou threshold for non max suppression (NMS).
    nms_max_detections: (int scalar) maximum output size for NMS.
    voting_iou_thresh: (float scalar) iou threshold for box voting.

  Returns:
    BoxList of refined boxes.

  Raises:
    ValueError: if
      a) nms_iou_thresh or voting_iou_thresh is not in [0, 1].
      b) pool_boxes is not a BoxList.
      c) pool_boxes does not have a scores and classes field.
  """
  if not 0.0 <= nms_iou_thresh <= 1.0:
    raise ValueError('nms_iou_thresh must be between 0 and 1')
  if not 0.0 <= voting_iou_thresh <= 1.0:
    raise ValueError('voting_iou_thresh must be between 0 and 1')
  if not isinstance(pool_boxes, box_list.BoxList):
    raise ValueError('pool_boxes must be a BoxList')
  if not pool_boxes.has_field('scores'):
    raise ValueError('pool_boxes must have a \'scores\' field')
  if not pool_boxes.has_field('classes'):
    raise ValueError('pool_boxes must have a \'classes\' field')

  refined_boxes = []
  for i in range(num_classes):
    boxes_class = filter_field_value_equals(pool_boxes, 'classes', i)
    refined_boxes_class = refine_boxes(boxes_class, nms_iou_thresh,
                                       nms_max_detections, voting_iou_thresh)
    refined_boxes.append(refined_boxes_class)
  return sort_by_field(concatenate(refined_boxes), 'scores')


def refine_boxes(pool_boxes,
                 nms_iou_thresh,
                 nms_max_detections,
                 voting_iou_thresh=0.5):
  """Refines a pool of boxes using non max suppression and box voting.

  Args:
    pool_boxes: (BoxList) A collection of boxes to be refined. pool_boxes must
      have a rank 1 'scores' field.
    nms_iou_thresh: (float scalar) iou threshold for non max suppression (NMS).
    nms_max_detections: (int scalar) maximum output size for NMS.
    voting_iou_thresh: (float scalar) iou threshold for box voting.

  Returns:
    BoxList of refined boxes.

  Raises:
    ValueError: if
      a) nms_iou_thresh or voting_iou_thresh is not in [0, 1].
      b) pool_boxes is not a BoxList.
      c) pool_boxes does not have a scores field.
  """
  if not 0.0 <= nms_iou_thresh <= 1.0:
    raise ValueError('nms_iou_thresh must be between 0 and 1')
  if not 0.0 <= voting_iou_thresh <= 1.0:
    raise ValueError('voting_iou_thresh must be between 0 and 1')
  if not isinstance(pool_boxes, box_list.BoxList):
    raise ValueError('pool_boxes must be a BoxList')
  if not pool_boxes.has_field('scores'):
    raise ValueError('pool_boxes must have a \'scores\' field')

  nms_boxes = non_max_suppression(pool_boxes, nms_iou_thresh,
                                  nms_max_detections)
  return box_voting(nms_boxes, pool_boxes, voting_iou_thresh)


def box_voting(selected_boxes, pool_boxes, iou_thresh=0.5):
  """Performs box voting as described in S. Gidaris and N.

  Komodakis, ICCV 2015.

  Performs box voting as described in 'Object detection via a multi-region &
  semantic segmentation-aware CNN model', Gidaris and Komodakis, ICCV 2015. For
  each box 'B' in selected_boxes, we find the set 'S' of boxes in pool_boxes
  with iou overlap >= iou_thresh. The location of B is set to the weighted
  average location of boxes in S (scores are used for weighting). And the score
  of B is set to the average score of boxes in S.

  Args:
    selected_boxes: BoxList containing a subset of boxes in pool_boxes. These
      boxes are usually selected from pool_boxes using non max suppression.
    pool_boxes: BoxList containing a set of (possibly redundant) boxes.
    iou_thresh: (float scalar) iou threshold for matching boxes in
      selected_boxes and pool_boxes.

  Returns:
    BoxList containing averaged locations and scores for each box in
    selected_boxes.

  Raises:
    ValueError: if
      a) selected_boxes or pool_boxes is not a BoxList.
      b) if iou_thresh is not in [0, 1].
      c) pool_boxes does not have a scores field.
  """
  if not 0.0 <= iou_thresh <= 1.0:
    raise ValueError('iou_thresh must be between 0 and 1')
  if not isinstance(selected_boxes, box_list.BoxList):
    raise ValueError('selected_boxes must be a BoxList')
  if not isinstance(pool_boxes, box_list.BoxList):
    raise ValueError('pool_boxes must be a BoxList')
  if not pool_boxes.has_field('scores'):
    raise ValueError('pool_boxes must have a \'scores\' field')

  iou_ = iou(selected_boxes, pool_boxes)
  match_indicator = tf.cast(tf.greater(iou_, iou_thresh), dtype=tf.float32)
  num_matches = tf.reduce_sum(match_indicator, 1)
  # TODO(kbanoop): Handle the case where some boxes in selected_boxes do not
  # match to any boxes in pool_boxes. For such boxes without any matches, we
  # should return the original boxes without voting.
  match_assert = tf.Assert(
      tf.reduce_all(tf.greater(num_matches, 0)),
      'Each box in selected_boxes must match with at least one box '
      'in pool_boxes.')

  scores = tf.expand_dims(pool_boxes.get_field('scores'), 1)
  scores_assert = tf.Assert(
      tf.reduce_all(tf.greater_equal(scores, 0)),
      ['Scores must be non negative.'])

  with tf.control_dependencies([scores_assert, match_assert]):
    sum_scores = tf.matmul(match_indicator, scores)
  averaged_scores = tf.reshape(sum_scores, [-1]) / num_matches

  box_locations = tf.matmul(match_indicator,
                            pool_boxes.get() * scores) / sum_scores
  averaged_boxes = box_list.BoxList(box_locations)
  _copy_extra_fields(averaged_boxes, selected_boxes)
  averaged_boxes.add_field('scores', averaged_scores)
  return averaged_boxes


def get_minimal_coverage_box(boxlist, default_box=None, scope=None):
  """Creates a single bounding box which covers all boxes in the boxlist.

  Args:
    boxlist: A Boxlist.
    default_box: A [1, 4] float32 tensor. If no boxes are present in `boxlist`,
      this default box will be returned. If None, will use a default box of
      [[0., 0., 1., 1.]].
    scope: Name scope.

  Returns:
    A [1, 4] float32 tensor with a bounding box that tightly covers all the
    boxes in the box list. If the boxlist does not contain any boxes, the
    default box is returned.
  """
  with tf.name_scope(scope or 'CreateCoverageBox'):
    num_boxes = boxlist.num_boxes()

    def coverage_box(bboxes):
      y_min, x_min, y_max, x_max = tf.split(
          value=bboxes, num_or_size_splits=4, axis=1)
      y_min_coverage = tf.reduce_min(y_min, axis=0)
      x_min_coverage = tf.reduce_min(x_min, axis=0)
      y_max_coverage = tf.reduce_max(y_max, axis=0)
      x_max_coverage = tf.reduce_max(x_max, axis=0)
      return tf.stack(
          [y_min_coverage, x_min_coverage, y_max_coverage, x_max_coverage],
          axis=1)

    default_box = default_box or tf.constant([[0., 0., 1., 1.]])
    return tf.cond(
        tf.greater_equal(num_boxes, 1),
        true_fn=lambda: coverage_box(boxlist.get()),
        false_fn=lambda: default_box)


def sample_boxes_by_jittering(boxlist,
                              num_boxes_to_sample,
                              stddev=0.1,
                              scope=None):
  """Samples num_boxes_to_sample boxes by jittering around boxlist boxes.

  It is possible that this function might generate boxes with size 0. The larger
  the stddev, this is more probable. For a small stddev of 0.1 this probability
  is very small.

  Args:
    boxlist: A boxlist containing N boxes in normalized coordinates.
    num_boxes_to_sample: A positive integer containing the number of boxes to
      sample.
    stddev: Standard deviation. This is used to draw random offsets for the box
      corners from a normal distribution. The offset is multiplied by the box
      size so will be larger in terms of pixels for larger boxes.
    scope: Name scope.

  Returns:
    sampled_boxlist: A boxlist containing num_boxes_to_sample boxes in
      normalized coordinates.
  """
  with tf.name_scope(scope or 'SampleBoxesByJittering'):
    num_boxes = boxlist.num_boxes()
    box_indices = tf.random_uniform([num_boxes_to_sample],
                                    minval=0,
                                    maxval=num_boxes,
                                    dtype=tf.int32)
    sampled_boxes = tf.gather(boxlist.get(), box_indices)
    sampled_boxes_height = sampled_boxes[:, 2] - sampled_boxes[:, 0]
    sampled_boxes_width = sampled_boxes[:, 3] - sampled_boxes[:, 1]
    rand_miny_gaussian = tf.random_normal([num_boxes_to_sample], stddev=stddev)
    rand_minx_gaussian = tf.random_normal([num_boxes_to_sample], stddev=stddev)
    rand_maxy_gaussian = tf.random_normal([num_boxes_to_sample], stddev=stddev)
    rand_maxx_gaussian = tf.random_normal([num_boxes_to_sample], stddev=stddev)
    miny = rand_miny_gaussian * sampled_boxes_height + sampled_boxes[:, 0]
    minx = rand_minx_gaussian * sampled_boxes_width + sampled_boxes[:, 1]
    maxy = rand_maxy_gaussian * sampled_boxes_height + sampled_boxes[:, 2]
    maxx = rand_maxx_gaussian * sampled_boxes_width + sampled_boxes[:, 3]
    maxy = tf.maximum(miny, maxy)
    maxx = tf.maximum(minx, maxx)
    sampled_boxes = tf.stack([miny, minx, maxy, maxx], axis=1)
    sampled_boxes = tf.maximum(tf.minimum(sampled_boxes, 1.0), 0.0)
    return box_list.BoxList(sampled_boxes)
