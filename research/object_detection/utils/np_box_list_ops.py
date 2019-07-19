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

"""Bounding Box List operations for Numpy BoxLists.

Example box operations that are supported:
  * Areas: compute bounding box areas
  * IOU: pairwise intersection-over-union scores
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range

from object_detection.utils import np_box_list
from object_detection.utils import np_box_ops


class SortOrder(object):
  """Enum class for sort order.

  Attributes:
    ascend: ascend order.
    descend: descend order.
  """
  ASCEND = 1
  DESCEND = 2


def area(boxlist):
  """Computes area of boxes.

  Args:
    boxlist: BoxList holding N boxes

  Returns:
    a numpy array with shape [N*1] representing box areas
  """
  y_min, x_min, y_max, x_max = boxlist.get_coordinates()
  return (y_max - y_min) * (x_max - x_min)


def intersection(boxlist1, boxlist2):
  """Compute pairwise intersection areas between boxes.

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes

  Returns:
    a numpy array with shape [N*M] representing pairwise intersection area
  """
  return np_box_ops.intersection(boxlist1.get(), boxlist2.get())


def iou(boxlist1, boxlist2):
  """Computes pairwise intersection-over-union between box collections.

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes

  Returns:
    a numpy array with shape [N, M] representing pairwise iou scores.
  """
  return np_box_ops.iou(boxlist1.get(), boxlist2.get())


def ioa(boxlist1, boxlist2):
  """Computes pairwise intersection-over-area between box collections.

  Intersection-over-area (ioa) between two boxes box1 and box2 is defined as
  their intersection area over box2's area. Note that ioa is not symmetric,
  that is, IOA(box1, box2) != IOA(box2, box1).

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes

  Returns:
    a numpy array with shape [N, M] representing pairwise ioa scores.
  """
  return np_box_ops.ioa(boxlist1.get(), boxlist2.get())


def gather(boxlist, indices, fields=None):
  """Gather boxes from BoxList according to indices and return new BoxList.

  By default, gather returns boxes corresponding to the input index list, as
  well as all additional fields stored in the boxlist (indexing into the
  first dimension).  However one can optionally only gather from a
  subset of fields.

  Args:
    boxlist: BoxList holding N boxes
    indices: a 1-d numpy array of type int_
    fields: (optional) list of fields to also gather from.  If None (default),
        all fields are gathered from.  Pass an empty fields list to only gather
        the box coordinates.

  Returns:
    subboxlist: a BoxList corresponding to the subset of the input BoxList
        specified by indices

  Raises:
    ValueError: if specified field is not contained in boxlist or if the
        indices are not of type int_
  """
  if indices.size:
    if np.amax(indices) >= boxlist.num_boxes() or np.amin(indices) < 0:
      raise ValueError('indices are out of valid range.')
  subboxlist = np_box_list.BoxList(boxlist.get()[indices, :])
  if fields is None:
    fields = boxlist.get_extra_fields()
  for field in fields:
    extra_field_data = boxlist.get_field(field)
    subboxlist.add_field(field, extra_field_data[indices, ...])
  return subboxlist


def sort_by_field(boxlist, field, order=SortOrder.DESCEND):
  """Sort boxes and associated fields according to a scalar field.

  A common use case is reordering the boxes according to descending scores.

  Args:
    boxlist: BoxList holding N boxes.
    field: A BoxList field for sorting and reordering the BoxList.
    order: (Optional) 'descend' or 'ascend'. Default is descend.

  Returns:
    sorted_boxlist: A sorted BoxList with the field in the specified order.

  Raises:
    ValueError: if specified field does not exist or is not of single dimension.
    ValueError: if the order is not either descend or ascend.
  """
  if not boxlist.has_field(field):
    raise ValueError('Field ' + field + ' does not exist')
  if len(boxlist.get_field(field).shape) != 1:
    raise ValueError('Field ' + field + 'should be single dimension.')
  if order != SortOrder.DESCEND and order != SortOrder.ASCEND:
    raise ValueError('Invalid sort order')

  field_to_sort = boxlist.get_field(field)
  sorted_indices = np.argsort(field_to_sort)
  if order == SortOrder.DESCEND:
    sorted_indices = sorted_indices[::-1]
  return gather(boxlist, sorted_indices)


def non_max_suppression(boxlist,
                        max_output_size=10000,
                        iou_threshold=1.0,
                        score_threshold=-10.0):
  """Non maximum suppression.

  This op greedily selects a subset of detection bounding boxes, pruning
  away boxes that have high IOU (intersection over union) overlap (> thresh)
  with already selected boxes. In each iteration, the detected bounding box with
  highest score in the available pool is selected.

  Args:
    boxlist: BoxList holding N boxes.  Must contain a 'scores' field
      representing detection scores. All scores belong to the same class.
    max_output_size: maximum number of retained boxes
    iou_threshold: intersection over union threshold.
    score_threshold: minimum score threshold. Remove the boxes with scores
                     less than this value. Default value is set to -10. A very
                     low threshold to pass pretty much all the boxes, unless
                     the user sets a different score threshold.

  Returns:
    a BoxList holding M boxes where M <= max_output_size
  Raises:
    ValueError: if 'scores' field does not exist
    ValueError: if threshold is not in [0, 1]
    ValueError: if max_output_size < 0
  """
  if not boxlist.has_field('scores'):
    raise ValueError('Field scores does not exist')
  if iou_threshold < 0. or iou_threshold > 1.0:
    raise ValueError('IOU threshold must be in [0, 1]')
  if max_output_size < 0:
    raise ValueError('max_output_size must be bigger than 0.')

  boxlist = filter_scores_greater_than(boxlist, score_threshold)
  if boxlist.num_boxes() == 0:
    return boxlist

  boxlist = sort_by_field(boxlist, 'scores')

  # Prevent further computation if NMS is disabled.
  if iou_threshold == 1.0:
    if boxlist.num_boxes() > max_output_size:
      selected_indices = np.arange(max_output_size)
      return gather(boxlist, selected_indices)
    else:
      return boxlist

  boxes = boxlist.get()
  num_boxes = boxlist.num_boxes()
  # is_index_valid is True only for all remaining valid boxes,
  is_index_valid = np.full(num_boxes, 1, dtype=bool)
  selected_indices = []
  num_output = 0
  for i in range(num_boxes):
    if num_output < max_output_size:
      if is_index_valid[i]:
        num_output += 1
        selected_indices.append(i)
        is_index_valid[i] = False
        valid_indices = np.where(is_index_valid)[0]
        if valid_indices.size == 0:
          break

        intersect_over_union = np_box_ops.iou(
            np.expand_dims(boxes[i, :], axis=0), boxes[valid_indices, :])
        intersect_over_union = np.squeeze(intersect_over_union, axis=0)
        is_index_valid[valid_indices] = np.logical_and(
            is_index_valid[valid_indices],
            intersect_over_union <= iou_threshold)
  return gather(boxlist, np.array(selected_indices))


def multi_class_non_max_suppression(boxlist, score_thresh, iou_thresh,
                                    max_output_size):
  """Multi-class version of non maximum suppression.

  This op greedily selects a subset of detection bounding boxes, pruning
  away boxes that have high IOU (intersection over union) overlap (> thresh)
  with already selected boxes.  It operates independently for each class for
  which scores are provided (via the scores field of the input box_list),
  pruning boxes with score less than a provided threshold prior to
  applying NMS.

  Args:
    boxlist: BoxList holding N boxes.  Must contain a 'scores' field
      representing detection scores.  This scores field is a tensor that can
      be 1 dimensional (in the case of a single class) or 2-dimensional, which
      which case we assume that it takes the shape [num_boxes, num_classes].
      We further assume that this rank is known statically and that
      scores.shape[1] is also known (i.e., the number of classes is fixed
      and known at graph construction time).
    score_thresh: scalar threshold for score (low scoring boxes are removed).
    iou_thresh: scalar threshold for IOU (boxes that that high IOU overlap
      with previously selected boxes are removed).
    max_output_size: maximum number of retained boxes per class.

  Returns:
    a BoxList holding M boxes with a rank-1 scores field representing
      corresponding scores for each box with scores sorted in decreasing order
      and a rank-1 classes field representing a class label for each box.
  Raises:
    ValueError: if iou_thresh is not in [0, 1] or if input boxlist does not have
      a valid scores field.
  """
  if not 0 <= iou_thresh <= 1.0:
    raise ValueError('thresh must be between 0 and 1')
  if not isinstance(boxlist, np_box_list.BoxList):
    raise ValueError('boxlist must be a BoxList')
  if not boxlist.has_field('scores'):
    raise ValueError('input boxlist must have \'scores\' field')
  scores = boxlist.get_field('scores')
  if len(scores.shape) == 1:
    scores = np.reshape(scores, [-1, 1])
  elif len(scores.shape) == 2:
    if scores.shape[1] is None:
      raise ValueError('scores field must have statically defined second '
                       'dimension')
  else:
    raise ValueError('scores field must be of rank 1 or 2')
  num_boxes = boxlist.num_boxes()
  num_scores = scores.shape[0]
  num_classes = scores.shape[1]

  if num_boxes != num_scores:
    raise ValueError('Incorrect scores field length: actual vs expected.')

  selected_boxes_list = []
  for class_idx in range(num_classes):
    boxlist_and_class_scores = np_box_list.BoxList(boxlist.get())
    class_scores = np.reshape(scores[0:num_scores, class_idx], [-1])
    boxlist_and_class_scores.add_field('scores', class_scores)
    boxlist_filt = filter_scores_greater_than(boxlist_and_class_scores,
                                              score_thresh)
    nms_result = non_max_suppression(boxlist_filt,
                                     max_output_size=max_output_size,
                                     iou_threshold=iou_thresh,
                                     score_threshold=score_thresh)
    nms_result.add_field(
        'classes', np.zeros_like(nms_result.get_field('scores')) + class_idx)
    selected_boxes_list.append(nms_result)
  selected_boxes = concatenate(selected_boxes_list)
  sorted_boxes = sort_by_field(selected_boxes, 'scores')
  return sorted_boxes


def scale(boxlist, y_scale, x_scale):
  """Scale box coordinates in x and y dimensions.

  Args:
    boxlist: BoxList holding N boxes
    y_scale: float
    x_scale: float

  Returns:
    boxlist: BoxList holding N boxes
  """
  y_min, x_min, y_max, x_max = np.array_split(boxlist.get(), 4, axis=1)
  y_min = y_scale * y_min
  y_max = y_scale * y_max
  x_min = x_scale * x_min
  x_max = x_scale * x_max
  scaled_boxlist = np_box_list.BoxList(np.hstack([y_min, x_min, y_max, x_max]))

  fields = boxlist.get_extra_fields()
  for field in fields:
    extra_field_data = boxlist.get_field(field)
    scaled_boxlist.add_field(field, extra_field_data)

  return scaled_boxlist


def clip_to_window(boxlist, window, filter_nonoverlapping=True):
  """Clip bounding boxes to a window.

  This op clips input bounding boxes (represented by bounding box
  corners) to a window, optionally filtering out boxes that do not
  overlap at all with the window.

  Args:
    boxlist: BoxList holding M_in boxes
    window: a numpy array of shape [4] representing the
            [y_min, x_min, y_max, x_max] window to which the op
            should clip boxes.
    filter_nonoverlapping: whether to filter out boxes that do not overlap at
      all with the window.

  Returns:
    a BoxList holding M_out boxes where M_out <= M_in
  """
  y_min, x_min, y_max, x_max = np.array_split(boxlist.get(), 4, axis=1)
  win_y_min = window[0]
  win_x_min = window[1]
  win_y_max = window[2]
  win_x_max = window[3]
  y_min_clipped = np.fmax(np.fmin(y_min, win_y_max), win_y_min)
  y_max_clipped = np.fmax(np.fmin(y_max, win_y_max), win_y_min)
  x_min_clipped = np.fmax(np.fmin(x_min, win_x_max), win_x_min)
  x_max_clipped = np.fmax(np.fmin(x_max, win_x_max), win_x_min)
  clipped = np_box_list.BoxList(
      np.hstack([y_min_clipped, x_min_clipped, y_max_clipped, x_max_clipped]))
  clipped = _copy_extra_fields(clipped, boxlist)
  if filter_nonoverlapping:
    areas = area(clipped)
    nonzero_area_indices = np.reshape(
        np.nonzero(np.greater(areas, 0.0)), [-1]).astype(np.int32)
    clipped = gather(clipped, nonzero_area_indices)
  return clipped


def prune_non_overlapping_boxes(boxlist1, boxlist2, minoverlap=0.0):
  """Prunes the boxes in boxlist1 that overlap less than thresh with boxlist2.

  For each box in boxlist1, we want its IOA to be more than minoverlap with
  at least one of the boxes in boxlist2. If it does not, we remove it.

  Args:
    boxlist1: BoxList holding N boxes.
    boxlist2: BoxList holding M boxes.
    minoverlap: Minimum required overlap between boxes, to count them as
                overlapping.

  Returns:
    A pruned boxlist with size [N', 4].
  """
  intersection_over_area = ioa(boxlist2, boxlist1)  # [M, N] tensor
  intersection_over_area = np.amax(intersection_over_area, axis=0)  # [N] tensor
  keep_bool = np.greater_equal(intersection_over_area, np.array(minoverlap))
  keep_inds = np.nonzero(keep_bool)[0]
  new_boxlist1 = gather(boxlist1, keep_inds)
  return new_boxlist1


def prune_outside_window(boxlist, window):
  """Prunes bounding boxes that fall outside a given window.

  This function prunes bounding boxes that even partially fall outside the given
  window. See also ClipToWindow which only prunes bounding boxes that fall
  completely outside the window, and clips any bounding boxes that partially
  overflow.

  Args:
    boxlist: a BoxList holding M_in boxes.
    window: a numpy array of size 4, representing [ymin, xmin, ymax, xmax]
            of the window.

  Returns:
    pruned_corners: a tensor with shape [M_out, 4] where M_out <= M_in.
    valid_indices: a tensor with shape [M_out] indexing the valid bounding boxes
     in the input tensor.
  """

  y_min, x_min, y_max, x_max = np.array_split(boxlist.get(), 4, axis=1)
  win_y_min = window[0]
  win_x_min = window[1]
  win_y_max = window[2]
  win_x_max = window[3]
  coordinate_violations = np.hstack([np.less(y_min, win_y_min),
                                     np.less(x_min, win_x_min),
                                     np.greater(y_max, win_y_max),
                                     np.greater(x_max, win_x_max)])
  valid_indices = np.reshape(
      np.where(np.logical_not(np.max(coordinate_violations, axis=1))), [-1])
  return gather(boxlist, valid_indices), valid_indices


def concatenate(boxlists, fields=None):
  """Concatenate list of BoxLists.

  This op concatenates a list of input BoxLists into a larger BoxList.  It also
  handles concatenation of BoxList fields as long as the field tensor shapes
  are equal except for the first dimension.

  Args:
    boxlists: list of BoxList objects
    fields: optional list of fields to also concatenate.  By default, all
      fields from the first BoxList in the list are included in the
      concatenation.

  Returns:
    a BoxList with number of boxes equal to
      sum([boxlist.num_boxes() for boxlist in BoxList])
  Raises:
    ValueError: if boxlists is invalid (i.e., is not a list, is empty, or
      contains non BoxList objects), or if requested fields are not contained in
      all boxlists
  """
  if not isinstance(boxlists, list):
    raise ValueError('boxlists should be a list')
  if not boxlists:
    raise ValueError('boxlists should have nonzero length')
  for boxlist in boxlists:
    if not isinstance(boxlist, np_box_list.BoxList):
      raise ValueError('all elements of boxlists should be BoxList objects')
  concatenated = np_box_list.BoxList(
      np.vstack([boxlist.get() for boxlist in boxlists]))
  if fields is None:
    fields = boxlists[0].get_extra_fields()
  for field in fields:
    first_field_shape = boxlists[0].get_field(field).shape
    first_field_shape = first_field_shape[1:]
    for boxlist in boxlists:
      if not boxlist.has_field(field):
        raise ValueError('boxlist must contain all requested fields')
      field_shape = boxlist.get_field(field).shape
      field_shape = field_shape[1:]
      if field_shape != first_field_shape:
        raise ValueError('field %s must have same shape for all boxlists '
                         'except for the 0th dimension.' % field)
    concatenated_field = np.concatenate(
        [boxlist.get_field(field) for boxlist in boxlists], axis=0)
    concatenated.add_field(field, concatenated_field)
  return concatenated


def filter_scores_greater_than(boxlist, thresh):
  """Filter to keep only boxes with score exceeding a given threshold.

  This op keeps the collection of boxes whose corresponding scores are
  greater than the input threshold.

  Args:
    boxlist: BoxList holding N boxes.  Must contain a 'scores' field
      representing detection scores.
    thresh: scalar threshold

  Returns:
    a BoxList holding M boxes where M <= N

  Raises:
    ValueError: if boxlist not a BoxList object or if it does not
      have a scores field
  """
  if not isinstance(boxlist, np_box_list.BoxList):
    raise ValueError('boxlist must be a BoxList')
  if not boxlist.has_field('scores'):
    raise ValueError('input boxlist must have \'scores\' field')
  scores = boxlist.get_field('scores')
  if len(scores.shape) > 2:
    raise ValueError('Scores should have rank 1 or 2')
  if len(scores.shape) == 2 and scores.shape[1] != 1:
    raise ValueError('Scores should have rank 1 or have shape '
                     'consistent with [None, 1]')
  high_score_indices = np.reshape(np.where(np.greater(scores, thresh)),
                                  [-1]).astype(np.int32)
  return gather(boxlist, high_score_indices)


def change_coordinate_frame(boxlist, window):
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
    window: a size 4 1-D numpy array.

  Returns:
    Returns a BoxList object with N boxes.
  """
  win_height = window[2] - window[0]
  win_width = window[3] - window[1]
  boxlist_new = scale(
      np_box_list.BoxList(boxlist.get() -
                          [window[0], window[1], window[0], window[1]]),
      1.0 / win_height, 1.0 / win_width)
  _copy_extra_fields(boxlist_new, boxlist)

  return boxlist_new


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


def _update_valid_indices_by_removing_high_iou_boxes(
    selected_indices, is_index_valid, intersect_over_union, threshold):
  max_iou = np.max(intersect_over_union[:, selected_indices], axis=1)
  return np.logical_and(is_index_valid, max_iou <= threshold)
