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
"""Post-processing operations on detected boxes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import standard_fields as fields
from object_detection.utils import shape_utils

_NMS_TILE_SIZE = 512


def batch_iou(boxes1, boxes2):
  """Calculates the overlap between proposal and ground truth boxes.

  Some `boxes2` may have been padded.  The returned `iou` tensor for these
  boxes will be -1.

  Args:
    boxes1: a tensor with a shape of [batch_size, N, 4]. N is the number of
      proposals before groundtruth assignment. The last dimension is the pixel
      coordinates in [ymin, xmin, ymax, xmax] form.
    boxes2: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4]. This
      tensor might have paddings with a negative value.

  Returns:
    iou: a tensor with as a shape of [batch_size, N, MAX_NUM_INSTANCES].
  """
  with tf.name_scope('BatchIOU'):
    y1_min, x1_min, y1_max, x1_max = tf.split(
        value=boxes1, num_or_size_splits=4, axis=2)
    y2_min, x2_min, y2_max, x2_max = tf.split(
        value=boxes2, num_or_size_splits=4, axis=2)

    # Calculates the intersection area.
    intersection_xmin = tf.maximum(x1_min, tf.transpose(x2_min, [0, 2, 1]))
    intersection_xmax = tf.minimum(x1_max, tf.transpose(x2_max, [0, 2, 1]))
    intersection_ymin = tf.maximum(y1_min, tf.transpose(y2_min, [0, 2, 1]))
    intersection_ymax = tf.minimum(y1_max, tf.transpose(y2_max, [0, 2, 1]))
    intersection_area = tf.maximum(
        (intersection_xmax - intersection_xmin), 0) * tf.maximum(
            (intersection_ymax - intersection_ymin), 0)

    # Calculates the union area.
    area1 = (y1_max - y1_min) * (x1_max - x1_min)
    area2 = (y2_max - y2_min) * (x2_max - x2_min)
    # Adds a small epsilon to avoid divide-by-zero.
    union_area = area1 + tf.transpose(area2,
                                      [0, 2, 1]) - intersection_area + 1e-8

    # Calculates IoU.
    iou = intersection_area / union_area

    # Fills -1 for padded ground truth boxes.
    padding_mask = tf.logical_and(
        tf.less(intersection_xmax, 0), tf.less(intersection_ymax, 0))
    iou = tf.where(padding_mask, -tf.ones_like(iou), iou)

    return iou


def _self_suppression(iou, iou_threshold, loop_condition, iou_sum):
  """Bounding-boxes self-suppression loop body.

  Args:
    iou: A float Tensor with shape [1, num_boxes, max_num_instance]: IOUs.
    iou_threshold: A scalar, representing IOU threshold.
    loop_condition: The loop condition returned from last iteration.
    iou_sum: iou_sum_new returned from last iteration.

  Returns:
    iou_suppressed: A float Tensor with shape [1, num_boxes, max_num_instance],
                    IOU after suppression.
    iou_threshold: A scalar, representing IOU threshold.
    loop_condition: Bool Tensor of shape [], the loop condition.
    iou_sum_new: The new IOU sum.
  """
  del loop_condition
  can_suppress_others = tf.cast(
      tf.reshape(tf.reduce_max(iou, 1) <= iou_threshold, [1, -1, 1]), iou.dtype)
  iou_suppressed = tf.reshape(
      tf.cast(
          tf.reduce_max(can_suppress_others * iou, 1) <= iou_threshold,
          iou.dtype), [1, -1, 1]) * iou
  iou_sum_new = tf.reduce_sum(iou_suppressed, [1, 2])
  return [
      iou_suppressed, iou_threshold,
      tf.reduce_any(iou_sum - iou_sum_new > iou_threshold), iou_sum_new
  ]


def _cross_suppression(boxes, box_slice, iou_threshold, inner_idx):
  """Bounding-boxes cross-suppression loop body.

  Args:
    boxes: A float Tensor of shape [1, anchors, 4], representing boxes.
    box_slice: A float Tensor of shape [1, _NMS_TILE_SIZE, 4], the box tile
      returned from last iteration
    iou_threshold: A scalar, representing IOU threshold.
    inner_idx: A scalar, representing inner index.

  Returns:
    boxes: A float Tensor of shape [1, anchors, 4], representing boxes.
    ret_slice: A float Tensor of shape [1, _NMS_TILE_SIZE, 4], the box tile
               after suppression
    iou_threshold: A scalar, representing IOU threshold.
    inner_idx: A scalar, inner index incremented.
  """
  new_slice = tf.slice(boxes, [0, inner_idx * _NMS_TILE_SIZE, 0],
                       [1, _NMS_TILE_SIZE, 4])
  iou = batch_iou(new_slice, box_slice)
  ret_slice = tf.expand_dims(
      tf.cast(tf.reduce_all(iou < iou_threshold, [1]), box_slice.dtype),
      2) * box_slice
  return boxes, ret_slice, iou_threshold, inner_idx + 1


def _suppression_loop_body(boxes, iou_threshold, output_size, idx):
  """Process boxes in the range [idx*_NMS_TILE_SIZE, (idx+1)*_NMS_TILE_SIZE).

  Args:
    boxes: a tensor with a shape of [1, anchors, 4].
    iou_threshold: a float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    output_size: an int32 tensor of size [1]. Representing the number of
      selected boxes.
    idx: an integer scalar representing induction variable.

  Returns:
    boxes: updated boxes.
    iou_threshold: pass down iou_threshold to the next iteration.
    output_size: the updated output_size.
    idx: the updated induction variable.
  """
  num_tiles = tf.shape(boxes)[1] // _NMS_TILE_SIZE

  # Iterates over tiles that can possibly suppress the current tile.
  box_slice = tf.slice(boxes, [0, idx * _NMS_TILE_SIZE, 0],
                       [1, _NMS_TILE_SIZE, 4])
  _, box_slice, _, _ = tf.while_loop(
      lambda _boxes, _box_slice, _threshold, inner_idx: inner_idx < idx,
      _cross_suppression, [boxes, box_slice, iou_threshold,
                           tf.constant(0)])

  # Iterates over the current tile to compute self-suppression.
  iou = batch_iou(box_slice, box_slice)
  mask = tf.expand_dims(
      tf.reshape(tf.range(_NMS_TILE_SIZE), [1, -1]) > tf.reshape(
          tf.range(_NMS_TILE_SIZE), [-1, 1]), 0)
  iou *= tf.cast(tf.logical_and(mask, iou >= iou_threshold), iou.dtype)
  suppressed_iou, _, _, _ = tf.while_loop(
      lambda _iou, _threshold, loop_condition, _iou_sum: loop_condition,
      _self_suppression,
      [iou, iou_threshold,
       tf.constant(True),
       tf.reduce_sum(iou, [1, 2])])
  suppressed_box = tf.reduce_sum(suppressed_iou, 1) > 0
  box_slice *= tf.expand_dims(1.0 - tf.cast(suppressed_box, box_slice.dtype), 2)

  # Uses box_slice to update the input boxes.
  mask = tf.reshape(
      tf.cast(tf.equal(tf.range(num_tiles), idx), boxes.dtype), [1, -1, 1, 1])
  boxes = tf.tile(tf.expand_dims(box_slice, [1]),
                  [1, num_tiles, 1, 1]) * mask + tf.reshape(
                      boxes, [1, num_tiles, _NMS_TILE_SIZE, 4]) * (1 - mask)
  boxes = tf.reshape(boxes, [1, -1, 4])

  # Updates output_size.
  output_size += tf.reduce_sum(
      tf.cast(tf.reduce_any(box_slice > 0, [2]), tf.int32), [1])
  return boxes, iou_threshold, output_size, idx + 1


def partitioned_non_max_suppression_padded(boxes,
                                           scores,
                                           max_output_size,
                                           iou_threshold=0.5,
                                           score_threshold=float('-inf')):
  """A tiled version of [`tf.image.non_max_suppression_padded`](https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression_padded).

  The overall design of the algorithm is to handle boxes tile-by-tile:

  boxes = boxes.pad_to_multiple_of(tile_size)
  num_tiles = len(boxes) // tile_size
  output_boxes = []
  for i in range(num_tiles):
    box_tile = boxes[i*tile_size : (i+1)*tile_size]
    for j in range(i - 1):
      suppressing_tile = boxes[j*tile_size : (j+1)*tile_size]
      iou = batch_iou(box_tile, suppressing_tile)
      # if the box is suppressed in iou, clear it to a dot
      box_tile *= _update_boxes(iou)
    # Iteratively handle the diagonal tile.
    iou = _box_overlap(box_tile, box_tile)
    iou_changed = True
    while iou_changed:
      # boxes that are not suppressed by anything else
      suppressing_boxes = _get_suppressing_boxes(iou)
      # boxes that are suppressed by suppressing_boxes
      suppressed_boxes = _get_suppressed_boxes(iou, suppressing_boxes)
      # clear iou to 0 for boxes that are suppressed, as they cannot be used
      # to suppress other boxes any more
      new_iou = _clear_iou(iou, suppressed_boxes)
      iou_changed = (new_iou != iou)
      iou = new_iou
    # remaining boxes that can still suppress others, are selected boxes.
    output_boxes.append(_get_suppressing_boxes(iou))
    if len(output_boxes) >= max_output_size:
      break

  Args:
    boxes: A 2-D float `Tensor` of shape `[num_boxes, 4]`.
    scores: A 1-D float `Tensor` of shape `[num_boxes]` representing a single
      score corresponding to each box (each row of boxes).
    max_output_size: a scalar integer `Tensor` representing the maximum number
      of boxes to be selected by non max suppression.
    iou_threshold: a float representing the threshold for deciding whether boxes
      overlap too much with respect to IOU.
    score_threshold: A float representing the threshold for deciding when to
      remove boxes based on score.

  Returns:
    selected_indices: a tensor of shape [anchors].
    num_valid_boxes: a scalar int tensor.
    nms_proposals: a tensor with a shape of [anchors, 4]. It has
      same dtype as input boxes.
    nms_scores: a tensor with a shape of [anchors]. It has same
      dtype as input scores.
    argsort_ids: a tensor of shape [anchors], mapping from input order of boxes
      to output order of boxes.
  """
  num_boxes = tf.shape(boxes)[0]
  pad = tf.cast(
      tf.ceil(tf.cast(num_boxes, tf.float32) / _NMS_TILE_SIZE),
      tf.int32) * _NMS_TILE_SIZE - num_boxes

  scores, argsort_ids = tf.nn.top_k(scores, k=num_boxes, sorted=True)
  boxes = tf.gather(boxes, argsort_ids)
  num_boxes = tf.shape(boxes)[0]
  num_boxes += pad
  boxes = tf.pad(
      tf.cast(boxes, tf.float32), [[0, pad], [0, 0]], constant_values=-1)
  scores = tf.pad(tf.cast(scores, tf.float32), [[0, pad]])

  # mask boxes to -1 by score threshold
  scores_mask = tf.expand_dims(
      tf.cast(scores > score_threshold, boxes.dtype), axis=1)
  boxes = ((boxes + 1.) * scores_mask) - 1.

  boxes = tf.expand_dims(boxes, axis=0)
  scores = tf.expand_dims(scores, axis=0)

  def _loop_cond(unused_boxes, unused_threshold, output_size, idx):
    return tf.logical_and(
        tf.reduce_min(output_size) < max_output_size,
        idx < num_boxes // _NMS_TILE_SIZE)

  selected_boxes, _, output_size, _ = tf.while_loop(
      _loop_cond, _suppression_loop_body,
      [boxes, iou_threshold,
       tf.zeros([1], tf.int32),
       tf.constant(0)])
  idx = num_boxes - tf.cast(
      tf.nn.top_k(
          tf.cast(tf.reduce_any(selected_boxes > 0, [2]), tf.int32) *
          tf.expand_dims(tf.range(num_boxes, 0, -1), 0), max_output_size)[0],
      tf.int32)
  idx = tf.minimum(idx, num_boxes - 1 - pad)
  idx = tf.reshape(idx + tf.reshape(tf.range(1) * num_boxes, [-1, 1]), [-1])
  num_valid_boxes = tf.reduce_sum(output_size)
  return (idx, num_valid_boxes, tf.reshape(boxes, [-1, 4]),
          tf.reshape(scores, [-1]), argsort_ids)


def _validate_boxes_scores_iou_thresh(boxes, scores, iou_thresh,
                                      change_coordinate_frame, clip_window):
  """Validates boxes, scores and iou_thresh.

  This function validates the boxes, scores, iou_thresh
     and if change_coordinate_frame is True, clip_window must be specified.

  Args:
    boxes: A [k, q, 4] float32 tensor containing k detections. `q` can be either
      number of classes or 1 depending on whether a separate box is predicted
      per class.
    scores: A [k, num_classes] float32 tensor containing the scores for each of
      the k detections. The scores have to be non-negative when
      pad_to_max_output_size is True.
    iou_thresh: scalar threshold for IOU (new boxes that have high IOU overlap
      with previously selected boxes are removed).
    change_coordinate_frame: Whether to normalize coordinates after clipping
      relative to clip_window (this can only be set to True if a clip_window is
      provided)
    clip_window: A float32 tensor of the form [y_min, x_min, y_max, x_max]
      representing the window to clip and normalize boxes to before performing
      non-max suppression.

  Raises:
    ValueError: if iou_thresh is not in [0, 1] or if input boxlist does not
    have a valid scores field.
  """
  if not 0 <= iou_thresh <= 1.0:
    raise ValueError('iou_thresh must be between 0 and 1')
  if scores.shape.ndims != 2:
    raise ValueError('scores field must be of rank 2')
  if shape_utils.get_dim_as_int(scores.shape[1]) is None:
    raise ValueError('scores must have statically defined second ' 'dimension')
  if boxes.shape.ndims != 3:
    raise ValueError('boxes must be of rank 3.')
  if not (shape_utils.get_dim_as_int(
      boxes.shape[1]) == shape_utils.get_dim_as_int(scores.shape[1]) or
          shape_utils.get_dim_as_int(boxes.shape[1]) == 1):
    raise ValueError('second dimension of boxes must be either 1 or equal '
                     'to the second dimension of scores')
  if shape_utils.get_dim_as_int(boxes.shape[2]) != 4:
    raise ValueError('last dimension of boxes must be of size 4.')
  if change_coordinate_frame and clip_window is None:
    raise ValueError('if change_coordinate_frame is True, then a clip_window'
                     'must be specified.')


def _clip_window_prune_boxes(sorted_boxes, clip_window, pad_to_max_output_size,
                             change_coordinate_frame):
  """Prune boxes with zero area.

  Args:
    sorted_boxes: A BoxList containing k detections.
    clip_window: A float32 tensor of the form [y_min, x_min, y_max, x_max]
      representing the window to clip and normalize boxes to before performing
      non-max suppression.
    pad_to_max_output_size: flag indicating whether to pad to max output size or
      not.
    change_coordinate_frame: Whether to normalize coordinates after clipping
      relative to clip_window (this can only be set to True if a clip_window is
      provided).

  Returns:
    sorted_boxes: A BoxList containing k detections after pruning.
    num_valid_nms_boxes_cumulative: Number of valid NMS boxes
  """
  sorted_boxes = box_list_ops.clip_to_window(
      sorted_boxes,
      clip_window,
      filter_nonoverlapping=not pad_to_max_output_size)
  # Set the scores of boxes with zero area to -1 to keep the default
  # behaviour of pruning out zero area boxes.
  sorted_boxes_size = tf.shape(sorted_boxes.get())[0]
  non_zero_box_area = tf.cast(box_list_ops.area(sorted_boxes), tf.bool)
  sorted_boxes_scores = tf.where(
      non_zero_box_area, sorted_boxes.get_field(fields.BoxListFields.scores),
      -1 * tf.ones(sorted_boxes_size))
  sorted_boxes.add_field(fields.BoxListFields.scores, sorted_boxes_scores)
  num_valid_nms_boxes_cumulative = tf.reduce_sum(
      tf.cast(tf.greater_equal(sorted_boxes_scores, 0), tf.int32))
  sorted_boxes = box_list_ops.sort_by_field(sorted_boxes,
                                            fields.BoxListFields.scores)
  if change_coordinate_frame:
    sorted_boxes = box_list_ops.change_coordinate_frame(sorted_boxes,
                                                        clip_window)
  return sorted_boxes, num_valid_nms_boxes_cumulative


class NullContextmanager(object):

  def __enter__(self):
    pass

  def __exit__(self, type_arg, value_arg, traceback_arg):
    return False


def multiclass_non_max_suppression(boxes,
                                   scores,
                                   score_thresh,
                                   iou_thresh,
                                   max_size_per_class,
                                   max_total_size=0,
                                   clip_window=None,
                                   change_coordinate_frame=False,
                                   masks=None,
                                   boundaries=None,
                                   pad_to_max_output_size=False,
                                   use_partitioned_nms=False,
                                   additional_fields=None,
                                   soft_nms_sigma=0.0,
                                   use_hard_nms=False,
                                   use_cpu_nms=False,
                                   scope=None):
  """Multi-class version of non maximum suppression.

  This op greedily selects a subset of detection bounding boxes, pruning
  away boxes that have high IOU (intersection over union) overlap (> thresh)
  with already selected boxes.  It operates independently for each class for
  which scores are provided (via the scores field of the input box_list),
  pruning boxes with score less than a provided threshold prior to
  applying NMS.

  Please note that this operation is performed on *all* classes, therefore any
  background classes should be removed prior to calling this function.

  Selected boxes are guaranteed to be sorted in decreasing order by score (but
  the sort is not guaranteed to be stable).

  Args:
    boxes: A [k, q, 4] float32 tensor containing k detections. `q` can be either
      number of classes or 1 depending on whether a separate box is predicted
      per class.
    scores: A [k, num_classes] float32 tensor containing the scores for each of
      the k detections. The scores have to be non-negative when
      pad_to_max_output_size is True.
    score_thresh: scalar threshold for score (low scoring boxes are removed).
    iou_thresh: scalar threshold for IOU (new boxes that have high IOU overlap
      with previously selected boxes are removed).
    max_size_per_class: maximum number of retained boxes per class.
    max_total_size: maximum number of boxes retained over all classes. By
      default returns all boxes retained after capping boxes per class.
    clip_window: A float32 tensor of the form [y_min, x_min, y_max, x_max]
      representing the window to clip and normalize boxes to before performing
      non-max suppression.
    change_coordinate_frame: Whether to normalize coordinates after clipping
      relative to clip_window (this can only be set to True if a clip_window
      is provided)
    masks: (optional) a [k, q, mask_height, mask_width] float32 tensor
      containing box masks. `q` can be either number of classes or 1 depending
      on whether a separate mask is predicted per class.
    boundaries: (optional) a [k, q, boundary_height, boundary_width] float32
      tensor containing box boundaries. `q` can be either number of classes or 1
      depending on whether a separate boundary is predicted per class.
    pad_to_max_output_size: If true, the output nmsed boxes are padded to be of
      length `max_size_per_class`. Defaults to false.
    use_partitioned_nms: If true, use partitioned version of
      non_max_suppression.
    additional_fields: (optional) If not None, a dictionary that maps keys to
      tensors whose first dimensions are all of size `k`. After non-maximum
      suppression, all tensors corresponding to the selected boxes will be
      added to resulting BoxList.
    soft_nms_sigma: A scalar float representing the Soft NMS sigma parameter;
      See Bodla et al, https://arxiv.org/abs/1704.04503).  When
      `soft_nms_sigma=0.0` (which is default), we fall back to standard (hard)
      NMS.  Soft NMS is currently only supported when pad_to_max_output_size is
      False.
    use_hard_nms: Enforce the usage of hard NMS.
    use_cpu_nms: Enforce NMS to run on CPU.
    scope: name scope.

  Returns:
    A tuple of sorted_boxes and num_valid_nms_boxes. The sorted_boxes is a
      BoxList holds M boxes with a rank-1 scores field representing
      corresponding scores for each box with scores sorted in decreasing order
      and a rank-1 classes field representing a class label for each box. The
      num_valid_nms_boxes is a 0-D integer tensor representing the number of
      valid elements in `BoxList`, with the valid elements appearing first.

  Raises:
    ValueError: if iou_thresh is not in [0, 1] or if input boxlist does not have
      a valid scores field.
    ValueError: if Soft NMS (tf.image.non_max_suppression_with_scores) is not
      supported in the current TF version and `soft_nms_sigma` is nonzero.
  """
  _validate_boxes_scores_iou_thresh(boxes, scores, iou_thresh,
                                    change_coordinate_frame, clip_window)
  if pad_to_max_output_size and soft_nms_sigma != 0.0:
    raise ValueError('Soft NMS (soft_nms_sigma != 0.0) is currently not '
                     'supported when pad_to_max_output_size is True.')

  with tf.name_scope(scope, 'MultiClassNonMaxSuppression'), tf.device(
      'cpu:0') if use_cpu_nms else NullContextmanager():
    num_scores = tf.shape(scores)[0]
    num_classes = shape_utils.get_dim_as_int(scores.get_shape()[1])

    selected_boxes_list = []
    num_valid_nms_boxes_cumulative = tf.constant(0)
    per_class_boxes_list = tf.unstack(boxes, axis=1)
    if masks is not None:
      per_class_masks_list = tf.unstack(masks, axis=1)
    if boundaries is not None:
      per_class_boundaries_list = tf.unstack(boundaries, axis=1)
    boxes_ids = (range(num_classes) if len(per_class_boxes_list) > 1
                 else [0] * num_classes)
    for class_idx, boxes_idx in zip(range(num_classes), boxes_ids):
      per_class_boxes = per_class_boxes_list[boxes_idx]
      boxlist_and_class_scores = box_list.BoxList(per_class_boxes)
      class_scores = tf.reshape(
          tf.slice(scores, [0, class_idx], tf.stack([num_scores, 1])), [-1])

      boxlist_and_class_scores.add_field(fields.BoxListFields.scores,
                                         class_scores)
      if masks is not None:
        per_class_masks = per_class_masks_list[boxes_idx]
        boxlist_and_class_scores.add_field(fields.BoxListFields.masks,
                                           per_class_masks)
      if boundaries is not None:
        per_class_boundaries = per_class_boundaries_list[boxes_idx]
        boxlist_and_class_scores.add_field(fields.BoxListFields.boundaries,
                                           per_class_boundaries)
      if additional_fields is not None:
        for key, tensor in additional_fields.items():
          boxlist_and_class_scores.add_field(key, tensor)

      nms_result = None
      selected_scores = None
      if pad_to_max_output_size:
        max_selection_size = max_size_per_class
        if use_partitioned_nms:
          (selected_indices, num_valid_nms_boxes,
           boxlist_and_class_scores.data['boxes'],
           boxlist_and_class_scores.data['scores'],
           _) = partitioned_non_max_suppression_padded(
               boxlist_and_class_scores.get(),
               boxlist_and_class_scores.get_field(fields.BoxListFields.scores),
               max_selection_size,
               iou_threshold=iou_thresh,
               score_threshold=score_thresh)
        else:
          selected_indices, num_valid_nms_boxes = (
              tf.image.non_max_suppression_padded(
                  boxlist_and_class_scores.get(),
                  boxlist_and_class_scores.get_field(
                      fields.BoxListFields.scores),
                  max_selection_size,
                  iou_threshold=iou_thresh,
                  score_threshold=score_thresh,
                  pad_to_max_output_size=True))
        nms_result = box_list_ops.gather(boxlist_and_class_scores,
                                         selected_indices)
        selected_scores = nms_result.get_field(fields.BoxListFields.scores)
      else:
        max_selection_size = tf.minimum(max_size_per_class,
                                        boxlist_and_class_scores.num_boxes())
        if (hasattr(tf.image, 'non_max_suppression_with_scores') and
            tf.compat.forward_compatible(2019, 6, 6) and not use_hard_nms):
          (selected_indices, selected_scores
          ) = tf.image.non_max_suppression_with_scores(
              boxlist_and_class_scores.get(),
              boxlist_and_class_scores.get_field(fields.BoxListFields.scores),
              max_selection_size,
              iou_threshold=iou_thresh,
              score_threshold=score_thresh,
              soft_nms_sigma=soft_nms_sigma)
          num_valid_nms_boxes = tf.shape(selected_indices)[0]
          selected_indices = tf.concat(
              [selected_indices,
               tf.zeros(max_selection_size-num_valid_nms_boxes, tf.int32)], 0)
          selected_scores = tf.concat(
              [selected_scores,
               tf.zeros(max_selection_size-num_valid_nms_boxes,
                        tf.float32)], -1)
          nms_result = box_list_ops.gather(boxlist_and_class_scores,
                                           selected_indices)
        else:
          if soft_nms_sigma != 0:
            raise ValueError('Soft NMS not supported in current TF version!')
          selected_indices = tf.image.non_max_suppression(
              boxlist_and_class_scores.get(),
              boxlist_and_class_scores.get_field(fields.BoxListFields.scores),
              max_selection_size,
              iou_threshold=iou_thresh,
              score_threshold=score_thresh)
          num_valid_nms_boxes = tf.shape(selected_indices)[0]
          selected_indices = tf.concat(
              [selected_indices,
               tf.zeros(max_selection_size-num_valid_nms_boxes, tf.int32)], 0)
          nms_result = box_list_ops.gather(boxlist_and_class_scores,
                                           selected_indices)
          selected_scores = nms_result.get_field(fields.BoxListFields.scores)
      # Make the scores -1 for invalid boxes.
      valid_nms_boxes_indices = tf.less(
          tf.range(max_selection_size), num_valid_nms_boxes)

      nms_result.add_field(
          fields.BoxListFields.scores,
          tf.where(valid_nms_boxes_indices,
                   selected_scores, -1*tf.ones(max_selection_size)))
      num_valid_nms_boxes_cumulative += num_valid_nms_boxes

      nms_result.add_field(
          fields.BoxListFields.classes, (tf.zeros_like(
              nms_result.get_field(fields.BoxListFields.scores)) + class_idx))
      selected_boxes_list.append(nms_result)
    selected_boxes = box_list_ops.concatenate(selected_boxes_list)
    sorted_boxes = box_list_ops.sort_by_field(selected_boxes,
                                              fields.BoxListFields.scores)
    if clip_window is not None:
      # When pad_to_max_output_size is False, it prunes the boxes with zero
      # area.
      sorted_boxes, num_valid_nms_boxes_cumulative = _clip_window_prune_boxes(
          sorted_boxes, clip_window, pad_to_max_output_size,
          change_coordinate_frame)

    if max_total_size:
      max_total_size = tf.minimum(max_total_size, sorted_boxes.num_boxes())
      sorted_boxes = box_list_ops.gather(sorted_boxes, tf.range(max_total_size))
      num_valid_nms_boxes_cumulative = tf.where(
          max_total_size > num_valid_nms_boxes_cumulative,
          num_valid_nms_boxes_cumulative, max_total_size)
    # Select only the valid boxes if pad_to_max_output_size is False.
    if not pad_to_max_output_size:
      sorted_boxes = box_list_ops.gather(
          sorted_boxes, tf.range(num_valid_nms_boxes_cumulative))

    return sorted_boxes, num_valid_nms_boxes_cumulative


def class_agnostic_non_max_suppression(boxes,
                                       scores,
                                       score_thresh,
                                       iou_thresh,
                                       max_classes_per_detection=1,
                                       max_total_size=0,
                                       clip_window=None,
                                       change_coordinate_frame=False,
                                       masks=None,
                                       boundaries=None,
                                       pad_to_max_output_size=False,
                                       use_partitioned_nms=False,
                                       additional_fields=None,
                                       soft_nms_sigma=0.0,
                                       scope=None):
  """Class-agnostic version of non maximum suppression.

  This op greedily selects a subset of detection bounding boxes, pruning
  away boxes that have high IOU (intersection over union) overlap (> thresh)
  with already selected boxes.  It operates on all the boxes using
  max scores across all classes for which scores are provided (via the scores
  field of the input box_list), pruning boxes with score less than a provided
  threshold prior to applying NMS.

  Please note that this operation is performed in a class-agnostic way,
  therefore any background classes should be removed prior to calling this
  function.

  Selected boxes are guaranteed to be sorted in decreasing order by score (but
  the sort is not guaranteed to be stable).

  Args:
    boxes: A [k, q, 4] float32 tensor containing k detections. `q` can be either
      number of classes or 1 depending on whether a separate box is predicted
      per class.
    scores: A [k, num_classes] float32 tensor containing the scores for each of
      the k detections. The scores have to be non-negative when
      pad_to_max_output_size is True.
    score_thresh: scalar threshold for score (low scoring boxes are removed).
    iou_thresh: scalar threshold for IOU (new boxes that have high IOU overlap
      with previously selected boxes are removed).
    max_classes_per_detection: maximum number of retained classes per detection
      box in class-agnostic NMS.
    max_total_size: maximum number of boxes retained over all classes. By
      default returns all boxes retained after capping boxes per class.
    clip_window: A float32 tensor of the form [y_min, x_min, y_max, x_max]
      representing the window to clip and normalize boxes to before performing
      non-max suppression.
    change_coordinate_frame: Whether to normalize coordinates after clipping
      relative to clip_window (this can only be set to True if a clip_window is
      provided)
    masks: (optional) a [k, q, mask_height, mask_width] float32 tensor
      containing box masks. `q` can be either number of classes or 1 depending
      on whether a separate mask is predicted per class.
    boundaries: (optional) a [k, q, boundary_height, boundary_width] float32
      tensor containing box boundaries. `q` can be either number of classes or 1
      depending on whether a separate boundary is predicted per class.
    pad_to_max_output_size: If true, the output nmsed boxes are padded to be of
      length `max_size_per_class`. Defaults to false.
    use_partitioned_nms: If true, use partitioned version of
      non_max_suppression.
    additional_fields: (optional) If not None, a dictionary that maps keys to
      tensors whose first dimensions are all of size `k`. After non-maximum
      suppression, all tensors corresponding to the selected boxes will be added
      to resulting BoxList.
    soft_nms_sigma: A scalar float representing the Soft NMS sigma parameter;
      See Bodla et al, https://arxiv.org/abs/1704.04503).  When
      `soft_nms_sigma=0.0` (which is default), we fall back to standard (hard)
      NMS.  Soft NMS is currently only supported when pad_to_max_output_size is
      False.
    scope: name scope.

  Returns:
    A tuple of sorted_boxes and num_valid_nms_boxes. The sorted_boxes is a
      BoxList holds M boxes with a rank-1 scores field representing
      corresponding scores for each box with scores sorted in decreasing order
      and a rank-1 classes field representing a class label for each box. The
      num_valid_nms_boxes is a 0-D integer tensor representing the number of
      valid elements in `BoxList`, with the valid elements appearing first.

  Raises:
    ValueError: if iou_thresh is not in [0, 1] or if input boxlist does not have
      a valid scores field or if non-zero soft_nms_sigma is provided when
      pad_to_max_output_size is True.
  """
  _validate_boxes_scores_iou_thresh(boxes, scores, iou_thresh,
                                    change_coordinate_frame, clip_window)
  if pad_to_max_output_size and soft_nms_sigma != 0.0:
    raise ValueError('Soft NMS (soft_nms_sigma != 0.0) is currently not '
                     'supported when pad_to_max_output_size is True.')

  if max_classes_per_detection > 1:
    raise ValueError('Max classes per detection box >1 not supported.')
  q = shape_utils.get_dim_as_int(boxes.shape[1])
  if q > 1:
    class_ids = tf.expand_dims(
        tf.argmax(scores, axis=1, output_type=tf.int32), axis=1)
    boxes = tf.batch_gather(boxes, class_ids)
    if masks is not None:
      masks = tf.batch_gather(masks, class_ids)
    if boundaries is not None:
      boundaries = tf.batch_gather(boundaries, class_ids)
  boxes = tf.squeeze(boxes, axis=[1])
  if masks is not None:
    masks = tf.squeeze(masks, axis=[1])
  if boundaries is not None:
    boundaries = tf.squeeze(boundaries, axis=[1])

  with tf.name_scope(scope, 'ClassAgnosticNonMaxSuppression'):
    boxlist_and_class_scores = box_list.BoxList(boxes)
    max_scores = tf.reduce_max(scores, axis=-1)
    classes_with_max_scores = tf.argmax(scores, axis=-1)
    boxlist_and_class_scores.add_field(fields.BoxListFields.scores, max_scores)
    if masks is not None:
      boxlist_and_class_scores.add_field(fields.BoxListFields.masks, masks)
    if boundaries is not None:
      boxlist_and_class_scores.add_field(fields.BoxListFields.boundaries,
                                         boundaries)

    if additional_fields is not None:
      for key, tensor in additional_fields.items():
        boxlist_and_class_scores.add_field(key, tensor)

    nms_result = None
    selected_scores = None
    if pad_to_max_output_size:
      max_selection_size = max_total_size
      if use_partitioned_nms:
        (selected_indices, num_valid_nms_boxes,
         boxlist_and_class_scores.data['boxes'],
         boxlist_and_class_scores.data['scores'],
         argsort_ids) = partitioned_non_max_suppression_padded(
             boxlist_and_class_scores.get(),
             boxlist_and_class_scores.get_field(fields.BoxListFields.scores),
             max_selection_size,
             iou_threshold=iou_thresh,
             score_threshold=score_thresh)
        classes_with_max_scores = tf.gather(classes_with_max_scores,
                                            argsort_ids)
      else:
        selected_indices, num_valid_nms_boxes = (
            tf.image.non_max_suppression_padded(
                boxlist_and_class_scores.get(),
                boxlist_and_class_scores.get_field(fields.BoxListFields.scores),
                max_selection_size,
                iou_threshold=iou_thresh,
                score_threshold=score_thresh,
                pad_to_max_output_size=True))
      nms_result = box_list_ops.gather(boxlist_and_class_scores,
                                       selected_indices)
      selected_scores = nms_result.get_field(fields.BoxListFields.scores)
    else:
      max_selection_size = tf.minimum(max_total_size,
                                      boxlist_and_class_scores.num_boxes())
      if (hasattr(tf.image, 'non_max_suppression_with_scores') and
          tf.compat.forward_compatible(2019, 6, 6)):
        (selected_indices, selected_scores
        ) = tf.image.non_max_suppression_with_scores(
            boxlist_and_class_scores.get(),
            boxlist_and_class_scores.get_field(fields.BoxListFields.scores),
            max_selection_size,
            iou_threshold=iou_thresh,
            score_threshold=score_thresh,
            soft_nms_sigma=soft_nms_sigma)
        num_valid_nms_boxes = tf.shape(selected_indices)[0]
        selected_indices = tf.concat([
            selected_indices,
            tf.zeros(max_selection_size - num_valid_nms_boxes, tf.int32)
        ], 0)
        selected_scores = tf.concat(
            [selected_scores,
             tf.zeros(max_selection_size-num_valid_nms_boxes, tf.float32)], -1)
        nms_result = box_list_ops.gather(boxlist_and_class_scores,
                                         selected_indices)
      else:
        if soft_nms_sigma != 0:
          raise ValueError('Soft NMS not supported in current TF version!')
        selected_indices = tf.image.non_max_suppression(
            boxlist_and_class_scores.get(),
            boxlist_and_class_scores.get_field(fields.BoxListFields.scores),
            max_selection_size,
            iou_threshold=iou_thresh,
            score_threshold=score_thresh)
        num_valid_nms_boxes = tf.shape(selected_indices)[0]
        selected_indices = tf.concat(
            [selected_indices,
             tf.zeros(max_selection_size-num_valid_nms_boxes, tf.int32)], 0)
        nms_result = box_list_ops.gather(boxlist_and_class_scores,
                                         selected_indices)
        selected_scores = nms_result.get_field(fields.BoxListFields.scores)
    valid_nms_boxes_indices = tf.less(
        tf.range(max_selection_size), num_valid_nms_boxes)
    nms_result.add_field(
        fields.BoxListFields.scores,
        tf.where(valid_nms_boxes_indices,
                 selected_scores, -1*tf.ones(max_selection_size)))

    selected_classes = tf.gather(classes_with_max_scores, selected_indices)
    selected_classes = tf.cast(selected_classes, tf.float32)
    nms_result.add_field(fields.BoxListFields.classes, selected_classes)
    selected_boxes = nms_result
    sorted_boxes = box_list_ops.sort_by_field(selected_boxes,
                                              fields.BoxListFields.scores)

    if clip_window is not None:
      # When pad_to_max_output_size is False, it prunes the boxes with zero
      # area.
      sorted_boxes, num_valid_nms_boxes = _clip_window_prune_boxes(
          sorted_boxes, clip_window, pad_to_max_output_size,
          change_coordinate_frame)

    if max_total_size:
      max_total_size = tf.minimum(max_total_size, sorted_boxes.num_boxes())
      sorted_boxes = box_list_ops.gather(sorted_boxes, tf.range(max_total_size))
      num_valid_nms_boxes = tf.where(max_total_size > num_valid_nms_boxes,
                                     num_valid_nms_boxes, max_total_size)
    # Select only the valid boxes if pad_to_max_output_size is False.
    if not pad_to_max_output_size:
      sorted_boxes = box_list_ops.gather(sorted_boxes,
                                         tf.range(num_valid_nms_boxes))

    return sorted_boxes, num_valid_nms_boxes


def batch_multiclass_non_max_suppression(boxes,
                                         scores,
                                         score_thresh,
                                         iou_thresh,
                                         max_size_per_class,
                                         max_total_size=0,
                                         clip_window=None,
                                         change_coordinate_frame=False,
                                         num_valid_boxes=None,
                                         masks=None,
                                         additional_fields=None,
                                         soft_nms_sigma=0.0,
                                         scope=None,
                                         use_static_shapes=False,
                                         use_partitioned_nms=False,
                                         parallel_iterations=32,
                                         use_class_agnostic_nms=False,
                                         max_classes_per_detection=1,
                                         use_dynamic_map_fn=False,
                                         use_combined_nms=False,
                                         use_hard_nms=False,
                                         use_cpu_nms=False):
  """Multi-class version of non maximum suppression that operates on a batch.

  This op is similar to `multiclass_non_max_suppression` but operates on a batch
  of boxes and scores. See documentation for `multiclass_non_max_suppression`
  for details.

  Args:
    boxes: A [batch_size, num_anchors, q, 4] float32 tensor containing
      detections. If `q` is 1 then same boxes are used for all classes
      otherwise, if `q` is equal to number of classes, class-specific boxes are
      used.
    scores: A [batch_size, num_anchors, num_classes] float32 tensor containing
      the scores for each of the `num_anchors` detections. The scores have to be
      non-negative when use_static_shapes is set True.
    score_thresh: scalar threshold for score (low scoring boxes are removed).
    iou_thresh: scalar threshold for IOU (new boxes that have high IOU overlap
      with previously selected boxes are removed).
    max_size_per_class: maximum number of retained boxes per class.
    max_total_size: maximum number of boxes retained over all classes. By
      default returns all boxes retained after capping boxes per class.
    clip_window: A float32 tensor of shape [batch_size, 4]  where each entry is
      of the form [y_min, x_min, y_max, x_max] representing the window to clip
      boxes to before performing non-max suppression. This argument can also be
      a tensor of shape [4] in which case, the same clip window is applied to
      all images in the batch. If clip_widow is None, all boxes are used to
      perform non-max suppression.
    change_coordinate_frame: Whether to normalize coordinates after clipping
      relative to clip_window (this can only be set to True if a clip_window is
      provided)
    num_valid_boxes: (optional) a Tensor of type `int32`. A 1-D tensor of shape
      [batch_size] representing the number of valid boxes to be considered for
      each image in the batch.  This parameter allows for ignoring zero
      paddings.
    masks: (optional) a [batch_size, num_anchors, q, mask_height, mask_width]
      float32 tensor containing box masks. `q` can be either number of classes
      or 1 depending on whether a separate mask is predicted per class.
    additional_fields: (optional) If not None, a dictionary that maps keys to
      tensors whose dimensions are [batch_size, num_anchors, ...].
    soft_nms_sigma: A scalar float representing the Soft NMS sigma parameter;
      See Bodla et al, https://arxiv.org/abs/1704.04503).  When
      `soft_nms_sigma=0.0` (which is default), we fall back to standard (hard)
      NMS.  Soft NMS is currently only supported when pad_to_max_output_size is
      False.
    scope: tf scope name.
    use_static_shapes: If true, the output nmsed boxes are padded to be of
      length `max_size_per_class` and it doesn't clip boxes to max_total_size.
      Defaults to false.
    use_partitioned_nms: If true, use partitioned version of
      non_max_suppression.
    parallel_iterations: (optional) number of batch items to process in
      parallel.
    use_class_agnostic_nms: If true, this uses class-agnostic non max
      suppression
    max_classes_per_detection: Maximum number of retained classes per detection
      box in class-agnostic NMS.
    use_dynamic_map_fn: If true, images in the batch will be processed within a
      dynamic loop. Otherwise, a static loop will be used if possible.
    use_combined_nms: If true, it uses tf.image.combined_non_max_suppression (
      multi-class version of NMS that operates on a batch).
      It greedily selects a subset of detection bounding boxes, pruning away
      boxes that have high IOU (intersection over union) overlap (> thresh) with
      already selected boxes. It operates independently for each batch.
      Within each batch, it operates independently for each class for which
      scores are provided (via the scores field of the input box_list),
      pruning boxes with score less than a provided threshold prior to applying
      NMS. This operation is performed on *all* batches and *all* classes
      in the batch, therefore any background classes should be removed prior to
      calling this function.
      Masks and additional fields are not supported.
      See argument checks in the code below for unsupported arguments.
    use_hard_nms: Enforce the usage of hard NMS.
    use_cpu_nms: Enforce NMS to run on CPU.

  Returns:
    'nmsed_boxes': A [batch_size, max_detections, 4] float32 tensor
      containing the non-max suppressed boxes.
    'nmsed_scores': A [batch_size, max_detections] float32 tensor containing
      the scores for the boxes.
    'nmsed_classes': A [batch_size, max_detections] float32 tensor
      containing the class for boxes.
    'nmsed_masks': (optional) a
      [batch_size, max_detections, mask_height, mask_width] float32 tensor
      containing masks for each selected box. This is set to None if input
      `masks` is None.
    'nmsed_additional_fields': (optional) a dictionary of
      [batch_size, max_detections, ...] float32 tensors corresponding to the
      tensors specified in the input `additional_fields`. This is not returned
      if input `additional_fields` is None.
    'num_detections': A [batch_size] int32 tensor indicating the number of
      valid detections per batch item. Only the top num_detections[i] entries in
      nms_boxes[i], nms_scores[i] and nms_class[i] are valid. The rest of the
      entries are zero paddings.

  Raises:
    ValueError: if `q` in boxes.shape is not 1 or not equal to number of
      classes as inferred from scores.shape.
  """
  if use_combined_nms:
    if change_coordinate_frame:
      raise ValueError(
          'change_coordinate_frame (normalizing coordinates'
          ' relative to clip_window) is not supported by combined_nms.')
    if num_valid_boxes is not None:
      raise ValueError('num_valid_boxes is not supported by combined_nms.')
    if masks is not None:
      raise ValueError('masks is not supported by combined_nms.')
    if soft_nms_sigma != 0.0:
      raise ValueError('Soft NMS is not supported by combined_nms.')
    if use_class_agnostic_nms:
      raise ValueError('class-agnostic NMS is not supported by combined_nms.')
    if clip_window is not None:
      tf.logging.warning(
          'clip_window is not supported by combined_nms unless it is'
          ' [0. 0. 1. 1.] for each image.')
    if additional_fields is not None:
      tf.logging.warning('additional_fields is not supported by combined_nms.')
    if parallel_iterations != 32:
      tf.logging.warning('Number of batch items to be processed in parallel is'
                         ' not configurable by combined_nms.')
    if max_classes_per_detection > 1:
      tf.logging.warning(
          'max_classes_per_detection is not configurable by combined_nms.')

    with tf.name_scope(scope, 'CombinedNonMaxSuppression'):
      (batch_nmsed_boxes, batch_nmsed_scores, batch_nmsed_classes,
       batch_num_detections) = tf.image.combined_non_max_suppression(
           boxes=boxes,
           scores=scores,
           max_output_size_per_class=max_size_per_class,
           max_total_size=max_total_size,
           iou_threshold=iou_thresh,
           score_threshold=score_thresh,
           pad_per_class=use_static_shapes)
      # Not supported by combined_non_max_suppression.
      batch_nmsed_masks = None
      # Not supported by combined_non_max_suppression.
      batch_nmsed_additional_fields = None
      return (batch_nmsed_boxes, batch_nmsed_scores, batch_nmsed_classes,
              batch_nmsed_masks, batch_nmsed_additional_fields,
              batch_num_detections)

  q = shape_utils.get_dim_as_int(boxes.shape[2])
  num_classes = shape_utils.get_dim_as_int(scores.shape[2])
  if q != 1 and q != num_classes:
    raise ValueError('third dimension of boxes must be either 1 or equal '
                     'to the third dimension of scores.')
  if change_coordinate_frame and clip_window is None:
    raise ValueError('if change_coordinate_frame is True, then a clip_window'
                     'must be specified.')
  original_masks = masks

  # Create ordered dictionary using the sorted keys from
  # additional fields to ensure getting the same key value assignment
  # in _single_image_nms_fn(). The dictionary is thus a sorted version of
  # additional_fields.
  if additional_fields is None:
    ordered_additional_fields = collections.OrderedDict()
  else:
    ordered_additional_fields = collections.OrderedDict(
        sorted(additional_fields.items(), key=lambda item: item[0]))

  with tf.name_scope(scope, 'BatchMultiClassNonMaxSuppression'):
    boxes_shape = boxes.shape
    batch_size = shape_utils.get_dim_as_int(boxes_shape[0])
    num_anchors = shape_utils.get_dim_as_int(boxes_shape[1])

    if batch_size is None:
      batch_size = tf.shape(boxes)[0]
    if num_anchors is None:
      num_anchors = tf.shape(boxes)[1]

    # If num valid boxes aren't provided, create one and mark all boxes as
    # valid.
    if num_valid_boxes is None:
      num_valid_boxes = tf.ones([batch_size], dtype=tf.int32) * num_anchors

    # If masks aren't provided, create dummy masks so we can only have one copy
    # of _single_image_nms_fn and discard the dummy masks after map_fn.
    if masks is None:
      masks_shape = tf.stack([batch_size, num_anchors, q, 1, 1])
      masks = tf.zeros(masks_shape)

    if clip_window is None:
      clip_window = tf.stack([
          tf.reduce_min(boxes[:, :, :, 0]),
          tf.reduce_min(boxes[:, :, :, 1]),
          tf.reduce_max(boxes[:, :, :, 2]),
          tf.reduce_max(boxes[:, :, :, 3])
      ])
    if clip_window.shape.ndims == 1:
      clip_window = tf.tile(tf.expand_dims(clip_window, 0), [batch_size, 1])

    def _single_image_nms_fn(args):
      """Runs NMS on a single image and returns padded output.

      Args:
        args: A list of tensors consisting of the following:
          per_image_boxes - A [num_anchors, q, 4] float32 tensor containing
            detections. If `q` is 1 then same boxes are used for all classes
            otherwise, if `q` is equal to number of classes, class-specific
            boxes are used.
          per_image_scores - A [num_anchors, num_classes] float32 tensor
            containing the scores for each of the `num_anchors` detections.
          per_image_masks - A [num_anchors, q, mask_height, mask_width] float32
            tensor containing box masks. `q` can be either number of classes
            or 1 depending on whether a separate mask is predicted per class.
          per_image_clip_window - A 1D float32 tensor of the form
            [ymin, xmin, ymax, xmax] representing the window to clip the boxes
            to.
          per_image_additional_fields - (optional) A variable number of float32
            tensors each with size [num_anchors, ...].
          per_image_num_valid_boxes - A tensor of type `int32`. A 1-D tensor of
            shape [batch_size] representing the number of valid boxes to be
            considered for each image in the batch.  This parameter allows for
            ignoring zero paddings.

      Returns:
        'nmsed_boxes': A [max_detections, 4] float32 tensor containing the
          non-max suppressed boxes.
        'nmsed_scores': A [max_detections] float32 tensor containing the scores
          for the boxes.
        'nmsed_classes': A [max_detections] float32 tensor containing the class
          for boxes.
        'nmsed_masks': (optional) a [max_detections, mask_height, mask_width]
          float32 tensor containing masks for each selected box. This is set to
          None if input `masks` is None.
        'nmsed_additional_fields':  (optional) A variable number of float32
          tensors each with size [max_detections, ...] corresponding to the
          input `per_image_additional_fields`.
        'num_detections': A [batch_size] int32 tensor indicating the number of
          valid detections per batch item. Only the top num_detections[i]
          entries in nms_boxes[i], nms_scores[i] and nms_class[i] are valid. The
          rest of the entries are zero paddings.
      """
      per_image_boxes = args[0]
      per_image_scores = args[1]
      per_image_masks = args[2]
      per_image_clip_window = args[3]
      # Make sure that the order of elements passed in args is aligned with
      # the iteration order of ordered_additional_fields
      per_image_additional_fields = {
          key: value
          for key, value in zip(ordered_additional_fields, args[4:-1])
      }
      per_image_num_valid_boxes = args[-1]
      if use_static_shapes:
        total_proposals = tf.shape(per_image_scores)
        per_image_scores = tf.where(
            tf.less(tf.range(total_proposals[0]), per_image_num_valid_boxes),
            per_image_scores,
            tf.fill(total_proposals, np.finfo('float32').min))
      else:
        per_image_boxes = tf.reshape(
            tf.slice(per_image_boxes, 3 * [0],
                     tf.stack([per_image_num_valid_boxes, -1, -1])), [-1, q, 4])
        per_image_scores = tf.reshape(
            tf.slice(per_image_scores, [0, 0],
                     tf.stack([per_image_num_valid_boxes, -1])),
            [-1, num_classes])
        per_image_masks = tf.reshape(
            tf.slice(per_image_masks, 4 * [0],
                     tf.stack([per_image_num_valid_boxes, -1, -1, -1])),
            [-1, q, shape_utils.get_dim_as_int(per_image_masks.shape[2]),
             shape_utils.get_dim_as_int(per_image_masks.shape[3])])
        if per_image_additional_fields is not None:
          for key, tensor in per_image_additional_fields.items():
            additional_field_shape = tensor.get_shape()
            additional_field_dim = len(additional_field_shape)
            per_image_additional_fields[key] = tf.reshape(
                tf.slice(
                    per_image_additional_fields[key],
                    additional_field_dim * [0],
                    tf.stack([per_image_num_valid_boxes] +
                             (additional_field_dim - 1) * [-1])), [-1] + [
                                 shape_utils.get_dim_as_int(dim)
                                 for dim in additional_field_shape[1:]
                             ])
      if use_class_agnostic_nms:
        nmsed_boxlist, num_valid_nms_boxes = class_agnostic_non_max_suppression(
            per_image_boxes,
            per_image_scores,
            score_thresh,
            iou_thresh,
            max_classes_per_detection,
            max_total_size,
            clip_window=per_image_clip_window,
            change_coordinate_frame=change_coordinate_frame,
            masks=per_image_masks,
            pad_to_max_output_size=use_static_shapes,
            use_partitioned_nms=use_partitioned_nms,
            additional_fields=per_image_additional_fields,
            soft_nms_sigma=soft_nms_sigma)
      else:
        nmsed_boxlist, num_valid_nms_boxes = multiclass_non_max_suppression(
            per_image_boxes,
            per_image_scores,
            score_thresh,
            iou_thresh,
            max_size_per_class,
            max_total_size,
            clip_window=per_image_clip_window,
            change_coordinate_frame=change_coordinate_frame,
            masks=per_image_masks,
            pad_to_max_output_size=use_static_shapes,
            use_partitioned_nms=use_partitioned_nms,
            additional_fields=per_image_additional_fields,
            soft_nms_sigma=soft_nms_sigma,
            use_hard_nms=use_hard_nms,
            use_cpu_nms=use_cpu_nms)

      if not use_static_shapes:
        nmsed_boxlist = box_list_ops.pad_or_clip_box_list(
            nmsed_boxlist, max_total_size)
      num_detections = num_valid_nms_boxes
      nmsed_boxes = nmsed_boxlist.get()
      nmsed_scores = nmsed_boxlist.get_field(fields.BoxListFields.scores)
      nmsed_classes = nmsed_boxlist.get_field(fields.BoxListFields.classes)
      nmsed_masks = nmsed_boxlist.get_field(fields.BoxListFields.masks)
      nmsed_additional_fields = []
      # Sorting is needed here to ensure that the values stored in
      # nmsed_additional_fields are always kept in the same order
      # across different execution runs.
      for key in sorted(per_image_additional_fields.keys()):
        nmsed_additional_fields.append(nmsed_boxlist.get_field(key))
      return ([nmsed_boxes, nmsed_scores, nmsed_classes, nmsed_masks] +
              nmsed_additional_fields + [num_detections])

    num_additional_fields = 0
    if ordered_additional_fields:
      num_additional_fields = len(ordered_additional_fields)
    num_nmsed_outputs = 4 + num_additional_fields

    if use_dynamic_map_fn:
      map_fn = tf.map_fn
    else:
      map_fn = shape_utils.static_or_dynamic_map_fn

    batch_outputs = map_fn(
        _single_image_nms_fn,
        elems=([boxes, scores, masks, clip_window] +
               list(ordered_additional_fields.values()) + [num_valid_boxes]),
        dtype=(num_nmsed_outputs * [tf.float32] + [tf.int32]),
        parallel_iterations=parallel_iterations)

    batch_nmsed_boxes = batch_outputs[0]
    batch_nmsed_scores = batch_outputs[1]
    batch_nmsed_classes = batch_outputs[2]
    batch_nmsed_masks = batch_outputs[3]
    batch_nmsed_values = batch_outputs[4:-1]

    batch_nmsed_additional_fields = {}
    if num_additional_fields > 0:
      # Sort the keys to ensure arranging elements in same order as
      # in _single_image_nms_fn.
      batch_nmsed_keys = list(ordered_additional_fields.keys())
      for i in range(len(batch_nmsed_keys)):
        batch_nmsed_additional_fields[
            batch_nmsed_keys[i]] = batch_nmsed_values[i]

    batch_num_detections = batch_outputs[-1]

    if original_masks is None:
      batch_nmsed_masks = None

    if not ordered_additional_fields:
      batch_nmsed_additional_fields = None

    return (batch_nmsed_boxes, batch_nmsed_scores, batch_nmsed_classes,
            batch_nmsed_masks, batch_nmsed_additional_fields,
            batch_num_detections)
