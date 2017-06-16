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

import tensorflow as tf

from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.core import standard_fields as fields


def multiclass_non_max_suppression(boxes,
                                   scores,
                                   score_thresh,
                                   iou_thresh,
                                   max_size_per_class,
                                   max_total_size=0,
                                   clip_window=None,
                                   change_coordinate_frame=False,
                                   masks=None,
                                   additional_fields=None,
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

  Args:
    boxes: A [k, q, 4] float32 tensor containing k detections. `q` can be either
      number of classes or 1 depending on whether a separate box is predicted
      per class.
    scores: A [k, num_classes] float32 tensor containing the scores for each of
      the k detections.
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
    additional_fields: (optional) If not None, a dictionary that maps keys to
      tensors whose first dimensions are all of size `k`. After non-maximum
      suppression, all tensors corresponding to the selected boxes will be
      added to resulting BoxList.
    scope: name scope.

  Returns:
    a BoxList holding M boxes with a rank-1 scores field representing
      corresponding scores for each box with scores sorted in decreasing order
      and a rank-1 classes field representing a class label for each box.
      If masks, keypoints, keypoint_heatmaps is not None, the boxlist will
      contain masks, keypoints, keypoint_heatmaps corresponding to boxes.

  Raises:
    ValueError: if iou_thresh is not in [0, 1] or if input boxlist does not have
      a valid scores field.
  """
  if not 0 <= iou_thresh <= 1.0:
    raise ValueError('iou_thresh must be between 0 and 1')
  if scores.shape.ndims != 2:
    raise ValueError('scores field must be of rank 2')
  if scores.shape[1].value is None:
    raise ValueError('scores must have statically defined second '
                     'dimension')
  if boxes.shape.ndims != 3:
    raise ValueError('boxes must be of rank 3.')
  if not (boxes.shape[1].value == scores.shape[1].value or
          boxes.shape[1].value == 1):
    raise ValueError('second dimension of boxes must be either 1 or equal '
                     'to the second dimension of scores')
  if boxes.shape[2].value != 4:
    raise ValueError('last dimension of boxes must be of size 4.')
  if change_coordinate_frame and clip_window is None:
    raise ValueError('if change_coordinate_frame is True, then a clip_window'
                     'must be specified.')

  with tf.name_scope(scope, 'MultiClassNonMaxSuppression'):
    num_boxes = tf.shape(boxes)[0]
    num_scores = tf.shape(scores)[0]
    num_classes = scores.get_shape()[1]

    length_assert = tf.Assert(
        tf.equal(num_boxes, num_scores),
        ['Incorrect scores field length: actual vs expected.',
         num_scores, num_boxes])

    selected_boxes_list = []
    per_class_boxes_list = tf.unstack(boxes, axis=1)
    if masks is not None:
      per_class_masks_list = tf.unstack(masks, axis=1)
    boxes_ids = (range(num_classes) if len(per_class_boxes_list) > 1
                 else [0] * num_classes)
    for class_idx, boxes_idx in zip(range(num_classes), boxes_ids):
      per_class_boxes = per_class_boxes_list[boxes_idx]
      boxlist_and_class_scores = box_list.BoxList(per_class_boxes)
      with tf.control_dependencies([length_assert]):
        class_scores = tf.reshape(
            tf.slice(scores, [0, class_idx], tf.stack([num_scores, 1])), [-1])
      boxlist_and_class_scores.add_field(fields.BoxListFields.scores,
                                         class_scores)
      if masks is not None:
        per_class_masks = per_class_masks_list[boxes_idx]
        boxlist_and_class_scores.add_field(fields.BoxListFields.masks,
                                           per_class_masks)
      if additional_fields is not None:
        for key, tensor in additional_fields.iteritems():
          boxlist_and_class_scores.add_field(key, tensor)
      boxlist_filtered = box_list_ops.filter_greater_than(
          boxlist_and_class_scores, score_thresh)
      if clip_window is not None:
        boxlist_filtered = box_list_ops.clip_to_window(
            boxlist_filtered, clip_window)
        if change_coordinate_frame:
          boxlist_filtered = box_list_ops.change_coordinate_frame(
              boxlist_filtered, clip_window)
      max_selection_size = tf.minimum(max_size_per_class,
                                      boxlist_filtered.num_boxes())
      selected_indices = tf.image.non_max_suppression(
          boxlist_filtered.get(),
          boxlist_filtered.get_field(fields.BoxListFields.scores),
          max_selection_size,
          iou_threshold=iou_thresh)
      nms_result = box_list_ops.gather(boxlist_filtered, selected_indices)
      nms_result.add_field(
          fields.BoxListFields.classes, (tf.zeros_like(
              nms_result.get_field(fields.BoxListFields.scores)) + class_idx))
      selected_boxes_list.append(nms_result)
    selected_boxes = box_list_ops.concatenate(selected_boxes_list)
    sorted_boxes = box_list_ops.sort_by_field(selected_boxes,
                                              fields.BoxListFields.scores)
    if max_total_size:
      max_total_size = tf.minimum(max_total_size,
                                  sorted_boxes.num_boxes())
      sorted_boxes = box_list_ops.gather(sorted_boxes,
                                         tf.range(max_total_size))
    return sorted_boxes


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
                                         scope=None):
  """Multi-class version of non maximum suppression that operates on a batch.

  This op is similar to `multiclass_non_max_suppression` but operates on a batch
  of boxes and scores. See documentation for `multiclass_non_max_suppression`
  for details.

  Args:
    boxes: A [batch_size, num_anchors, q, 4] float32 tensor containing
      detections. If `q` is 1 then same boxes are used for all classes
        otherwise, if `q` is equal to number of classes, class-specific boxes
        are used.
    scores: A [batch_size, num_anchors, num_classes] float32 tensor containing
      the scores for each of the `num_anchors` detections.
    score_thresh: scalar threshold for score (low scoring boxes are removed).
    iou_thresh: scalar threshold for IOU (new boxes that have high IOU overlap
      with previously selected boxes are removed).
    max_size_per_class: maximum number of retained boxes per class.
    max_total_size: maximum number of boxes retained over all classes. By
      default returns all boxes retained after capping boxes per class.
    clip_window: A float32 tensor of the form [y_min, x_min, y_max, x_max]
      representing the window to clip boxes to before performing non-max
      suppression.
    change_coordinate_frame: Whether to normalize coordinates after clipping
      relative to clip_window (this can only be set to True if a clip_window
      is provided)
    num_valid_boxes: (optional) a Tensor of type `int32`. A 1-D tensor of shape
      [batch_size] representing the number of valid boxes to be considered
        for each image in the batch.  This parameter allows for ignoring zero
        paddings.
    masks: (optional) a [batch_size, num_anchors, q, mask_height, mask_width]
      float32 tensor containing box masks. `q` can be either number of classes
      or 1 depending on whether a separate mask is predicted per class.
    scope: tf scope name.

  Returns:
    A dictionary containing the following entries:
    'detection_boxes': A [batch_size, max_detections, 4] float32 tensor
      containing the non-max suppressed boxes.
    'detection_scores': A [bath_size, max_detections] float32 tensor containing
      the scores for the boxes.
    'detection_classes': A [batch_size, max_detections] float32 tensor
      containing the class for boxes.
    'num_detections': A [batchsize] float32 tensor indicating the number of
      valid detections per batch item. Only the top num_detections[i] entries in
      nms_boxes[i], nms_scores[i] and nms_class[i] are valid. the rest of the
      entries are zero paddings.
    'detection_masks': (optional) a
      [batch_size, max_detections, mask_height, mask_width] float32 tensor
      containing masks for each selected box.

  Raises:
    ValueError: if iou_thresh is not in [0, 1] or if input boxlist does not have
      a valid scores field.
  """
  q = boxes.shape[2].value
  num_classes = scores.shape[2].value
  if q != 1 and q != num_classes:
    raise ValueError('third dimension of boxes must be either 1 or equal '
                     'to the third dimension of scores')

  with tf.name_scope(scope, 'BatchMultiClassNonMaxSuppression'):
    per_image_boxes_list = tf.unstack(boxes)
    per_image_scores_list = tf.unstack(scores)
    num_valid_boxes_list = len(per_image_boxes_list) * [None]
    per_image_masks_list = len(per_image_boxes_list) * [None]
    if num_valid_boxes is not None:
      num_valid_boxes_list = tf.unstack(num_valid_boxes)
    if masks is not None:
      per_image_masks_list = tf.unstack(masks)

    detection_boxes_list = []
    detection_scores_list = []
    detection_classes_list = []
    num_detections_list = []
    detection_masks_list = []
    for (per_image_boxes, per_image_scores, per_image_masks, num_valid_boxes
        ) in zip(per_image_boxes_list, per_image_scores_list,
                 per_image_masks_list, num_valid_boxes_list):
      if num_valid_boxes is not None:
        per_image_boxes = tf.reshape(
            tf.slice(per_image_boxes, 3*[0],
                     tf.stack([num_valid_boxes, -1, -1])), [-1, q, 4])
        per_image_scores = tf.reshape(
            tf.slice(per_image_scores, [0, 0],
                     tf.stack([num_valid_boxes, -1])), [-1, num_classes])
        if masks is not None:
          per_image_masks = tf.reshape(
              tf.slice(per_image_masks, 4*[0],
                       tf.stack([num_valid_boxes, -1, -1, -1])),
              [-1, q, masks.shape[3].value, masks.shape[4].value])
      nmsed_boxlist = multiclass_non_max_suppression(
          per_image_boxes,
          per_image_scores,
          score_thresh,
          iou_thresh,
          max_size_per_class,
          max_total_size,
          masks=per_image_masks,
          clip_window=clip_window,
          change_coordinate_frame=change_coordinate_frame)
      num_detections_list.append(tf.to_float(nmsed_boxlist.num_boxes()))
      padded_boxlist = box_list_ops.pad_or_clip_box_list(nmsed_boxlist,
                                                         max_total_size)
      detection_boxes_list.append(padded_boxlist.get())
      detection_scores_list.append(
          padded_boxlist.get_field(fields.BoxListFields.scores))
      detection_classes_list.append(
          padded_boxlist.get_field(fields.BoxListFields.classes))
      if masks is not None:
        detection_masks_list.append(
            padded_boxlist.get_field(fields.BoxListFields.masks))

    nms_dict = {
        'detection_boxes': tf.stack(detection_boxes_list),
        'detection_scores': tf.stack(detection_scores_list),
        'detection_classes': tf.stack(detection_classes_list),
        'num_detections': tf.stack(num_detections_list)
    }
    if masks is not None:
      nms_dict['detection_masks'] = tf.stack(detection_masks_list)
    return nms_dict
