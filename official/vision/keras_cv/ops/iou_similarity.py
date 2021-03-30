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

"""Region Similarity Calculators."""

import tensorflow as tf


def area(box):
  """Computes area of boxes.

  B: batch_size
  N: number of boxes

  Args:
    box: a float Tensor with [N, 4], or [B, N, 4].

  Returns:
    a float Tensor with [N], or [B, N]
  """
  with tf.name_scope('Area'):
    y_min, x_min, y_max, x_max = tf.split(
        value=box, num_or_size_splits=4, axis=-1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), axis=-1)


def intersection(gt_boxes, boxes):
  """Compute pairwise intersection areas between boxes.

  B: batch_size
  N: number of groundtruth boxes.
  M: number of anchor boxes.

  Args:
    gt_boxes: a float Tensor with [N, 4], or [B, N, 4]
    boxes: a float Tensor with [M, 4], or [B, M, 4]

  Returns:
    a float Tensor with shape [N, M] or [B, N, M] representing pairwise
      intersections.
  """
  with tf.name_scope('Intersection'):
    y_min1, x_min1, y_max1, x_max1 = tf.split(
        value=gt_boxes, num_or_size_splits=4, axis=-1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(
        value=boxes, num_or_size_splits=4, axis=-1)

    boxes_rank = len(boxes.shape)
    perm = [1, 0] if boxes_rank == 2 else [0, 2, 1]
    # [N, M] or [B, N, M]
    y_min_max = tf.minimum(y_max1, tf.transpose(y_max2, perm))
    y_max_min = tf.maximum(y_min1, tf.transpose(y_min2, perm))
    x_min_max = tf.minimum(x_max1, tf.transpose(x_max2, perm))
    x_max_min = tf.maximum(x_min1, tf.transpose(x_min2, perm))

    intersect_heights = y_min_max - y_max_min
    intersect_widths = x_min_max - x_max_min
    zeros_t = tf.cast(0, intersect_heights.dtype)
    intersect_heights = tf.maximum(zeros_t, intersect_heights)
    intersect_widths = tf.maximum(zeros_t, intersect_widths)
    return intersect_heights * intersect_widths


def iou(gt_boxes, boxes):
  """Computes pairwise intersection-over-union between box collections.

  Args:
    gt_boxes: a float Tensor with [N, 4].
    boxes: a float Tensor with [M, 4].

  Returns:
    a Tensor with shape [N, M] representing pairwise iou scores.
  """
  with tf.name_scope('IOU'):
    intersections = intersection(gt_boxes, boxes)
    gt_boxes_areas = area(gt_boxes)
    boxes_areas = area(boxes)
    boxes_rank = len(boxes_areas.shape)
    boxes_axis = 1 if (boxes_rank == 2) else 0
    gt_boxes_areas = tf.expand_dims(gt_boxes_areas, -1)
    boxes_areas = tf.expand_dims(boxes_areas, boxes_axis)
    unions = gt_boxes_areas + boxes_areas
    unions = unions - intersections
    return tf.where(
        tf.equal(intersections, 0.0), tf.zeros_like(intersections),
        tf.truediv(intersections, unions))


class IouSimilarity:
  """Class to compute similarity based on Intersection over Union (IOU) metric.

  """

  def __init__(self, mask_val=-1):
    self.mask_val = mask_val

  def __call__(self, boxes_1, boxes_2, boxes_1_masks=None, boxes_2_masks=None):
    """Compute pairwise IOU similarity between ground truth boxes and anchors.

    B: batch_size
    N: Number of groundtruth boxes.
    M: Number of anchor boxes.

    Args:
      boxes_1: a float Tensor with M or B * M boxes.
      boxes_2: a float Tensor with N or B * N boxes, the rank must be less than
        or equal to rank of `boxes_1`.
      boxes_1_masks: a boolean Tensor with M or B * M boxes. Optional.
      boxes_2_masks: a boolean Tensor with N or B * N boxes. Optional.

    Returns:
      A Tensor with shape [M, N] or [B, M, N] representing pairwise
        iou scores, anchor per row and groundtruth_box per colulmn.

    Input shape:
      boxes_1: [N, 4], or [B, N, 4]
      boxes_2: [M, 4], or [B, M, 4]
      boxes_1_masks: [N, 1], or [B, N, 1]
      boxes_2_masks: [M, 1], or [B, M, 1]

    Output shape:
      [M, N], or [B, M, N]
    """
    boxes_1_rank = len(boxes_1.shape)
    boxes_2_rank = len(boxes_2.shape)
    if boxes_1_rank < 2 or boxes_1_rank > 3:
      raise ValueError(
          '`groudtruth_boxes` must be rank 2 or 3, got {}'.format(boxes_1_rank))
    if boxes_2_rank < 2 or boxes_2_rank > 3:
      raise ValueError(
          '`anchors` must be rank 2 or 3, got {}'.format(boxes_2_rank))
    if boxes_1_rank < boxes_2_rank:
      raise ValueError('`groundtruth_boxes` is unbatched while `anchors` is '
                       'batched is not a valid use case, got groundtruth_box '
                       'rank {}, and anchors rank {}'.format(
                           boxes_1_rank, boxes_2_rank))

    result = iou(boxes_1, boxes_2)
    if boxes_1_masks is None and boxes_2_masks is None:
      return result
    background_mask = None
    mask_val_t = tf.cast(self.mask_val, result.dtype) * tf.ones_like(result)
    perm = [1, 0] if boxes_2_rank == 2 else [0, 2, 1]
    if boxes_1_masks is not None and boxes_2_masks is not None:
      background_mask = tf.logical_or(boxes_1_masks,
                                      tf.transpose(boxes_2_masks, perm))
    elif boxes_1_masks is not None:
      background_mask = boxes_1_masks
    else:
      background_mask = tf.logical_or(
          tf.zeros(tf.shape(boxes_2)[:-1], dtype=tf.bool),
          tf.transpose(boxes_2_masks, perm))
    return tf.where(background_mask, mask_val_t, result)
