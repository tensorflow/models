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

"""nms computation."""

import tensorflow as tf, tf_keras

from official.projects.yolo.ops import box_ops

NMS_TILE_SIZE = 512


# pylint: disable=missing-function-docstring
def aggregated_comparative_iou(boxes1, boxes2=None, iou_type=0):
  k = tf.shape(boxes1)[-2]

  boxes1 = tf.expand_dims(boxes1, axis=-2)
  boxes1 = tf.tile(boxes1, [1, 1, k, 1])

  if boxes2 is not None:
    boxes2 = tf.expand_dims(boxes2, axis=-2)
    boxes2 = tf.tile(boxes2, [1, 1, k, 1])
    boxes2 = tf.transpose(boxes2, perm=(0, 2, 1, 3))
  else:
    boxes2 = tf.transpose(boxes1, perm=(0, 2, 1, 3))

  if iou_type == 0:  # diou
    _, iou = box_ops.compute_diou(boxes1, boxes2)
  elif iou_type == 1:  # giou
    _, iou = box_ops.compute_giou(boxes1, boxes2)
  else:
    iou = box_ops.compute_iou(boxes1, boxes2, yxyx=True)
  return iou


# pylint: disable=missing-function-docstring
def sort_drop(objectness, box, classificationsi, k):
  objectness, ind = tf.math.top_k(objectness, k=k)

  ind_m = tf.ones_like(ind) * tf.expand_dims(
      tf.range(0,
               tf.shape(objectness)[0]), axis=-1)
  bind = tf.stack([tf.reshape(ind_m, [-1]), tf.reshape(ind, [-1])], axis=-1)

  box = tf.gather_nd(box, bind)
  classifications = tf.gather_nd(classificationsi, bind)

  bsize = tf.shape(ind)[0]
  box = tf.reshape(box, [bsize, k, -1])
  classifications = tf.reshape(classifications, [bsize, k, -1])
  return objectness, box, classifications


# pylint: disable=missing-function-docstring
def segment_nms(boxes, classes, confidence, k, iou_thresh):
  mrange = tf.range(k)
  mask_x = tf.tile(
      tf.transpose(tf.expand_dims(mrange, axis=-1), perm=[1, 0]), [k, 1])
  mask_y = tf.tile(tf.expand_dims(mrange, axis=-1), [1, k])
  mask_diag = tf.expand_dims(mask_x > mask_y, axis=0)

  iou = aggregated_comparative_iou(boxes, iou_type=0)

  # duplicate boxes
  iou_mask = iou >= iou_thresh
  iou_mask = tf.logical_and(mask_diag, iou_mask)
  iou *= tf.cast(iou_mask, iou.dtype)

  can_suppress_others = 1 - tf.cast(
      tf.reduce_any(iou_mask, axis=-2), boxes.dtype)

  raw = tf.cast(can_suppress_others, boxes.dtype)

  boxes *= tf.expand_dims(raw, axis=-1)
  confidence *= tf.cast(raw, confidence.dtype)
  classes *= tf.cast(raw, classes.dtype)

  return boxes, classes, confidence


# pylint: disable=missing-function-docstring
def nms(boxes,
        classes,
        confidence,
        k,
        pre_nms_thresh,
        nms_thresh,
        limit_pre_thresh=False,
        use_classes=True):
  if limit_pre_thresh:
    confidence, boxes, classes = sort_drop(confidence, boxes, classes, k)

  mask = tf.fill(
      tf.shape(confidence), tf.cast(pre_nms_thresh, dtype=confidence.dtype))
  mask = tf.math.ceil(tf.nn.relu(confidence - mask))
  confidence = confidence * mask
  mask = tf.expand_dims(mask, axis=-1)
  boxes = boxes * mask
  classes = classes * mask

  if use_classes:
    confidence = tf.reduce_max(classes, axis=-1)
  confidence, boxes, classes = sort_drop(confidence, boxes, classes, k)

  classes = tf.cast(tf.argmax(classes, axis=-1), tf.float32)
  boxes, classes, confidence = segment_nms(boxes, classes, confidence, k,
                                           nms_thresh)
  confidence, boxes, classes = sort_drop(confidence, boxes, classes, k)
  classes = tf.squeeze(classes, axis=-1)
  return boxes, classes, confidence
