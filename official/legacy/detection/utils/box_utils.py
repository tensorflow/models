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

"""Utility functions for bounding box processing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf, tf_keras

EPSILON = 1e-8
BBOX_XFORM_CLIP = np.log(1000. / 16.)


def visualize_images_with_bounding_boxes(images, box_outputs, step,
                                         summary_writer):
  """Records subset of evaluation images with bounding boxes."""
  image_shape = tf.shape(images[0])
  image_height = tf.cast(image_shape[0], tf.float32)
  image_width = tf.cast(image_shape[1], tf.float32)
  normalized_boxes = normalize_boxes(box_outputs, [image_height, image_width])

  bounding_box_color = tf.constant([[1.0, 1.0, 0.0, 1.0]])
  image_summary = tf.image.draw_bounding_boxes(images, normalized_boxes,
                                               bounding_box_color)
  with summary_writer.as_default():
    tf.summary.image('bounding_box_summary', image_summary, step=step)
    summary_writer.flush()


def yxyx_to_xywh(boxes):
  """Converts boxes from ymin, xmin, ymax, xmax to xmin, ymin, width, height.

  Args:
    boxes: a numpy array whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.

  Returns:
    boxes: a numpy array whose shape is the same as `boxes` in new format.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError('boxes.shape[-1] is {:d}, but must be 4.'.format(
        boxes.shape[-1]))

  boxes_ymin = boxes[..., 0]
  boxes_xmin = boxes[..., 1]
  boxes_width = boxes[..., 3] - boxes[..., 1]
  boxes_height = boxes[..., 2] - boxes[..., 0]
  new_boxes = np.stack([boxes_xmin, boxes_ymin, boxes_width, boxes_height],
                       axis=-1)

  return new_boxes


def jitter_boxes(boxes, noise_scale=0.025):
  """Jitter the box coordinates by some noise distribution.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates of
      boxes in ymin, xmin, ymax, xmax order.
    noise_scale: a python float which specifies the magnitude of noise. The rule
      of thumb is to set this between (0, 0.1]. The default value is found to
      mimic the noisy detections best empirically.

  Returns:
    jittered_boxes: a tensor whose shape is the same as `boxes` representing
      the jittered boxes.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError('boxes.shape[-1] is {:d}, but must be 4.'.format(
        boxes.shape[-1]))

  with tf.name_scope('jitter_boxes'):
    bbox_jitters = tf.random.normal(boxes.get_shape(), stddev=noise_scale)
    ymin = boxes[..., 0:1]
    xmin = boxes[..., 1:2]
    ymax = boxes[..., 2:3]
    xmax = boxes[..., 3:4]
    width = xmax - xmin
    height = ymax - ymin
    new_center_x = (xmin + xmax) / 2.0 + bbox_jitters[..., 0:1] * width
    new_center_y = (ymin + ymax) / 2.0 + bbox_jitters[..., 1:2] * height
    new_width = width * tf.math.exp(bbox_jitters[..., 2:3])
    new_height = height * tf.math.exp(bbox_jitters[..., 3:4])
    jittered_boxes = tf.concat([
        new_center_y - new_height * 0.5, new_center_x - new_width * 0.5,
        new_center_y + new_height * 0.5, new_center_x + new_width * 0.5
    ],
                               axis=-1)

    return jittered_boxes


def normalize_boxes(boxes, image_shape):
  """Converts boxes to the normalized coordinates.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates of
      boxes in ymin, xmin, ymax, xmax order.
    image_shape: a list of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].

  Returns:
    normalized_boxes: a tensor whose shape is the same as `boxes` representing
      the normalized boxes.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError('boxes.shape[-1] is {:d}, but must be 4.'.format(
        boxes.shape[-1]))

  with tf.name_scope('normalize_boxes'):
    if isinstance(image_shape, list) or isinstance(image_shape, tuple):
      height, width = image_shape
    else:
      image_shape = tf.cast(image_shape, dtype=boxes.dtype)
      height = image_shape[..., 0:1]
      width = image_shape[..., 1:2]

    ymin = boxes[..., 0:1] / height
    xmin = boxes[..., 1:2] / width
    ymax = boxes[..., 2:3] / height
    xmax = boxes[..., 3:4] / width

    normalized_boxes = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
    return normalized_boxes


def denormalize_boxes(boxes, image_shape):
  """Converts boxes normalized by [height, width] to pixel coordinates.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates of
      boxes in ymin, xmin, ymax, xmax order.
    image_shape: a list of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].

  Returns:
    denormalized_boxes: a tensor whose shape is the same as `boxes` representing
      the denormalized boxes.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  with tf.name_scope('denormalize_boxes'):
    if isinstance(image_shape, list) or isinstance(image_shape, tuple):
      height, width = image_shape
    else:
      image_shape = tf.cast(image_shape, dtype=boxes.dtype)
      height, width = tf.split(image_shape, 2, axis=-1)

    ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis=-1)
    ymin = ymin * height
    xmin = xmin * width
    ymax = ymax * height
    xmax = xmax * width

    denormalized_boxes = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
    return denormalized_boxes


def clip_boxes(boxes, image_shape):
  """Clips boxes to image boundaries.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates of
      boxes in ymin, xmin, ymax, xmax order.
    image_shape: a list of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].

  Returns:
    clipped_boxes: a tensor whose shape is the same as `boxes` representing the
      clipped boxes.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError('boxes.shape[-1] is {:d}, but must be 4.'.format(
        boxes.shape[-1]))

  with tf.name_scope('clip_boxes'):
    if isinstance(image_shape, list) or isinstance(image_shape, tuple):
      height, width = image_shape
      max_length = [height - 1.0, width - 1.0, height - 1.0, width - 1.0]
    else:
      image_shape = tf.cast(image_shape, dtype=boxes.dtype)
      height, width = tf.unstack(image_shape, axis=-1)
      max_length = tf.stack(
          [height - 1.0, width - 1.0, height - 1.0, width - 1.0], axis=-1)

    clipped_boxes = tf.math.maximum(tf.math.minimum(boxes, max_length), 0.0)
    return clipped_boxes


def compute_outer_boxes(boxes, image_shape, scale=1.0):
  """Compute outer box encloses an object with a margin.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates of
      boxes in ymin, xmin, ymax, xmax order.
    image_shape: a list of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].
    scale: a float number specifying the scale of output outer boxes to input
      `boxes`.

  Returns:
    outer_boxes: a tensor whose shape is the same as `boxes` representing the
      outer boxes.
  """
  if scale < 1.0:
    raise ValueError(
        'scale is {}, but outer box scale must be greater than 1.0.'.format(
            scale))
  centers_y = (boxes[..., 0] + boxes[..., 2]) / 2.0
  centers_x = (boxes[..., 1] + boxes[..., 3]) / 2.0
  box_height = (boxes[..., 2] - boxes[..., 0]) * scale
  box_width = (boxes[..., 3] - boxes[..., 1]) * scale
  outer_boxes = tf.stack([
      centers_y - box_height / 2.0, centers_x - box_width / 2.0,
      centers_y + box_height / 2.0, centers_x + box_width / 2.0
  ],
                         axis=1)
  outer_boxes = clip_boxes(outer_boxes, image_shape)
  return outer_boxes


def encode_boxes(boxes, anchors, weights=None):
  """Encode boxes to targets.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates of
      boxes in ymin, xmin, ymax, xmax order.
    anchors: a tensor whose shape is the same as, or `broadcastable` to `boxes`,
      representing the coordinates of anchors in ymin, xmin, ymax, xmax order.
    weights: None or a list of four float numbers used to scale coordinates.

  Returns:
    encoded_boxes: a tensor whose shape is the same as `boxes` representing the
      encoded box targets.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError('boxes.shape[-1] is {:d}, but must be 4.'.format(
        boxes.shape[-1]))

  with tf.name_scope('encode_boxes'):
    boxes = tf.cast(boxes, dtype=anchors.dtype)
    ymin = boxes[..., 0:1]
    xmin = boxes[..., 1:2]
    ymax = boxes[..., 2:3]
    xmax = boxes[..., 3:4]
    box_h = ymax - ymin + 1.0
    box_w = xmax - xmin + 1.0
    box_yc = ymin + 0.5 * box_h
    box_xc = xmin + 0.5 * box_w

    anchor_ymin = anchors[..., 0:1]
    anchor_xmin = anchors[..., 1:2]
    anchor_ymax = anchors[..., 2:3]
    anchor_xmax = anchors[..., 3:4]
    anchor_h = anchor_ymax - anchor_ymin + 1.0
    anchor_w = anchor_xmax - anchor_xmin + 1.0
    anchor_yc = anchor_ymin + 0.5 * anchor_h
    anchor_xc = anchor_xmin + 0.5 * anchor_w

    encoded_dy = (box_yc - anchor_yc) / anchor_h
    encoded_dx = (box_xc - anchor_xc) / anchor_w
    encoded_dh = tf.math.log(box_h / anchor_h)
    encoded_dw = tf.math.log(box_w / anchor_w)
    if weights:
      encoded_dy *= weights[0]
      encoded_dx *= weights[1]
      encoded_dh *= weights[2]
      encoded_dw *= weights[3]

    encoded_boxes = tf.concat([encoded_dy, encoded_dx, encoded_dh, encoded_dw],
                              axis=-1)
    return encoded_boxes


def decode_boxes(encoded_boxes, anchors, weights=None):
  """Decode boxes.

  Args:
    encoded_boxes: a tensor whose last dimension is 4 representing the
      coordinates of encoded boxes in ymin, xmin, ymax, xmax order.
    anchors: a tensor whose shape is the same as, or `broadcastable` to `boxes`,
      representing the coordinates of anchors in ymin, xmin, ymax, xmax order.
    weights: None or a list of four float numbers used to scale coordinates.

  Returns:
    encoded_boxes: a tensor whose shape is the same as `boxes` representing the
      decoded box targets.
  """
  if encoded_boxes.shape[-1] != 4:
    raise ValueError('encoded_boxes.shape[-1] is {:d}, but must be 4.'.format(
        encoded_boxes.shape[-1]))

  with tf.name_scope('decode_boxes'):
    encoded_boxes = tf.cast(encoded_boxes, dtype=anchors.dtype)
    dy = encoded_boxes[..., 0:1]
    dx = encoded_boxes[..., 1:2]
    dh = encoded_boxes[..., 2:3]
    dw = encoded_boxes[..., 3:4]
    if weights:
      dy /= weights[0]
      dx /= weights[1]
      dh /= weights[2]
      dw /= weights[3]
    dh = tf.math.minimum(dh, BBOX_XFORM_CLIP)
    dw = tf.math.minimum(dw, BBOX_XFORM_CLIP)

    anchor_ymin = anchors[..., 0:1]
    anchor_xmin = anchors[..., 1:2]
    anchor_ymax = anchors[..., 2:3]
    anchor_xmax = anchors[..., 3:4]
    anchor_h = anchor_ymax - anchor_ymin + 1.0
    anchor_w = anchor_xmax - anchor_xmin + 1.0
    anchor_yc = anchor_ymin + 0.5 * anchor_h
    anchor_xc = anchor_xmin + 0.5 * anchor_w

    decoded_boxes_yc = dy * anchor_h + anchor_yc
    decoded_boxes_xc = dx * anchor_w + anchor_xc
    decoded_boxes_h = tf.math.exp(dh) * anchor_h
    decoded_boxes_w = tf.math.exp(dw) * anchor_w

    decoded_boxes_ymin = decoded_boxes_yc - 0.5 * decoded_boxes_h
    decoded_boxes_xmin = decoded_boxes_xc - 0.5 * decoded_boxes_w
    decoded_boxes_ymax = decoded_boxes_ymin + decoded_boxes_h - 1.0
    decoded_boxes_xmax = decoded_boxes_xmin + decoded_boxes_w - 1.0

    decoded_boxes = tf.concat([
        decoded_boxes_ymin, decoded_boxes_xmin, decoded_boxes_ymax,
        decoded_boxes_xmax
    ],
                              axis=-1)
    return decoded_boxes


def encode_boxes_lrtb(boxes, anchors, weights=None):
  """Encode boxes to targets on lrtb (=left,right,top,bottom) format.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    anchors: a tensor whose shape is the same as, or `broadcastable` to `boxes`,
      representing the coordinates of anchors in ymin, xmin, ymax, xmax order.
    weights: None or a list of four float numbers used to scale coordinates.

  Returns:
    encoded_boxes_lrtb: a tensor whose shape is the same as `boxes` representing
      the encoded box targets. The box targets encode the left, right, top,
      bottom distances from an anchor location to the four borders of the
      matched groundtruth bounding box.
    center_targets: centerness targets defined by the left, right, top, and
      bottom distance targets. The centerness is defined as the deviation of the
      anchor location from the groundtruth object center. Formally, centerness =
      sqrt(min(left, right)/max(left, right)*min(top, bottom)/max(top, bottom)).

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError(
        'boxes.shape[-1] is {:d}, but must be 4.'.format(boxes.shape[-1]))

  with tf.name_scope('encode_boxes_lrtb'):
    boxes = tf.cast(boxes, dtype=anchors.dtype)
    ymin = boxes[..., 0:1]
    xmin = boxes[..., 1:2]
    ymax = boxes[..., 2:3]
    xmax = boxes[..., 3:4]
    # box_h = ymax - ymin + 1.0
    # box_w = xmax - xmin + 1.0
    box_h = ymax - ymin
    box_w = xmax - xmin

    anchor_ymin = anchors[..., 0:1]
    anchor_xmin = anchors[..., 1:2]
    anchor_ymax = anchors[..., 2:3]
    anchor_xmax = anchors[..., 3:4]
    # anchor_h = anchor_ymax - anchor_ymin + 1.0
    # anchor_w = anchor_xmax - anchor_xmin + 1.0
    anchor_h = anchor_ymax - anchor_ymin
    anchor_w = anchor_xmax - anchor_xmin
    anchor_yc = anchor_ymin + 0.5 * anchor_h
    anchor_xc = anchor_xmin + 0.5 * anchor_w

    box_h += EPSILON
    box_w += EPSILON
    anchor_h += EPSILON
    anchor_w += EPSILON

    left = (anchor_xc - xmin) / anchor_w
    right = (xmax - anchor_xc) / anchor_w
    top = (anchor_yc - ymin) / anchor_h
    bottom = (ymax - anchor_yc) / anchor_h

    # Create centerness target. {
    lrtb_targets = tf.concat([left, right, top, bottom], axis=-1)
    valid_match = tf.greater(tf.reduce_min(lrtb_targets, -1), 0.0)

    # Centerness score.
    left_right = tf.concat([left, right], axis=-1)

    left_right = tf.where(tf.stack([valid_match, valid_match], -1),
                          left_right, tf.zeros_like(left_right))
    top_bottom = tf.concat([top, bottom], axis=-1)
    top_bottom = tf.where(tf.stack([valid_match, valid_match], -1),
                          top_bottom, tf.zeros_like(top_bottom))
    center_targets = tf.sqrt(
        (tf.reduce_min(left_right, -1) /
         (tf.reduce_max(left_right, -1) + EPSILON)) *
        (tf.reduce_min(top_bottom, -1) /
         (tf.reduce_max(top_bottom, -1) + EPSILON)))
    center_targets = tf.where(valid_match,
                              center_targets,
                              tf.zeros_like(center_targets))
    if weights:
      left *= weights[0]
      right *= weights[1]
      top *= weights[2]
      bottom *= weights[3]

    encoded_boxes_lrtb = tf.concat(
        [left, right, top, bottom],
        axis=-1)

    return encoded_boxes_lrtb, center_targets


def decode_boxes_lrtb(encoded_boxes_lrtb, anchors, weights=None):
  """Decode boxes.

  Args:
    encoded_boxes_lrtb: a tensor whose last dimension is 4 representing the
      coordinates of encoded boxes in left, right, top, bottom order.
    anchors: a tensor whose shape is the same as, or `broadcastable` to `boxes`,
      representing the coordinates of anchors in ymin, xmin, ymax, xmax order.
    weights: None or a list of four float numbers used to scale coordinates.

  Returns:
    decoded_boxes_lrtb: a tensor whose shape is the same as `boxes` representing
      the decoded box targets in lrtb (=left,right,top,bottom) format. The box
      decoded box coordinates represent the left, right, top, and bottom
      distances from an anchor location to the four borders of the matched
      groundtruth bounding box.
  """
  if encoded_boxes_lrtb.shape[-1] != 4:
    raise ValueError(
        'encoded_boxes_lrtb.shape[-1] is {:d}, but must be 4.'
        .format(encoded_boxes_lrtb.shape[-1]))

  with tf.name_scope('decode_boxes_lrtb'):
    encoded_boxes_lrtb = tf.cast(encoded_boxes_lrtb, dtype=anchors.dtype)
    left = encoded_boxes_lrtb[..., 0:1]
    right = encoded_boxes_lrtb[..., 1:2]
    top = encoded_boxes_lrtb[..., 2:3]
    bottom = encoded_boxes_lrtb[..., 3:4]
    if weights:
      left /= weights[0]
      right /= weights[1]
      top /= weights[2]
      bottom /= weights[3]

    anchor_ymin = anchors[..., 0:1]
    anchor_xmin = anchors[..., 1:2]
    anchor_ymax = anchors[..., 2:3]
    anchor_xmax = anchors[..., 3:4]

    anchor_h = anchor_ymax - anchor_ymin
    anchor_w = anchor_xmax - anchor_xmin
    anchor_yc = anchor_ymin + 0.5 * anchor_h
    anchor_xc = anchor_xmin + 0.5 * anchor_w
    anchor_h += EPSILON
    anchor_w += EPSILON

    decoded_boxes_ymin = anchor_yc - top * anchor_h
    decoded_boxes_xmin = anchor_xc - left * anchor_w
    decoded_boxes_ymax = anchor_yc + bottom * anchor_h
    decoded_boxes_xmax = anchor_xc + right * anchor_w

    decoded_boxes_lrtb = tf.concat(
        [decoded_boxes_ymin, decoded_boxes_xmin,
         decoded_boxes_ymax, decoded_boxes_xmax],
        axis=-1)
    return decoded_boxes_lrtb


def filter_boxes(boxes, scores, image_shape, min_size_threshold):
  """Filter and remove boxes that are too small or fall outside the image.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates of
      boxes in ymin, xmin, ymax, xmax order.
    scores: a tensor whose shape is the same as tf.shape(boxes)[:-1]
      representing the original scores of the boxes.
    image_shape: a tensor whose shape is the same as, or `broadcastable` to
      `boxes` except the last dimension, which is 2, representing [height,
      width] of the scaled image.
    min_size_threshold: a float representing the minimal box size in each side
      (w.r.t. the scaled image). Boxes whose sides are smaller than it will be
      filtered out.

  Returns:
    filtered_boxes: a tensor whose shape is the same as `boxes` but with
      the position of the filtered boxes are filled with 0.
    filtered_scores: a tensor whose shape is the same as 'scores' but with
      the positinon of the filtered boxes filled with 0.
  """
  if boxes.shape[-1] != 4:
    raise ValueError('boxes.shape[1] is {:d}, but must be 4.'.format(
        boxes.shape[-1]))

  with tf.name_scope('filter_boxes'):
    if isinstance(image_shape, list) or isinstance(image_shape, tuple):
      height, width = image_shape
    else:
      image_shape = tf.cast(image_shape, dtype=boxes.dtype)
      height = image_shape[..., 0]
      width = image_shape[..., 1]

    ymin = boxes[..., 0]
    xmin = boxes[..., 1]
    ymax = boxes[..., 2]
    xmax = boxes[..., 3]

    h = ymax - ymin + 1.0
    w = xmax - xmin + 1.0
    yc = ymin + 0.5 * h
    xc = xmin + 0.5 * w

    min_size = tf.cast(
        tf.math.maximum(min_size_threshold, 1.0), dtype=boxes.dtype)

    filtered_size_mask = tf.math.logical_and(
        tf.math.greater(h, min_size), tf.math.greater(w, min_size))
    filtered_center_mask = tf.logical_and(
        tf.math.logical_and(tf.math.greater(yc, 0.0), tf.math.less(yc, height)),
        tf.math.logical_and(tf.math.greater(xc, 0.0), tf.math.less(xc, width)))
    filtered_mask = tf.math.logical_and(filtered_size_mask,
                                        filtered_center_mask)

    filtered_scores = tf.where(filtered_mask, scores, tf.zeros_like(scores))
    filtered_boxes = tf.cast(
        tf.expand_dims(filtered_mask, axis=-1), dtype=boxes.dtype) * boxes

    return filtered_boxes, filtered_scores


def filter_boxes_by_scores(boxes, scores, min_score_threshold):
  """Filter and remove boxes whose scores are smaller than the threshold.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates of
      boxes in ymin, xmin, ymax, xmax order.
    scores: a tensor whose shape is the same as tf.shape(boxes)[:-1]
      representing the original scores of the boxes.
    min_score_threshold: a float representing the minimal box score threshold.
      Boxes whose score are smaller than it will be filtered out.

  Returns:
    filtered_boxes: a tensor whose shape is the same as `boxes` but with
      the position of the filtered boxes are filled with -1.
    filtered_scores: a tensor whose shape is the same as 'scores' but with
      the
  """
  if boxes.shape[-1] != 4:
    raise ValueError('boxes.shape[1] is {:d}, but must be 4.'.format(
        boxes.shape[-1]))

  with tf.name_scope('filter_boxes_by_scores'):
    filtered_mask = tf.math.greater(scores, min_score_threshold)
    filtered_scores = tf.where(filtered_mask, scores, -tf.ones_like(scores))
    filtered_boxes = tf.cast(
        tf.expand_dims(filtered_mask, axis=-1), dtype=boxes.dtype) * boxes

    return filtered_boxes, filtered_scores


def top_k_boxes(boxes, scores, k):
  """Sort and select top k boxes according to the scores.

  Args:
    boxes: a tensor of shape [batch_size, N, 4] representing the coordiante of
      the boxes. N is the number of boxes per image.
    scores: a tensor of shsape [batch_size, N] representing the socre of the
      boxes.
    k: an integer or a tensor indicating the top k number.

  Returns:
    selected_boxes: a tensor of shape [batch_size, k, 4] representing the
      selected top k box coordinates.
    selected_scores: a tensor of shape [batch_size, k] representing the selected
      top k box scores.
  """
  with tf.name_scope('top_k_boxes'):
    selected_scores, top_k_indices = tf.nn.top_k(scores, k=k, sorted=True)

    batch_size, _ = scores.get_shape().as_list()
    if batch_size == 1:
      selected_boxes = tf.squeeze(
          tf.gather(boxes, top_k_indices, axis=1), axis=1)
    else:
      top_k_indices_shape = tf.shape(top_k_indices)
      batch_indices = (
          tf.expand_dims(tf.range(top_k_indices_shape[0]), axis=-1) *
          tf.ones([1, top_k_indices_shape[-1]], dtype=tf.int32))
      gather_nd_indices = tf.stack([batch_indices, top_k_indices], axis=-1)
      selected_boxes = tf.gather_nd(boxes, gather_nd_indices)

    return selected_boxes, selected_scores


def bbox_overlap(boxes, gt_boxes):
  """Calculates the overlap between proposal and ground truth boxes.

  Some `gt_boxes` may have been padded.  The returned `iou` tensor for these
  boxes will be -1.

  Args:
    boxes: a tensor with a shape of [batch_size, N, 4]. N is the number of
      proposals before groundtruth assignment (e.g., rpn_post_nms_topn). The
      last dimension is the pixel coordinates in [ymin, xmin, ymax, xmax] form.
    gt_boxes: a tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4]. This
      tensor might have paddings with a negative value.

  Returns:
    iou: a tensor with as a shape of [batch_size, N, MAX_NUM_INSTANCES].
  """
  with tf.name_scope('bbox_overlap'):
    bb_y_min, bb_x_min, bb_y_max, bb_x_max = tf.split(
        value=boxes, num_or_size_splits=4, axis=2)
    gt_y_min, gt_x_min, gt_y_max, gt_x_max = tf.split(
        value=gt_boxes, num_or_size_splits=4, axis=2)

    # Calculates the intersection area.
    i_xmin = tf.math.maximum(bb_x_min, tf.transpose(gt_x_min, [0, 2, 1]))
    i_xmax = tf.math.minimum(bb_x_max, tf.transpose(gt_x_max, [0, 2, 1]))
    i_ymin = tf.math.maximum(bb_y_min, tf.transpose(gt_y_min, [0, 2, 1]))
    i_ymax = tf.math.minimum(bb_y_max, tf.transpose(gt_y_max, [0, 2, 1]))
    i_area = tf.math.maximum((i_xmax - i_xmin), 0) * tf.math.maximum(
        (i_ymax - i_ymin), 0)

    # Calculates the union area.
    bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
    gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)
    # Adds a small epsilon to avoid divide-by-zero.
    u_area = bb_area + tf.transpose(gt_area, [0, 2, 1]) - i_area + 1e-8

    # Calculates IoU.
    iou = i_area / u_area

    # Fills -1 for IoU entries between the padded ground truth boxes.
    gt_invalid_mask = tf.less(
        tf.reduce_max(gt_boxes, axis=-1, keepdims=True), 0.0)
    padding_mask = tf.logical_or(
        tf.zeros_like(bb_x_min, dtype=tf.bool),
        tf.transpose(gt_invalid_mask, [0, 2, 1]))
    iou = tf.where(padding_mask, -tf.ones_like(iou), iou)

    return iou


def get_non_empty_box_indices(boxes):
  """Get indices for non-empty boxes."""
  # Selects indices if box height or width is 0.
  height = boxes[:, 2] - boxes[:, 0]
  width = boxes[:, 3] - boxes[:, 1]
  indices = tf.where(
      tf.logical_and(tf.greater(height, 0), tf.greater(width, 0)))
  return indices[:, 0]
