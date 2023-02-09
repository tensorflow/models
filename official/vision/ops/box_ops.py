# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Box related ops."""

# Import libraries
import numpy as np
import tensorflow as tf


EPSILON = 1e-8
BBOX_XFORM_CLIP = np.log(1000. / 16.)


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
    raise ValueError(
        'boxes.shape[-1] is {:d}, but must be 4.'.format(boxes.shape[-1]))

  boxes_ymin = boxes[..., 0]
  boxes_xmin = boxes[..., 1]
  boxes_width = boxes[..., 3] - boxes[..., 1]
  boxes_height = boxes[..., 2] - boxes[..., 0]
  new_boxes = np.stack(
      [boxes_xmin, boxes_ymin, boxes_width, boxes_height], axis=-1)

  return new_boxes


def yxyx_to_cycxhw(boxes):
  """Converts box corner coordinates to center plus height and width terms.

  Args:
    boxes: a `Tensor` with last dimension of 4, representing the coordinates of
      boxes in ymin, xmin, ymax, xmax order.

  Returns:
    boxes: a `Tensor` with the same shape as the inputted boxes, in the format
      of cy, cx, height, width.

  Raises:
    ValueError: if the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError('Last dimension of boxes must be 4 but is {:d}'.format(
        boxes.shape[-1]))

  boxes_ycenter = (boxes[..., 0] + boxes[..., 2]) / 2
  boxes_xcenter = (boxes[..., 1] + boxes[..., 3]) / 2
  boxes_height = boxes[..., 2] - boxes[..., 0]
  boxes_width = boxes[..., 3] - boxes[..., 1]

  new_boxes = tf.stack(
      [boxes_ycenter, boxes_xcenter, boxes_height, boxes_width], axis=-1)
  return new_boxes


def cycxhw_to_yxyx(boxes):
  """Converts box center coordinates plus height and width terms to corner.

  Args:
    boxes: a numpy array whose last dimension is 4 representing the coordinates
      of boxes in cy, cx, height, width order.

  Returns:
    boxes: a numpy array whose shape is the same as `boxes` in new format.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError(
        'boxes.shape[-1] is {:d}, but must be 4.'.format(boxes.shape[-1]))

  boxes_ymin = boxes[..., 0] - boxes[..., 2] / 2
  boxes_xmin = boxes[..., 1] - boxes[..., 3] / 2
  boxes_ymax = boxes[..., 0] + boxes[..., 2] / 2
  boxes_xmax = boxes[..., 1] + boxes[..., 3] / 2
  new_boxes = tf.stack([
      boxes_ymin, boxes_xmin, boxes_ymax, boxes_xmax], axis=-1)
  return new_boxes


def jitter_boxes(boxes, noise_scale=0.025):
  """Jitters the box coordinates by some noise distribution.

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
    raise ValueError(
        'boxes.shape[-1] is {:d}, but must be 4.'.format(boxes.shape[-1]))

  with tf.name_scope('jitter_boxes'):
    bbox_jitters = tf.random.normal(tf.shape(boxes), stddev=noise_scale)
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
    jittered_boxes = tf.concat(
        [new_center_y - new_height * 0.5, new_center_x - new_width * 0.5,
         new_center_y + new_height * 0.5, new_center_x + new_width * 0.5],
        axis=-1)

    return jittered_boxes


def normalize_boxes(boxes, image_shape):
  """Converts boxes to the normalized coordinates.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
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
    raise ValueError(
        'boxes.shape[-1] is {:d}, but must be 4.'.format(boxes.shape[-1]))

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
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
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


def horizontal_flip_boxes(normalized_boxes):
  """Flips normalized boxes horizontally.

  Args:
    normalized_boxes: the boxes in normalzied coordinates.

  Returns:
    horizontally flipped boxes.
  """
  if normalized_boxes.shape[-1] != 4:
    raise ValueError('boxes.shape[-1] is {:d}, but must be 4.'.format(
        normalized_boxes.shape[-1]))

  with tf.name_scope('horizontal_flip_boxes'):
    ymin, xmin, ymax, xmax = tf.split(
        value=normalized_boxes, num_or_size_splits=4, axis=-1)
    flipped_xmin = tf.subtract(1.0, xmax)
    flipped_xmax = tf.subtract(1.0, xmin)
    flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], axis=-1)
    return flipped_boxes


def clip_boxes(boxes, image_shape):
  """Clips boxes to image boundaries.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
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
    raise ValueError(
        'boxes.shape[-1] is {:d}, but must be 4.'.format(boxes.shape[-1]))

  with tf.name_scope('clip_boxes'):
    if isinstance(image_shape, list) or isinstance(image_shape, tuple):
      height, width = image_shape
      max_length = [height, width, height, width]
    else:
      image_shape = tf.cast(image_shape, dtype=boxes.dtype)
      height, width = tf.unstack(image_shape, axis=-1)
      max_length = tf.stack([height, width, height, width], axis=-1)

    clipped_boxes = tf.math.maximum(tf.math.minimum(boxes, max_length), 0.0)
    return clipped_boxes


def compute_outer_boxes(boxes, image_shape, scale=1.0):
  """Computes outer box encloses an object with a margin.

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
  if scale == 1.0:
    return boxes
  centers_y = (boxes[..., 0] + boxes[..., 2]) / 2.0
  centers_x = (boxes[..., 1] + boxes[..., 3]) / 2.0
  box_height = (boxes[..., 2] - boxes[..., 0]) * scale
  box_width = (boxes[..., 3] - boxes[..., 1]) * scale
  outer_boxes = tf.stack(
      [centers_y - box_height / 2.0, centers_x - box_width / 2.0,
       centers_y + box_height / 2.0, centers_x + box_width / 2.0],
      axis=-1)
  outer_boxes = clip_boxes(outer_boxes, image_shape)
  return outer_boxes


def encode_boxes(boxes, anchors, weights=None):
  """Encodes boxes to targets.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
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
    raise ValueError(
        'boxes.shape[-1] is {:d}, but must be 4.'.format(boxes.shape[-1]))

  with tf.name_scope('encode_boxes'):
    boxes = tf.cast(boxes, dtype=anchors.dtype)
    ymin = boxes[..., 0:1]
    xmin = boxes[..., 1:2]
    ymax = boxes[..., 2:3]
    xmax = boxes[..., 3:4]
    box_h = ymax - ymin
    box_w = xmax - xmin
    box_yc = ymin + 0.5 * box_h
    box_xc = xmin + 0.5 * box_w

    anchor_ymin = anchors[..., 0:1]
    anchor_xmin = anchors[..., 1:2]
    anchor_ymax = anchors[..., 2:3]
    anchor_xmax = anchors[..., 3:4]
    anchor_h = anchor_ymax - anchor_ymin
    anchor_w = anchor_xmax - anchor_xmin
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

    encoded_boxes = tf.concat(
        [encoded_dy, encoded_dx, encoded_dh, encoded_dw], axis=-1)
    return encoded_boxes


def decode_boxes(encoded_boxes, anchors, weights=None):
  """Decodes boxes.

  Args:
    encoded_boxes: a tensor whose last dimension is 4 representing the
      coordinates of encoded boxes in dy, dx, dh, dw in order.
    anchors: a tensor whose shape is the same as, or `broadcastable` to `boxes`,
      representing the coordinates of anchors in ymin, xmin, ymax, xmax order.
    weights: None or a list of four float numbers used to scale coordinates.

  Returns:
    decoded_boxes: a tensor whose shape is the same as `boxes` representing the
      decoded box targets.
  """
  if encoded_boxes.shape[-1] != 4:
    raise ValueError(
        'encoded_boxes.shape[-1] is {:d}, but must be 4.'
        .format(encoded_boxes.shape[-1]))

  with tf.name_scope('decode_boxes'):
    encoded_boxes = tf.cast(encoded_boxes, dtype=anchors.dtype)
    dy, dx, dh, dw = tf.split(encoded_boxes, 4, -1)
    if weights:
      dy /= weights[0]
      dx /= weights[1]
      dh /= weights[2]
      dw /= weights[3]
    dh = tf.math.minimum(dh, BBOX_XFORM_CLIP)
    dw = tf.math.minimum(dw, BBOX_XFORM_CLIP)

    anchor_ymin, anchor_xmin, anchor_ymax, anchor_xmax = tf.split(
        anchors, 4, -1)
    anchor_h = anchor_ymax - anchor_ymin
    anchor_w = anchor_xmax - anchor_xmin
    anchor_yc = anchor_ymin + 0.5 * anchor_h
    anchor_xc = anchor_xmin + 0.5 * anchor_w

    decoded_boxes_yc = dy * anchor_h + anchor_yc
    decoded_boxes_xc = dx * anchor_w + anchor_xc
    decoded_boxes_h = tf.math.exp(dh) * anchor_h
    decoded_boxes_w = tf.math.exp(dw) * anchor_w

    decoded_boxes_ymin = decoded_boxes_yc - 0.5 * decoded_boxes_h
    decoded_boxes_xmin = decoded_boxes_xc - 0.5 * decoded_boxes_w
    decoded_boxes_ymax = decoded_boxes_ymin + decoded_boxes_h
    decoded_boxes_xmax = decoded_boxes_xmin + decoded_boxes_w

    decoded_boxes = tf.concat(
        [decoded_boxes_ymin, decoded_boxes_xmin,
         decoded_boxes_ymax, decoded_boxes_xmax],
        axis=-1)
    return decoded_boxes


def filter_boxes(boxes, scores, image_shape, min_size_threshold):
  """Filters and remove boxes that are too small or fall outside the image.

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
    raise ValueError(
        'boxes.shape[1] is {:d}, but must be 4.'.format(boxes.shape[-1]))

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

    h = ymax - ymin
    w = xmax - xmin
    yc = ymin + 0.5 * h
    xc = xmin + 0.5 * w

    min_size = tf.cast(
        tf.math.maximum(min_size_threshold, 0.0), dtype=boxes.dtype)

    filtered_size_mask = tf.math.logical_and(
        tf.math.greater(h, min_size), tf.math.greater(w, min_size))
    filtered_center_mask = tf.logical_and(
        tf.math.logical_and(tf.math.greater(yc, 0.0), tf.math.less(yc, height)),
        tf.math.logical_and(tf.math.greater(xc, 0.0), tf.math.less(xc, width)))
    filtered_mask = tf.math.logical_and(
        filtered_size_mask, filtered_center_mask)

    filtered_scores = tf.where(filtered_mask, scores, tf.zeros_like(scores))
    filtered_boxes = tf.cast(
        tf.expand_dims(filtered_mask, axis=-1), dtype=boxes.dtype) * boxes
    return filtered_boxes, filtered_scores


def filter_boxes_by_scores(boxes, scores, min_score_threshold):
  """Filters and remove boxes whose scores are smaller than the threshold.

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


def gather_instances(selected_indices, instances, *aux_instances):
  """Gathers instances by indices.

  Args:
    selected_indices: a Tensor of shape [batch, K] which indicates the selected
      indices in instance dimension (2nd dimension).
    instances: a Tensor of shape [batch, N, ...] where the 2nd dimension is
      the instance dimension to be selected from.
    *aux_instances: the additional Tensors whose shapes are in [batch, N, ...]
      which are the tensors to be selected from using the `selected_indices`.

  Returns:
    selected_instances: the tensor of shape [batch, K, ...] which corresponds to
      the selected instances of the `instances` tensor.
    selected_aux_instances: the additional tensors of shape [batch, K, ...]
      which corresponds to the selected instances of the `aus_instances`
      tensors.
  """
  batch_size = instances.shape[0]
  if batch_size == 1:
    selected_instances = tf.squeeze(
        tf.gather(instances, selected_indices, axis=1), axis=1)
    if aux_instances:
      selected_aux_instances = [
          tf.squeeze(
              tf.gather(a, selected_indices, axis=1), axis=1)
          for a in aux_instances
      ]
      return tuple([selected_instances] + selected_aux_instances)
    else:
      return selected_instances
  else:
    indices_shape = tf.shape(selected_indices)
    batch_indices = (
        tf.expand_dims(tf.range(indices_shape[0]), axis=-1) *
        tf.ones([1, indices_shape[-1]], dtype=tf.int32))
    gather_nd_indices = tf.stack(
        [batch_indices, selected_indices], axis=-1)
    selected_instances = tf.gather_nd(instances, gather_nd_indices)
    if aux_instances:
      selected_aux_instances = [
          tf.gather_nd(a, gather_nd_indices) for a in aux_instances
      ]
      return tuple([selected_instances] + selected_aux_instances)
    else:
      return selected_instances


def top_k_boxes(boxes, scores, k):
  """Sorts and select top k boxes according to the scores.

  Args:
    boxes: a tensor of shape [batch_size, N, 4] representing the coordinate of
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
    selected_boxes = gather_instances(top_k_indices, boxes)
    return selected_boxes, selected_scores


def get_non_empty_box_indices(boxes):
  """Gets indices for non-empty boxes."""
  # Selects indices if box height or width is 0.
  height = boxes[:, 2] - boxes[:, 0]
  width = boxes[:, 3] - boxes[:, 1]
  indices = tf.where(tf.logical_and(tf.greater(height, 0),
                                    tf.greater(width, 0)))
  return indices[:, 0]


def bbox_overlap(boxes, gt_boxes):
  """Calculates the overlap between proposal and ground truth boxes.

  Some `boxes` or `gt_boxes` may have been padded.  The returned `iou` tensor
  for these boxes will be -1.

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
    i_area = (
        tf.math.maximum((i_xmax - i_xmin), 0) *
        tf.math.maximum((i_ymax - i_ymin), 0))

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

    # Fills -1 for invalid (-1) boxes.
    boxes_invalid_mask = tf.less(
        tf.reduce_max(boxes, axis=-1, keepdims=True), 0.0)
    iou = tf.where(boxes_invalid_mask, -tf.ones_like(iou), iou)

    return iou


def bbox_generalized_overlap(boxes, gt_boxes):
  """Calculates the GIOU between proposal and ground truth boxes.

  The generalized intersection of union is an adjustment of the traditional IOU
  metric which provides continuous updates even for predictions with no overlap.
  This metric is defined in https://giou.stanford.edu/GIoU.pdf. Note, some
  `gt_boxes` may have been padded. The returned `giou` tensor for these boxes
  will be -1.

  Args:
    boxes: a `Tensor` with a shape of [batch_size, N, 4]. N is the number of
      proposals before groundtruth assignment (e.g., rpn_post_nms_topn). The
      last dimension is the pixel coordinates in [ymin, xmin, ymax, xmax] form.
    gt_boxes: a `Tensor` with a shape of [batch_size, max_num_instances, 4].
      This tensor may have paddings with a negative value and will also be in
      the [ymin, xmin, ymax, xmax] format.

  Returns:
    giou: a `Tensor` with as a shape of [batch_size, N, max_num_instances].
  """
  with tf.name_scope('bbox_generalized_overlap'):
    assert boxes.shape.as_list(
    )[-1] == 4, 'Boxes must be defined by 4 coordinates.'
    assert gt_boxes.shape.as_list(
    )[-1] == 4, 'Groundtruth boxes must be defined by 4 coordinates.'

    bb_y_min, bb_x_min, bb_y_max, bb_x_max = tf.split(
        value=boxes, num_or_size_splits=4, axis=2)
    gt_y_min, gt_x_min, gt_y_max, gt_x_max = tf.split(
        value=gt_boxes, num_or_size_splits=4, axis=2)

    # Calculates the hull area for each pair of boxes, with one from
    # boxes and the other from gt_boxes.
    # Outputs for coordinates are of shape [batch_size, N, max_num_instances]
    h_xmin = tf.minimum(bb_x_min, tf.transpose(gt_x_min, [0, 2, 1]))
    h_xmax = tf.maximum(bb_x_max, tf.transpose(gt_x_max, [0, 2, 1]))
    h_ymin = tf.minimum(bb_y_min, tf.transpose(gt_y_min, [0, 2, 1]))
    h_ymax = tf.maximum(bb_y_max, tf.transpose(gt_y_max, [0, 2, 1]))
    h_area = tf.maximum((h_xmax - h_xmin), 0) * tf.maximum((h_ymax - h_ymin), 0)
    # Add a small epsilon to avoid divide-by-zero.
    h_area = h_area + 1e-8

    # Calculates the intersection area.
    i_xmin = tf.maximum(bb_x_min, tf.transpose(gt_x_min, [0, 2, 1]))
    i_xmax = tf.minimum(bb_x_max, tf.transpose(gt_x_max, [0, 2, 1]))
    i_ymin = tf.maximum(bb_y_min, tf.transpose(gt_y_min, [0, 2, 1]))
    i_ymax = tf.minimum(bb_y_max, tf.transpose(gt_y_max, [0, 2, 1]))
    i_area = tf.maximum((i_xmax - i_xmin), 0) * tf.maximum((i_ymax - i_ymin), 0)

    # Calculates the union area.
    bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
    gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)

    # Adds a small epsilon to avoid divide-by-zero.
    u_area = bb_area + tf.transpose(gt_area, [0, 2, 1]) - i_area + 1e-8

    # Calculates IoU.
    iou = i_area / u_area
    # Calculates GIoU.
    giou = iou - (h_area - u_area) / h_area

    # Fills -1 for GIoU entries between the padded ground truth boxes.
    gt_invalid_mask = tf.less(
        tf.reduce_max(gt_boxes, axis=-1, keepdims=True), 0.0)
    padding_mask = tf.broadcast_to(
        tf.transpose(gt_invalid_mask, [0, 2, 1]), tf.shape(giou))
    giou = tf.where(padding_mask, -tf.ones_like(giou), giou)
    return giou


def bbox_intersection_over_area(boxes, gt_boxes):
  """Calculates IoAs (intersection over area) between proposal and ground truth boxes.

  Some `boxes` or `gt_boxes` may have been padded.  The returned `iou` tensor
  for these boxes will be -1.

  Args:
    boxes: a tensor with a shape of [batch_size, N, 4]. N is the number of
      proposals before groundtruth assignment (e.g., rpn_post_nms_topn). The
      last dimension is the pixel coordinates in [ymin, xmin, ymax, xmax] form.
    gt_boxes: a tensor with a shape of [batch_size, M, 4]. This tensor might
      have paddings with a negative value.

  Returns:
    ioa: a tensor with as a shape of [batch_size, N, M].
  """
  with tf.name_scope('bbox_overlap'):
    bb_y_min, bb_x_min, bb_y_max, bb_x_max = tf.split(
        value=boxes, num_or_size_splits=4, axis=2
    )
    gt_y_min, gt_x_min, gt_y_max, gt_x_max = tf.split(
        value=gt_boxes, num_or_size_splits=4, axis=2
    )

    # Calculates the intersection area.
    i_xmin = tf.math.maximum(bb_x_min, tf.transpose(gt_x_min, [0, 2, 1]))
    i_xmax = tf.math.minimum(bb_x_max, tf.transpose(gt_x_max, [0, 2, 1]))
    i_ymin = tf.math.maximum(bb_y_min, tf.transpose(gt_y_min, [0, 2, 1]))
    i_ymax = tf.math.minimum(bb_y_max, tf.transpose(gt_y_max, [0, 2, 1]))
    i_area = tf.math.maximum((i_xmax - i_xmin), 0) * tf.math.maximum(
        (i_ymax - i_ymin), 0
    )

    bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
    ioa = tf.math.divide_no_nan(i_area, bb_area)

    # Fills -1 for IoA entries between the padded ground truth boxes.
    gt_invalid_mask = tf.less(
        tf.reduce_max(gt_boxes, axis=-1, keepdims=True), 0.0
    )
    padding_mask = tf.logical_or(
        tf.zeros_like(bb_x_min, dtype=tf.bool),
        tf.transpose(gt_invalid_mask, [0, 2, 1]),
    )
    ioa = tf.where(padding_mask, -1., ioa)

    # Fills -1 for invalid (-1) boxes.
    boxes_invalid_mask = tf.less(
        tf.reduce_max(boxes, axis=-1, keepdims=True), 0.0
    )
    ioa = tf.where(boxes_invalid_mask, -1., ioa)

    return ioa


def box_matching(boxes, gt_boxes, gt_classes):
  """Matches boxes to groundtruth boxes.

  Given the proposal boxes and the groundtruth boxes and classes, perform the
  groundtruth matching by taking the argmax of the IoU between boxes and
  groundtruth boxes.

  Args:
    boxes: a tensor of shape of [batch_size, N, 4] representing the box
      coordiantes to be matched to groundtruth boxes.
    gt_boxes: a tensor of shape of [batch_size, MAX_INSTANCES, 4] representing
      the groundtruth box coordinates. It is padded with -1s to indicate the
      invalid boxes.
    gt_classes: [batch_size, MAX_INSTANCES] representing the groundtruth box
      classes. It is padded with -1s to indicate the invalid classes.

  Returns:
    matched_gt_boxes: a tensor of shape of [batch_size, N, 4], representing
      the matched groundtruth box coordinates for each input box. If the box
      does not overlap with any groundtruth boxes, the matched boxes of it
      will be set to all 0s.
    matched_gt_classes: a tensor of shape of [batch_size, N], representing
      the matched groundtruth classes for each input box. If the box does not
      overlap with any groundtruth boxes, the matched box classes of it will
      be set to 0, which corresponds to the background class.
    matched_gt_indices: a tensor of shape of [batch_size, N], representing
      the indices of the matched groundtruth boxes in the original gt_boxes
      tensor. If the box does not overlap with any groundtruth boxes, the
      index of the matched groundtruth will be set to -1.
    matched_iou: a tensor of shape of [batch_size, N], representing the IoU
      between the box and its matched groundtruth box. The matched IoU is the
      maximum IoU of the box and all the groundtruth boxes.
    iou: a tensor of shape of [batch_size, N, K], representing the IoU matrix
      between boxes and the groundtruth boxes. The IoU between a box and the
      invalid groundtruth boxes whose coordinates are [-1, -1, -1, -1] is -1.
  """
  # Compute IoU between boxes and gt_boxes.
  # iou <- [batch_size, N, K]
  iou = bbox_overlap(boxes, gt_boxes)

  # max_iou <- [batch_size, N]
  # 0.0 -> no match to gt, or -1.0 match to no gt
  matched_iou = tf.reduce_max(iou, axis=-1)

  # background_box_mask <- bool, [batch_size, N]
  background_box_mask = tf.less_equal(matched_iou, 0.0)

  argmax_iou_indices = tf.argmax(iou, axis=-1, output_type=tf.int32)

  matched_gt_boxes, matched_gt_classes = gather_instances(
      argmax_iou_indices, gt_boxes, gt_classes)
  matched_gt_boxes = tf.where(
      tf.tile(tf.expand_dims(background_box_mask, axis=-1), [1, 1, 4]),
      tf.zeros_like(matched_gt_boxes, dtype=matched_gt_boxes.dtype),
      matched_gt_boxes)
  matched_gt_classes = tf.where(
      background_box_mask,
      tf.zeros_like(matched_gt_classes),
      matched_gt_classes)

  matched_gt_indices = tf.where(
      background_box_mask,
      -tf.ones_like(argmax_iou_indices),
      argmax_iou_indices)

  return (matched_gt_boxes, matched_gt_classes, matched_gt_indices,
          matched_iou, iou)


def bbox2mask(bbox: tf.Tensor,
              *,
              image_height: int,
              image_width: int,
              dtype: tf.DType = tf.bool) -> tf.Tensor:
  """Converts bounding boxes to bitmasks.

  Args:
    bbox: A tensor in shape (..., 4) with arbitrary numbers of batch dimensions,
      representing the absolute coordinates (ymin, xmin, ymax, xmax) for each
      bounding box.
    image_height: an integer representing the height of the image.
    image_width: an integer representing the width of the image.
    dtype: DType of the output bitmasks.

  Returns:
    A tensor in shape (..., height, width) which stores the bitmasks created
    from the bounding boxes. For example:

    >>> bbox2mask(tf.constant([[1,2,4,4]]),
                  image_height=5,
                  image_width=5,
                  dtype=tf.int32)
    <tf.Tensor: shape=(1, 5, 5), dtype=int32, numpy=
    array([[[0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0]]], dtype=int32)>
  """
  bbox_shape = bbox.get_shape().as_list()
  if bbox_shape[-1] != 4:
    raise ValueError(
        'Expected the last dimension of `bbox` has size == 4, but the shape '
        'of `bbox` was: %s' % bbox_shape)

  # (..., 1)
  ymin = bbox[..., 0:1]
  xmin = bbox[..., 1:2]
  ymax = bbox[..., 2:3]
  xmax = bbox[..., 3:4]
  # (..., 1, width)
  ymin = tf.expand_dims(tf.repeat(ymin, repeats=image_width, axis=-1), axis=-2)
  # (..., height, 1)
  xmin = tf.expand_dims(tf.repeat(xmin, repeats=image_height, axis=-1), axis=-1)
  # (..., 1, width)
  ymax = tf.expand_dims(tf.repeat(ymax, repeats=image_width, axis=-1), axis=-2)
  # (..., height, 1)
  xmax = tf.expand_dims(tf.repeat(xmax, repeats=image_height, axis=-1), axis=-1)

  # (height, 1)
  y_grid = tf.expand_dims(tf.range(image_height, dtype=bbox.dtype), axis=-1)
  # (1, width)
  x_grid = tf.expand_dims(tf.range(image_width, dtype=bbox.dtype), axis=-2)

  # (..., height, width)
  ymin_mask = y_grid >= ymin
  xmin_mask = x_grid >= xmin
  ymax_mask = y_grid < ymax
  xmax_mask = x_grid < xmax
  return tf.cast(ymin_mask & xmin_mask & ymax_mask & xmax_mask, dtype)
