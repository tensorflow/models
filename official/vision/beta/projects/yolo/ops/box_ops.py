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

"""Yolo box ops."""
import math
import tensorflow as tf
from official.vision.beta.projects.yolo.ops import math_ops


def yxyx_to_xcycwh(box: tf.Tensor):
  """Converts boxes from yxyx to x_center, y_center, width, height.

  Args:
    box: any `Tensor` whose last dimension is 4 representing the coordinates of
      boxes in ymin, xmin, ymax, xmax.

  Returns:
    box: a `Tensor` whose shape is the same as `box` in new format.
  """
  with tf.name_scope('yxyx_to_xcycwh'):
    ymin, xmin, ymax, xmax = tf.split(box, 4, axis=-1)
    x_center = (xmax + xmin) / 2
    y_center = (ymax + ymin) / 2
    width = xmax - xmin
    height = ymax - ymin
    box = tf.concat([x_center, y_center, width, height], axis=-1)
  return box


def xcycwh_to_yxyx(box: tf.Tensor):
  """Converts boxes from x_center, y_center, width, height to yxyx format.

  Args:
    box: any `Tensor` whose last dimension is 4 representing the coordinates of
      boxes in x_center, y_center, width, height.

  Returns:
    box: a `Tensor` whose shape is the same as `box` in new format.
  """
  with tf.name_scope('xcycwh_to_yxyx'):
    xy, wh = tf.split(box, 2, axis=-1)
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2
    x_min, y_min = tf.split(xy_min, 2, axis=-1)
    x_max, y_max = tf.split(xy_max, 2, axis=-1)
    box = tf.concat([y_min, x_min, y_max, x_max], axis=-1)
  return box


def intersect_and_union(box1, box2, yxyx=False):
  """Calculates the intersection and union between box1 and box2.

  Args:
    box1: any `Tensor` whose last dimension is 4 representing the coordinates of
      boxes.
    box2: any `Tensor` whose last dimension is 4 representing the coordinates of
      boxes.
    yxyx: a `bool` indicating whether the input box is of the format x_center
      y_center, width, height or y_min, x_min, y_max, x_max.

  Returns:
    intersection: a `Tensor` who represents the intersection.
    union: a `Tensor` who represents the union.
  """
  if not yxyx:
    box1_area = tf.reduce_prod(tf.split(box1, 2, axis=-1)[-1], axis=-1)
    box2_area = tf.reduce_prod(tf.split(box2, 2, axis=-1)[-1], axis=-1)
    box1 = xcycwh_to_yxyx(box1)
    box2 = xcycwh_to_yxyx(box2)

  b1mi, b1ma = tf.split(box1, 2, axis=-1)
  b2mi, b2ma = tf.split(box2, 2, axis=-1)
  intersect_mins = tf.math.maximum(b1mi, b2mi)
  intersect_maxes = tf.math.minimum(b1ma, b2ma)
  intersect_wh = tf.math.maximum(intersect_maxes - intersect_mins, 0.0)
  intersection = tf.reduce_prod(intersect_wh, axis=-1)

  if yxyx:
    box1_area = tf.reduce_prod(b1ma - b1mi, axis=-1)
    box2_area = tf.reduce_prod(b2ma - b2mi, axis=-1)
  union = box1_area + box2_area - intersection
  return intersection, union


def smallest_encompassing_box(box1, box2, yxyx=False, clip=False):
  """Calculates the smallest box that encompasses box1 and box2.

  Args:
    box1: any `Tensor` whose last dimension is 4 representing the coordinates of
      boxes.
    box2: any `Tensor` whose last dimension is 4 representing the coordinates of
      boxes.
    yxyx: a `bool` indicating whether the input box is of the format x_center
      y_center, width, height or y_min, x_min, y_max, x_max.
    clip: a `bool`, whether or not to clip boxes.

  Returns:
    box_c: a `Tensor` whose last dimension is 4 representing the coordinates of
      boxes, the return format is y_min, x_min, y_max, x_max if yxyx is set to
      to True. In other words it will match the input format.
  """
  if not yxyx:
    box1 = xcycwh_to_yxyx(box1)
    box2 = xcycwh_to_yxyx(box2)

  b1mi, b1ma = tf.split(box1, 2, axis=-1)
  b2mi, b2ma = tf.split(box2, 2, axis=-1)

  bcmi = tf.math.minimum(b1mi, b2mi)
  bcma = tf.math.maximum(b1ma, b2ma)
  box_c = tf.concat([bcmi, bcma], axis=-1)

  if not yxyx:
    box_c = yxyx_to_xcycwh(box_c)

  if clip:
    bca = tf.reduce_prod(bcma - bcmi, keepdims=True, axis=-1)
    box_c = tf.where(bca <= 0.0, tf.zeros_like(box_c), box_c)
  return bcmi, bcma, box_c


def compute_iou(box1, box2, yxyx=False):
  """Calculates the intersection over union between box1 and box2.

  Args:
    box1: any `Tensor` whose last dimension is 4 representing the coordinates of
      boxes.
    box2: any `Tensor` whose last dimension is 4 representing the coordinates of
      boxes.
    yxyx: a `bool` indicating whether the input box is of the format x_center
      y_center, width, height or y_min, x_min, y_max, x_max.

  Returns:
    iou: a `Tensor` who represents the intersection over union.
  """
  with tf.name_scope('iou'):
    intersection, union = intersect_and_union(box1, box2, yxyx=yxyx)
    iou = math_ops.divide_no_nan(intersection, union)
  return iou


def compute_giou(box1, box2, yxyx=False):
  """Calculates the General intersection over union between box1 and box2.

  Args:
    box1: any `Tensor` whose last dimension is 4 representing the coordinates of
      boxes.
    box2: any `Tensor` whose last dimension is 4 representing the coordinates of
      boxes.
    yxyx: a `bool` indicating whether the input box is of the format x_center
      y_center, width, height or y_min, x_min, y_max, x_max.

  Returns:
    giou: a `Tensor` who represents the General intersection over union.
  """
  with tf.name_scope('giou'):
    if not yxyx:
      yxyx1 = xcycwh_to_yxyx(box1)
      yxyx2 = xcycwh_to_yxyx(box2)
    else:
      yxyx1, yxyx2 = box1, box2

    cmi, cma, _ = smallest_encompassing_box(yxyx1, yxyx2, yxyx=True)
    intersection, union = intersect_and_union(yxyx1, yxyx2, yxyx=True)
    iou = math_ops.divide_no_nan(intersection, union)

    bcwh = cma - cmi
    c = tf.math.reduce_prod(bcwh, axis=-1)

    regularization = math_ops.divide_no_nan((c - union), c)
    giou = iou - regularization
  return iou, giou


def compute_diou(box1, box2, beta=1.0, yxyx=False):
  """Calculates the distance intersection over union between box1 and box2.

  Args:
    box1: any `Tensor` whose last dimension is 4 representing the coordinates of
      boxes.
    box2: any `Tensor` whose last dimension is 4 representing the coordinates of
      boxes.
    beta: a `float` indicating the amount to scale the distance iou
      regularization term.
    yxyx: a `bool` indicating whether the input box is of the format x_center
      y_center, width, height or y_min, x_min, y_max, x_max.

  Returns:
    diou: a `Tensor` who represents the distance intersection over union.
  """
  with tf.name_scope('diou'):
    # compute center distance
    if not yxyx:
      xycc1, xycc2 = box1, box2
      yxyx1 = xcycwh_to_yxyx(box1)
      yxyx2 = xcycwh_to_yxyx(box2)
    else:
      yxyx1, yxyx2 = box1, box2
      xycc1 = yxyx_to_xcycwh(box1)
      xycc2 = yxyx_to_xcycwh(box2)

    cmi, cma, _ = smallest_encompassing_box(yxyx1, yxyx2, yxyx=True)
    intersection, union = intersect_and_union(yxyx1, yxyx2, yxyx=True)
    iou = math_ops.divide_no_nan(intersection, union)

    b1xy, _ = tf.split(xycc1, 2, axis=-1)
    b2xy, _ = tf.split(xycc2, 2, axis=-1)
    bcwh = cma - cmi

    center_dist = tf.reduce_sum((b1xy - b2xy)**2, axis=-1)
    c_diag = tf.reduce_sum(bcwh**2, axis=-1)

    regularization = math_ops.divide_no_nan(center_dist, c_diag)
    diou = iou - regularization**beta
  return iou, diou


def compute_ciou(box1, box2, yxyx=False, darknet=False):
  """Calculates the complete intersection over union between box1 and box2.

  Args:
    box1: any `Tensor` whose last dimension is 4 representing the coordinates of
      boxes.
    box2: any `Tensor` whose last dimension is 4 representing the coordinates of
      boxes.
    yxyx: a `bool` indicating whether the input box is of the format x_center
      y_center, width, height or y_min, x_min, y_max, x_max.
    darknet: a `bool` indicating whether the calling function is the YOLO
      darknet loss.

  Returns:
    ciou: a `Tensor` who represents the complete intersection over union.
  """
  with tf.name_scope('ciou'):
    if not yxyx:
      xycc1, xycc2 = box1, box2
      yxyx1 = xcycwh_to_yxyx(box1)
      yxyx2 = xcycwh_to_yxyx(box2)
    else:
      yxyx1, yxyx2 = box1, box2
      xycc1 = yxyx_to_xcycwh(box1)
      xycc2 = yxyx_to_xcycwh(box2)

    # Build the smallest encomapssing box.
    cmi, cma, _ = smallest_encompassing_box(yxyx1, yxyx2, yxyx=True)
    intersection, union = intersect_and_union(yxyx1, yxyx2, yxyx=True)
    iou = math_ops.divide_no_nan(intersection, union)

    b1xy, b1w, b1h = tf.split(xycc1, [2, 1, 1], axis=-1)
    b2xy, b2w, b2h = tf.split(xycc2, [2, 1, 1], axis=-1)
    bchw = cma - cmi

    # Center regularization
    center_dist = tf.reduce_sum((b1xy - b2xy)**2, axis=-1)
    c_diag = tf.reduce_sum(bchw**2, axis=-1)
    regularization = math_ops.divide_no_nan(center_dist, c_diag)

    # Computer aspect ratio consistency
    terma = math_ops.divide_no_nan(b1w, b1h)  # gt
    termb = math_ops.divide_no_nan(b2w, b2h)  # pred
    arcterm = tf.squeeze(
        tf.math.pow(tf.math.atan(termb) - tf.math.atan(terma), 2), axis=-1)
    v = (4 / math.pi**2) * arcterm

    # Compute the aspect ratio weight, should be treated as a constant
    a = tf.stop_gradient(math_ops.divide_no_nan(v, 1 - iou + v))

    if darknet:
      grad_scale = tf.stop_gradient(tf.square(b2w) + tf.square(b2h))
      v *= tf.squeeze(grad_scale, axis=-1)

    ciou = iou - regularization - (v * a)
  return iou, ciou


def aggregated_comparitive_iou(boxes1, boxes2=None, iou_type=0, beta=0.6):
  """Calculates the IOU between two set of boxes.

  Similar to bbox_overlap but far more versitile.

  Args:
    boxes1: a `Tensor` of shape [batch size, N, 4] representing the coordinates
      of boxes.
    boxes2: a `Tensor` of shape [batch size, N, 4] representing the coordinates
      of boxes.
    iou_type: `integer` representing the iou version to use, 0 is distance iou,
      1 is the general iou, 2 is the complete iou, any other number uses the
      standard iou.
    beta: `float` for the scaling quantity to apply to distance iou
      regularization.

  Returns:
    iou: a `Tensor` who represents the intersection over union in of the
      expected/input type.
  """
  boxes1 = tf.expand_dims(boxes1, axis=-2)

  if boxes2 is not None:
    boxes2 = tf.expand_dims(boxes2, axis=-3)
  else:
    boxes2 = tf.transpose(boxes1, perm=(0, 2, 1, 3))

  if iou_type == 0 or iou_type == 'diou':  # diou
    _, iou = compute_diou(boxes1, boxes2, beta=beta, yxyx=True)
  elif iou_type == 1 or iou_type == 'giou':  # giou
    _, iou = compute_giou(boxes1, boxes2, yxyx=True)
  elif iou_type == 2 or iou_type == 'ciou':  # ciou
    _, iou = compute_ciou(boxes1, boxes2, yxyx=True)
  else:
    iou = compute_iou(boxes1, boxes2, yxyx=True)
  return iou
