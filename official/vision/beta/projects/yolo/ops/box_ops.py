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

"""Bounding box utils."""

import math

import tensorflow as tf


def yxyx_to_xcycwh(box: tf.Tensor):
  """Converts boxes from ymin, xmin, ymax, xmax.

  to x_center, y_center, width, height.

  Args:
    box: `Tensor` whose shape is [..., 4] and represents the coordinates
      of boxes in ymin, xmin, ymax, xmax.

  Returns:
    `Tensor` whose shape is [..., 4] and contains the new format.

  Raises:
    ValueError: If the last dimension of box is not 4 or if box's dtype isn't
      a floating point type.
  """
  with tf.name_scope('yxyx_to_xcycwh'):
    ymin, xmin, ymax, xmax = tf.split(box, 4, axis=-1)
    x_center = (xmax + xmin) / 2
    y_center = (ymax + ymin) / 2
    width = xmax - xmin
    height = ymax - ymin
    box = tf.concat([x_center, y_center, width, height], axis=-1)
  return box


def xcycwh_to_yxyx(box: tf.Tensor, split_min_max: bool = False):
  """Converts boxes from x_center, y_center, width, height.

  to ymin, xmin, ymax, xmax.

  Args:
    box: a `Tensor` whose shape is [..., 4] and represents the coordinates
      of boxes in x_center, y_center, width, height.
    split_min_max: bool, whether or not to split x, y min and max values.

  Returns:
    box: a `Tensor` whose shape is [..., 4] and contains the new format.

  Raises:
    ValueError: If the last dimension of box is not 4 or if box's dtype isn't
      a floating point type.
  """
  with tf.name_scope('xcycwh_to_yxyx'):
    xy, wh = tf.split(box, 2, axis=-1)
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2
    x_min, y_min = tf.split(xy_min, 2, axis=-1)
    x_max, y_max = tf.split(xy_max, 2, axis=-1)
    box = tf.concat([y_min, x_min, y_max, x_max], axis=-1)
    if split_min_max:
      box = tf.split(box, 2, axis=-1)
  return box


def xcycwh_to_xyxy(box: tf.Tensor, split_min_max: bool = False):
  """Converts boxes from x_center, y_center, width, height to.

  xmin, ymin, xmax, ymax.

  Args:
    box: box: a `Tensor` whose shape is [..., 4] and represents the
      coordinates of boxes in x_center, y_center, width, height.
    split_min_max: bool, whether or not to split x, y min and max values.

  Returns:
    box: a `Tensor` whose shape is [..., 4] and contains the new format.

  Raises:
    ValueError: If the last dimension of box is not 4 or if box's dtype isn't
      a floating point type.
  """
  with tf.name_scope('xcycwh_to_yxyx'):
    xy, wh = tf.split(box, 2, axis=-1)
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2
    box = (xy_min, xy_max)
    if not split_min_max:
      box = tf.concat(box, axis=-1)
  return box


def center_distance(center_1: tf.Tensor, center_2: tf.Tensor):
  """Calculates the squared distance between two points.

  This function is mathematically equivalent to the following code, but has
  smaller rounding errors.

  tf.norm(center_1 - center_2, axis=-1)**2

  Args:
    center_1: a `Tensor` whose shape is [..., 2] and represents a point.
    center_2: a `Tensor` whose shape is [..., 2] and represents a point.

  Returns:
    dist: a `Tensor` whose shape is [...] and value represents the squared
      distance between center_1 and center_2.

  Raises:
    ValueError: If the last dimension of either center_1 or center_2 is not 2.
  """
  with tf.name_scope('center_distance'):
    dist = (center_1[..., 0] - center_2[..., 0])**2 + (center_1[..., 1] -
                                                       center_2[..., 1])**2
  return dist


def compute_iou(box1, box2, yxyx=False):
  """Calculates the intersection of union between box1 and box2.

  Args:
    box1: a `Tensor` whose shape is [..., 4] and represents the coordinates of
      boxes in x_center, y_center, width, height.
    box2: a `Tensor` whose shape is [..., 4] and represents the coordinates of
      boxes in x_center, y_center, width, height.
    yxyx: `bool`, whether or not box1, and box2 are in yxyx format.

  Returns:
    iou: a `Tensor` whose shape is [...] and value represents the intersection
      over union.

  Raises:
    ValueError: If the last dimension of either box1 or box2 is not 4.
  """
  # Get box corners
  with tf.name_scope('iou'):
    if not yxyx:
      box1 = xcycwh_to_yxyx(box1)
      box2 = xcycwh_to_yxyx(box2)

    b1mi, b1ma = tf.split(box1, 2, axis=-1)
    b2mi, b2ma = tf.split(box2, 2, axis=-1)
    intersect_mins = tf.math.maximum(b1mi, b2mi)
    intersect_maxes = tf.math.minimum(b1ma, b2ma)
    intersect_wh = tf.math.maximum(intersect_maxes - intersect_mins,
                                   tf.zeros_like(intersect_mins))
    intersection = tf.reduce_prod(
        intersect_wh, axis=-1)  # intersect_wh[..., 0] * intersect_wh[..., 1]

    box1_area = tf.math.abs(tf.reduce_prod(b1ma - b1mi, axis=-1))
    box2_area = tf.math.abs(tf.reduce_prod(b2ma - b2mi, axis=-1))
    union = box1_area + box2_area - intersection

    iou = intersection / (union + 1e-7)
    iou = tf.clip_by_value(iou, clip_value_min=0.0, clip_value_max=1.0)
  return iou


def compute_giou(box1, box2):
  """Calculates the generalized intersection of union between box1 and box2.

  Args:
    box1: a `Tensor` whose shape is [..., 4] and represents the coordinates of
      boxes in x_center, y_center, width, height.
    box2: a `Tensor` whose shape is [..., 4] and represents the coordinates of
      boxes in x_center, y_center, width, height.

  Returns:
    iou: a `Tensor` whose shape is [...] and value represents the generalized
      intersection over union.

  Raises:
    ValueError: If the last dimension of either box1 or box2 is not 4.
  """
  with tf.name_scope('giou'):
    # get box corners
    box1 = xcycwh_to_yxyx(box1)
    box2 = xcycwh_to_yxyx(box2)

    # compute IOU
    intersect_mins = tf.math.maximum(box1[..., 0:2], box2[..., 0:2])
    intersect_maxes = tf.math.minimum(box1[..., 2:4], box2[..., 2:4])
    intersect_wh = tf.math.maximum(intersect_maxes - intersect_mins,
                                   tf.zeros_like(intersect_mins))
    intersection = intersect_wh[..., 0] * intersect_wh[..., 1]

    box1_area = tf.math.abs(
        tf.reduce_prod(box1[..., 2:4] - box1[..., 0:2], axis=-1))
    box2_area = tf.math.abs(
        tf.reduce_prod(box2[..., 2:4] - box2[..., 0:2], axis=-1))
    union = box1_area + box2_area - intersection

    iou = tf.math.divide_no_nan(intersection, union)
    iou = tf.clip_by_value(iou, clip_value_min=0.0, clip_value_max=1.0)

    # find the smallest box to encompase both box1 and box2
    c_mins = tf.math.minimum(box1[..., 0:2], box2[..., 0:2])
    c_maxes = tf.math.maximum(box1[..., 2:4], box2[..., 2:4])
    c = tf.math.abs(tf.reduce_prod(c_mins - c_maxes, axis=-1))

    # compute giou
    giou = iou - tf.math.divide_no_nan((c - union), c)
  return iou, giou


def compute_diou(box1, box2):
  """Calculates the distance intersection of union between box1 and box2.

  Args:
    box1: a `Tensor` whose shape is [..., 4] and represents the coordinates of
      boxes in x_center, y_center, width, height.
    box2: a `Tensor` whose shape is [..., 4] and represents the coordinates of
      boxes in x_center, y_center, width, height.

  Returns:
    iou: a `Tensor` whose shape is [...] and value represents the distance
      intersection over union.

  Raises:
    ValueError: If the last dimension of either box1 or box2 is not 4.
  """
  with tf.name_scope('diou'):
    # compute center distance
    dist = center_distance(box1[..., 0:2], box2[..., 0:2])

    # get box corners
    box1 = xcycwh_to_yxyx(box1)
    box2 = xcycwh_to_yxyx(box2)

    # compute IOU
    intersect_mins = tf.math.maximum(box1[..., 0:2], box2[..., 0:2])
    intersect_maxes = tf.math.minimum(box1[..., 2:4], box2[..., 2:4])
    intersect_wh = tf.math.maximum(intersect_maxes - intersect_mins,
                                   tf.zeros_like(intersect_mins))
    intersection = intersect_wh[..., 0] * intersect_wh[..., 1]

    box1_area = tf.math.abs(
        tf.reduce_prod(box1[..., 2:4] - box1[..., 0:2], axis=-1))
    box2_area = tf.math.abs(
        tf.reduce_prod(box2[..., 2:4] - box2[..., 0:2], axis=-1))
    union = box1_area + box2_area - intersection

    iou = tf.math.divide_no_nan(intersection, union)
    iou = tf.clip_by_value(iou, clip_value_min=0.0, clip_value_max=1.0)

    # compute max diagnal of the smallest enclosing box
    c_mins = tf.math.minimum(box1[..., 0:2], box2[..., 0:2])
    c_maxes = tf.math.maximum(box1[..., 2:4], box2[..., 2:4])

    diag_dist = tf.reduce_sum((c_maxes - c_mins)**2, axis=-1)

    regularization = tf.math.divide_no_nan(dist, diag_dist)
    diou = iou + regularization
  return iou, diou


def compute_ciou(box1, box2):
  """Calculates the complete intersection of union between box1 and box2.

  Args:
    box1: a `Tensor` whose shape is [..., 4] and represents the coordinates
      of boxes in x_center, y_center, width, height.
    box2: a `Tensor` whose shape is [..., 4] and represents the coordinates of
      boxes in x_center, y_center, width, height.

  Returns:
    iou: a `Tensor` whose shape is [...] and value represents the complete
      intersection over union.

  Raises:
    ValueError: If the last dimension of either box1 or box2 is not 4.
  """
  with tf.name_scope('ciou'):
    # compute DIOU and IOU
    iou, diou = compute_diou(box1, box2)

    # computer aspect ratio consistency
    arcterm = (
        tf.math.atan(tf.math.divide_no_nan(box1[..., 2], box1[..., 3])) -
        tf.math.atan(tf.math.divide_no_nan(box2[..., 2], box2[..., 3])))**2
    v = 4 * arcterm / (math.pi)**2

    # compute IOU regularization
    a = tf.math.divide_no_nan(v, ((1 - iou) + v))
    ciou = diou + v * a
  return iou, ciou
