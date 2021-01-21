""" bounding box utils file """

# import libraries
import tensorflow as tf
import math
from typing import Union, Tuple


def yxyx_to_xcycwh(box: tf.Tensor):
  """Converts boxes from ymin, xmin, ymax, xmax to x_center, y_center, width,
      height.
    Args:
        box: a `Tensor` whose shape is [4,] and represents the coordinates of
            boxes in ymin, xmin, ymax, xmax.
    Returns:
        box: a `Tensor` whose shape is [4,] and contains the new format.
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
  """Converts boxes from x_center, y_center, width, height to ymin, xmin, ymax,
      xmax.
    Args:
        box: a `Tensor` whose shape is [4,] and represents the coordinates of
            boxes in x_center, y_center, width, height.
    Returns:
        box: a `Tensor` whose shape is [4,] and contains the new format.
    """
  with tf.name_scope('xcycwh_to_yxyx'):
    xy, wh = tf.split(box, 2, axis=-1)
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2
    box = tf.stack(
        [xy_min[..., 1], xy_min[..., 0], xy_max[..., 1], xy_max[..., 0]],
        axis=-1)
    if split_min_max:
      box = tf.split(box, 2, axis=-1)
  return box


def xcycwh_to_xyxy(box: tf.Tensor, split_min_max: bool = False):
  """Converts boxes from x_center, y_center, width, height to xmin, ymin, xmax,
      ymax.
    Args:
        box: box: a `Tensor` whose shape is [4,] and represents the coordinates
            of boxes in x_center, y_center, width, height.
    Returns:
        box: a `Tensor` whose shape is [4,] and contains the new format.
    """
  with tf.name_scope('xcycwh_to_yxyx'):
    xy, wh = tf.split(box, 2, axis=-1)
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2
    box = (xy_min, xy_max)
    if not split_min_max:
      box = tf.concat(box, axis=-1)
  return box


def intersection_and_union(box1: tf.Tensor, box2: tf.Tensor):
  """Calculates the intersection and union between between box1 and box2.
    Args:
        box1: a `Tensor` with a shape of [batch_size, N, 4]. N is the number of
            proposals before groundtruth assignment. The last dimension is the
            pixel coordinates in [ymin, xmin, ymax, xmax].
        box2: a `Tensor` with a shape of [batch_size, N, 4]. N is the number of
            proposals before groundtruth assignment. The last dimension is the
            pixel coordinates in [ymin, xmin, ymax, xmax].
    Returns:
        intersection: a `Tensor` whose shape is [batch_size, N].
        union: a `Tensor` whose shape is [batch_size, N].
    """
  with tf.name_scope('intersection_and_union'):
    intersect_mins = tf.math.maximum(box1[..., 0:2], box2[..., 0:2])
    intersect_maxes = tf.math.minimum(box1[..., 2:4], box2[..., 2:4])
    intersect_wh = tf.math.maximum(intersect_maxes - intersect_mins,
                                   tf.zeros_like(intersect_mins))
    intersection = intersect_wh[..., 0] * intersect_wh[..., 1]

    box1_area = get_area(box1)
    box2_area = get_area(box2)
    union = box1_area + box2_area - intersection
  return intersection, union


def get_area(box: Union[tf.Tensor, Tuple],
             xywh: bool = False,
             use_tuple: bool = False):
  """Calculates the area of the box.
    Args:
        box: box: a `Tensor` whose shape is [4,].
        xywh: a `bool` who flags the format of the box.
        use_tuple: a `bool` that flags the type of box.
    Returns:
        area: a `Tensor` whose shape is [] and value represents the area of the
            box.
    """
  with tf.name_scope('box_area'):
    if use_tuple:
      area = get_area_tuple(box=box, xywh=xywh)
    else:
      area = get_area_tensor(box=box, xywh=xywh)
  return area


def get_area_tensor(box: tf.Tensor, xywh: bool = False):
  """Calculates the area of the box.
    Args:
        box: a `Tensor` whose shape is [4,].
        xywh: a `bool` who flags the format of the box.
    Returns:
        area: a `Tensor` whose shape is [] and value represents the area of the
            box.
    """
  with tf.name_scope('tensor_area'):
    if xywh:
      area = tf.reduce_prod(box[..., 2:4], axis=-1)
    else:
      area = tf.math.abs(tf.reduce_prod(box[..., 2:4] - box[..., 0:2], axis=-1))
  return area


def get_area_tuple(box: Tuple, xywh: bool = False):
  """Calculates the area of the box.
    Args:
        box: box: a `Tuple` whose shape is [4,].
        xywh: a `bool` who flags the format of the box.
    Returns:
        area: a `Tensor` whose shape is [] and value represents the area of the
            box.
    """
  with tf.name_scope('tuple_area'):
    if xywh:
      area = tf.reduce_prod(box[1], axis=-1)
    else:
      area = tf.math.abs(tf.reduce_prod(box[1] - box[0], axis=-1))
  return area


def center_distance(center_1: tf.Tensor, center_2: tf.Tensor):
  """Calculates the squared distance between two points.
    Args:
        center_1: a `Tensor` whose shape is [2,] and represents a point.
        center_2: a `Tensor` whose shape is [2,] and represents a point.
    Returns:
        dist: a `Tensor` whose shape is [] and value represents the squared
            distance between center_1 and center_2.
    """
  with tf.name_scope('center_distance'):
    dist = (center_1[..., 0] - center_2[..., 0])**2 + (center_1[..., 1] -
                                                       center_2[..., 1])**2
  return dist


def aspect_ratio_consistancy(w_gt: tf.Tensor, h_gt: tf.Tensor, w: tf.Tensor,
                             h: tf.Tensor):
  """Calculates the consistency aspect ratio.
    Args:
        w_gt: a `Tensor` whose shape is [] and value represents the width of the ground
            truth box.
        h_gt: a `Tensor` whose shape is [] and value represents the height of the ground
            truth box.
        w_gt: a `Tensor` whose shape is [] and value represents the width of the proposed
            box.
        h_gt: a `Tensor` whose shape is [] and value represents the height of the proposed
            box.
    Returns:
        consistency: a `Tensor` whose shape is [] and value represents the consistency of
            aspect ratio
    """
  arcterm = (tf.math.atan(tf.math.divide_no_nan(w_gt, h_gt)) -
             tf.math.atan(tf.math.divide_no_nan(w, h)))**2
  consistency = 4 * arcterm / (math.pi)**2
  return consistency


def compute_iou(box1, box2):
  """Calculates the intersection of union between box1 and box2.
    Args:
        box1: a `Tensor` whose shape is [4,] and represents the coordinates of boxes in
            x_center, y_center, width, height.
        box2: a `Tensor` whose shape is [4,] and represents the coordinates of boxes in
            x_center, y_center, width, height.
    Returns:
        iou: a `Tensor` whose shape is [] and value represents the intersection over union.
    """
  # get box corners
  with tf.name_scope('iou'):
    box1 = xcycwh_to_yxyx(box1)
    box2 = xcycwh_to_yxyx(box2)
    intersection, union = intersection_and_union(box1, box2)

    iou = tf.math.divide_no_nan(intersection, union)
    iou = tf.clip_by_value(iou, clip_value_min=0.0, clip_value_max=1.0)
  return iou


def compute_giou(box1, box2):
  """Calculates the generalized intersection of union between box1 and box2.
    Args:
        box1: a `Tensor` whose shape is [4,] and represents the coordinates of boxes in
            x_center, y_center, width, height.
        box2: a `Tensor` whose shape is [4,] and represents the coordinates of boxes in
            x_center, y_center, width, height.
    Returns:
        iou: a `Tensor` whose shape is [] and value represents the generalized intersection over union.
    """
  with tf.name_scope('giou'):
    # get box corners
    box1 = xcycwh_to_yxyx(box1)
    box2 = xcycwh_to_yxyx(box2)

    # compute IOU
    intersection, union = intersection_and_union(box1, box2)
    iou = tf.math.divide_no_nan(intersection, union)
    iou = tf.clip_by_value(iou, clip_value_min=0.0, clip_value_max=1.0)

    # find the smallest box to encompase both box1 and box2
    c_mins = tf.math.minimum(box1[..., 0:2], box2[..., 0:2])
    c_maxes = tf.math.maximum(box1[..., 2:4], box2[..., 2:4])
    c = get_area((c_mins, c_maxes), use_tuple=True)

    # compute giou
    giou = iou - tf.math.divide_no_nan((c - union), c)
  return iou, giou


def compute_diou(box1, box2):
  """Calculates the distance intersection of union between box1 and box2.
    Args:
        box1: a `Tensor` whose shape is [4,] and represents the coordinates of boxes in
            x_center, y_center, width, height.
        box2: a `Tensor` whose shape is [4,] and represents the coordinates of boxes in
            x_center, y_center, width, height.
    Returns:
        iou: a `Tensor` whose shape is [] and value represents the distance intersection over union.
    """
  with tf.name_scope('diou'):
    # compute center distance
    dist = center_distance(box1[..., 0:2], box2[..., 0:2])

    # get box corners
    box1 = xcycwh_to_yxyx(box1)
    box2 = xcycwh_to_yxyx(box2)

    # compute IOU
    intersection, union = intersection_and_union(box1, box2)
    iou = tf.math.divide_no_nan(intersection, union)
    iou = tf.clip_by_value(iou, clip_value_min=0.0, clip_value_max=1.0)

    # compute max diagnal of the smallest enclosing box
    c_mins = tf.math.minimum(box1[..., 0:2], box2[..., 0:2])
    c_maxes = tf.math.maximum(box1[..., 2:4], box2[..., 2:4])
    diag_dist = center_distance(c_mins, c_maxes)

    regularization = tf.math.divide_no_nan(dist, diag_dist)
    diou = iou + regularization
  return iou, diou


def compute_ciou(box1, box2):
  """Calculates the complete intersection of union between box1 and box2.
    Args:
        box1: a `Tensor` whose shape is [4,] and represents the coordinates of boxes in
            x_center, y_center, width, height.
        box2: a `Tensor` whose shape is [4,] and represents the coordinates of boxes in
            x_center, y_center, width, height.
    Returns:
        iou: a `Tensor` whose shape is [] and value represents the complete intersection over union.
    """
  with tf.name_scope('ciou'):
    # compute DIOU and IOU
    iou, diou = compute_diou(box1, box2)

    # computer aspect ratio consistency
    v = aspect_ratio_consistancy(box1[..., 2], box1[..., 3], box2[..., 2],
                                 box2[..., 3])

    # compute IOU regularization
    a = tf.math.divide_no_nan(v, ((1 - iou) + v))
    ciou = diou + v * a
  return iou, ciou
