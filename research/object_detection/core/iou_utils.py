# Copyright 2020 Google Research. All Rights Reserved.
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
"""IoU utils for box regression with iou losses.
Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression.
https://arxiv.org/pdf/1911.08287.pdf
"""
import math
from typing import Union, Text
import numpy as np
import tensorflow as tf
FloatType = Union[tf.Tensor, float, np.float32, np.float64]


def _get_v(b1_height: FloatType, b1_width: FloatType, b2_height: FloatType,
           b2_width: FloatType) -> tf.Tensor:
  """Get the consistency measurement of aspect ratio for ciou."""

  @tf.custom_gradient
  def _get_grad_v(height, width):
    """backpropogate gradient."""
    arctan = tf.atan(tf.math.divide_no_nan(b1_width, b1_height)) - tf.atan(
        tf.math.divide_no_nan(width, height))
    v = 4 * ((arctan / math.pi)**2)

    def _grad_v(dv):
      """Grad for eager mode."""
      gdw = dv * 8 * arctan * height / (math.pi**2)
      gdh = -dv * 8 * arctan * width / (math.pi**2)
      return [gdh, gdw]

    def _grad_v_graph(dv, variables):
      """Grad for graph mode."""
      gdw = dv * 8 * arctan * height / (math.pi**2)
      gdh = -dv * 8 * arctan * width / (math.pi**2)
      return [gdh, gdw], tf.gradients(v, variables, grad_ys=dv)

    if tf.compat.v1.executing_eagerly_outside_functions():
      return v, _grad_v
    return v, _grad_v_graph

  return _get_grad_v(b2_height, b2_width)


def _iou_per_anchor(pred_boxes: FloatType,
                    target_boxes: FloatType,
                    iou_type: Text = 'iou') -> tf.Tensor:
  """Computing the IoU for a single anchor.
  Args:
    pred_boxes: predicted boxes, with coordinate [y_min, x_min, y_max, x_max].
    target_boxes: target boxes, with coordinate [y_min, x_min, y_max, x_max].
    iou_type: one of ['iou', 'ciou', 'diou', 'giou'].
  Returns:
    IoU loss float `Tensor`.
  """
  # t_ denotes target boxes and p_ denotes predicted boxes.
  t_ymin, t_xmin, t_ymax, t_xmax = target_boxes
  p_ymin, p_xmin, p_ymax, p_xmax = pred_boxes

  zero = tf.convert_to_tensor(0.0, t_ymin.dtype)
  p_width = tf.maximum(zero, p_xmax - p_xmin)
  p_height = tf.maximum(zero, p_ymax - p_ymin)
  t_width = tf.maximum(zero, t_xmax - t_xmin)
  t_height = tf.maximum(zero, t_ymax - t_ymin)
  p_area = p_width * p_height
  t_area = t_width * t_height

  intersect_ymin = tf.maximum(p_ymin, t_ymin)
  intersect_xmin = tf.maximum(p_xmin, t_xmin)
  intersect_ymax = tf.minimum(p_ymax, t_ymax)
  intersect_xmax = tf.minimum(p_xmax, t_xmax)
  intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
  intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
  intersect_area = intersect_width * intersect_height

  union_area = p_area + t_area - intersect_area
  iou_v = tf.math.divide_no_nan(intersect_area, union_area)
  if iou_type == 'iou':
    return iou_v  # iou is the simplest form.

  enclose_ymin = tf.minimum(p_ymin, t_ymin)
  enclose_xmin = tf.minimum(p_xmin, t_xmin)
  enclose_ymax = tf.maximum(p_ymax, t_ymax)
  enclose_xmax = tf.maximum(p_xmax, t_xmax)

  assert iou_type in ('giou', 'diou', 'ciou')
  if iou_type == 'giou':  # giou is the generalized iou.
    enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
    enclose_area = enclose_width * enclose_height
    giou_v = iou_v - tf.math.divide_no_nan(
        (enclose_area - union_area), enclose_area)
    return giou_v

  assert iou_type in ('diou', 'ciou')
  p_center = tf.stack([(p_ymin + p_ymax) / 2, (p_xmin + p_xmax) / 2])
  t_center = tf.stack([(t_ymin + t_ymax) / 2, (t_xmin + t_xmax) / 2])
  euclidean = tf.linalg.norm(t_center - p_center)
  diag_length = tf.linalg.norm(
      [enclose_ymax - enclose_ymin, enclose_xmax - enclose_xmin])
  diou_v = iou_v - tf.math.divide_no_nan(euclidean**2, diag_length**2)
  if iou_type == 'diou':  # diou is the distance iou.
    return diou_v

  assert iou_type == 'ciou'
  v = _get_v(p_height, p_width, t_height, t_width)
  alpha = tf.math.divide_no_nan(v, ((1 - iou_v) + v))
  return diou_v - alpha * v  # the last one is ciou.

iou_dict = {0: 'iou', 1: 'ciou', 2: 'diou', 3: 'giou'}
def iou_loss(pred_boxes: FloatType,
             target_boxes: FloatType,
             iou_type = 0) -> tf.Tensor:
  """A unified interface for computing various IoU losses.
  Let B and B_gt denotes the pred_box and B_gt is the target box (ground truth):
    IoU = |B & B_gt| / |B | B_gt|
    GIoU = IoU - |C - B U B_gt| / C, where C is the smallest box covering B and
    B_gt.
    DIoU = IoU - E(B, B_gt)^2 / c^2, E is the Euclidean distance of the center
    points of B and B_gt, and c is the diagonal length of the smallest box
    covering the two boxes
    CIoU = IoU - DIoU - a * v, where a is a positive trade-off parameter, and
    v measures the consistency of aspect ratio:
      v = (arctan(w_gt / h_gt) - arctan(w / h)) * 4 / pi^2
    where (w_gt, h_gt) and (w, h) are the width and height of the target and
    predicted box respectively.
  The returned loss is computed as 1 - one of {IoU, GIoU, DIoU, CIoU}.
  Args:
    pred_boxes: predicted boxes, with coordinate [y_min, x_min, y_max, x_max]*.
      It can be multiple anchors, with each anchor box has four coordinates.
    target_boxes: target boxes, with coordinate [y_min, x_min, y_max, x_max]*.
      It can be multiple anchors, with each anchor box has four coordinates.
    iou_type: one of ['iou', 'ciou', 'diou', 'giou'].
  Returns:
    IoU loss float `Tensor`.
  """
  iou_type = iou_dict[iou_type]
  if iou_type not in ('iou', 'ciou', 'diou', 'giou'):
    raise ValueError(
        'Unknown loss_type {}, not iou/ciou/diou/giou'.format(iou_type))

  pred_boxes = tf.convert_to_tensor(pred_boxes, tf.float32)
  target_boxes = tf.cast(target_boxes, pred_boxes.dtype)

  # t_ denotes target boxes and p_ denotes predicted boxes: (y, x, y_max, x_max)
  pred_boxes_list = tf.unstack(pred_boxes, None, axis=-1)
  target_boxes_list = tf.unstack(target_boxes, None, axis=-1)
  assert len(pred_boxes_list) == len(target_boxes_list)
  assert len(pred_boxes_list) % 4 == 0

  iou_loss_list = []
  for i in range(0, len(pred_boxes_list), 4):
    pred_boxes = pred_boxes_list[i:i + 4]
    target_boxes = target_boxes_list[i:i + 4]

    # Compute mask.
    t_ymin, t_xmin, t_ymax, t_xmax = target_boxes
    mask = tf.math.logical_and(t_ymax > t_ymin, t_xmax > t_xmin)
    mask = tf.cast(mask, t_ymin.dtype)
    # Loss should be mask * (1 - iou) = mask - masked_iou.
    pred_boxes = [b * mask for b in pred_boxes]
    target_boxes = [b * mask for b in target_boxes]
    iou_loss_list.append(
        mask *
        (1 - tf.squeeze(_iou_per_anchor(pred_boxes, target_boxes, iou_type))))
  #   return iou_loss_list[0]
  # return tf.reduce_sum(tf.stack(iou_loss_list), 0)
  return tf.convert_to_tensor(iou_loss_list, tf.float32)