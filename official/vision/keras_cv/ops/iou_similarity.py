# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Region Similarity Calculators."""

import tensorflow as tf


def area(box):
  """Computes area of boxes.

  Args:
    box: a float Tensor with [N, 4].

  Returns:
    a float tensor with [N].
  """
  with tf.name_scope('Area'):
    y_min, x_min, y_max, x_max = tf.split(
        value=box, num_or_size_splits=4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


def intersection(box1, box2):
  """Compute pairwise intersection areas between boxes.

  Args:
    box1: a float Tensor with [N, 4].
    box2: a float Tensor with [M, 4].

  Returns:
    a float tensor with shape [N, M] representing pairwise intersections
  """
  with tf.name_scope('Intersection'):
    y_min1, x_min1, y_max1, x_max1 = tf.split(
        value=box1, num_or_size_splits=4, axis=1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(
        value=box2, num_or_size_splits=4, axis=1)
    y_min_max = tf.minimum(y_max1, tf.transpose(a=y_max2))
    y_max_min = tf.maximum(y_min1, tf.transpose(a=y_min2))
    intersect_heights = tf.maximum(0.0, y_min_max - y_max_min)
    x_min_max = tf.minimum(x_max1, tf.transpose(a=x_max2))
    x_max_min = tf.maximum(x_min1, tf.transpose(a=x_min2))
    intersect_widths = tf.maximum(0.0, x_min_max - x_max_min)
    return intersect_heights * intersect_widths


def iou(box1, box2):
  """Computes pairwise intersection-over-union between box collections.

  Args:
    box1: a float Tensor with [N, 4].
    box2: a float Tensor with [M, 4].

  Returns:
    a tensor with shape [N, M] representing pairwise iou scores.
  """
  intersections = intersection(box1, box2)
  areas1 = area(box1)
  areas2 = area(box2)
  unions = (
      tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
  return tf.where(
      tf.equal(intersections, 0.0), tf.zeros_like(intersections),
      tf.truediv(intersections, unions))


class IouSimilarity():
  """Class to compute similarity based on Intersection over Union (IOU) metric.

  """

  def __call__(self, groundtruth_boxes, anchors):
    """Compute pairwise IOU similarity between ground truth boxes and anchors.

    Args:
      groundtruth_boxes: a float Tensor with N boxes.
      anchors: a float Tensor with M boxes.

    Returns:
      A tensor with shape [N, M] representing pairwise iou scores.

    Input shape:
      groundtruth_boxes: [N, 4]
      anchors: [M, 4]

    Output shape:
      [N, M]
    """
    with tf.name_scope('IOU'):
      return iou(groundtruth_boxes, anchors)
