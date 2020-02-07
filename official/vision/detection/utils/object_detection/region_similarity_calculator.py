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

"""Region Similarity Calculators for BoxLists.

Region Similarity Calculators compare a pairwise measure of similarity
between the boxes in two BoxLists.
"""
from abc import ABCMeta
from abc import abstractmethod

import tensorflow.compat.v2 as tf


def area(boxlist, scope=None):
  """Computes area of boxes.

  Args:
    boxlist: BoxList holding N boxes
    scope: name scope.

  Returns:
    a tensor with shape [N] representing box areas.
  """
  if not scope:
    scope = 'Area'
  with tf.name_scope(scope):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist.get(), num_or_size_splits=4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])


def intersection(boxlist1, boxlist2, scope=None):
  """Compute pairwise intersection areas between boxes.

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.

  Returns:
    a tensor with shape [N, M] representing pairwise intersections
  """
  if not scope:
    scope = 'Intersection'
  with tf.name_scope(scope):
    y_min1, x_min1, y_max1, x_max1 = tf.split(
        value=boxlist1.get(), num_or_size_splits=4, axis=1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(
        value=boxlist2.get(), num_or_size_splits=4, axis=1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(a=y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(a=y_min2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(a=x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(a=x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def iou(boxlist1, boxlist2, scope=None):
  """Computes pairwise intersection-over-union between box collections.

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.

  Returns:
    a tensor with shape [N, M] representing pairwise iou scores.
  """
  if not scope:
    scope = 'IOU'
  with tf.name_scope(scope):
    intersections = intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0), tf.zeros_like(intersections),
        tf.truediv(intersections, unions))


class RegionSimilarityCalculator(object):
  """Abstract base class for region similarity calculator."""
  __metaclass__ = ABCMeta

  def compare(self, boxlist1, boxlist2, scope=None):
    """Computes matrix of pairwise similarity between BoxLists.

    This op (to be overriden) computes a measure of pairwise similarity between
    the boxes in the given BoxLists. Higher values indicate more similarity.

    Note that this method simply measures similarity and does not explicitly
    perform a matching.

    Args:
      boxlist1: BoxList holding N boxes.
      boxlist2: BoxList holding M boxes.
      scope: Op scope name. Defaults to 'Compare' if None.

    Returns:
      a (float32) tensor of shape [N, M] with pairwise similarity score.
    """
    if not scope:
      scope = 'Compare'
    with tf.name_scope(scope) as scope:
      return self._compare(boxlist1, boxlist2)

  @abstractmethod
  def _compare(self, boxlist1, boxlist2):
    pass


class IouSimilarity(RegionSimilarityCalculator):
  """Class to compute similarity based on Intersection over Union (IOU) metric.

  This class computes pairwise similarity between two BoxLists based on IOU.
  """

  def _compare(self, boxlist1, boxlist2):
    """Compute pairwise IOU similarity between the two BoxLists.

    Args:
      boxlist1: BoxList holding N boxes.
      boxlist2: BoxList holding M boxes.

    Returns:
      A tensor with shape [N, M] representing pairwise iou scores.
    """
    return iou(boxlist1, boxlist2)
