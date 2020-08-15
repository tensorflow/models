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

"""Base anchor generator.

The job of the anchor generator is to create (or load) a collection
of bounding boxes to be used as anchors.

Generated anchors are assumed to match some convolutional grid or list of grid
shapes.  For example, we might want to generate anchors matching an 8x8
feature map and a 4x4 feature map.  If we place 3 anchors per grid location
on the first feature map and 6 anchors per grid location on the second feature
map, then 3*8*8 + 6*4*4 = 288 anchors are generated in total.

To support fully convolutional settings, feature map shapes are passed
dynamically at generation time.  The number of anchors to place at each location
is static --- implementations of AnchorGenerator must always be able return
the number of anchors that it uses per location for each feature map.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod

import six
from six.moves import zip
import tensorflow.compat.v1 as tf


class AnchorGenerator(six.with_metaclass(ABCMeta, object)):
  """Abstract base class for anchor generators."""

  @abstractmethod
  def name_scope(self):
    """Name scope.

    Must be defined by implementations.

    Returns:
      a string representing the name scope of the anchor generation operation.
    """
    pass

  @property
  def check_num_anchors(self):
    """Whether to dynamically check the number of anchors generated.

    Can be overridden by implementations that would like to disable this
    behavior.

    Returns:
      a boolean controlling whether the Generate function should dynamically
      check the number of anchors generated against the mathematically
      expected number of anchors.
    """
    return True

  @abstractmethod
  def num_anchors_per_location(self):
    """Returns the number of anchors per spatial location.

    Returns:
      a list of integers, one for each expected feature map to be passed to
      the `generate` function.
    """
    pass

  def generate(self, feature_map_shape_list, **params):
    """Generates a collection of bounding boxes to be used as anchors.

    TODO(rathodv): remove **params from argument list and make stride and
      offsets (for multiple_grid_anchor_generator) constructor arguments.

    Args:
      feature_map_shape_list: list of (height, width) pairs in the format
        [(height_0, width_0), (height_1, width_1), ...] that the generated
        anchors must align with.  Pairs can be provided as 1-dimensional
        integer tensors of length 2 or simply as tuples of integers.
      **params: parameters for anchor generation op

    Returns:
      boxes_list: a list of BoxLists each holding anchor boxes corresponding to
        the input feature map shapes.

    Raises:
      ValueError: if the number of feature map shapes does not match the length
        of NumAnchorsPerLocation.
    """
    if self.check_num_anchors and (
        len(feature_map_shape_list) != len(self.num_anchors_per_location())):
      raise ValueError('Number of feature maps is expected to equal the length '
                       'of `num_anchors_per_location`.')
    with tf.name_scope(self.name_scope()):
      anchors_list = self._generate(feature_map_shape_list, **params)
      if self.check_num_anchors:
        with tf.control_dependencies([
            self._assert_correct_number_of_anchors(
                anchors_list, feature_map_shape_list)]):
          for item in anchors_list:
            item.set(tf.identity(item.get()))
      return anchors_list

  @abstractmethod
  def _generate(self, feature_map_shape_list, **params):
    """To be overridden by implementations.

    Args:
      feature_map_shape_list: list of (height, width) pairs in the format
        [(height_0, width_0), (height_1, width_1), ...] that the generated
        anchors must align with.
      **params: parameters for anchor generation op

    Returns:
      boxes_list: a list of BoxList, each holding a collection of N anchor
        boxes.
    """
    pass

  def anchor_index_to_feature_map_index(self, boxlist_list):
    """Returns a 1-D array of feature map indices for each anchor.

    Args:
      boxlist_list: a list of Boxlist, each holding a collection of N anchor
        boxes. This list is produced in self.generate().

    Returns:
      A [num_anchors] integer array, where each element indicates which feature
      map index the anchor belongs to.
    """
    feature_map_indices_list = []
    for i, boxes in enumerate(boxlist_list):
      feature_map_indices_list.append(
          i * tf.ones([boxes.num_boxes()], dtype=tf.int32))
    return tf.concat(feature_map_indices_list, axis=0)

  def _assert_correct_number_of_anchors(self, anchors_list,
                                        feature_map_shape_list):
    """Assert that correct number of anchors was generated.

    Args:
      anchors_list: A list of box_list.BoxList object holding anchors generated.
      feature_map_shape_list: list of (height, width) pairs in the format
        [(height_0, width_0), (height_1, width_1), ...] that the generated
        anchors must align with.
    Returns:
      Op that raises InvalidArgumentError if the number of anchors does not
        match the number of expected anchors.
    """
    expected_num_anchors = 0
    actual_num_anchors = 0
    for num_anchors_per_location, feature_map_shape, anchors in zip(
        self.num_anchors_per_location(), feature_map_shape_list, anchors_list):
      expected_num_anchors += (num_anchors_per_location
                               * feature_map_shape[0]
                               * feature_map_shape[1])
      actual_num_anchors += anchors.num_boxes()
    return tf.assert_equal(expected_num_anchors, actual_num_anchors)
