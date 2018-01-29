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
from abc import ABCMeta
from abc import abstractmethod

import tensorflow as tf


class AnchorGenerator(object):
  """Abstract base class for anchor generators."""
  __metaclass__ = ABCMeta

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

    TODO: remove **params from argument list and make stride and offsets (for
        multiple_grid_anchor_generator) constructor arguments.

    Args:
      feature_map_shape_list: list of (height, width) pairs in the format
        [(height_0, width_0), (height_1, width_1), ...] that the generated
        anchors must align with.  Pairs can be provided as 1-dimensional
        integer tensors of length 2 or simply as tuples of integers.
      **params: parameters for anchor generation op

    Returns:
      boxes: a BoxList holding a collection of N anchor boxes
    Raises:
      ValueError: if the number of feature map shapes does not match the length
        of NumAnchorsPerLocation.
    """
    if self.check_num_anchors and (
        len(feature_map_shape_list) != len(self.num_anchors_per_location())):
      raise ValueError('Number of feature maps is expected to equal the length '
                       'of `num_anchors_per_location`.')
    with tf.name_scope(self.name_scope()):
      anchors = self._generate(feature_map_shape_list, **params)
      if self.check_num_anchors:
        with tf.control_dependencies([
            self._assert_correct_number_of_anchors(
                anchors, feature_map_shape_list)]):
          anchors.set(tf.identity(anchors.get()))
      return anchors

  @abstractmethod
  def _generate(self, feature_map_shape_list, **params):
    """To be overridden by implementations.

    Args:
      feature_map_shape_list: list of (height, width) pairs in the format
        [(height_0, width_0), (height_1, width_1), ...] that the generated
        anchors must align with.
      **params: parameters for anchor generation op

    Returns:
      boxes: a BoxList holding a collection of N anchor boxes
    """
    pass

  def _assert_correct_number_of_anchors(self, anchors, feature_map_shape_list):
    """Assert that correct number of anchors was generated.

    Args:
      anchors: box_list.BoxList object holding anchors generated
      feature_map_shape_list: list of (height, width) pairs in the format
        [(height_0, width_0), (height_1, width_1), ...] that the generated
        anchors must align with.
    Returns:
      Op that raises InvalidArgumentError if the number of anchors does not
        match the number of expected anchors.
    """
    expected_num_anchors = 0
    for num_anchors_per_location, feature_map_shape in zip(
        self.num_anchors_per_location(), feature_map_shape_list):
      expected_num_anchors += (num_anchors_per_location
                               * feature_map_shape[0]
                               * feature_map_shape[1])
    return tf.assert_equal(expected_num_anchors, anchors.num_boxes())
