# Lint as: python3
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
"""Tests for context_rcnn_lib."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from absl.testing import parameterized
import tensorflow.compat.v1 as tf

from object_detection.meta_architectures import context_rcnn_lib
from object_detection.utils import test_case
from object_detection.utils import tf_version

_NEGATIVE_PADDING_VALUE = -100000


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class ContextRcnnLibTest(parameterized.TestCase, test_case.TestCase,
                         tf.test.TestCase):
  """Tests for the functions in context_rcnn_lib."""

  def test_compute_valid_mask(self):
    num_elements = tf.constant(3, tf.int32)
    num_valid_elementss = tf.constant((1, 2), tf.int32)
    valid_mask = context_rcnn_lib.compute_valid_mask(num_valid_elementss,
                                                     num_elements)
    expected_valid_mask = tf.constant([[1, 0, 0], [1, 1, 0]], tf.float32)
    self.assertAllEqual(valid_mask, expected_valid_mask)

  def test_filter_weight_value(self):
    weights = tf.ones((2, 3, 2), tf.float32) * 4
    values = tf.ones((2, 2, 4), tf.float32)
    valid_mask = tf.constant([[True, True], [True, False]], tf.bool)

    filtered_weights, filtered_values = context_rcnn_lib.filter_weight_value(
        weights, values, valid_mask)
    expected_weights = tf.constant([[[4, 4], [4, 4], [4, 4]],
                                    [[4, _NEGATIVE_PADDING_VALUE],
                                     [4, _NEGATIVE_PADDING_VALUE],
                                     [4, _NEGATIVE_PADDING_VALUE]]])

    expected_values = tf.constant([[[1, 1, 1, 1], [1, 1, 1, 1]],
                                   [[1, 1, 1, 1], [0, 0, 0, 0]]])
    self.assertAllEqual(filtered_weights, expected_weights)
    self.assertAllEqual(filtered_values, expected_values)

    # Changes the valid_mask so the results will be different.
    valid_mask = tf.constant([[True, True], [False, False]], tf.bool)

    filtered_weights, filtered_values = context_rcnn_lib.filter_weight_value(
        weights, values, valid_mask)
    expected_weights = tf.constant(
        [[[4, 4], [4, 4], [4, 4]],
         [[_NEGATIVE_PADDING_VALUE, _NEGATIVE_PADDING_VALUE],
          [_NEGATIVE_PADDING_VALUE, _NEGATIVE_PADDING_VALUE],
          [_NEGATIVE_PADDING_VALUE, _NEGATIVE_PADDING_VALUE]]])

    expected_values = tf.constant([[[1, 1, 1, 1], [1, 1, 1, 1]],
                                   [[0, 0, 0, 0], [0, 0, 0, 0]]])
    self.assertAllEqual(filtered_weights, expected_weights)
    self.assertAllEqual(filtered_values, expected_values)

  @parameterized.parameters((2, True, True), (2, False, True),
                            (10, True, False), (10, False, False))
  def test_project_features(self, projection_dimension, is_training, normalize):
    features = tf.ones([2, 3, 4], tf.float32)
    projected_features = context_rcnn_lib.project_features(
        features,
        projection_dimension,
        is_training=is_training,
        normalize=normalize)

    # Makes sure the shape is correct.
    self.assertAllEqual(projected_features.shape, [2, 3, projection_dimension])

  @parameterized.parameters(
      (2, 10, 1),
      (3, 10, 2),
      (4, 20, 3),
      (5, 20, 4),
      (7, 20, 5),
  )
  def test_attention_block(self, bottleneck_dimension, output_dimension,
                           attention_temperature):
    input_features = tf.ones([2, 3, 4], tf.float32)
    context_features = tf.ones([2, 2, 3], tf.float32)
    valid_mask = tf.constant([[True, True], [False, False]], tf.bool)
    box_valid_mask = tf.constant([[True, True, True], [False, False, False]],
                                 tf.bool)
    is_training = False
    output_features = context_rcnn_lib.attention_block(
        input_features, context_features, bottleneck_dimension,
        output_dimension, attention_temperature,
        keys_values_valid_mask=valid_mask,
        queries_valid_mask=box_valid_mask,
        is_training=is_training)

    # Makes sure the shape is correct.
    self.assertAllEqual(output_features.shape, [2, 3, output_dimension])

  @parameterized.parameters(True, False)
  def test_compute_box_context_attention(self, is_training):
    box_features = tf.ones([2 * 3, 4, 4, 4], tf.float32)
    context_features = tf.ones([2, 5, 6], tf.float32)
    valid_context_size = tf.constant((2, 3), tf.int32)
    num_proposals = tf.constant((2, 3), tf.int32)
    bottleneck_dimension = 10
    attention_temperature = 1
    attention_features = context_rcnn_lib._compute_box_context_attention(
        box_features, num_proposals, context_features, valid_context_size,
        bottleneck_dimension, attention_temperature, is_training,
        max_num_proposals=3)
    # Makes sure the shape is correct.
    self.assertAllEqual(attention_features.shape, [2, 3, 1, 1, 4])

  @parameterized.parameters(True, False)
  def test_compute_box_context_attention_with_self_attention(self, is_training):
    box_features = tf.ones([2 * 3, 4, 4, 4], tf.float32)
    context_features = tf.ones([2, 5, 6], tf.float32)
    valid_context_size = tf.constant((2, 3), tf.int32)
    num_proposals = tf.constant((2, 3), tf.int32)
    bottleneck_dimension = 10
    attention_temperature = 1
    attention_features = context_rcnn_lib._compute_box_context_attention(
        box_features, num_proposals, context_features, valid_context_size,
        bottleneck_dimension, attention_temperature, is_training,
        max_num_proposals=3,
        use_self_attention=True)
    # Makes sure the shape is correct.
    self.assertAllEqual(attention_features.shape, [2, 3, 1, 1, 4])

  @parameterized.parameters(True, False)
  def test_compute_box_context_attention_with_layers_and_heads(
      self, is_training):
    box_features = tf.ones([2 * 3, 4, 4, 4], tf.float32)
    context_features = tf.ones([2, 5, 6], tf.float32)
    valid_context_size = tf.constant((2, 3), tf.int32)
    num_proposals = tf.constant((2, 3), tf.int32)
    bottleneck_dimension = 10
    attention_temperature = 1
    attention_features = context_rcnn_lib._compute_box_context_attention(
        box_features, num_proposals, context_features, valid_context_size,
        bottleneck_dimension, attention_temperature, is_training,
        max_num_proposals=3,
        num_attention_layers=3,
        num_attention_heads=3)
    # Makes sure the shape is correct.
    self.assertAllEqual(attention_features.shape, [2, 3, 1, 1, 4])

if __name__ == '__main__':
  tf.test.main()
