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
"""Tests for ssd resnet v1 FPN feature extractors."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
from six.moves import zip
import tensorflow.compat.v1 as tf

from object_detection.models import ssd_feature_extractor_test
from object_detection.utils import test_utils


class SSDResnetFPNFeatureExtractorTestBase(
    ssd_feature_extractor_test.SsdFeatureExtractorTestBase):
  """Helper test class for SSD Resnet v1 FPN feature extractors."""

  @abc.abstractmethod
  def _resnet_scope_name(self):
    pass

  @abc.abstractmethod
  def _fpn_scope_name(self):
    return 'fpn'

  @abc.abstractmethod
  def _create_feature_extractor(self,
                                depth_multiplier,
                                pad_to_multiple,
                                use_explicit_padding=False,
                                min_depth=32,
                                use_keras=False):
    pass

  def test_extract_features_returns_correct_shapes_256(self):
    image_height = 256
    image_width = 256
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 32, 32, 256), (2, 16, 16, 256),
                                  (2, 8, 8, 256), (2, 4, 4, 256),
                                  (2, 2, 2, 256)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, use_keras=self.is_tf2())

  def test_extract_features_returns_correct_shapes_with_dynamic_inputs(
      self):
    image_height = 256
    image_width = 256
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 32, 32, 256), (2, 16, 16, 256),
                                  (2, 8, 8, 256), (2, 4, 4, 256),
                                  (2, 2, 2, 256)]
    self.check_extract_features_returns_correct_shapes_with_dynamic_inputs(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, use_keras=self.is_tf2())

  def test_extract_features_returns_correct_shapes_with_depth_multiplier(
      self):
    image_height = 256
    image_width = 256
    depth_multiplier = 0.5
    expected_num_channels = int(256 * depth_multiplier)
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 32, 32, expected_num_channels),
                                  (2, 16, 16, expected_num_channels),
                                  (2, 8, 8, expected_num_channels),
                                  (2, 4, 4, expected_num_channels),
                                  (2, 2, 2, expected_num_channels)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, use_keras=self.is_tf2())

  def test_extract_features_returns_correct_shapes_with_min_depth(
      self):
    image_height = 256
    image_width = 256
    depth_multiplier = 1.0
    pad_to_multiple = 1
    min_depth = 320
    expected_feature_map_shape = [(2, 32, 32, min_depth),
                                  (2, 16, 16, min_depth),
                                  (2, 8, 8, min_depth),
                                  (2, 4, 4, min_depth),
                                  (2, 2, 2, min_depth)]

    with test_utils.GraphContextOrNone() as g:
      image_tensor = tf.random.uniform([2, image_height, image_width, 3])
      feature_extractor = self._create_feature_extractor(
          depth_multiplier, pad_to_multiple, min_depth=min_depth,
          use_keras=self.is_tf2())

    def graph_fn():
      if self.is_tf2():
        return feature_extractor(image_tensor)
      return feature_extractor.extract_features(image_tensor)

    feature_maps = self.execute(graph_fn, [], graph=g)
    for feature_map, expected_shape in zip(feature_maps,
                                           expected_feature_map_shape):
      self.assertAllEqual(feature_map.shape, expected_shape)

  def test_extract_features_returns_correct_shapes_with_pad_to_multiple(
      self):
    image_height = 254
    image_width = 254
    depth_multiplier = 1.0
    pad_to_multiple = 32
    expected_feature_map_shape = [(2, 32, 32, 256), (2, 16, 16, 256),
                                  (2, 8, 8, 256), (2, 4, 4, 256),
                                  (2, 2, 2, 256)]

    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, use_keras=self.is_tf2())

  def test_extract_features_raises_error_with_invalid_image_size(
      self):
    image_height = 32
    image_width = 32
    depth_multiplier = 1.0
    pad_to_multiple = 1
    self.check_extract_features_raises_error_with_invalid_image_size(
        image_height, image_width, depth_multiplier, pad_to_multiple,
        use_keras=self.is_tf2())

  def test_preprocess_returns_correct_value_range(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1
    pad_to_multiple = 1
    test_image_np = np.random.rand(4, image_height, image_width, 3)
    with test_utils.GraphContextOrNone() as g:
      test_image = tf.constant(test_image_np)
      feature_extractor = self._create_feature_extractor(
          depth_multiplier, pad_to_multiple, use_keras=self.is_tf2())

    def graph_fn():
      preprocessed_image = feature_extractor.preprocess(test_image)
      return preprocessed_image

    preprocessed_image_out = self.execute(graph_fn, [], graph=g)
    self.assertAllClose(preprocessed_image_out,
                        test_image_np - [[123.68, 116.779, 103.939]])

  def test_variables_only_created_in_scope(self):
    if self.is_tf2():
      self.skipTest('test_variables_only_created_in_scope is only tf1')
    depth_multiplier = 1
    pad_to_multiple = 1
    scope_name = self._resnet_scope_name()
    self.check_feature_extractor_variables_under_scope(
        depth_multiplier,
        pad_to_multiple,
        scope_name,
        use_keras=self.is_tf2())

  def test_variable_count(self):
    if self.is_tf2():
      self.skipTest('test_variable_count is only tf1')
    depth_multiplier = 1
    pad_to_multiple = 1
    variables = self.get_feature_extractor_variables(
        depth_multiplier,
        pad_to_multiple,
        use_keras=self.is_tf2())
    # The number of expected variables in resnet_v1_50, resnet_v1_101,
    # and resnet_v1_152 is 279, 534, and 789 respectively.
    expected_variables_len = 279
    scope_name = self._resnet_scope_name()
    if scope_name in ('ResNet101V1_FPN', 'resnet_v1_101'):
      expected_variables_len = 534
    elif scope_name in ('ResNet152V1_FPN', 'resnet_v1_152'):
      expected_variables_len = 789
    self.assertEqual(len(variables), expected_variables_len)
