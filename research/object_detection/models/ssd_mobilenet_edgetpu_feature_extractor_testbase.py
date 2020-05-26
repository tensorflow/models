# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Base test class for ssd_mobilenet_edgetpu_feature_extractor."""

import abc

import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.models import ssd_feature_extractor_test


class _SsdMobilenetEdgeTPUFeatureExtractorTestBase(
    ssd_feature_extractor_test.SsdFeatureExtractorTestBase):
  """Base class for MobilenetEdgeTPU tests."""

  @abc.abstractmethod
  def _get_input_sizes(self):
    """Return feature map sizes for the two inputs to SSD head."""
    pass

  def test_extract_features_returns_correct_shapes_128(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1.0
    pad_to_multiple = 1
    input_feature_sizes = self._get_input_sizes()
    expected_feature_map_shape = [(2, 8, 8, input_feature_sizes[0]),
                                  (2, 4, 4, input_feature_sizes[1]),
                                  (2, 2, 2, 512), (2, 1, 1, 256), (2, 1, 1,
                                                                   256),
                                  (2, 1, 1, 128)]
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_keras=False)

  def test_extract_features_returns_correct_shapes_299(self):
    image_height = 299
    image_width = 299
    depth_multiplier = 1.0
    pad_to_multiple = 1
    input_feature_sizes = self._get_input_sizes()
    expected_feature_map_shape = [(2, 19, 19, input_feature_sizes[0]),
                                  (2, 10, 10, input_feature_sizes[1]),
                                  (2, 5, 5, 512), (2, 3, 3, 256), (2, 2, 2,
                                                                   256),
                                  (2, 1, 1, 128)]
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_keras=False)

  def test_extract_features_returns_correct_shapes_with_pad_to_multiple(self):
    image_height = 299
    image_width = 299
    depth_multiplier = 1.0
    pad_to_multiple = 32
    input_feature_sizes = self._get_input_sizes()
    expected_feature_map_shape = [(2, 20, 20, input_feature_sizes[0]),
                                  (2, 10, 10, input_feature_sizes[1]),
                                  (2, 5, 5, 512), (2, 3, 3, 256), (2, 2, 2,
                                                                   256),
                                  (2, 1, 1, 128)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape)

  def test_preprocess_returns_correct_value_range(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1
    pad_to_multiple = 1
    test_image = np.random.rand(4, image_height, image_width, 3)
    feature_extractor = self._create_feature_extractor(
        depth_multiplier, pad_to_multiple, use_keras=False)
    preprocessed_image = feature_extractor.preprocess(test_image)
    self.assertTrue(np.all(np.less_equal(np.abs(preprocessed_image), 1.0)))

  def test_has_fused_batchnorm(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1
    pad_to_multiple = 1
    image_placeholder = tf.placeholder(tf.float32,
                                       [1, image_height, image_width, 3])
    feature_extractor = self._create_feature_extractor(
        depth_multiplier, pad_to_multiple, use_keras=False)
    preprocessed_image = feature_extractor.preprocess(image_placeholder)
    _ = feature_extractor.extract_features(preprocessed_image)
    self.assertTrue(any('FusedBatchNorm' in op.type
                        for op in tf.get_default_graph().get_operations()))
