# Lint as: python2, python3
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
"""Tests for ssd_mobilenet_v2_nas_fpn_feature_extractor."""
import unittest
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.models import ssd_feature_extractor_test
from object_detection.models import ssd_mobilenet_v2_mnasfpn_feature_extractor as mnasfpn_feature_extractor
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class SsdMobilenetV2MnasFPNFeatureExtractorTest(
    ssd_feature_extractor_test.SsdFeatureExtractorTestBase):

  def _create_feature_extractor(self,
                                depth_multiplier,
                                pad_to_multiple,
                                use_explicit_padding=False):
    min_depth = 16
    is_training = True
    fpn_num_filters = 48
    return mnasfpn_feature_extractor.SSDMobileNetV2MnasFPNFeatureExtractor(
        is_training,
        depth_multiplier,
        min_depth,
        pad_to_multiple,
        self.conv_hyperparams_fn,
        additional_layer_depth=fpn_num_filters,
        use_explicit_padding=use_explicit_padding)

  def test_extract_features_returns_correct_shapes_320_256(self):
    image_height = 320
    image_width = 256
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 40, 32, 48), (2, 20, 16, 48),
                                  (2, 10, 8, 48), (2, 5, 4, 48)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, use_explicit_padding=False)
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, use_explicit_padding=True)

  def test_extract_features_returns_correct_shapes_enforcing_min_depth(self):
    image_height = 256
    image_width = 256
    depth_multiplier = 0.5**12
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 32, 32, 16), (2, 16, 16, 16),
                                  (2, 8, 8, 16), (2, 4, 4, 16)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, use_explicit_padding=False)
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, use_explicit_padding=True)

  def test_preprocess_returns_correct_value_range(self):
    image_height = 320
    image_width = 320
    depth_multiplier = 1
    pad_to_multiple = 1
    test_image = np.random.rand(2, image_height, image_width, 3)
    feature_extractor = self._create_feature_extractor(depth_multiplier,
                                                       pad_to_multiple)
    preprocessed_image = feature_extractor.preprocess(test_image)
    self.assertTrue(np.all(np.less_equal(np.abs(preprocessed_image), 1.0)))


if __name__ == '__main__':
  tf.test.main()
