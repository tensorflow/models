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

"""Tests for embedded_ssd_mobilenet_v1_feature_extractor."""
import unittest
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.models import embedded_ssd_mobilenet_v1_feature_extractor
from object_detection.models import ssd_feature_extractor_test
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class EmbeddedSSDMobileNetV1FeatureExtractorTest(
    ssd_feature_extractor_test.SsdFeatureExtractorTestBase):

  def _create_feature_extractor(self, depth_multiplier, pad_to_multiple,
                                is_training=True):
    """Constructs a new feature extractor.

    Args:
      depth_multiplier: float depth multiplier for feature extractor
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      is_training: whether the network is in training mode.

    Returns:
      an ssd_meta_arch.SSDFeatureExtractor object.
    """
    min_depth = 32
    return (embedded_ssd_mobilenet_v1_feature_extractor.
            EmbeddedSSDMobileNetV1FeatureExtractor(
                is_training, depth_multiplier, min_depth, pad_to_multiple,
                self.conv_hyperparams_fn,
                override_base_feature_extractor_hyperparams=True))

  def test_extract_features_returns_correct_shapes_256(self):
    image_height = 256
    image_width = 256
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 16, 16, 512), (2, 8, 8, 1024),
                                  (2, 4, 4, 512), (2, 2, 2, 256),
                                  (2, 1, 1, 256)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape)

  def test_extract_features_returns_correct_shapes_with_dynamic_inputs(self):
    image_height = 256
    image_width = 256
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 16, 16, 512), (2, 8, 8, 1024),
                                  (2, 4, 4, 512), (2, 2, 2, 256),
                                  (2, 1, 1, 256)]
    self.check_extract_features_returns_correct_shapes_with_dynamic_inputs(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape)

  def test_extract_features_returns_correct_shapes_enforcing_min_depth(self):
    image_height = 256
    image_width = 256
    depth_multiplier = 0.5**12
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 16, 16, 32), (2, 8, 8, 32), (2, 4, 4, 32),
                                  (2, 2, 2, 32), (2, 1, 1, 32)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape)

  def test_extract_features_returns_correct_shapes_with_pad_to_multiple_of_1(
      self):
    image_height = 256
    image_width = 256
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 16, 16, 512), (2, 8, 8, 1024),
                                  (2, 4, 4, 512), (2, 2, 2, 256),
                                  (2, 1, 1, 256)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape)

  def test_extract_features_raises_error_with_pad_to_multiple_not_1(self):
    depth_multiplier = 1.0
    pad_to_multiple = 2
    with self.assertRaises(ValueError):
      _ = self._create_feature_extractor(depth_multiplier, pad_to_multiple)

  def test_extract_features_raises_error_with_invalid_image_size(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1.0
    pad_to_multiple = 1
    self.check_extract_features_raises_error_with_invalid_image_size(
        image_height, image_width, depth_multiplier, pad_to_multiple)

  def test_preprocess_returns_correct_value_range(self):
    image_height = 256
    image_width = 256
    depth_multiplier = 1
    pad_to_multiple = 1
    test_image = np.random.rand(4, image_height, image_width, 3)
    feature_extractor = self._create_feature_extractor(depth_multiplier,
                                                       pad_to_multiple)
    preprocessed_image = feature_extractor.preprocess(test_image)
    self.assertTrue(np.all(np.less_equal(np.abs(preprocessed_image), 1.0)))

  def test_variables_only_created_in_scope(self):
    depth_multiplier = 1
    pad_to_multiple = 1
    scope_name = 'MobilenetV1'
    self.check_feature_extractor_variables_under_scope(
        depth_multiplier, pad_to_multiple, scope_name)


if __name__ == '__main__':
  tf.test.main()
