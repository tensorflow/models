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

"""Tests for object_detection.models.ssd_inception_v2_feature_extractor."""
import unittest
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.models import ssd_feature_extractor_test
from object_detection.models import ssd_inception_v2_feature_extractor
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class SsdInceptionV2FeatureExtractorTest(
    ssd_feature_extractor_test.SsdFeatureExtractorTestBase):

  def _create_feature_extractor(self,
                                depth_multiplier,
                                pad_to_multiple,
                                use_explicit_padding=False,
                                num_layers=6,
                                is_training=True):
    """Constructs a SsdInceptionV2FeatureExtractor.

    Args:
      depth_multiplier: float depth multiplier for feature extractor
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
        inputs so that the output dimensions are the same as if 'SAME' padding
        were used.
      num_layers: number of SSD layers.
      is_training: whether the network is in training mode.

    Returns:
      an ssd_inception_v2_feature_extractor.SsdInceptionV2FeatureExtractor.
    """
    min_depth = 32
    return ssd_inception_v2_feature_extractor.SSDInceptionV2FeatureExtractor(
        is_training,
        depth_multiplier,
        min_depth,
        pad_to_multiple,
        self.conv_hyperparams_fn,
        num_layers=num_layers,
        override_base_feature_extractor_hyperparams=True)

  def test_extract_features_returns_correct_shapes_128(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 8, 8, 576), (2, 4, 4, 1024),
                                  (2, 2, 2, 512), (2, 1, 1, 256),
                                  (2, 1, 1, 256), (2, 1, 1, 128)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape)

  def test_extract_features_returns_correct_shapes_with_dynamic_inputs(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 8, 8, 576), (2, 4, 4, 1024),
                                  (2, 2, 2, 512), (2, 1, 1, 256),
                                  (2, 1, 1, 256), (2, 1, 1, 128)]
    self.check_extract_features_returns_correct_shapes_with_dynamic_inputs(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape)

  def test_extract_features_returns_correct_shapes_299(self):
    image_height = 299
    image_width = 299
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 19, 19, 576), (2, 10, 10, 1024),
                                  (2, 5, 5, 512), (2, 3, 3, 256),
                                  (2, 2, 2, 256), (2, 1, 1, 128)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape)

  def test_extract_features_returns_correct_shapes_enforcing_min_depth(self):
    image_height = 299
    image_width = 299
    depth_multiplier = 0.5**12
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 19, 19, 128), (2, 10, 10, 128),
                                  (2, 5, 5, 32), (2, 3, 3, 32),
                                  (2, 2, 2, 32), (2, 1, 1, 32)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape)

  def test_extract_features_returns_correct_shapes_with_pad_to_multiple(self):
    image_height = 299
    image_width = 299
    depth_multiplier = 1.0
    pad_to_multiple = 32
    expected_feature_map_shape = [(2, 20, 20, 576), (2, 10, 10, 1024),
                                  (2, 5, 5, 512), (2, 3, 3, 256),
                                  (2, 2, 2, 256), (2, 1, 1, 128)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape)

  def test_extract_features_raises_error_with_invalid_image_size(self):
    image_height = 32
    image_width = 32
    depth_multiplier = 1.0
    pad_to_multiple = 1
    self.check_extract_features_raises_error_with_invalid_image_size(
        image_height, image_width, depth_multiplier, pad_to_multiple)

  def test_preprocess_returns_correct_value_range(self):
    image_height = 128
    image_width = 128
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
    scope_name = 'InceptionV2'
    self.check_feature_extractor_variables_under_scope(
        depth_multiplier, pad_to_multiple, scope_name)

  def test_extract_features_with_fewer_layers(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 8, 8, 576), (2, 4, 4, 1024),
                                  (2, 2, 2, 512), (2, 1, 1, 256)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, num_layers=4)


if __name__ == '__main__':
  tf.test.main()
