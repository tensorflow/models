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
"""Tests for SSD Mobilenet V1 feature extractors.

By using parameterized test decorator, this test serves for both Slim-based and
Keras-based Mobilenet V1 feature extractors in SSD.
"""
import unittest
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.models import ssd_feature_extractor_test
from object_detection.models import ssd_mobilenet_v1_keras_feature_extractor
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class SsdMobilenetV1FeatureExtractorTest(
    ssd_feature_extractor_test.SsdFeatureExtractorTestBase):

  def _create_feature_extractor(self,
                                depth_multiplier,
                                pad_to_multiple,
                                use_explicit_padding=False,
                                num_layers=6,
                                is_training=False,
                                use_keras=False):
    """Constructs a new feature extractor.

    Args:
      depth_multiplier: float depth multiplier for feature extractor
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
        inputs so that the output dimensions are the same as if 'SAME' padding
        were used.
      num_layers: number of SSD layers.
      is_training: whether the network is in training mode.
      use_keras: if True builds a keras-based feature extractor, if False builds
        a slim-based one.

    Returns:
      an ssd_meta_arch.SSDFeatureExtractor object.
    """
    del use_keras
    min_depth = 32
    return (ssd_mobilenet_v1_keras_feature_extractor
            .SSDMobileNetV1KerasFeatureExtractor(
                is_training=is_training,
                depth_multiplier=depth_multiplier,
                min_depth=min_depth,
                pad_to_multiple=pad_to_multiple,
                conv_hyperparams=self._build_conv_hyperparams(
                    add_batch_norm=False),
                freeze_batchnorm=False,
                inplace_batchnorm_update=False,
                use_explicit_padding=use_explicit_padding,
                num_layers=num_layers,
                name='MobilenetV1'))

  def test_extract_features_returns_correct_shapes_128(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 8, 8, 512), (2, 4, 4, 1024),
                                  (2, 2, 2, 512), (2, 1, 1, 256),
                                  (2, 1, 1, 256), (2, 1, 1, 128)]
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=False,
        use_keras=True)
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=True,
        use_keras=True)

  def test_extract_features_returns_correct_shapes_299(self):
    image_height = 299
    image_width = 299
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 19, 19, 512), (2, 10, 10, 1024),
                                  (2, 5, 5, 512), (2, 3, 3, 256),
                                  (2, 2, 2, 256), (2, 1, 1, 128)]
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=False,
        use_keras=True)
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=True,
        use_keras=True)

  def test_extract_features_with_dynamic_image_shape(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 8, 8, 512), (2, 4, 4, 1024),
                                  (2, 2, 2, 512), (2, 1, 1, 256),
                                  (2, 1, 1, 256), (2, 1, 1, 128)]
    self.check_extract_features_returns_correct_shapes_with_dynamic_inputs(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=False,
        use_keras=True)
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=True,
        use_keras=True)

  def test_extract_features_returns_correct_shapes_enforcing_min_depth(
      self):
    image_height = 299
    image_width = 299
    depth_multiplier = 0.5**12
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 19, 19, 32), (2, 10, 10, 32),
                                  (2, 5, 5, 32), (2, 3, 3, 32), (2, 2, 2, 32),
                                  (2, 1, 1, 32)]
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=False,
        use_keras=True)
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=True,
        use_keras=True)

  def test_extract_features_returns_correct_shapes_with_pad_to_multiple(
      self):
    image_height = 299
    image_width = 299
    depth_multiplier = 1.0
    pad_to_multiple = 32
    expected_feature_map_shape = [(2, 20, 20, 512), (2, 10, 10, 1024),
                                  (2, 5, 5, 512), (2, 3, 3, 256),
                                  (2, 2, 2, 256), (2, 1, 1, 128)]
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=False,
        use_keras=True)
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=True,
        use_keras=True)

  def test_extract_features_raises_error_with_invalid_image_size(
      self):
    image_height = 32
    image_width = 32
    depth_multiplier = 1.0
    pad_to_multiple = 1
    self.check_extract_features_raises_error_with_invalid_image_size(
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        use_keras=True)

  def test_preprocess_returns_correct_value_range(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1
    pad_to_multiple = 1
    test_image = np.random.rand(2, image_height, image_width, 3)
    feature_extractor = self._create_feature_extractor(
        depth_multiplier, pad_to_multiple, use_keras=True)
    preprocessed_image = feature_extractor.preprocess(test_image)
    self.assertTrue(np.all(np.less_equal(np.abs(preprocessed_image), 1.0)))

  def test_extract_features_with_fewer_layers(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 8, 8, 512), (2, 4, 4, 1024),
                                  (2, 2, 2, 512), (2, 1, 1, 256)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, use_explicit_padding=False, num_layers=4,
        use_keras=True)


if __name__ == '__main__':
  tf.test.main()
