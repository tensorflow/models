# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for ssd_mobilenet_v1_fpn_feature_extractor.

By using parameterized test decorator, this test serves for both Slim-based and
Keras-based Mobilenet V1 FPN feature extractors in SSD.
"""
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from object_detection.models import ssd_feature_extractor_test
from object_detection.models import ssd_mobilenet_v1_fpn_feature_extractor
from object_detection.models import ssd_mobilenet_v1_fpn_keras_feature_extractor

slim = tf.contrib.slim


@parameterized.parameters(
    {'use_keras': False},
    {'use_keras': True},
)
class SsdMobilenetV1FpnFeatureExtractorTest(
    ssd_feature_extractor_test.SsdFeatureExtractorTestBase):

  def _create_feature_extractor(self, depth_multiplier, pad_to_multiple,
                                is_training=True, use_explicit_padding=False,
                                use_keras=False):
    """Constructs a new feature extractor.

    Args:
      depth_multiplier: float depth multiplier for feature extractor
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      is_training: whether the network is in training mode.
      use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
        inputs so that the output dimensions are the same as if 'SAME' padding
        were used.
      use_keras: if True builds a keras-based feature extractor, if False builds
        a slim-based one.
    Returns:
      an ssd_meta_arch.SSDFeatureExtractor object.
    """
    min_depth = 32
    if use_keras:
      return (ssd_mobilenet_v1_fpn_keras_feature_extractor.
              SSDMobileNetV1FpnKerasFeatureExtractor(
                  is_training=is_training,
                  depth_multiplier=depth_multiplier,
                  min_depth=min_depth,
                  pad_to_multiple=pad_to_multiple,
                  conv_hyperparams=self._build_conv_hyperparams(
                      add_batch_norm=False),
                  freeze_batchnorm=False,
                  inplace_batchnorm_update=False,
                  use_explicit_padding=use_explicit_padding,
                  use_depthwise=True,
                  name='MobilenetV1_FPN'))
    else:
      return (ssd_mobilenet_v1_fpn_feature_extractor.
              SSDMobileNetV1FpnFeatureExtractor(
                  is_training,
                  depth_multiplier,
                  min_depth,
                  pad_to_multiple,
                  self.conv_hyperparams_fn,
                  use_depthwise=True,
                  use_explicit_padding=use_explicit_padding))

  def test_extract_features_returns_correct_shapes_256(self, use_keras):
    image_height = 256
    image_width = 256
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 32, 32, 256), (2, 16, 16, 256),
                                  (2, 8, 8, 256), (2, 4, 4, 256),
                                  (2, 2, 2, 256)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, use_explicit_padding=False,
        use_keras=use_keras)
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, use_explicit_padding=True,
        use_keras=use_keras)

  def test_extract_features_returns_correct_shapes_384(self, use_keras):
    image_height = 320
    image_width = 320
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 40, 40, 256), (2, 20, 20, 256),
                                  (2, 10, 10, 256), (2, 5, 5, 256),
                                  (2, 3, 3, 256)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, use_explicit_padding=False,
        use_keras=use_keras)
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, use_explicit_padding=True,
        use_keras=use_keras)

  def test_extract_features_with_dynamic_image_shape(self, use_keras):
    image_height = 256
    image_width = 256
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 32, 32, 256), (2, 16, 16, 256),
                                  (2, 8, 8, 256), (2, 4, 4, 256),
                                  (2, 2, 2, 256)]
    self.check_extract_features_returns_correct_shapes_with_dynamic_inputs(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, use_explicit_padding=False,
        use_keras=use_keras)
    self.check_extract_features_returns_correct_shapes_with_dynamic_inputs(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, use_explicit_padding=True,
        use_keras=use_keras)

  def test_extract_features_returns_correct_shapes_with_pad_to_multiple(
      self, use_keras):
    image_height = 299
    image_width = 299
    depth_multiplier = 1.0
    pad_to_multiple = 32
    expected_feature_map_shape = [(2, 40, 40, 256), (2, 20, 20, 256),
                                  (2, 10, 10, 256), (2, 5, 5, 256),
                                  (2, 3, 3, 256)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, use_explicit_padding=False,
        use_keras=use_keras)
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, use_explicit_padding=True,
        use_keras=use_keras)

  def test_extract_features_returns_correct_shapes_enforcing_min_depth(
      self, use_keras):
    image_height = 256
    image_width = 256
    depth_multiplier = 0.5**12
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 32, 32, 32), (2, 16, 16, 32),
                                  (2, 8, 8, 32), (2, 4, 4, 32),
                                  (2, 2, 2, 32)]
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, use_explicit_padding=False,
        use_keras=use_keras)
    self.check_extract_features_returns_correct_shape(
        2, image_height, image_width, depth_multiplier, pad_to_multiple,
        expected_feature_map_shape, use_explicit_padding=True,
        use_keras=use_keras)

  def test_extract_features_raises_error_with_invalid_image_size(
      self, use_keras):
    image_height = 32
    image_width = 32
    depth_multiplier = 1.0
    pad_to_multiple = 1
    self.check_extract_features_raises_error_with_invalid_image_size(
        image_height, image_width, depth_multiplier, pad_to_multiple,
        use_keras=use_keras)

  def test_preprocess_returns_correct_value_range(self, use_keras):
    image_height = 256
    image_width = 256
    depth_multiplier = 1
    pad_to_multiple = 1
    test_image = np.random.rand(2, image_height, image_width, 3)
    feature_extractor = self._create_feature_extractor(depth_multiplier,
                                                       pad_to_multiple,
                                                       use_keras=use_keras)
    preprocessed_image = feature_extractor.preprocess(test_image)
    self.assertTrue(np.all(np.less_equal(np.abs(preprocessed_image), 1.0)))

  def test_variables_only_created_in_scope(self, use_keras):
    depth_multiplier = 1
    pad_to_multiple = 1
    scope_name = 'MobilenetV1'
    self.check_feature_extractor_variables_under_scope(
        depth_multiplier, pad_to_multiple, scope_name, use_keras=use_keras)

  def test_variable_count(self, use_keras):
    depth_multiplier = 1
    pad_to_multiple = 1
    variables = self.get_feature_extractor_variables(
        depth_multiplier, pad_to_multiple, use_keras=use_keras)
    self.assertEqual(len(variables), 153)

  def test_fused_batchnorm(self, use_keras):
    image_height = 256
    image_width = 256
    depth_multiplier = 1
    pad_to_multiple = 1
    image_placeholder = tf.placeholder(tf.float32,
                                       [1, image_height, image_width, 3])
    feature_extractor = self._create_feature_extractor(depth_multiplier,
                                                       pad_to_multiple,
                                                       use_keras=use_keras)
    preprocessed_image = feature_extractor.preprocess(image_placeholder)
    if use_keras:
      _ = feature_extractor(preprocessed_image)
    else:
      _ = feature_extractor.extract_features(preprocessed_image)

    self.assertTrue(
        any(op.type == 'FusedBatchNorm'
            for op in tf.get_default_graph().get_operations()))


if __name__ == '__main__':
  tf.test.main()
