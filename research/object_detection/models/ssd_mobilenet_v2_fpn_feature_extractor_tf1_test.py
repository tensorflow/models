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

"""Tests for ssd_mobilenet_v2_fpn_feature_extractor.

By using parameterized test decorator, this test serves for both Slim-based and
Keras-based Mobilenet V2 FPN feature extractors in SSD.
"""
import unittest
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.models import ssd_feature_extractor_test
from object_detection.models import ssd_mobilenet_v2_fpn_feature_extractor
from object_detection.utils import tf_version


@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
@parameterized.parameters(
    {
        'use_depthwise': False
    },
    {
        'use_depthwise': True
    },
)
class SsdMobilenetV2FpnFeatureExtractorTest(
    ssd_feature_extractor_test.SsdFeatureExtractorTestBase):

  def _create_feature_extractor(self,
                                depth_multiplier,
                                pad_to_multiple,
                                is_training=True,
                                use_explicit_padding=False,
                                use_keras=False,
                                use_depthwise=False):
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
      use_depthwise: Whether to use depthwise convolutions.
    Returns:
      an ssd_meta_arch.SSDFeatureExtractor object.
    """
    del use_keras
    min_depth = 32
    return (ssd_mobilenet_v2_fpn_feature_extractor
            .SSDMobileNetV2FpnFeatureExtractor(
                is_training,
                depth_multiplier,
                min_depth,
                pad_to_multiple,
                self.conv_hyperparams_fn,
                use_depthwise=use_depthwise,
                use_explicit_padding=use_explicit_padding))

  def test_extract_features_returns_correct_shapes_256(self, use_depthwise):
    use_keras = False
    image_height = 256
    image_width = 256
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 32, 32, 256), (2, 16, 16, 256),
                                  (2, 8, 8, 256), (2, 4, 4, 256),
                                  (2, 2, 2, 256)]
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=False,
        use_keras=use_keras,
        use_depthwise=use_depthwise)
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=True,
        use_keras=use_keras,
        use_depthwise=use_depthwise)

  def test_extract_features_returns_correct_shapes_384(self, use_depthwise):
    use_keras = False
    image_height = 320
    image_width = 320
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 40, 40, 256), (2, 20, 20, 256),
                                  (2, 10, 10, 256), (2, 5, 5, 256),
                                  (2, 3, 3, 256)]
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=False,
        use_keras=use_keras,
        use_depthwise=use_depthwise)
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=True,
        use_keras=use_keras,
        use_depthwise=use_depthwise)

  def test_extract_features_returns_correct_shapes_4_channels(self,
                                                              use_depthwise):
    use_keras = False
    image_height = 320
    image_width = 320
    num_channels = 4
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 40, 40, 256), (2, 20, 20, 256),
                                  (2, 10, 10, 256), (2, 5, 5, 256),
                                  (2, 3, 3, 256)]
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=False,
        use_keras=use_keras,
        use_depthwise=use_depthwise,
        num_channels=num_channels)
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=True,
        use_keras=use_keras,
        use_depthwise=use_depthwise,
        num_channels=num_channels)

  def test_extract_features_with_dynamic_image_shape(self,
                                                     use_depthwise):
    use_keras = False
    image_height = 256
    image_width = 256
    depth_multiplier = 1.0
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 32, 32, 256), (2, 16, 16, 256),
                                  (2, 8, 8, 256), (2, 4, 4, 256),
                                  (2, 2, 2, 256)]
    self.check_extract_features_returns_correct_shapes_with_dynamic_inputs(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=False,
        use_keras=use_keras,
        use_depthwise=use_depthwise)
    self.check_extract_features_returns_correct_shapes_with_dynamic_inputs(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=True,
        use_keras=use_keras,
        use_depthwise=use_depthwise)

  def test_extract_features_returns_correct_shapes_with_pad_to_multiple(
      self, use_depthwise):
    use_keras = False
    image_height = 299
    image_width = 299
    depth_multiplier = 1.0
    pad_to_multiple = 32
    expected_feature_map_shape = [(2, 40, 40, 256), (2, 20, 20, 256),
                                  (2, 10, 10, 256), (2, 5, 5, 256),
                                  (2, 3, 3, 256)]
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=False,
        use_keras=use_keras,
        use_depthwise=use_depthwise)
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=True,
        use_keras=use_keras,
        use_depthwise=use_depthwise)

  def test_extract_features_returns_correct_shapes_enforcing_min_depth(
      self, use_depthwise):
    use_keras = False
    image_height = 256
    image_width = 256
    depth_multiplier = 0.5**12
    pad_to_multiple = 1
    expected_feature_map_shape = [(2, 32, 32, 32), (2, 16, 16, 32),
                                  (2, 8, 8, 32), (2, 4, 4, 32),
                                  (2, 2, 2, 32)]
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=False,
        use_keras=use_keras,
        use_depthwise=use_depthwise)
    self.check_extract_features_returns_correct_shape(
        2,
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        expected_feature_map_shape,
        use_explicit_padding=True,
        use_keras=use_keras,
        use_depthwise=use_depthwise)

  def test_extract_features_raises_error_with_invalid_image_size(
      self, use_depthwise):
    use_keras = False
    image_height = 32
    image_width = 32
    depth_multiplier = 1.0
    pad_to_multiple = 1
    self.check_extract_features_raises_error_with_invalid_image_size(
        image_height,
        image_width,
        depth_multiplier,
        pad_to_multiple,
        use_keras=use_keras,
        use_depthwise=use_depthwise)

  def test_preprocess_returns_correct_value_range(self,
                                                  use_depthwise):
    use_keras = False
    image_height = 256
    image_width = 256
    depth_multiplier = 1
    pad_to_multiple = 1
    test_image = np.random.rand(2, image_height, image_width, 3)
    feature_extractor = self._create_feature_extractor(
        depth_multiplier,
        pad_to_multiple,
        use_keras=use_keras,
        use_depthwise=use_depthwise)
    preprocessed_image = feature_extractor.preprocess(test_image)
    self.assertTrue(np.all(np.less_equal(np.abs(preprocessed_image), 1.0)))

  def test_variables_only_created_in_scope(self, use_depthwise):
    use_keras = False
    depth_multiplier = 1
    pad_to_multiple = 1
    scope_name = 'MobilenetV2'
    self.check_feature_extractor_variables_under_scope(
        depth_multiplier,
        pad_to_multiple,
        scope_name,
        use_keras=use_keras,
        use_depthwise=use_depthwise)

  def test_fused_batchnorm(self, use_depthwise):
    use_keras = False
    image_height = 256
    image_width = 256
    depth_multiplier = 1
    pad_to_multiple = 1
    image_placeholder = tf.placeholder(tf.float32,
                                       [1, image_height, image_width, 3])
    feature_extractor = self._create_feature_extractor(
        depth_multiplier,
        pad_to_multiple,
        use_keras=use_keras,
        use_depthwise=use_depthwise)
    preprocessed_image = feature_extractor.preprocess(image_placeholder)
    _ = feature_extractor.extract_features(preprocessed_image)
    self.assertTrue(
        any('FusedBatchNorm' in op.type
            for op in tf.get_default_graph().get_operations()))

  def test_variable_count(self, use_depthwise):
    use_keras = False
    depth_multiplier = 1
    pad_to_multiple = 1
    variables = self.get_feature_extractor_variables(
        depth_multiplier,
        pad_to_multiple,
        use_keras=use_keras,
        use_depthwise=use_depthwise)
    expected_variables_len = 274
    if use_depthwise:
      expected_variables_len = 278
    self.assertEqual(len(variables), expected_variables_len)

  def test_get_expected_feature_map_variable_names(self,
                                                   use_depthwise):
    use_keras = False
    depth_multiplier = 1.0
    pad_to_multiple = 1

    slim_expected_feature_maps_variables = set([
        # Slim Mobilenet V2 feature maps
        'MobilenetV2/expanded_conv_4/depthwise/depthwise_weights',
        'MobilenetV2/expanded_conv_7/depthwise/depthwise_weights',
        'MobilenetV2/expanded_conv_14/depthwise/depthwise_weights',
        'MobilenetV2/Conv_1/weights',
        # FPN layers
        'MobilenetV2/fpn/bottom_up_Conv2d_20/weights',
        'MobilenetV2/fpn/bottom_up_Conv2d_21/weights',
        'MobilenetV2/fpn/smoothing_1/weights',
        'MobilenetV2/fpn/smoothing_2/weights',
        'MobilenetV2/fpn/projection_1/weights',
        'MobilenetV2/fpn/projection_2/weights',
        'MobilenetV2/fpn/projection_3/weights',
    ])
    slim_expected_feature_maps_variables_with_depthwise = set([
        # Slim Mobilenet V2 feature maps
        'MobilenetV2/expanded_conv_4/depthwise/depthwise_weights',
        'MobilenetV2/expanded_conv_7/depthwise/depthwise_weights',
        'MobilenetV2/expanded_conv_14/depthwise/depthwise_weights',
        'MobilenetV2/Conv_1/weights',
        # FPN layers
        'MobilenetV2/fpn/bottom_up_Conv2d_20/pointwise_weights',
        'MobilenetV2/fpn/bottom_up_Conv2d_20/depthwise_weights',
        'MobilenetV2/fpn/bottom_up_Conv2d_21/pointwise_weights',
        'MobilenetV2/fpn/bottom_up_Conv2d_21/depthwise_weights',
        'MobilenetV2/fpn/smoothing_1/depthwise_weights',
        'MobilenetV2/fpn/smoothing_1/pointwise_weights',
        'MobilenetV2/fpn/smoothing_2/depthwise_weights',
        'MobilenetV2/fpn/smoothing_2/pointwise_weights',
        'MobilenetV2/fpn/projection_1/weights',
        'MobilenetV2/fpn/projection_2/weights',
        'MobilenetV2/fpn/projection_3/weights',
    ])

    g = tf.Graph()
    with g.as_default():
      preprocessed_inputs = tf.placeholder(tf.float32, (4, None, None, 3))
      feature_extractor = self._create_feature_extractor(
          depth_multiplier,
          pad_to_multiple,
          use_keras=use_keras,
          use_depthwise=use_depthwise)

      _ = feature_extractor.extract_features(preprocessed_inputs)
      expected_feature_maps_variables = slim_expected_feature_maps_variables
      if use_depthwise:
        expected_feature_maps_variables = (
            slim_expected_feature_maps_variables_with_depthwise)
      actual_variable_set = set([
          var.op.name for var in g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      ])
      variable_intersection = expected_feature_maps_variables.intersection(
          actual_variable_set)
      self.assertSetEqual(expected_feature_maps_variables,
                          variable_intersection)


if __name__ == '__main__':
  tf.test.main()
