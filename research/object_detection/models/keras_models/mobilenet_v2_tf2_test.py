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

"""Tests for mobilenet_v2."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unittest
import numpy as np
from six.moves import zip
import tensorflow.compat.v1 as tf

from google.protobuf import text_format

from object_detection.builders import hyperparams_builder
from object_detection.models.keras_models import mobilenet_v2
from object_detection.models.keras_models import model_utils
from object_detection.models.keras_models import test_utils
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case
from object_detection.utils import tf_version

_layers_to_check = [
    'Conv1_relu',
    'block_1_expand_relu', 'block_1_depthwise_relu', 'block_1_project_BN',
    'block_2_expand_relu', 'block_2_depthwise_relu', 'block_2_project_BN',
    'block_3_expand_relu', 'block_3_depthwise_relu', 'block_3_project_BN',
    'block_4_expand_relu', 'block_4_depthwise_relu', 'block_4_project_BN',
    'block_5_expand_relu', 'block_5_depthwise_relu', 'block_5_project_BN',
    'block_6_expand_relu', 'block_6_depthwise_relu', 'block_6_project_BN',
    'block_7_expand_relu', 'block_7_depthwise_relu', 'block_7_project_BN',
    'block_8_expand_relu', 'block_8_depthwise_relu', 'block_8_project_BN',
    'block_9_expand_relu', 'block_9_depthwise_relu', 'block_9_project_BN',
    'block_10_expand_relu', 'block_10_depthwise_relu', 'block_10_project_BN',
    'block_11_expand_relu', 'block_11_depthwise_relu', 'block_11_project_BN',
    'block_12_expand_relu', 'block_12_depthwise_relu', 'block_12_project_BN',
    'block_13_expand_relu', 'block_13_depthwise_relu', 'block_13_project_BN',
    'block_14_expand_relu', 'block_14_depthwise_relu', 'block_14_project_BN',
    'block_15_expand_relu', 'block_15_depthwise_relu', 'block_15_project_BN',
    'block_16_expand_relu', 'block_16_depthwise_relu', 'block_16_project_BN',
    'out_relu']


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class MobilenetV2Test(test_case.TestCase):

  def _build_conv_hyperparams(self):
    conv_hyperparams = hyperparams_pb2.Hyperparams()
    conv_hyperparams_text_proto = """
      activation: RELU_6
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
      batch_norm {
        train: true,
        scale: false,
        center: true,
        decay: 0.2,
        epsilon: 0.1,
      }
    """
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)
    return hyperparams_builder.KerasLayerHyperparams(conv_hyperparams)

  def _create_application_with_layer_outputs(
      self, layer_names, batchnorm_training,
      conv_hyperparams=None,
      use_explicit_padding=False,
      alpha=1.0,
      min_depth=None,
      conv_defs=None):
    """Constructs Keras mobilenetv2 that extracts intermediate layer outputs."""
    # Have to clear the Keras backend to ensure isolation in layer naming
    tf.keras.backend.clear_session()
    if not layer_names:
      layer_names = _layers_to_check
    full_model = mobilenet_v2.mobilenet_v2(
        batchnorm_training=batchnorm_training,
        conv_hyperparams=conv_hyperparams,
        weights=None,
        use_explicit_padding=use_explicit_padding,
        alpha=alpha,
        min_depth=min_depth,
        include_top=False,
        conv_defs=conv_defs)
    layer_outputs = [full_model.get_layer(name=layer).output
                     for layer in layer_names]
    return tf.keras.Model(
        inputs=full_model.inputs,
        outputs=layer_outputs)

  def _check_returns_correct_shape(
      self, batch_size, image_height, image_width, depth_multiplier,
      expected_feature_map_shapes, use_explicit_padding=False, min_depth=None,
      layer_names=None, conv_defs=None):
    model = self._create_application_with_layer_outputs(
        layer_names=layer_names,
        batchnorm_training=False,
        use_explicit_padding=use_explicit_padding,
        min_depth=min_depth,
        alpha=depth_multiplier,
        conv_defs=conv_defs)

    image_tensor = np.random.rand(batch_size, image_height, image_width,
                                  3).astype(np.float32)
    feature_maps = model([image_tensor])

    for feature_map, expected_shape in zip(feature_maps,
                                           expected_feature_map_shapes):
      self.assertAllEqual(feature_map.shape, expected_shape)

  def _check_returns_correct_shapes_with_dynamic_inputs(
      self, batch_size, image_height, image_width, depth_multiplier,
      expected_feature_map_shapes, use_explicit_padding=False,
      layer_names=None):
    height = tf.random.uniform([], minval=image_height, maxval=image_height+1,
                               dtype=tf.int32)
    width = tf.random.uniform([], minval=image_width, maxval=image_width+1,
                              dtype=tf.int32)
    image_tensor = tf.random.uniform([batch_size, height, width,
                                      3], dtype=tf.float32)
    model = self._create_application_with_layer_outputs(
        layer_names=layer_names,
        batchnorm_training=False, use_explicit_padding=use_explicit_padding,
        alpha=depth_multiplier)
    feature_maps = model(image_tensor)
    for feature_map, expected_shape in zip(feature_maps,
                                           expected_feature_map_shapes):
      self.assertAllEqual(feature_map.shape, expected_shape)

  def _get_variables(self, depth_multiplier, layer_names=None):
    tf.keras.backend.clear_session()
    model = self._create_application_with_layer_outputs(
        layer_names=layer_names,
        batchnorm_training=False, use_explicit_padding=False,
        alpha=depth_multiplier)
    preprocessed_inputs = tf.random.uniform([2, 40, 40, 3])
    model(preprocessed_inputs)
    return model.variables

  def test_returns_correct_shapes_128(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1.0
    expected_feature_map_shape = (
        test_utils.moblenet_v2_expected_feature_map_shape_128)

    self._check_returns_correct_shape(
        2, image_height, image_width, depth_multiplier,
        expected_feature_map_shape)

  def test_returns_correct_shapes_128_explicit_padding(
      self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1.0
    expected_feature_map_shape = (
        test_utils.moblenet_v2_expected_feature_map_shape_128_explicit_padding)
    self._check_returns_correct_shape(
        2, image_height, image_width, depth_multiplier,
        expected_feature_map_shape, use_explicit_padding=True)

  def test_returns_correct_shapes_with_dynamic_inputs(
      self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1.0
    expected_feature_map_shape = (
        test_utils.mobilenet_v2_expected_feature_map_shape_with_dynamic_inputs)
    self._check_returns_correct_shapes_with_dynamic_inputs(
        2, image_height, image_width, depth_multiplier,
        expected_feature_map_shape)

  def test_returns_correct_shapes_299(self):
    image_height = 299
    image_width = 299
    depth_multiplier = 1.0
    expected_feature_map_shape = (
        test_utils.moblenet_v2_expected_feature_map_shape_299)
    self._check_returns_correct_shape(
        2, image_height, image_width, depth_multiplier,
        expected_feature_map_shape)

  def test_returns_correct_shapes_enforcing_min_depth(
      self):
    image_height = 299
    image_width = 299
    depth_multiplier = 0.5**12
    expected_feature_map_shape = (
        test_utils.moblenet_v2_expected_feature_map_shape_enforcing_min_depth)
    self._check_returns_correct_shape(
        2, image_height, image_width, depth_multiplier,
        expected_feature_map_shape, min_depth=32)

  def test_returns_correct_shapes_with_conv_defs(
      self):
    image_height = 299
    image_width = 299
    depth_multiplier = 1.0
    conv_1 = model_utils.ConvDefs(
        conv_name='Conv_1', filters=256)
    conv_defs = [conv_1]

    expected_feature_map_shape = (
        test_utils.moblenet_v2_expected_feature_map_shape_with_conv_defs)
    self._check_returns_correct_shape(
        2, image_height, image_width, depth_multiplier,
        expected_feature_map_shape, conv_defs=conv_defs)

  def test_hyperparam_override(self):
    hyperparams = self._build_conv_hyperparams()
    model = mobilenet_v2.mobilenet_v2(
        batchnorm_training=True,
        conv_hyperparams=hyperparams,
        weights=None,
        use_explicit_padding=False,
        alpha=1.0,
        min_depth=32,
        include_top=False)
    hyperparams.params()
    bn_layer = model.get_layer(name='block_5_project_BN')
    self.assertAllClose(bn_layer.momentum, 0.2)
    self.assertAllClose(bn_layer.epsilon, 0.1)

  def test_variable_count(self):
    depth_multiplier = 1
    variables = self._get_variables(depth_multiplier)
    self.assertEqual(len(variables), 260)


if __name__ == '__main__':
  tf.test.main()
