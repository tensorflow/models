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

"""Tests for mobilenet_v1.py.

This test mainly focuses on comparing slim MobilenetV1 and Keras MobilenetV1 for
object detection. To verify the consistency of the two models, we compare:
  1. Output shape of each layer given different inputs
  2. Number of global variables

We also visualize the model structure via Tensorboard, and compare the model
layout and the parameters of each Op to make sure the two implementations are
consistent.
"""

import itertools
import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from object_detection.builders import hyperparams_builder
from object_detection.models.keras_models import mobilenet_v1
from object_detection.models.keras_models import model_utils
from object_detection.models.keras_models import test_utils
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case

_KERAS_LAYERS_TO_CHECK = [
    'conv1_relu',
    'conv_dw_1_relu', 'conv_pw_1_relu',
    'conv_dw_2_relu', 'conv_pw_2_relu',
    'conv_dw_3_relu', 'conv_pw_3_relu',
    'conv_dw_4_relu', 'conv_pw_4_relu',
    'conv_dw_5_relu', 'conv_pw_5_relu',
    'conv_dw_6_relu', 'conv_pw_6_relu',
    'conv_dw_7_relu', 'conv_pw_7_relu',
    'conv_dw_8_relu', 'conv_pw_8_relu',
    'conv_dw_9_relu', 'conv_pw_9_relu',
    'conv_dw_10_relu', 'conv_pw_10_relu',
    'conv_dw_11_relu', 'conv_pw_11_relu',
    'conv_dw_12_relu', 'conv_pw_12_relu',
    'conv_dw_13_relu', 'conv_pw_13_relu',
]

_NUM_CHANNELS = 3
_BATCH_SIZE = 2


class MobilenetV1Test(test_case.TestCase):

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
    """Constructs Keras MobilenetV1 that extracts intermediate layer outputs."""
    if not layer_names:
      layer_names = _KERAS_LAYERS_TO_CHECK
    full_model = mobilenet_v1.mobilenet_v1(
        batchnorm_training=batchnorm_training,
        conv_hyperparams=conv_hyperparams,
        weights=None,
        use_explicit_padding=use_explicit_padding,
        alpha=alpha,
        min_depth=min_depth,
        conv_defs=conv_defs,
        include_top=False)
    layer_outputs = [full_model.get_layer(name=layer).output
                     for layer in layer_names]
    return tf.keras.Model(
        inputs=full_model.inputs,
        outputs=layer_outputs)

  def _check_returns_correct_shape(
      self, image_height, image_width, depth_multiplier,
      expected_feature_map_shape, use_explicit_padding=False, min_depth=8,
      layer_names=None, conv_defs=None):
    def graph_fn(image_tensor):
      model = self._create_application_with_layer_outputs(
          layer_names=layer_names,
          batchnorm_training=False,
          use_explicit_padding=use_explicit_padding,
          min_depth=min_depth,
          alpha=depth_multiplier,
          conv_defs=conv_defs)
      return model(image_tensor)

    image_tensor = np.random.rand(_BATCH_SIZE, image_height, image_width,
                                  _NUM_CHANNELS).astype(np.float32)
    feature_maps = self.execute(graph_fn, [image_tensor])

    for feature_map, expected_shape in itertools.izip(
        feature_maps, expected_feature_map_shape):
      self.assertAllEqual(feature_map.shape, expected_shape)

  def _check_returns_correct_shapes_with_dynamic_inputs(
      self, image_height, image_width, depth_multiplier,
      expected_feature_map_shape, use_explicit_padding=False, min_depth=8,
      layer_names=None):
    def graph_fn(image_height, image_width):
      image_tensor = tf.random_uniform([_BATCH_SIZE, image_height, image_width,
                                        _NUM_CHANNELS], dtype=tf.float32)
      model = self._create_application_with_layer_outputs(
          layer_names=layer_names,
          batchnorm_training=False,
          use_explicit_padding=use_explicit_padding,
          alpha=depth_multiplier)
      return model(image_tensor)

    feature_maps = self.execute_cpu(graph_fn, [
        np.array(image_height, dtype=np.int32),
        np.array(image_width, dtype=np.int32)
    ])

    for feature_map, expected_shape in itertools.izip(
        feature_maps, expected_feature_map_shape):
      self.assertAllEqual(feature_map.shape, expected_shape)

  def _get_variables(self, depth_multiplier, layer_names=None):
    g = tf.Graph()
    with g.as_default():
      preprocessed_inputs = tf.placeholder(
          tf.float32, (4, None, None, _NUM_CHANNELS))
      model = self._create_application_with_layer_outputs(
          layer_names=layer_names,
          batchnorm_training=False, use_explicit_padding=False,
          alpha=depth_multiplier)
      model(preprocessed_inputs)
      return g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

  def test_returns_correct_shapes_128(self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1.0
    expected_feature_map_shape = (
        test_utils.moblenet_v1_expected_feature_map_shape_128)
    self._check_returns_correct_shape(
        image_height, image_width, depth_multiplier, expected_feature_map_shape)

  def test_returns_correct_shapes_128_explicit_padding(
      self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1.0
    expected_feature_map_shape = (
        test_utils.moblenet_v1_expected_feature_map_shape_128_explicit_padding)
    self._check_returns_correct_shape(
        image_height, image_width, depth_multiplier, expected_feature_map_shape,
        use_explicit_padding=True)

  def test_returns_correct_shapes_with_dynamic_inputs(
      self):
    image_height = 128
    image_width = 128
    depth_multiplier = 1.0
    expected_feature_map_shape = (
        test_utils.mobilenet_v1_expected_feature_map_shape_with_dynamic_inputs)
    self._check_returns_correct_shapes_with_dynamic_inputs(
        image_height, image_width, depth_multiplier, expected_feature_map_shape)

  def test_returns_correct_shapes_299(self):
    image_height = 299
    image_width = 299
    depth_multiplier = 1.0
    expected_feature_map_shape = (
        test_utils.moblenet_v1_expected_feature_map_shape_299)
    self._check_returns_correct_shape(
        image_height, image_width, depth_multiplier, expected_feature_map_shape)

  def test_returns_correct_shapes_enforcing_min_depth(
      self):
    image_height = 299
    image_width = 299
    depth_multiplier = 0.5**12
    expected_feature_map_shape = (
        test_utils.moblenet_v1_expected_feature_map_shape_enforcing_min_depth)
    self._check_returns_correct_shape(
        image_height, image_width, depth_multiplier, expected_feature_map_shape)

  def test_returns_correct_shapes_with_conv_defs(
      self):
    image_height = 299
    image_width = 299
    depth_multiplier = 1.0
    conv_def_block_12 = model_utils.ConvDefs(
        conv_name='conv_pw_12', filters=512)
    conv_def_block_13 = model_utils.ConvDefs(
        conv_name='conv_pw_13', filters=256)
    conv_defs = [conv_def_block_12, conv_def_block_13]

    expected_feature_map_shape = (
        test_utils.moblenet_v1_expected_feature_map_shape_with_conv_defs)
    self._check_returns_correct_shape(
        image_height, image_width, depth_multiplier, expected_feature_map_shape,
        conv_defs=conv_defs)

  def test_hyperparam_override(self):
    hyperparams = self._build_conv_hyperparams()
    model = mobilenet_v1.mobilenet_v1(
        batchnorm_training=True,
        conv_hyperparams=hyperparams,
        weights=None,
        use_explicit_padding=False,
        alpha=1.0,
        min_depth=32,
        include_top=False)
    hyperparams.params()
    bn_layer = model.get_layer(name='conv_pw_5_bn')
    self.assertAllClose(bn_layer.momentum, 0.2)
    self.assertAllClose(bn_layer.epsilon, 0.1)

  def test_variable_count(self):
    depth_multiplier = 1
    variables = self._get_variables(depth_multiplier)
    # 135 is the number of variables from slim MobilenetV1 model.
    self.assertEqual(len(variables), 135)


if __name__ == '__main__':
  tf.test.main()
