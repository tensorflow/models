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
"""Tests for resnet_v1.py.

This test mainly focuses on comparing slim resnet v1 and Keras resnet v1 for
object detection. To verify the consistency of the two models, we compare:
  1. Output shape of each layer given different inputs.
  2. Number of global variables.
"""
import unittest

from absl.testing import parameterized
import numpy as np
from six.moves import zip
import tensorflow.compat.v1 as tf

from google.protobuf import text_format
from object_detection.builders import hyperparams_builder
from object_detection.models.keras_models import resnet_v1
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case
from object_detection.utils import tf_version

_EXPECTED_SHAPES_224_RESNET50 = {
    'conv2_block3_out': (4, 56, 56, 256),
    'conv3_block4_out': (4, 28, 28, 512),
    'conv4_block6_out': (4, 14, 14, 1024),
    'conv5_block3_out': (4, 7, 7, 2048),
}

_EXPECTED_SHAPES_224_RESNET101 = {
    'conv2_block3_out': (4, 56, 56, 256),
    'conv3_block4_out': (4, 28, 28, 512),
    'conv4_block23_out': (4, 14, 14, 1024),
    'conv5_block3_out': (4, 7, 7, 2048),
}

_EXPECTED_SHAPES_224_RESNET152 = {
    'conv2_block3_out': (4, 56, 56, 256),
    'conv3_block8_out': (4, 28, 28, 512),
    'conv4_block36_out': (4, 14, 14, 1024),
    'conv5_block3_out': (4, 7, 7, 2048),
}

_RESNET_NAMES = ['resnet_v1_50', 'resnet_v1_101', 'resnet_v1_152']
_RESNET_MODELS = [
    resnet_v1.resnet_v1_50, resnet_v1.resnet_v1_101, resnet_v1.resnet_v1_152
]
_RESNET_SHAPES = [
    _EXPECTED_SHAPES_224_RESNET50, _EXPECTED_SHAPES_224_RESNET101,
    _EXPECTED_SHAPES_224_RESNET152
]

_NUM_CHANNELS = 3
_BATCH_SIZE = 4


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class ResnetV1Test(test_case.TestCase):

  def _build_conv_hyperparams(self):
    conv_hyperparams = hyperparams_pb2.Hyperparams()
    conv_hyperparams_text_proto = """
      activation: RELU_6,
      regularizer {
        l2_regularizer {
          weight: 0.0004
        }
      }
      initializer {
        truncated_normal_initializer {
          stddev: 0.03
          mean: 0.0
        }
      }
      batch_norm {
        scale: true,
        decay: 0.997,
        epsilon: 0.001,
      }
    """
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)
    return hyperparams_builder.KerasLayerHyperparams(conv_hyperparams)

  def _create_application_with_layer_outputs(self,
                                             model_index,
                                             batchnorm_training,
                                             batchnorm_scale=True,
                                             weight_decay=0.0001,
                                             default_batchnorm_momentum=0.997,
                                             default_batchnorm_epsilon=1e-5):
    """Constructs Keras resnet_v1 that extracts layer outputs."""
    # Have to clear the Keras backend to ensure isolation in layer naming
    tf.keras.backend.clear_session()
    layer_names = _RESNET_SHAPES[model_index].keys()
    full_model = _RESNET_MODELS[model_index](
        batchnorm_training=batchnorm_training,
        weights=None,
        batchnorm_scale=batchnorm_scale,
        weight_decay=weight_decay,
        default_batchnorm_momentum=default_batchnorm_momentum,
        default_batchnorm_epsilon=default_batchnorm_epsilon,
        include_top=False)

    layer_outputs = [
        full_model.get_layer(name=layer).output for layer in layer_names
    ]
    return tf.keras.Model(inputs=full_model.inputs, outputs=layer_outputs)

  def _check_returns_correct_shape(self,
                                   image_height,
                                   image_width,
                                   model_index,
                                   expected_feature_map_shape,
                                   batchnorm_training=True,
                                   batchnorm_scale=True,
                                   weight_decay=0.0001,
                                   default_batchnorm_momentum=0.997,
                                   default_batchnorm_epsilon=1e-5):
    model = self._create_application_with_layer_outputs(
        model_index=model_index,
        batchnorm_training=batchnorm_training,
        batchnorm_scale=batchnorm_scale,
        weight_decay=weight_decay,
        default_batchnorm_momentum=default_batchnorm_momentum,
        default_batchnorm_epsilon=default_batchnorm_epsilon)

    image_tensor = np.random.rand(_BATCH_SIZE, image_height, image_width,
                                  _NUM_CHANNELS).astype(np.float32)
    feature_maps = model(image_tensor)
    layer_names = _RESNET_SHAPES[model_index].keys()
    for feature_map, layer_name in zip(feature_maps, layer_names):
      expected_shape = _RESNET_SHAPES[model_index][layer_name]
      self.assertAllEqual(feature_map.shape, expected_shape)

  def _get_variables(self, model_index):
    tf.keras.backend.clear_session()
    model = self._create_application_with_layer_outputs(
        model_index, batchnorm_training=False)
    preprocessed_inputs = tf.random.uniform([2, 40, 40, _NUM_CHANNELS])
    model(preprocessed_inputs)
    return model.variables

  def test_returns_correct_shapes_224(self):
    image_height = 224
    image_width = 224
    for model_index, _ in enumerate(_RESNET_NAMES):
      expected_feature_map_shape = _RESNET_SHAPES[model_index]
      self._check_returns_correct_shape(image_height, image_width, model_index,
                                        expected_feature_map_shape)

  def test_hyperparam_override(self):
    for model_name in _RESNET_MODELS:
      model = model_name(
          batchnorm_training=True,
          default_batchnorm_momentum=0.2,
          default_batchnorm_epsilon=0.1,
          weights=None,
          include_top=False)
      bn_layer = model.get_layer(name='conv1_bn')
      self.assertAllClose(bn_layer.momentum, 0.2)
      self.assertAllClose(bn_layer.epsilon, 0.1)

  def test_variable_count(self):
    # The number of variables from slim resnetv1-* model.
    variable_nums = [265, 520, 775]
    for model_index, var_num in enumerate(variable_nums):
      variables = self._get_variables(model_index)
      self.assertEqual(len(variables), var_num)


class ResnetShapeTest(test_case.TestCase, parameterized.TestCase):

  @unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
  @parameterized.parameters(
      {
          'resnet_type':
              'resnet_v1_34',
          'output_layer_names': [
              'conv2_block3_out', 'conv3_block4_out', 'conv4_block6_out',
              'conv5_block3_out'
          ]
      }, {
          'resnet_type':
              'resnet_v1_18',
          'output_layer_names': [
              'conv2_block2_out', 'conv3_block2_out', 'conv4_block2_out',
              'conv5_block2_out'
          ]
      })
  def test_output_shapes(self, resnet_type, output_layer_names):
    if resnet_type == 'resnet_v1_34':
      model = resnet_v1.resnet_v1_34(input_shape=(64, 64, 3), weights=None)
    else:
      model = resnet_v1.resnet_v1_18(input_shape=(64, 64, 3), weights=None)
    outputs = [
        model.get_layer(output_layer_name).output
        for output_layer_name in output_layer_names
    ]
    resnet_model = tf.keras.models.Model(inputs=model.input, outputs=outputs)
    outputs = resnet_model(np.zeros((2, 64, 64, 3), dtype=np.float32))

    # Check the shape of 'conv2_block3_out':
    self.assertEqual(outputs[0].shape, [2, 16, 16, 64])
    # Check the shape of 'conv3_block4_out':
    self.assertEqual(outputs[1].shape, [2, 8, 8, 128])
    # Check the shape of 'conv4_block6_out':
    self.assertEqual(outputs[2].shape, [2, 4, 4, 256])
    # Check the shape of 'conv5_block3_out':
    self.assertEqual(outputs[3].shape, [2, 2, 2, 512])


if __name__ == '__main__':
  tf.test.main()
