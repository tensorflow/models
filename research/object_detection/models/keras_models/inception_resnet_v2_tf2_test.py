# Lint as: python2, python3
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

"""Tests for inception_resnet_v2.py.

This test mainly focuses on comparing slim inception resnet v2 and Keras
inception resnet v2 for object detection. To verify the consistency of the two
models, we compare:
  1. Output shape of each layer given different inputs
  2. Number of global variables

We also visualize the model structure via Tensorboard, and compare the model
layout and the parameters of each Op to make sure the two implementations are
consistent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unittest
import numpy as np
from six.moves import zip
import tensorflow.compat.v1 as tf

from object_detection.models.keras_models import inception_resnet_v2
from object_detection.utils import test_case
from object_detection.utils import tf_version

_KERAS_TO_SLIM_ENDPOINT_NAMES = {
    'activation': 'Conv2d_1a_3x3',
    'activation_1': 'Conv2d_2a_3x3',
    'activation_2': 'Conv2d_2b_3x3',
    'activation_3': 'Conv2d_3b_1x1',
    'activation_4': 'Conv2d_4a_3x3',
    'max_pooling2d': 'MaxPool_3a_3x3',
    'max_pooling2d_1': 'MaxPool_5a_3x3',
    'mixed_5b': 'Mixed_5b',
    'mixed_6a': 'Mixed_6a',
    'block17_20_ac': 'PreAuxLogits',
    'mixed_7a': 'Mixed_7a',
    'conv_7b_ac': 'Conv2d_7b_1x1',
}

_SLIM_ENDPOINT_SHAPES_128 = {
    'Conv2d_1a_3x3': (2, 64, 64, 32),
    'Conv2d_2a_3x3': (2, 64, 64, 32),
    'Conv2d_2b_3x3': (2, 64, 64, 64),
    'Conv2d_3b_1x1': (2, 32, 32, 80),
    'Conv2d_4a_3x3': (2, 32, 32, 192),
    'Conv2d_7b_1x1': (2, 4, 4, 1536),
    'MaxPool_3a_3x3': (2, 32, 32, 64),
    'MaxPool_5a_3x3': (2, 16, 16, 192),
    'Mixed_5b': (2, 16, 16, 320),
    'Mixed_6a': (2, 8, 8, 1088),
    'Mixed_7a': (2, 4, 4, 2080),
    'PreAuxLogits': (2, 8, 8, 1088)}
_SLIM_ENDPOINT_SHAPES_128_STRIDE_8 = {
    'Conv2d_1a_3x3': (2, 64, 64, 32),
    'Conv2d_2a_3x3': (2, 64, 64, 32),
    'Conv2d_2b_3x3': (2, 64, 64, 64),
    'Conv2d_3b_1x1': (2, 32, 32, 80),
    'Conv2d_4a_3x3': (2, 32, 32, 192),
    'MaxPool_3a_3x3': (2, 32, 32, 64),
    'MaxPool_5a_3x3': (2, 16, 16, 192),
    'Mixed_5b': (2, 16, 16, 320),
    'Mixed_6a': (2, 16, 16, 1088),
    'PreAuxLogits': (2, 16, 16, 1088)}
_SLIM_ENDPOINT_SHAPES_128_ALIGN_FEATURE_MAPS_FALSE = {
    'Conv2d_1a_3x3': (2, 63, 63, 32),
    'Conv2d_2a_3x3': (2, 61, 61, 32),
    'Conv2d_2b_3x3': (2, 61, 61, 64),
    'Conv2d_3b_1x1': (2, 30, 30, 80),
    'Conv2d_4a_3x3': (2, 28, 28, 192),
    'Conv2d_7b_1x1': (2, 2, 2, 1536),
    'MaxPool_3a_3x3': (2, 30, 30, 64),
    'MaxPool_5a_3x3': (2, 13, 13, 192),
    'Mixed_5b': (2, 13, 13, 320),
    'Mixed_6a': (2, 6, 6, 1088),
    'Mixed_7a': (2, 2, 2, 2080),
    'PreAuxLogits': (2, 6, 6, 1088)}
_SLIM_ENDPOINT_SHAPES_299 = {}
_SLIM_ENDPOINT_SHAPES_299_STRIDE_8 = {}
_SLIM_ENDPOINT_SHAPES_299_ALIGN_FEATURE_MAPS_FALSE = {}

_KERAS_LAYERS_TO_CHECK = list(_KERAS_TO_SLIM_ENDPOINT_NAMES.keys())

_NUM_CHANNELS = 3
_BATCH_SIZE = 2


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class InceptionResnetV2Test(test_case.TestCase):

  def _create_application_with_layer_outputs(
      self, layer_names, batchnorm_training,
      output_stride=16,
      align_feature_maps=False,
      batchnorm_scale=False,
      weight_decay=0.00004,
      default_batchnorm_momentum=0.9997,
      default_batchnorm_epsilon=0.001,):
    """Constructs Keras inception_resnet_v2 that extracts layer outputs."""
    # Have to clear the Keras backend to ensure isolation in layer naming
    tf.keras.backend.clear_session()
    if not layer_names:
      layer_names = _KERAS_LAYERS_TO_CHECK
    full_model = inception_resnet_v2.inception_resnet_v2(
        batchnorm_training=batchnorm_training,
        output_stride=output_stride,
        align_feature_maps=align_feature_maps,
        weights=None,
        batchnorm_scale=batchnorm_scale,
        weight_decay=weight_decay,
        default_batchnorm_momentum=default_batchnorm_momentum,
        default_batchnorm_epsilon=default_batchnorm_epsilon,
        include_top=False)
    layer_outputs = [full_model.get_layer(name=layer).output
                     for layer in layer_names]
    return tf.keras.Model(
        inputs=full_model.inputs,
        outputs=layer_outputs)

  def _check_returns_correct_shape(
      self, image_height, image_width,
      expected_feature_map_shape, layer_names=None, batchnorm_training=True,
      output_stride=16,
      align_feature_maps=False,
      batchnorm_scale=False,
      weight_decay=0.00004,
      default_batchnorm_momentum=0.9997,
      default_batchnorm_epsilon=0.001,):
    if not layer_names:
      layer_names = _KERAS_LAYERS_TO_CHECK
    model = self._create_application_with_layer_outputs(
        layer_names=layer_names,
        batchnorm_training=batchnorm_training,
        output_stride=output_stride,
        align_feature_maps=align_feature_maps,
        batchnorm_scale=batchnorm_scale,
        weight_decay=weight_decay,
        default_batchnorm_momentum=default_batchnorm_momentum,
        default_batchnorm_epsilon=default_batchnorm_epsilon)

    image_tensor = np.random.rand(_BATCH_SIZE, image_height, image_width,
                                  _NUM_CHANNELS).astype(np.float32)
    feature_maps = model(image_tensor)

    for feature_map, layer_name in zip(feature_maps, layer_names):
      endpoint_name = _KERAS_TO_SLIM_ENDPOINT_NAMES[layer_name]
      expected_shape = expected_feature_map_shape[endpoint_name]
      self.assertAllEqual(feature_map.shape, expected_shape)

  def _get_variables(self, layer_names=None):
    tf.keras.backend.clear_session()
    model = self._create_application_with_layer_outputs(
        layer_names=layer_names,
        batchnorm_training=False)
    preprocessed_inputs = tf.random.uniform([4, 40, 40, _NUM_CHANNELS])
    model(preprocessed_inputs)
    return model.variables

  def test_returns_correct_shapes_128(self):
    image_height = 128
    image_width = 128
    expected_feature_map_shape = (
        _SLIM_ENDPOINT_SHAPES_128)
    self._check_returns_correct_shape(
        image_height, image_width, expected_feature_map_shape,
        align_feature_maps=True)

  def test_returns_correct_shapes_128_output_stride_8(self):
    image_height = 128
    image_width = 128
    expected_feature_map_shape = (
        _SLIM_ENDPOINT_SHAPES_128_STRIDE_8)

    # Output stride of 8 not defined beyond 'block17_20_ac', which is
    # PreAuxLogits in slim. So, we exclude those layers in our Keras vs Slim
    # comparison.
    excluded_layers = {'mixed_7a', 'conv_7b_ac'}
    layer_names = [l for l in _KERAS_LAYERS_TO_CHECK
                   if l not in excluded_layers]
    self._check_returns_correct_shape(
        image_height, image_width, expected_feature_map_shape,
        layer_names=layer_names, output_stride=8, align_feature_maps=True)

  def test_returns_correct_shapes_128_align_feature_maps_false(
      self):
    image_height = 128
    image_width = 128
    expected_feature_map_shape = (
        _SLIM_ENDPOINT_SHAPES_128_ALIGN_FEATURE_MAPS_FALSE)
    self._check_returns_correct_shape(
        image_height, image_width, expected_feature_map_shape,
        align_feature_maps=False)

  def test_hyperparam_override(self):
    model = inception_resnet_v2.inception_resnet_v2(
        batchnorm_training=True,
        default_batchnorm_momentum=0.2,
        default_batchnorm_epsilon=0.1,
        weights=None,
        include_top=False)
    bn_layer = model.get_layer(name='freezable_batch_norm')
    self.assertAllClose(bn_layer.momentum, 0.2)
    self.assertAllClose(bn_layer.epsilon, 0.1)

  def test_variable_count(self):
    variables = self._get_variables()
    # 896 is the number of variables from slim inception resnet v2 model.
    self.assertEqual(len(variables), 896)


if __name__ == '__main__':
  tf.test.main()
