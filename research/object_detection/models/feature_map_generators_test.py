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

"""Tests for feature map generators."""
import unittest
from absl.testing import parameterized

import numpy as np
import tensorflow.compat.v1 as tf

from google.protobuf import text_format

from object_detection.builders import hyperparams_builder
from object_detection.models import feature_map_generators
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case
from object_detection.utils import test_utils
from object_detection.utils import tf_version

INCEPTION_V2_LAYOUT = {
    'from_layer': ['Mixed_3c', 'Mixed_4c', 'Mixed_5c', '', '', ''],
    'layer_depth': [-1, -1, -1, 512, 256, 256],
    'anchor_strides': [16, 32, 64, -1, -1, -1],
    'layer_target_norm': [20.0, -1, -1, -1, -1, -1],
}

INCEPTION_V3_LAYOUT = {
    'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', ''],
    'layer_depth': [-1, -1, -1, 512, 256, 128],
    'anchor_strides': [16, 32, 64, -1, -1, -1],
    'aspect_ratios': [1.0, 2.0, 1.0/2, 3.0, 1.0/3]
}

EMBEDDED_SSD_MOBILENET_V1_LAYOUT = {
    'from_layer': ['Conv2d_11_pointwise', 'Conv2d_13_pointwise', '', '', ''],
    'layer_depth': [-1, -1, 512, 256, 256],
    'conv_kernel_size': [-1, -1, 3, 3, 2],
}

SSD_MOBILENET_V1_WEIGHT_SHARED_LAYOUT = {
    'from_layer': ['Conv2d_13_pointwise', '', '', ''],
    'layer_depth': [-1, 256, 256, 256],
}


class MultiResolutionFeatureMapGeneratorTest(test_case.TestCase):

  def _build_conv_hyperparams(self):
    conv_hyperparams = hyperparams_pb2.Hyperparams()
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)
    return hyperparams_builder.KerasLayerHyperparams(conv_hyperparams)

  def _build_feature_map_generator(self, feature_map_layout,
                                   pool_residual=False):
    if tf_version.is_tf2():
      return feature_map_generators.KerasMultiResolutionFeatureMaps(
          feature_map_layout=feature_map_layout,
          depth_multiplier=1,
          min_depth=32,
          insert_1x1_conv=True,
          freeze_batchnorm=False,
          is_training=True,
          conv_hyperparams=self._build_conv_hyperparams(),
          name='FeatureMaps'
      )
    else:
      def feature_map_generator(image_features):
        return feature_map_generators.multi_resolution_feature_maps(
            feature_map_layout=feature_map_layout,
            depth_multiplier=1,
            min_depth=32,
            insert_1x1_conv=True,
            image_features=image_features,
            pool_residual=pool_residual)
      return feature_map_generator

  def test_get_expected_feature_map_shapes_with_inception_v2(self):
    with test_utils.GraphContextOrNone() as g:
      image_features = {
          'Mixed_3c': tf.random_uniform([4, 28, 28, 256], dtype=tf.float32),
          'Mixed_4c': tf.random_uniform([4, 14, 14, 576], dtype=tf.float32),
          'Mixed_5c': tf.random_uniform([4, 7, 7, 1024], dtype=tf.float32)
      }
      feature_map_generator = self._build_feature_map_generator(
          feature_map_layout=INCEPTION_V2_LAYOUT)
    def graph_fn():
      feature_maps = feature_map_generator(image_features)
      return feature_maps

    expected_feature_map_shapes = {
        'Mixed_3c': (4, 28, 28, 256),
        'Mixed_4c': (4, 14, 14, 576),
        'Mixed_5c': (4, 7, 7, 1024),
        'Mixed_5c_2_Conv2d_3_3x3_s2_512': (4, 4, 4, 512),
        'Mixed_5c_2_Conv2d_4_3x3_s2_256': (4, 2, 2, 256),
        'Mixed_5c_2_Conv2d_5_3x3_s2_256': (4, 1, 1, 256)}
    out_feature_maps = self.execute(graph_fn, [], g)
    out_feature_map_shapes = dict(
        (key, value.shape) for key, value in out_feature_maps.items())
    self.assertDictEqual(expected_feature_map_shapes, out_feature_map_shapes)

  def test_get_expected_feature_map_shapes_with_inception_v2_use_depthwise(
      self):
    with test_utils.GraphContextOrNone() as g:
      image_features = {
          'Mixed_3c': tf.random_uniform([4, 28, 28, 256], dtype=tf.float32),
          'Mixed_4c': tf.random_uniform([4, 14, 14, 576], dtype=tf.float32),
          'Mixed_5c': tf.random_uniform([4, 7, 7, 1024], dtype=tf.float32)
      }
      layout_copy = INCEPTION_V2_LAYOUT.copy()
      layout_copy['use_depthwise'] = True
      feature_map_generator = self._build_feature_map_generator(
          feature_map_layout=layout_copy)
    def graph_fn():
      return feature_map_generator(image_features)

    expected_feature_map_shapes = {
        'Mixed_3c': (4, 28, 28, 256),
        'Mixed_4c': (4, 14, 14, 576),
        'Mixed_5c': (4, 7, 7, 1024),
        'Mixed_5c_2_Conv2d_3_3x3_s2_512': (4, 4, 4, 512),
        'Mixed_5c_2_Conv2d_4_3x3_s2_256': (4, 2, 2, 256),
        'Mixed_5c_2_Conv2d_5_3x3_s2_256': (4, 1, 1, 256)}
    out_feature_maps = self.execute(graph_fn, [], g)
    out_feature_map_shapes = dict(
        (key, value.shape) for key, value in out_feature_maps.items())
    self.assertDictEqual(expected_feature_map_shapes, out_feature_map_shapes)

  def test_get_expected_feature_map_shapes_use_explicit_padding(self):
    with test_utils.GraphContextOrNone() as g:
      image_features = {
          'Mixed_3c': tf.random_uniform([4, 28, 28, 256], dtype=tf.float32),
          'Mixed_4c': tf.random_uniform([4, 14, 14, 576], dtype=tf.float32),
          'Mixed_5c': tf.random_uniform([4, 7, 7, 1024], dtype=tf.float32)
      }
      layout_copy = INCEPTION_V2_LAYOUT.copy()
      layout_copy['use_explicit_padding'] = True
      feature_map_generator = self._build_feature_map_generator(
          feature_map_layout=layout_copy,
      )
    def graph_fn():
      return feature_map_generator(image_features)

    expected_feature_map_shapes = {
        'Mixed_3c': (4, 28, 28, 256),
        'Mixed_4c': (4, 14, 14, 576),
        'Mixed_5c': (4, 7, 7, 1024),
        'Mixed_5c_2_Conv2d_3_3x3_s2_512': (4, 4, 4, 512),
        'Mixed_5c_2_Conv2d_4_3x3_s2_256': (4, 2, 2, 256),
        'Mixed_5c_2_Conv2d_5_3x3_s2_256': (4, 1, 1, 256)}
    out_feature_maps = self.execute(graph_fn, [], g)
    out_feature_map_shapes = dict(
        (key, value.shape) for key, value in out_feature_maps.items())
    self.assertDictEqual(expected_feature_map_shapes, out_feature_map_shapes)

  def test_get_expected_feature_map_shapes_with_inception_v3(self):
    with test_utils.GraphContextOrNone() as g:
      image_features = {
          'Mixed_5d': tf.random_uniform([4, 35, 35, 256], dtype=tf.float32),
          'Mixed_6e': tf.random_uniform([4, 17, 17, 576], dtype=tf.float32),
          'Mixed_7c': tf.random_uniform([4, 8, 8, 1024], dtype=tf.float32)
      }

      feature_map_generator = self._build_feature_map_generator(
          feature_map_layout=INCEPTION_V3_LAYOUT,
      )
    def graph_fn():
      return feature_map_generator(image_features)

    expected_feature_map_shapes = {
        'Mixed_5d': (4, 35, 35, 256),
        'Mixed_6e': (4, 17, 17, 576),
        'Mixed_7c': (4, 8, 8, 1024),
        'Mixed_7c_2_Conv2d_3_3x3_s2_512': (4, 4, 4, 512),
        'Mixed_7c_2_Conv2d_4_3x3_s2_256': (4, 2, 2, 256),
        'Mixed_7c_2_Conv2d_5_3x3_s2_128': (4, 1, 1, 128)}
    out_feature_maps = self.execute(graph_fn, [], g)
    out_feature_map_shapes = dict(
        (key, value.shape) for key, value in out_feature_maps.items())
    self.assertDictEqual(expected_feature_map_shapes, out_feature_map_shapes)

  def test_get_expected_feature_map_shapes_with_embedded_ssd_mobilenet_v1(
      self):
    with test_utils.GraphContextOrNone() as g:
      image_features = {
          'Conv2d_11_pointwise': tf.random_uniform([4, 16, 16, 512],
                                                   dtype=tf.float32),
          'Conv2d_13_pointwise': tf.random_uniform([4, 8, 8, 1024],
                                                   dtype=tf.float32),
      }

      feature_map_generator = self._build_feature_map_generator(
          feature_map_layout=EMBEDDED_SSD_MOBILENET_V1_LAYOUT,
      )
    def graph_fn():
      return feature_map_generator(image_features)

    expected_feature_map_shapes = {
        'Conv2d_11_pointwise': (4, 16, 16, 512),
        'Conv2d_13_pointwise': (4, 8, 8, 1024),
        'Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_512': (4, 4, 4, 512),
        'Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256': (4, 2, 2, 256),
        'Conv2d_13_pointwise_2_Conv2d_4_2x2_s2_256': (4, 1, 1, 256)}
    out_feature_maps = self.execute(graph_fn, [], g)
    out_feature_map_shapes = dict(
        (key, value.shape) for key, value in out_feature_maps.items())
    self.assertDictEqual(expected_feature_map_shapes, out_feature_map_shapes)

  def test_feature_map_shapes_with_pool_residual_ssd_mobilenet_v1(
      self):
    with test_utils.GraphContextOrNone() as g:
      image_features = {
          'Conv2d_13_pointwise': tf.random_uniform([4, 8, 8, 1024],
                                                   dtype=tf.float32),
      }

      feature_map_generator = self._build_feature_map_generator(
          feature_map_layout=SSD_MOBILENET_V1_WEIGHT_SHARED_LAYOUT,
          pool_residual=True
      )
    def graph_fn():
      return feature_map_generator(image_features)

    expected_feature_map_shapes = {
        'Conv2d_13_pointwise': (4, 8, 8, 1024),
        'Conv2d_13_pointwise_2_Conv2d_1_3x3_s2_256': (4, 4, 4, 256),
        'Conv2d_13_pointwise_2_Conv2d_2_3x3_s2_256': (4, 2, 2, 256),
        'Conv2d_13_pointwise_2_Conv2d_3_3x3_s2_256': (4, 1, 1, 256)}
    out_feature_maps = self.execute(graph_fn, [], g)
    out_feature_map_shapes = dict(
        (key, value.shape) for key, value in out_feature_maps.items())
    self.assertDictEqual(expected_feature_map_shapes, out_feature_map_shapes)

  def test_get_expected_variable_names_with_inception_v2(self):
    with test_utils.GraphContextOrNone() as g:
      image_features = {
          'Mixed_3c': tf.random_uniform([4, 28, 28, 256], dtype=tf.float32),
          'Mixed_4c': tf.random_uniform([4, 14, 14, 576], dtype=tf.float32),
          'Mixed_5c': tf.random_uniform([4, 7, 7, 1024], dtype=tf.float32)
      }
      feature_map_generator = self._build_feature_map_generator(
          feature_map_layout=INCEPTION_V2_LAYOUT,
      )
    def graph_fn():
      return feature_map_generator(image_features)

    self.execute(graph_fn, [], g)
    expected_slim_variables = set([
        'Mixed_5c_1_Conv2d_3_1x1_256/weights',
        'Mixed_5c_1_Conv2d_3_1x1_256/biases',
        'Mixed_5c_2_Conv2d_3_3x3_s2_512/weights',
        'Mixed_5c_2_Conv2d_3_3x3_s2_512/biases',
        'Mixed_5c_1_Conv2d_4_1x1_128/weights',
        'Mixed_5c_1_Conv2d_4_1x1_128/biases',
        'Mixed_5c_2_Conv2d_4_3x3_s2_256/weights',
        'Mixed_5c_2_Conv2d_4_3x3_s2_256/biases',
        'Mixed_5c_1_Conv2d_5_1x1_128/weights',
        'Mixed_5c_1_Conv2d_5_1x1_128/biases',
        'Mixed_5c_2_Conv2d_5_3x3_s2_256/weights',
        'Mixed_5c_2_Conv2d_5_3x3_s2_256/biases',
    ])

    expected_keras_variables = set([
        'FeatureMaps/Mixed_5c_1_Conv2d_3_1x1_256_conv/kernel',
        'FeatureMaps/Mixed_5c_1_Conv2d_3_1x1_256_conv/bias',
        'FeatureMaps/Mixed_5c_2_Conv2d_3_3x3_s2_512_conv/kernel',
        'FeatureMaps/Mixed_5c_2_Conv2d_3_3x3_s2_512_conv/bias',
        'FeatureMaps/Mixed_5c_1_Conv2d_4_1x1_128_conv/kernel',
        'FeatureMaps/Mixed_5c_1_Conv2d_4_1x1_128_conv/bias',
        'FeatureMaps/Mixed_5c_2_Conv2d_4_3x3_s2_256_conv/kernel',
        'FeatureMaps/Mixed_5c_2_Conv2d_4_3x3_s2_256_conv/bias',
        'FeatureMaps/Mixed_5c_1_Conv2d_5_1x1_128_conv/kernel',
        'FeatureMaps/Mixed_5c_1_Conv2d_5_1x1_128_conv/bias',
        'FeatureMaps/Mixed_5c_2_Conv2d_5_3x3_s2_256_conv/kernel',
        'FeatureMaps/Mixed_5c_2_Conv2d_5_3x3_s2_256_conv/bias',
    ])

    if tf_version.is_tf2():
      actual_variable_set = set(
          [var.name.split(':')[0] for var in feature_map_generator.variables])
      self.assertSetEqual(expected_keras_variables, actual_variable_set)
    else:
      with g.as_default():
        actual_variable_set = set(
            [var.op.name for var in tf.trainable_variables()])
      self.assertSetEqual(expected_slim_variables, actual_variable_set)

  def test_get_expected_variable_names_with_inception_v2_use_depthwise(
      self):
    with test_utils.GraphContextOrNone() as g:
      image_features = {
          'Mixed_3c': tf.random_uniform([4, 28, 28, 256], dtype=tf.float32),
          'Mixed_4c': tf.random_uniform([4, 14, 14, 576], dtype=tf.float32),
          'Mixed_5c': tf.random_uniform([4, 7, 7, 1024], dtype=tf.float32)
      }
      layout_copy = INCEPTION_V2_LAYOUT.copy()
      layout_copy['use_depthwise'] = True
      feature_map_generator = self._build_feature_map_generator(
          feature_map_layout=layout_copy,
      )
    def graph_fn():
      return feature_map_generator(image_features)
    self.execute(graph_fn, [], g)

    expected_slim_variables = set([
        'Mixed_5c_1_Conv2d_3_1x1_256/weights',
        'Mixed_5c_1_Conv2d_3_1x1_256/biases',
        'Mixed_5c_2_Conv2d_3_3x3_s2_512_depthwise/depthwise_weights',
        'Mixed_5c_2_Conv2d_3_3x3_s2_512_depthwise/biases',
        'Mixed_5c_2_Conv2d_3_3x3_s2_512/weights',
        'Mixed_5c_2_Conv2d_3_3x3_s2_512/biases',
        'Mixed_5c_1_Conv2d_4_1x1_128/weights',
        'Mixed_5c_1_Conv2d_4_1x1_128/biases',
        'Mixed_5c_2_Conv2d_4_3x3_s2_256_depthwise/depthwise_weights',
        'Mixed_5c_2_Conv2d_4_3x3_s2_256_depthwise/biases',
        'Mixed_5c_2_Conv2d_4_3x3_s2_256/weights',
        'Mixed_5c_2_Conv2d_4_3x3_s2_256/biases',
        'Mixed_5c_1_Conv2d_5_1x1_128/weights',
        'Mixed_5c_1_Conv2d_5_1x1_128/biases',
        'Mixed_5c_2_Conv2d_5_3x3_s2_256_depthwise/depthwise_weights',
        'Mixed_5c_2_Conv2d_5_3x3_s2_256_depthwise/biases',
        'Mixed_5c_2_Conv2d_5_3x3_s2_256/weights',
        'Mixed_5c_2_Conv2d_5_3x3_s2_256/biases',
    ])

    expected_keras_variables = set([
        'FeatureMaps/Mixed_5c_1_Conv2d_3_1x1_256_conv/kernel',
        'FeatureMaps/Mixed_5c_1_Conv2d_3_1x1_256_conv/bias',
        ('FeatureMaps/Mixed_5c_2_Conv2d_3_3x3_s2_512_depthwise_conv/'
         'depthwise_kernel'),
        ('FeatureMaps/Mixed_5c_2_Conv2d_3_3x3_s2_512_depthwise_conv/'
         'bias'),
        'FeatureMaps/Mixed_5c_2_Conv2d_3_3x3_s2_512_conv/kernel',
        'FeatureMaps/Mixed_5c_2_Conv2d_3_3x3_s2_512_conv/bias',
        'FeatureMaps/Mixed_5c_1_Conv2d_4_1x1_128_conv/kernel',
        'FeatureMaps/Mixed_5c_1_Conv2d_4_1x1_128_conv/bias',
        ('FeatureMaps/Mixed_5c_2_Conv2d_4_3x3_s2_256_depthwise_conv/'
         'depthwise_kernel'),
        ('FeatureMaps/Mixed_5c_2_Conv2d_4_3x3_s2_256_depthwise_conv/'
         'bias'),
        'FeatureMaps/Mixed_5c_2_Conv2d_4_3x3_s2_256_conv/kernel',
        'FeatureMaps/Mixed_5c_2_Conv2d_4_3x3_s2_256_conv/bias',
        'FeatureMaps/Mixed_5c_1_Conv2d_5_1x1_128_conv/kernel',
        'FeatureMaps/Mixed_5c_1_Conv2d_5_1x1_128_conv/bias',
        ('FeatureMaps/Mixed_5c_2_Conv2d_5_3x3_s2_256_depthwise_conv/'
         'depthwise_kernel'),
        ('FeatureMaps/Mixed_5c_2_Conv2d_5_3x3_s2_256_depthwise_conv/'
         'bias'),
        'FeatureMaps/Mixed_5c_2_Conv2d_5_3x3_s2_256_conv/kernel',
        'FeatureMaps/Mixed_5c_2_Conv2d_5_3x3_s2_256_conv/bias',
    ])

    if tf_version.is_tf2():
      actual_variable_set = set(
          [var.name.split(':')[0] for var in feature_map_generator.variables])
      self.assertSetEqual(expected_keras_variables, actual_variable_set)
    else:
      with g.as_default():
        actual_variable_set = set(
            [var.op.name for var in tf.trainable_variables()])
      self.assertSetEqual(expected_slim_variables, actual_variable_set)


@parameterized.parameters({'use_native_resize_op': True},
                          {'use_native_resize_op': False})
class FPNFeatureMapGeneratorTest(test_case.TestCase, parameterized.TestCase):

  def _build_conv_hyperparams(self):
    conv_hyperparams = hyperparams_pb2.Hyperparams()
    conv_hyperparams_text_proto = """
      regularizer {
        l2_regularizer {
        }
      }
      initializer {
        truncated_normal_initializer {
        }
      }
    """
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)
    return hyperparams_builder.KerasLayerHyperparams(conv_hyperparams)

  def _build_feature_map_generator(
      self, image_features, depth, use_bounded_activations=False,
      use_native_resize_op=False, use_explicit_padding=False,
      use_depthwise=False):
    if tf_version.is_tf2():
      return feature_map_generators.KerasFpnTopDownFeatureMaps(
          num_levels=len(image_features),
          depth=depth,
          is_training=True,
          conv_hyperparams=self._build_conv_hyperparams(),
          freeze_batchnorm=False,
          use_depthwise=use_depthwise,
          use_explicit_padding=use_explicit_padding,
          use_bounded_activations=use_bounded_activations,
          use_native_resize_op=use_native_resize_op,
          scope=None,
          name='FeatureMaps',
      )
    else:
      def feature_map_generator(image_features):
        return feature_map_generators.fpn_top_down_feature_maps(
            image_features=image_features,
            depth=depth,
            use_depthwise=use_depthwise,
            use_explicit_padding=use_explicit_padding,
            use_bounded_activations=use_bounded_activations,
            use_native_resize_op=use_native_resize_op)
      return feature_map_generator

  def test_get_expected_feature_map_shapes(
      self, use_native_resize_op):
    with test_utils.GraphContextOrNone() as g:
      image_features = [
          ('block2', tf.random_uniform([4, 8, 8, 256], dtype=tf.float32)),
          ('block3', tf.random_uniform([4, 4, 4, 256], dtype=tf.float32)),
          ('block4', tf.random_uniform([4, 2, 2, 256], dtype=tf.float32)),
          ('block5', tf.random_uniform([4, 1, 1, 256], dtype=tf.float32))
      ]
      feature_map_generator = self._build_feature_map_generator(
          image_features=image_features,
          depth=128,
          use_native_resize_op=use_native_resize_op)
    def graph_fn():
      return feature_map_generator(image_features)

    expected_feature_map_shapes = {
        'top_down_block2': (4, 8, 8, 128),
        'top_down_block3': (4, 4, 4, 128),
        'top_down_block4': (4, 2, 2, 128),
        'top_down_block5': (4, 1, 1, 128)
    }
    out_feature_maps = self.execute(graph_fn, [], g)
    out_feature_map_shapes = dict(
        (key, value.shape) for key, value in out_feature_maps.items())
    self.assertDictEqual(expected_feature_map_shapes, out_feature_map_shapes)

  def test_get_expected_feature_map_shapes_with_explicit_padding(
      self, use_native_resize_op):
    with test_utils.GraphContextOrNone() as g:
      image_features = [
          ('block2', tf.random_uniform([4, 8, 8, 256], dtype=tf.float32)),
          ('block3', tf.random_uniform([4, 4, 4, 256], dtype=tf.float32)),
          ('block4', tf.random_uniform([4, 2, 2, 256], dtype=tf.float32)),
          ('block5', tf.random_uniform([4, 1, 1, 256], dtype=tf.float32))
      ]
      feature_map_generator = self._build_feature_map_generator(
          image_features=image_features,
          depth=128,
          use_explicit_padding=True,
          use_native_resize_op=use_native_resize_op)
    def graph_fn():
      return feature_map_generator(image_features)

    expected_feature_map_shapes = {
        'top_down_block2': (4, 8, 8, 128),
        'top_down_block3': (4, 4, 4, 128),
        'top_down_block4': (4, 2, 2, 128),
        'top_down_block5': (4, 1, 1, 128)
    }
    out_feature_maps = self.execute(graph_fn, [], g)
    out_feature_map_shapes = dict(
        (key, value.shape) for key, value in out_feature_maps.items())
    self.assertDictEqual(expected_feature_map_shapes, out_feature_map_shapes)

  @unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
  def test_use_bounded_activations_add_operations(
      self, use_native_resize_op):
    with test_utils.GraphContextOrNone() as g:
      image_features = [('block2',
                         tf.random_uniform([4, 8, 8, 256], dtype=tf.float32)),
                        ('block3',
                         tf.random_uniform([4, 4, 4, 256], dtype=tf.float32)),
                        ('block4',
                         tf.random_uniform([4, 2, 2, 256], dtype=tf.float32)),
                        ('block5',
                         tf.random_uniform([4, 1, 1, 256], dtype=tf.float32))]
      feature_map_generator = self._build_feature_map_generator(
          image_features=image_features,
          depth=128,
          use_bounded_activations=True,
          use_native_resize_op=use_native_resize_op)
    def graph_fn():
      return feature_map_generator(image_features)
    self.execute(graph_fn, [], g)
    expected_added_operations = dict.fromkeys([
        'top_down/clip_by_value', 'top_down/clip_by_value_1',
        'top_down/clip_by_value_2', 'top_down/clip_by_value_3',
        'top_down/clip_by_value_4', 'top_down/clip_by_value_5',
        'top_down/clip_by_value_6'
    ])
    op_names = {op.name: None for op in g.get_operations()}
    self.assertDictContainsSubset(expected_added_operations, op_names)

  @unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
  def test_use_bounded_activations_clip_value(
      self, use_native_resize_op):
    tf_graph = tf.Graph()
    with tf_graph.as_default():
      image_features = [
          ('block2', 255 * tf.ones([4, 8, 8, 256], dtype=tf.float32)),
          ('block3', 255 * tf.ones([4, 4, 4, 256], dtype=tf.float32)),
          ('block4', 255 * tf.ones([4, 2, 2, 256], dtype=tf.float32)),
          ('block5', 255 * tf.ones([4, 1, 1, 256], dtype=tf.float32))
      ]
      feature_map_generator = self._build_feature_map_generator(
          image_features=image_features,
          depth=128,
          use_bounded_activations=True,
          use_native_resize_op=use_native_resize_op)
      feature_map_generator(image_features)

      expected_clip_by_value_ops = [
          'top_down/clip_by_value', 'top_down/clip_by_value_1',
          'top_down/clip_by_value_2', 'top_down/clip_by_value_3',
          'top_down/clip_by_value_4', 'top_down/clip_by_value_5',
          'top_down/clip_by_value_6'
      ]

      # Gathers activation tensors before and after clip_by_value operations.
      activations = {}
      for clip_by_value_op in expected_clip_by_value_ops:
        clip_input_tensor = tf_graph.get_operation_by_name(
            '{}/Minimum'.format(clip_by_value_op)).inputs[0]
        clip_output_tensor = tf_graph.get_tensor_by_name(
            '{}:0'.format(clip_by_value_op))
        activations.update({
            'before_{}'.format(clip_by_value_op): clip_input_tensor,
            'after_{}'.format(clip_by_value_op): clip_output_tensor,
        })

      expected_lower_bound = -feature_map_generators.ACTIVATION_BOUND
      expected_upper_bound = feature_map_generators.ACTIVATION_BOUND
      init_op = tf.global_variables_initializer()
      with self.test_session() as session:
        session.run(init_op)
        activations_output = session.run(activations)
        for clip_by_value_op in expected_clip_by_value_ops:
          # Before clipping, activations are beyound the expected bound because
          # of large input image_features values.
          activations_before_clipping = (
              activations_output['before_{}'.format(clip_by_value_op)])
          before_clipping_lower_bound = np.amin(activations_before_clipping)
          before_clipping_upper_bound = np.amax(activations_before_clipping)
          self.assertLessEqual(before_clipping_lower_bound,
                               expected_lower_bound)
          self.assertGreaterEqual(before_clipping_upper_bound,
                                  expected_upper_bound)

          # After clipping, activations are bounded as expectation.
          activations_after_clipping = (
              activations_output['after_{}'.format(clip_by_value_op)])
          after_clipping_lower_bound = np.amin(activations_after_clipping)
          after_clipping_upper_bound = np.amax(activations_after_clipping)
          self.assertGreaterEqual(after_clipping_lower_bound,
                                  expected_lower_bound)
          self.assertLessEqual(after_clipping_upper_bound, expected_upper_bound)

  def test_get_expected_feature_map_shapes_with_depthwise(
      self, use_native_resize_op):
    with test_utils.GraphContextOrNone() as g:
      image_features = [
          ('block2', tf.random_uniform([4, 8, 8, 256], dtype=tf.float32)),
          ('block3', tf.random_uniform([4, 4, 4, 256], dtype=tf.float32)),
          ('block4', tf.random_uniform([4, 2, 2, 256], dtype=tf.float32)),
          ('block5', tf.random_uniform([4, 1, 1, 256], dtype=tf.float32))
      ]
      feature_map_generator = self._build_feature_map_generator(
          image_features=image_features,
          depth=128,
          use_depthwise=True,
          use_native_resize_op=use_native_resize_op)
    def graph_fn():
      return feature_map_generator(image_features)

    expected_feature_map_shapes = {
        'top_down_block2': (4, 8, 8, 128),
        'top_down_block3': (4, 4, 4, 128),
        'top_down_block4': (4, 2, 2, 128),
        'top_down_block5': (4, 1, 1, 128)
    }
    out_feature_maps = self.execute(graph_fn, [], g)
    out_feature_map_shapes = dict(
        (key, value.shape) for key, value in out_feature_maps.items())
    self.assertDictEqual(expected_feature_map_shapes, out_feature_map_shapes)

  def test_get_expected_variable_names(
      self, use_native_resize_op):
    with test_utils.GraphContextOrNone() as g:
      image_features = [
          ('block2', tf.random_uniform([4, 8, 8, 256], dtype=tf.float32)),
          ('block3', tf.random_uniform([4, 4, 4, 256], dtype=tf.float32)),
          ('block4', tf.random_uniform([4, 2, 2, 256], dtype=tf.float32)),
          ('block5', tf.random_uniform([4, 1, 1, 256], dtype=tf.float32))
      ]
      feature_map_generator = self._build_feature_map_generator(
          image_features=image_features,
          depth=128,
          use_native_resize_op=use_native_resize_op)
    def graph_fn():
      return feature_map_generator(image_features)
    self.execute(graph_fn, [], g)
    expected_slim_variables = set([
        'projection_1/weights',
        'projection_1/biases',
        'projection_2/weights',
        'projection_2/biases',
        'projection_3/weights',
        'projection_3/biases',
        'projection_4/weights',
        'projection_4/biases',
        'smoothing_1/weights',
        'smoothing_1/biases',
        'smoothing_2/weights',
        'smoothing_2/biases',
        'smoothing_3/weights',
        'smoothing_3/biases',
    ])

    expected_keras_variables = set([
        'FeatureMaps/top_down/projection_1/kernel',
        'FeatureMaps/top_down/projection_1/bias',
        'FeatureMaps/top_down/projection_2/kernel',
        'FeatureMaps/top_down/projection_2/bias',
        'FeatureMaps/top_down/projection_3/kernel',
        'FeatureMaps/top_down/projection_3/bias',
        'FeatureMaps/top_down/projection_4/kernel',
        'FeatureMaps/top_down/projection_4/bias',
        'FeatureMaps/top_down/smoothing_1_conv/kernel',
        'FeatureMaps/top_down/smoothing_1_conv/bias',
        'FeatureMaps/top_down/smoothing_2_conv/kernel',
        'FeatureMaps/top_down/smoothing_2_conv/bias',
        'FeatureMaps/top_down/smoothing_3_conv/kernel',
        'FeatureMaps/top_down/smoothing_3_conv/bias'
    ])

    if tf_version.is_tf2():
      actual_variable_set = set(
          [var.name.split(':')[0] for var in feature_map_generator.variables])
      self.assertSetEqual(expected_keras_variables, actual_variable_set)
    else:
      with g.as_default():
        actual_variable_set = set(
            [var.op.name for var in tf.trainable_variables()])
      self.assertSetEqual(expected_slim_variables, actual_variable_set)

  def test_get_expected_variable_names_with_depthwise(
      self, use_native_resize_op):
    with test_utils.GraphContextOrNone() as g:
      image_features = [
          ('block2', tf.random_uniform([4, 8, 8, 256], dtype=tf.float32)),
          ('block3', tf.random_uniform([4, 4, 4, 256], dtype=tf.float32)),
          ('block4', tf.random_uniform([4, 2, 2, 256], dtype=tf.float32)),
          ('block5', tf.random_uniform([4, 1, 1, 256], dtype=tf.float32))
      ]
      feature_map_generator = self._build_feature_map_generator(
          image_features=image_features,
          depth=128,
          use_depthwise=True,
          use_native_resize_op=use_native_resize_op)
    def graph_fn():
      return feature_map_generator(image_features)
    self.execute(graph_fn, [], g)
    expected_slim_variables = set([
        'projection_1/weights',
        'projection_1/biases',
        'projection_2/weights',
        'projection_2/biases',
        'projection_3/weights',
        'projection_3/biases',
        'projection_4/weights',
        'projection_4/biases',
        'smoothing_1/depthwise_weights',
        'smoothing_1/pointwise_weights',
        'smoothing_1/biases',
        'smoothing_2/depthwise_weights',
        'smoothing_2/pointwise_weights',
        'smoothing_2/biases',
        'smoothing_3/depthwise_weights',
        'smoothing_3/pointwise_weights',
        'smoothing_3/biases',
    ])

    expected_keras_variables = set([
        'FeatureMaps/top_down/projection_1/kernel',
        'FeatureMaps/top_down/projection_1/bias',
        'FeatureMaps/top_down/projection_2/kernel',
        'FeatureMaps/top_down/projection_2/bias',
        'FeatureMaps/top_down/projection_3/kernel',
        'FeatureMaps/top_down/projection_3/bias',
        'FeatureMaps/top_down/projection_4/kernel',
        'FeatureMaps/top_down/projection_4/bias',
        'FeatureMaps/top_down/smoothing_1_depthwise_conv/depthwise_kernel',
        'FeatureMaps/top_down/smoothing_1_depthwise_conv/pointwise_kernel',
        'FeatureMaps/top_down/smoothing_1_depthwise_conv/bias',
        'FeatureMaps/top_down/smoothing_2_depthwise_conv/depthwise_kernel',
        'FeatureMaps/top_down/smoothing_2_depthwise_conv/pointwise_kernel',
        'FeatureMaps/top_down/smoothing_2_depthwise_conv/bias',
        'FeatureMaps/top_down/smoothing_3_depthwise_conv/depthwise_kernel',
        'FeatureMaps/top_down/smoothing_3_depthwise_conv/pointwise_kernel',
        'FeatureMaps/top_down/smoothing_3_depthwise_conv/bias'
    ])

    if tf_version.is_tf2():
      actual_variable_set = set(
          [var.name.split(':')[0] for var in feature_map_generator.variables])
      self.assertSetEqual(expected_keras_variables, actual_variable_set)
    else:
      with g.as_default():
        actual_variable_set = set(
            [var.op.name for var in tf.trainable_variables()])
      self.assertSetEqual(expected_slim_variables, actual_variable_set)


class GetDepthFunctionTest(tf.test.TestCase):

  def test_return_min_depth_when_multiplier_is_small(self):
    depth_fn = feature_map_generators.get_depth_fn(depth_multiplier=0.5,
                                                   min_depth=16)
    self.assertEqual(depth_fn(16), 16)

  def test_return_correct_depth_with_multiplier(self):
    depth_fn = feature_map_generators.get_depth_fn(depth_multiplier=0.5,
                                                   min_depth=16)
    self.assertEqual(depth_fn(64), 32)


@parameterized.parameters(
    {'replace_pool_with_conv': False},
    {'replace_pool_with_conv': True},
)
@unittest.skipIf(tf_version.is_tf2(), 'Skipping TF1.X only test.')
class PoolingPyramidFeatureMapGeneratorTest(tf.test.TestCase):

  def test_get_expected_feature_map_shapes(self, replace_pool_with_conv):
    image_features = {
        'image_features': tf.random_uniform([4, 19, 19, 1024])
    }
    feature_maps = feature_map_generators.pooling_pyramid_feature_maps(
        base_feature_map_depth=1024,
        num_layers=6,
        image_features=image_features,
        replace_pool_with_conv=replace_pool_with_conv)

    expected_pool_feature_map_shapes = {
        'Base_Conv2d_1x1_1024': (4, 19, 19, 1024),
        'MaxPool2d_0_2x2': (4, 10, 10, 1024),
        'MaxPool2d_1_2x2': (4, 5, 5, 1024),
        'MaxPool2d_2_2x2': (4, 3, 3, 1024),
        'MaxPool2d_3_2x2': (4, 2, 2, 1024),
        'MaxPool2d_4_2x2': (4, 1, 1, 1024),
    }

    expected_conv_feature_map_shapes = {
        'Base_Conv2d_1x1_1024': (4, 19, 19, 1024),
        'Conv2d_0_3x3_s2_1024': (4, 10, 10, 1024),
        'Conv2d_1_3x3_s2_1024': (4, 5, 5, 1024),
        'Conv2d_2_3x3_s2_1024': (4, 3, 3, 1024),
        'Conv2d_3_3x3_s2_1024': (4, 2, 2, 1024),
        'Conv2d_4_3x3_s2_1024': (4, 1, 1, 1024),
    }

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      out_feature_maps = sess.run(feature_maps)
      out_feature_map_shapes = {key: value.shape
                                for key, value in out_feature_maps.items()}
      if replace_pool_with_conv:
        self.assertDictEqual(expected_conv_feature_map_shapes,
                             out_feature_map_shapes)
      else:
        self.assertDictEqual(expected_pool_feature_map_shapes,
                             out_feature_map_shapes)

  def test_get_expected_variable_names(self, replace_pool_with_conv):
    image_features = {
        'image_features': tf.random_uniform([4, 19, 19, 1024])
    }
    feature_maps = feature_map_generators.pooling_pyramid_feature_maps(
        base_feature_map_depth=1024,
        num_layers=6,
        image_features=image_features,
        replace_pool_with_conv=replace_pool_with_conv)

    expected_pool_variables = set([
        'Base_Conv2d_1x1_1024/weights',
        'Base_Conv2d_1x1_1024/biases',
    ])

    expected_conv_variables = set([
        'Base_Conv2d_1x1_1024/weights',
        'Base_Conv2d_1x1_1024/biases',
        'Conv2d_0_3x3_s2_1024/weights',
        'Conv2d_0_3x3_s2_1024/biases',
        'Conv2d_1_3x3_s2_1024/weights',
        'Conv2d_1_3x3_s2_1024/biases',
        'Conv2d_2_3x3_s2_1024/weights',
        'Conv2d_2_3x3_s2_1024/biases',
        'Conv2d_3_3x3_s2_1024/weights',
        'Conv2d_3_3x3_s2_1024/biases',
        'Conv2d_4_3x3_s2_1024/weights',
        'Conv2d_4_3x3_s2_1024/biases',
    ])

    init_op = tf.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      sess.run(feature_maps)
      actual_variable_set = set(
          [var.op.name for var in tf.trainable_variables()])
      if replace_pool_with_conv:
        self.assertSetEqual(expected_conv_variables, actual_variable_set)
      else:
        self.assertSetEqual(expected_pool_variables, actual_variable_set)


if __name__ == '__main__':
  tf.test.main()
