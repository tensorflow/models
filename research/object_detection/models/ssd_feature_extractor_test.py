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

"""Base test class SSDFeatureExtractors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod

import numpy as np
from six.moves import zip
import tensorflow.compat.v1 as tf
import tf_slim as slim
from google.protobuf import text_format

from object_detection.builders import hyperparams_builder
from object_detection.protos import hyperparams_pb2
from object_detection.utils import test_case
from object_detection.utils import test_utils


class SsdFeatureExtractorTestBase(test_case.TestCase):

  def _build_conv_hyperparams(self, add_batch_norm=True):
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
    """
    if add_batch_norm:
      batch_norm_proto = """
        batch_norm {
          scale: false
        }
      """
      conv_hyperparams_text_proto += batch_norm_proto
    text_format.Merge(conv_hyperparams_text_proto, conv_hyperparams)
    return hyperparams_builder.KerasLayerHyperparams(conv_hyperparams)

  def conv_hyperparams_fn(self):
    with slim.arg_scope([]) as sc:
      return sc

  @abstractmethod
  def _create_feature_extractor(self,
                                depth_multiplier,
                                pad_to_multiple,
                                use_explicit_padding=False,
                                num_layers=6,
                                use_keras=False,
                                use_depthwise=False):
    """Constructs a new feature extractor.

    Args:
      depth_multiplier: float depth multiplier for feature extractor
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      use_explicit_padding: use 'VALID' padding for convolutions, but prepad
        inputs so that the output dimensions are the same as if 'SAME' padding
        were used.
      num_layers: number of SSD layers.
      use_keras: if True builds a keras-based feature extractor, if False builds
        a slim-based one.
      use_depthwise: Whether to use depthwise convolutions.
    Returns:
      an ssd_meta_arch.SSDFeatureExtractor or an
      ssd_meta_arch.SSDKerasFeatureExtractor object.
    """
    pass

  def _create_features(self,
                       depth_multiplier,
                       pad_to_multiple,
                       use_explicit_padding=False,
                       use_depthwise=False,
                       num_layers=6,
                       use_keras=False):
    kwargs = {}
    if use_explicit_padding:
      kwargs.update({'use_explicit_padding': use_explicit_padding})
    if use_depthwise:
      kwargs.update({'use_depthwise': use_depthwise})
    if num_layers != 6:
      kwargs.update({'num_layers': num_layers})
    if use_keras:
      kwargs.update({'use_keras': use_keras})
    feature_extractor = self._create_feature_extractor(
        depth_multiplier,
        pad_to_multiple,
        **kwargs)
    return feature_extractor

  def _extract_features(self,
                        image_tensor,
                        feature_extractor,
                        use_keras=False):
    if use_keras:
      feature_maps = feature_extractor(image_tensor)
    else:
      feature_maps = feature_extractor.extract_features(image_tensor)
    return feature_maps

  def check_extract_features_returns_correct_shape(self,
                                                   batch_size,
                                                   image_height,
                                                   image_width,
                                                   depth_multiplier,
                                                   pad_to_multiple,
                                                   expected_feature_map_shapes,
                                                   use_explicit_padding=False,
                                                   num_layers=6,
                                                   use_keras=False,
                                                   use_depthwise=False):
    with test_utils.GraphContextOrNone() as g:
      feature_extractor = self._create_features(
          depth_multiplier,
          pad_to_multiple,
          use_explicit_padding=use_explicit_padding,
          num_layers=num_layers,
          use_keras=use_keras,
          use_depthwise=use_depthwise)

    def graph_fn(image_tensor):
      return self._extract_features(
          image_tensor,
          feature_extractor,
          use_keras=use_keras)

    image_tensor = np.random.rand(batch_size, image_height, image_width,
                                  3).astype(np.float32)
    feature_maps = self.execute(graph_fn, [image_tensor], graph=g)
    for feature_map, expected_shape in zip(
        feature_maps, expected_feature_map_shapes):
      self.assertAllEqual(feature_map.shape, expected_shape)

  def check_extract_features_returns_correct_shapes_with_dynamic_inputs(
      self,
      batch_size,
      image_height,
      image_width,
      depth_multiplier,
      pad_to_multiple,
      expected_feature_map_shapes,
      use_explicit_padding=False,
      num_layers=6,
      use_keras=False,
      use_depthwise=False):

    with test_utils.GraphContextOrNone() as g:
      feature_extractor = self._create_features(
          depth_multiplier,
          pad_to_multiple,
          use_explicit_padding=use_explicit_padding,
          num_layers=num_layers,
          use_keras=use_keras,
          use_depthwise=use_depthwise)

    def graph_fn(image_height, image_width):
      image_tensor = tf.random_uniform([batch_size, image_height, image_width,
                                        3], dtype=tf.float32)
      return self._extract_features(
          image_tensor,
          feature_extractor,
          use_keras=use_keras)

    feature_maps = self.execute_cpu(graph_fn, [
        np.array(image_height, dtype=np.int32),
        np.array(image_width, dtype=np.int32)
    ], graph=g)
    for feature_map, expected_shape in zip(
        feature_maps, expected_feature_map_shapes):
      self.assertAllEqual(feature_map.shape, expected_shape)

  def check_extract_features_raises_error_with_invalid_image_size(
      self,
      image_height,
      image_width,
      depth_multiplier,
      pad_to_multiple,
      use_keras=False,
      use_depthwise=False):

    with test_utils.GraphContextOrNone() as g:
      batch = 4
      width = tf.random.uniform([], minval=image_width, maxval=image_width+1,
                                dtype=tf.int32)
      height = tf.random.uniform([], minval=image_height, maxval=image_height+1,
                                 dtype=tf.int32)
      shape = tf.stack([batch, height, width, 3])
      preprocessed_inputs = tf.random.uniform(shape)
      feature_extractor = self._create_features(
          depth_multiplier,
          pad_to_multiple,
          use_keras=use_keras,
          use_depthwise=use_depthwise)

    def graph_fn():
      feature_maps = self._extract_features(
          preprocessed_inputs,
          feature_extractor,
          use_keras=use_keras)
      return feature_maps
    if self.is_tf2():
      with self.assertRaises(ValueError):
        self.execute_cpu(graph_fn, [], graph=g)
    else:
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.execute_cpu(graph_fn, [], graph=g)

  def check_feature_extractor_variables_under_scope(self,
                                                    depth_multiplier,
                                                    pad_to_multiple,
                                                    scope_name,
                                                    use_keras=False,
                                                    use_depthwise=False):
    variables = self.get_feature_extractor_variables(
        depth_multiplier,
        pad_to_multiple,
        use_keras=use_keras,
        use_depthwise=use_depthwise)
    for variable in variables:
      self.assertTrue(variable.name.startswith(scope_name))

  def get_feature_extractor_variables(self,
                                      depth_multiplier,
                                      pad_to_multiple,
                                      use_keras=False,
                                      use_depthwise=False):
    g = tf.Graph()
    with g.as_default():
      feature_extractor = self._create_features(
          depth_multiplier,
          pad_to_multiple,
          use_keras=use_keras,
          use_depthwise=use_depthwise)
      preprocessed_inputs = tf.placeholder(tf.float32, (4, None, None, 3))
      self._extract_features(
          preprocessed_inputs,
          feature_extractor,
          use_keras=use_keras)
      return g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
