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

from abc import abstractmethod

import itertools
import numpy as np
import tensorflow as tf

from object_detection.utils import test_case


class SsdFeatureExtractorTestBase(test_case.TestCase):

  @abstractmethod
  def _create_feature_extractor(self, depth_multiplier, pad_to_multiple,
                                use_explicit_padding=False):
    """Constructs a new feature extractor.

    Args:
      depth_multiplier: float depth multiplier for feature extractor
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      use_explicit_padding: use 'VALID' padding for convolutions, but prepad
        inputs so that the output dimensions are the same as if 'SAME' padding
        were used.
    Returns:
      an ssd_meta_arch.SSDFeatureExtractor object.
    """
    pass

  def check_extract_features_returns_correct_shape(
      self, batch_size, image_height, image_width, depth_multiplier,
      pad_to_multiple, expected_feature_map_shapes, use_explicit_padding=False):
    def graph_fn(image_tensor):
      feature_extractor = self._create_feature_extractor(depth_multiplier,
                                                         pad_to_multiple,
                                                         use_explicit_padding)
      feature_maps = feature_extractor.extract_features(image_tensor)
      return feature_maps

    image_tensor = np.random.rand(batch_size, image_height, image_width,
                                  3).astype(np.float32)
    feature_maps = self.execute(graph_fn, [image_tensor])
    for feature_map, expected_shape in itertools.izip(
        feature_maps, expected_feature_map_shapes):
      self.assertAllEqual(feature_map.shape, expected_shape)

  def check_extract_features_returns_correct_shapes_with_dynamic_inputs(
      self, batch_size, image_height, image_width, depth_multiplier,
      pad_to_multiple, expected_feature_map_shapes, use_explicit_padding=False):
    def graph_fn(image_height, image_width):
      feature_extractor = self._create_feature_extractor(depth_multiplier,
                                                         pad_to_multiple,
                                                         use_explicit_padding)
      image_tensor = tf.random_uniform([batch_size, image_height, image_width,
                                        3], dtype=tf.float32)
      feature_maps = feature_extractor.extract_features(image_tensor)
      return feature_maps

    feature_maps = self.execute_cpu(graph_fn, [
        np.array(image_height, dtype=np.int32),
        np.array(image_width, dtype=np.int32)
    ])
    for feature_map, expected_shape in itertools.izip(
        feature_maps, expected_feature_map_shapes):
      self.assertAllEqual(feature_map.shape, expected_shape)

  def check_extract_features_raises_error_with_invalid_image_size(
      self, image_height, image_width, depth_multiplier, pad_to_multiple):
    feature_extractor = self._create_feature_extractor(depth_multiplier,
                                                       pad_to_multiple)
    preprocessed_inputs = tf.placeholder(tf.float32, (4, None, None, 3))
    feature_maps = feature_extractor.extract_features(preprocessed_inputs)
    test_preprocessed_image = np.random.rand(4, image_height, image_width, 3)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      with self.assertRaises(tf.errors.InvalidArgumentError):
        sess.run(feature_maps,
                 feed_dict={preprocessed_inputs: test_preprocessed_image})

  def check_feature_extractor_variables_under_scope(
      self, depth_multiplier, pad_to_multiple, scope_name):
    g = tf.Graph()
    with g.as_default():
      feature_extractor = self._create_feature_extractor(
          depth_multiplier, pad_to_multiple)
      preprocessed_inputs = tf.placeholder(tf.float32, (4, None, None, 3))
      feature_extractor.extract_features(preprocessed_inputs)
      variables = g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      for variable in variables:
        self.assertTrue(variable.name.startswith(scope_name))
