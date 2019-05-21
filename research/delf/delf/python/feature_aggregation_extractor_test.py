# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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
"""Tests for DELF feature aggregation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from delf import aggregation_config_pb2
from delf import feature_aggregation_extractor


class FeatureAggregationTest(tf.test.TestCase):

  def _CreateCodebook(self, checkpoint_path):
    """Creates codebook used in tests.

    Args:
      checkpoint_path: Directory where codebook is saved to.
    """
    with tf.Graph().as_default() as g, self.session(graph=g) as sess:
      codebook = tf.Variable(
          [[0.5, 0.5], [0.0, 0.0], [1.0, 0.0], [-0.5, -0.5], [0.0, 1.0]],
          name='clusters')
      saver = tf.compat.v1.train.Saver([codebook])
      sess.run(tf.compat.v1.global_variables_initializer())
      saver.save(sess, checkpoint_path)

  def setUp(self):
    self._codebook_path = os.path.join(tf.compat.v1.test.get_temp_dir(),
                                       'test_codebook')
    self._CreateCodebook(self._codebook_path)

  def testComputeNormalizedVladWorks(self):
    # Construct inputs.
    # 3 2-D features.
    features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0]], dtype=float)
    config = aggregation_config_pb2.AggregationConfig()
    config.codebook_size = 5
    config.feature_dimensionality = 2
    config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
    config.use_l2_normalization = True
    config.codebook_path = self._codebook_path
    config.num_assignments = 1

    # Run tested function.
    with tf.Graph().as_default() as g, self.session(graph=g) as sess:
      extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(
          sess, config)
      vlad, extra_output = extractor.Extract(features)

    # Define expected results.
    exp_vlad = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.316228, 0.316228, 0.632456, 0.632456
    ]
    exp_extra_output = -1

    # Compare actual and expected results.
    self.assertAllClose(vlad, exp_vlad)
    self.assertAllEqual(extra_output, exp_extra_output)

  def testComputeNormalizedVladWithBatchingWorks(self):
    # Construct inputs.
    # 3 2-D features.
    features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0]], dtype=float)
    config = aggregation_config_pb2.AggregationConfig()
    config.codebook_size = 5
    config.feature_dimensionality = 2
    config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
    config.use_l2_normalization = True
    config.codebook_path = self._codebook_path
    config.num_assignments = 1
    config.feature_batch_size = 2

    # Run tested function.
    with tf.Graph().as_default() as g, self.session(graph=g) as sess:
      extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(
          sess, config)
      vlad, extra_output = extractor.Extract(features)

    # Define expected results.
    exp_vlad = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.316228, 0.316228, 0.632456, 0.632456
    ]
    exp_extra_output = -1

    # Compare actual and expected results.
    self.assertAllClose(vlad, exp_vlad)
    self.assertAllEqual(extra_output, exp_extra_output)

  def testComputeUnnormalizedVladWorks(self):
    # Construct inputs.
    # 3 2-D features.
    features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0]], dtype=float)
    config = aggregation_config_pb2.AggregationConfig()
    config.codebook_size = 5
    config.feature_dimensionality = 2
    config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
    config.use_l2_normalization = False
    config.codebook_path = self._codebook_path
    config.num_assignments = 1

    # Run tested function.
    with tf.Graph().as_default() as g, self.session(graph=g) as sess:
      extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(
          sess, config)
      vlad, extra_output = extractor.Extract(features)

    # Define expected results.
    exp_vlad = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.5, 1.0, 1.0]
    exp_extra_output = -1

    # Compare actual and expected results.
    self.assertAllEqual(vlad, exp_vlad)
    self.assertAllEqual(extra_output, exp_extra_output)

  def testComputeUnnormalizedVladMultipleAssignmentWorks(self):
    # Construct inputs.
    # 3 2-D features.
    features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0]], dtype=float)
    config = aggregation_config_pb2.AggregationConfig()
    config.codebook_size = 5
    config.feature_dimensionality = 2
    config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
    config.use_l2_normalization = False
    config.codebook_path = self._codebook_path
    config.num_assignments = 3

    # Run tested function.
    with tf.Graph().as_default() as g, self.session(graph=g) as sess:
      extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(
          sess, config)
      vlad, extra_output = extractor.Extract(features)

    # Define expected results.
    exp_vlad = [1.0, 1.0, 0.0, 0.0, 0.0, 2.0, -0.5, 0.5, 0.0, 0.0]
    exp_extra_output = -1

    # Compare actual and expected results.
    self.assertAllEqual(vlad, exp_vlad)
    self.assertAllEqual(extra_output, exp_extra_output)

  def testComputeVladEmptyFeaturesWorks(self):
    # Construct inputs.
    # Empty feature array.
    features = np.array([[]])
    config = aggregation_config_pb2.AggregationConfig()
    config.codebook_size = 5
    config.feature_dimensionality = 2
    config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
    config.codebook_path = self._codebook_path

    # Run tested function.
    with tf.Graph().as_default() as g, self.session(graph=g) as sess:
      extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(
          sess, config)
      vlad, extra_output = extractor.Extract(features)

    # Define expected results.
    exp_vlad = np.zeros([10], dtype=float)
    exp_extra_output = -1

    # Compare actual and expected results.
    self.assertAllEqual(vlad, exp_vlad)
    self.assertAllEqual(extra_output, exp_extra_output)

  def testComputeUnnormalizedRvladWorks(self):
    # Construct inputs.
    # 4 2-D features: 3 in first region, 1 in second region.
    features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0], [0.0, 2.0]],
                        dtype=float)
    num_features_per_region = np.array([3, 1])
    config = aggregation_config_pb2.AggregationConfig()
    config.codebook_size = 5
    config.feature_dimensionality = 2
    config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
    config.use_l2_normalization = False
    config.codebook_path = self._codebook_path
    config.num_assignments = 1
    config.use_regional_aggregation = True

    # Run tested function.
    with tf.Graph().as_default() as g, self.session(graph=g) as sess:
      extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(
          sess, config)
      rvlad, extra_output = extractor.Extract(features, num_features_per_region)

    # Define expected results.
    exp_rvlad = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.158114, 0.158114, 0.316228, 0.816228
    ]
    exp_extra_output = -1

    # Compare actual and expected results.
    self.assertAllClose(rvlad, exp_rvlad)
    self.assertAllEqual(extra_output, exp_extra_output)

  def testComputeNormalizedRvladWorks(self):
    # Construct inputs.
    # 4 2-D features: 3 in first region, 1 in second region.
    features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0], [0.0, 2.0]],
                        dtype=float)
    num_features_per_region = np.array([3, 1])
    config = aggregation_config_pb2.AggregationConfig()
    config.codebook_size = 5
    config.feature_dimensionality = 2
    config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
    config.use_l2_normalization = True
    config.codebook_path = self._codebook_path
    config.num_assignments = 1
    config.use_regional_aggregation = True

    # Run tested function.
    with tf.Graph().as_default() as g, self.session(graph=g) as sess:
      extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(
          sess, config)
      rvlad, extra_output = extractor.Extract(features, num_features_per_region)

    # Define expected results.
    exp_rvlad = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.175011, 0.175011, 0.350021, 0.903453
    ]
    exp_extra_output = -1

    # Compare actual and expected results.
    self.assertAllClose(rvlad, exp_rvlad)
    self.assertAllEqual(extra_output, exp_extra_output)

  def testComputeRvladEmptyRegionsWorks(self):
    # Construct inputs.
    # Empty feature array.
    features = np.array([[]])
    num_features_per_region = np.array([])
    config = aggregation_config_pb2.AggregationConfig()
    config.codebook_size = 5
    config.feature_dimensionality = 2
    config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
    config.codebook_path = self._codebook_path
    config.use_regional_aggregation = True

    # Run tested function.
    with tf.Graph().as_default() as g, self.session(graph=g) as sess:
      extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(
          sess, config)
      rvlad, extra_output = extractor.Extract(features, num_features_per_region)

    # Define expected results.
    exp_rvlad = np.zeros([10], dtype=float)
    exp_extra_output = -1

    # Compare actual and expected results.
    self.assertAllEqual(rvlad, exp_rvlad)
    self.assertAllEqual(extra_output, exp_extra_output)

  def testComputeUnnormalizedRvladSomeEmptyRegionsWorks(self):
    # Construct inputs.
    # 4 2-D features: 0 in first region, 3 in second region, 0 in third region,
    # 1 in fourth region.
    features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0], [0.0, 2.0]],
                        dtype=float)
    num_features_per_region = np.array([0, 3, 0, 1])
    config = aggregation_config_pb2.AggregationConfig()
    config.codebook_size = 5
    config.feature_dimensionality = 2
    config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
    config.use_l2_normalization = False
    config.codebook_path = self._codebook_path
    config.num_assignments = 1
    config.use_regional_aggregation = True

    # Run tested function.
    with tf.Graph().as_default() as g, self.session(graph=g) as sess:
      extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(
          sess, config)
      rvlad, extra_output = extractor.Extract(features, num_features_per_region)

    # Define expected results.
    exp_rvlad = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.079057, 0.079057, 0.158114, 0.408114
    ]
    exp_extra_output = -1

    # Compare actual and expected results.
    self.assertAllClose(rvlad, exp_rvlad)
    self.assertAllEqual(extra_output, exp_extra_output)

  def testComputeNormalizedRvladSomeEmptyRegionsWorks(self):
    # Construct inputs.
    # 4 2-D features: 0 in first region, 3 in second region, 0 in third region,
    # 1 in fourth region.
    features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0], [0.0, 2.0]],
                        dtype=float)
    num_features_per_region = np.array([0, 3, 0, 1])
    config = aggregation_config_pb2.AggregationConfig()
    config.codebook_size = 5
    config.feature_dimensionality = 2
    config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
    config.use_l2_normalization = True
    config.codebook_path = self._codebook_path
    config.num_assignments = 1
    config.use_regional_aggregation = True

    # Run tested function.
    with tf.Graph().as_default() as g, self.session(graph=g) as sess:
      extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(
          sess, config)
      rvlad, extra_output = extractor.Extract(features, num_features_per_region)

    # Define expected results.
    exp_rvlad = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.175011, 0.175011, 0.350021, 0.903453
    ]
    exp_extra_output = -1

    # Compare actual and expected results.
    self.assertAllClose(rvlad, exp_rvlad)
    self.assertAllEqual(extra_output, exp_extra_output)

  def testComputeRvladMisconfiguredFeatures(self):
    # Construct inputs.
    # 4 2-D features: 3 in first region, 1 in second region.
    features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0], [0.0, 2.0]],
                        dtype=float)
    # Misconfigured number of features; there are only 4 features, but
    # sum(num_features_per_region) = 5.
    num_features_per_region = np.array([3, 2])
    config = aggregation_config_pb2.AggregationConfig()
    config.codebook_size = 5
    config.feature_dimensionality = 2
    config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD
    config.codebook_path = self._codebook_path
    config.use_regional_aggregation = True

    # Run tested function.
    with tf.Graph().as_default() as g, self.session(graph=g) as sess:
      extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(
          sess, config)
      with self.assertRaisesRegex(
          ValueError,
          r'Incorrect arguments: sum\(num_features_per_region\) and '
          r'features.shape\[0\] are different'):
        extractor.Extract(features, num_features_per_region)

  def testComputeAsmkWorks(self):
    # Construct inputs.
    # 3 2-D features.
    features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0]], dtype=float)
    config = aggregation_config_pb2.AggregationConfig()
    config.codebook_size = 5
    config.feature_dimensionality = 2
    config.aggregation_type = aggregation_config_pb2.AggregationConfig.ASMK
    config.codebook_path = self._codebook_path
    config.num_assignments = 1

    # Run tested function.
    with tf.Graph().as_default() as g, self.session(graph=g) as sess:
      extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(
          sess, config)
      asmk, visual_words = extractor.Extract(features)

    # Define expected results.
    exp_asmk = [-0.707107, 0.707107, 0.707107, 0.707107]
    exp_visual_words = [3, 4]

    # Compare actual and expected results.
    self.assertAllClose(asmk, exp_asmk)
    self.assertAllEqual(visual_words, exp_visual_words)

  def testComputeAsmkStarWorks(self):
    # Construct inputs.
    # 3 2-D features.
    features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0]], dtype=float)
    config = aggregation_config_pb2.AggregationConfig()
    config.codebook_size = 5
    config.feature_dimensionality = 2
    config.aggregation_type = aggregation_config_pb2.AggregationConfig.ASMK_STAR
    config.codebook_path = self._codebook_path
    config.num_assignments = 1

    # Run tested function.
    with tf.Graph().as_default() as g, self.session(graph=g) as sess:
      extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(
          sess, config)
      asmk_star, visual_words = extractor.Extract(features)

    # Define expected results.
    exp_asmk_star = [64, 192]
    exp_visual_words = [3, 4]

    # Compare actual and expected results.
    self.assertAllEqual(asmk_star, exp_asmk_star)
    self.assertAllEqual(visual_words, exp_visual_words)

  def testComputeAsmkMultipleAssignmentWorks(self):
    # Construct inputs.
    # 3 2-D features.
    features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0]], dtype=float)
    config = aggregation_config_pb2.AggregationConfig()
    config.codebook_size = 5
    config.feature_dimensionality = 2
    config.aggregation_type = aggregation_config_pb2.AggregationConfig.ASMK
    config.codebook_path = self._codebook_path
    config.num_assignments = 3

    # Run tested function.
    with tf.Graph().as_default() as g, self.session(graph=g) as sess:
      extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(
          sess, config)
      asmk, visual_words = extractor.Extract(features)

    # Define expected results.
    exp_asmk = [0.707107, 0.707107, 0.0, 1.0, -0.707107, 0.707107]
    exp_visual_words = [0, 2, 3]

    # Compare actual and expected results.
    self.assertAllClose(asmk, exp_asmk)
    self.assertAllEqual(visual_words, exp_visual_words)

  def testComputeRasmkWorks(self):
    # Construct inputs.
    # 4 2-D features: 3 in first region, 1 in second region.
    features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0], [0.0, 2.0]],
                        dtype=float)
    num_features_per_region = np.array([3, 1])
    config = aggregation_config_pb2.AggregationConfig()
    config.codebook_size = 5
    config.feature_dimensionality = 2
    config.aggregation_type = aggregation_config_pb2.AggregationConfig.ASMK
    config.codebook_path = self._codebook_path
    config.num_assignments = 1
    config.use_regional_aggregation = True

    # Run tested function.
    with tf.Graph().as_default() as g, self.session(graph=g) as sess:
      extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(
          sess, config)
      rasmk, visual_words = extractor.Extract(features, num_features_per_region)

    # Define expected results.
    exp_rasmk = [-0.707107, 0.707107, 0.361261, 0.932465]
    exp_visual_words = [3, 4]

    # Compare actual and expected results.
    self.assertAllClose(rasmk, exp_rasmk)
    self.assertAllEqual(visual_words, exp_visual_words)

  def testComputeRasmkStarWorks(self):
    # Construct inputs.
    # 4 2-D features: 3 in first region, 1 in second region.
    features = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 2.0], [0.0, 2.0]],
                        dtype=float)
    num_features_per_region = np.array([3, 1])
    config = aggregation_config_pb2.AggregationConfig()
    config.codebook_size = 5
    config.feature_dimensionality = 2
    config.aggregation_type = aggregation_config_pb2.AggregationConfig.ASMK_STAR
    config.codebook_path = self._codebook_path
    config.num_assignments = 1
    config.use_regional_aggregation = True

    # Run tested function.
    with tf.Graph().as_default() as g, self.session(graph=g) as sess:
      extractor = feature_aggregation_extractor.ExtractAggregatedRepresentation(
          sess, config)
      rasmk_star, visual_words = extractor.Extract(features,
                                                   num_features_per_region)

    # Define expected results.
    exp_rasmk_star = [64, 192]
    exp_visual_words = [3, 4]

    # Compare actual and expected results.
    self.assertAllEqual(rasmk_star, exp_rasmk_star)
    self.assertAllEqual(visual_words, exp_visual_words)

  def testComputeUnknownAggregation(self):
    # Construct inputs.
    config = aggregation_config_pb2.AggregationConfig()
    config.codebook_size = 5
    config.feature_dimensionality = 2
    config.aggregation_type = 0
    config.codebook_path = self._codebook_path
    config.use_regional_aggregation = True

    # Run tested function.
    with tf.Graph().as_default() as g, self.session(graph=g) as sess:
      with self.assertRaisesRegex(ValueError, 'Invalid aggregation type'):
        feature_aggregation_extractor.ExtractAggregatedRepresentation(
            sess, config)


if __name__ == '__main__':
  tf.test.main()
