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
"""Tests for DELF feature aggregation similarity."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from delf import aggregation_config_pb2
from delf import feature_aggregation_similarity


class FeatureAggregationSimilarityTest(tf.test.TestCase):

  def testComputeVladSimilarityWorks(self):
    # Construct inputs.
    vlad_1 = np.array([0, 1, 2, 3, 4])
    vlad_2 = np.array([5, 6, 7, 8, 9])
    config = aggregation_config_pb2.AggregationConfig()
    config.aggregation_type = aggregation_config_pb2.AggregationConfig.VLAD

    # Run tested function.
    similarity_computer = (
        feature_aggregation_similarity.SimilarityAggregatedRepresentation(
            config))
    similarity = similarity_computer.ComputeSimilarity(vlad_1, vlad_2)

    # Define expected results.
    exp_similarity = 80

    # Compare actual and expected results.
    self.assertAllEqual(similarity, exp_similarity)

  def testComputeAsmkSimilarityWorks(self):
    # Construct inputs.
    aggregated_descriptors_1 = np.array([
        0.0, 0.0, -0.707107, -0.707107, 0.5, 0.866025, 0.816497, 0.577350, 1.0,
        0.0
    ])
    visual_words_1 = np.array([0, 1, 2, 3, 4])
    aggregated_descriptors_2 = np.array(
        [0.0, 1.0, 1.0, 0.0, 0.707107, 0.707107])
    visual_words_2 = np.array([1, 2, 4])
    config = aggregation_config_pb2.AggregationConfig()
    config.codebook_size = 5
    config.feature_dimensionality = 2
    config.aggregation_type = aggregation_config_pb2.AggregationConfig.ASMK
    config.use_l2_normalization = True

    # Run tested function.
    similarity_computer = (
        feature_aggregation_similarity.SimilarityAggregatedRepresentation(
            config))
    similarity = similarity_computer.ComputeSimilarity(
        aggregated_descriptors_1, aggregated_descriptors_2, visual_words_1,
        visual_words_2)

    # Define expected results.
    exp_similarity = 0.123562

    # Compare actual and expected results.
    self.assertAllClose(similarity, exp_similarity)

  def testComputeAsmkSimilarityNoNormalizationWorks(self):
    # Construct inputs.
    aggregated_descriptors_1 = np.array([
        0.0, 0.0, -0.707107, -0.707107, 0.5, 0.866025, 0.816497, 0.577350, 1.0,
        0.0
    ])
    visual_words_1 = np.array([0, 1, 2, 3, 4])
    aggregated_descriptors_2 = np.array(
        [0.0, 1.0, 1.0, 0.0, 0.707107, 0.707107])
    visual_words_2 = np.array([1, 2, 4])
    config = aggregation_config_pb2.AggregationConfig()
    config.codebook_size = 5
    config.feature_dimensionality = 2
    config.aggregation_type = aggregation_config_pb2.AggregationConfig.ASMK
    config.use_l2_normalization = False

    # Run tested function.
    similarity_computer = (
        feature_aggregation_similarity.SimilarityAggregatedRepresentation(
            config))
    similarity = similarity_computer.ComputeSimilarity(
        aggregated_descriptors_1, aggregated_descriptors_2, visual_words_1,
        visual_words_2)

    # Define expected results.
    exp_similarity = 0.478554

    # Compare actual and expected results.
    self.assertAllClose(similarity, exp_similarity)

  def testComputeAsmkStarSimilarityWorks(self):
    # Construct inputs.
    aggregated_descriptors_1 = np.array([0, 0, 3, 3, 3], dtype='uint8')
    visual_words_1 = np.array([0, 1, 2, 3, 4])
    aggregated_descriptors_2 = np.array([1, 2, 3], dtype='uint8')
    visual_words_2 = np.array([1, 2, 4])
    config = aggregation_config_pb2.AggregationConfig()
    config.codebook_size = 5
    config.feature_dimensionality = 2
    config.aggregation_type = aggregation_config_pb2.AggregationConfig.ASMK_STAR
    config.use_l2_normalization = True

    # Run tested function.
    similarity_computer = (
        feature_aggregation_similarity.SimilarityAggregatedRepresentation(
            config))
    similarity = similarity_computer.ComputeSimilarity(
        aggregated_descriptors_1, aggregated_descriptors_2, visual_words_1,
        visual_words_2)

    # Define expected results.
    exp_similarity = 0.258199

    # Compare actual and expected results.
    self.assertAllClose(similarity, exp_similarity)


if __name__ == '__main__':
  tf.test.main()
