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
"""Local feature aggregation extraction.

For more details, please refer to the paper:
"Detect-to-Retrieve: Efficient Regional Aggregation for Image Search",
Proc. CVPR'19 (https://arxiv.org/abs/1812.01584).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from delf import aggregation_config_pb2

_CLUSTER_CENTERS_VAR_NAME = "clusters"
_NORM_SQUARED_TOLERANCE = 1e-12

# Aliases for aggregation types.
_VLAD = aggregation_config_pb2.AggregationConfig.VLAD
_ASMK = aggregation_config_pb2.AggregationConfig.ASMK
_ASMK_STAR = aggregation_config_pb2.AggregationConfig.ASMK_STAR


class ExtractAggregatedRepresentation(object):
  """Class for extraction of aggregated local feature representation.

  Args:
    aggregation_config: AggregationConfig object defining type of aggregation to
      use.

  Raises:
    ValueError: If aggregation type is invalid.
  """

  def __init__(self, aggregation_config):
    self._codebook_size = aggregation_config.codebook_size
    self._feature_dimensionality = aggregation_config.feature_dimensionality
    self._aggregation_type = aggregation_config.aggregation_type
    self._feature_batch_size = aggregation_config.feature_batch_size
    self._codebook_path = aggregation_config.codebook_path
    self._use_regional_aggregation = aggregation_config.use_regional_aggregation
    self._use_l2_normalization = aggregation_config.use_l2_normalization
    self._num_assignments = aggregation_config.num_assignments

    if self._aggregation_type  not in [_VLAD, _ASMK, _ASMK_STAR]:
      raise ValueError("Invalid aggregation type: %d" % self._aggregation_type)

    # Load codebook
    codebook = tf.Variable(
        tf.zeros([self._codebook_size, self._feature_dimensionality],
                 dtype=tf.float32),
        name=_CLUSTER_CENTERS_VAR_NAME)
    ckpt = tf.train.Checkpoint(codebook=codebook)
    ckpt.restore(self._codebook_path)

    self._codebook = codebook

  def Extract(self, features, num_features_per_region=None):
    """Extracts aggregated representation.

    Args:
      features: [N, D] float numpy array with N local feature descriptors.
      num_features_per_region: Required only if computing regional aggregated
        representations, otherwise optional. List of number of features per
        region, such that sum(num_features_per_region) = N. It indicates which
        features correspond to each region.

    Returns:
      aggregated_descriptors: 1-D numpy array.
      feature_visual_words: Used only for ASMK/ASMK* aggregation type. 1-D
        numpy array denoting visual words corresponding to the
        `aggregated_descriptors`.

    Raises:
      ValueError: If inputs are misconfigured.
    """
    features = tf.cast(features, dtype=tf.float32)

    if num_features_per_region is None:
      # Use dummy value since it is unused.
      num_features_per_region = []
    else:
      num_features_per_region = tf.cast(num_features_per_region, dtype=tf.int32)
      if len(num_features_per_region
            ) and sum(num_features_per_region) != features.shape[0]:
        raise ValueError(
            "Incorrect arguments: sum(num_features_per_region) and "
            "features.shape[0] are different: %d vs %d" %
            (sum(num_features_per_region), features.shape[0]))

    # Extract features based on desired options.
    if self._aggregation_type == _VLAD:
      # Feature visual words are unused in the case of VLAD, so just return
      # dummy constant.
      feature_visual_words = tf.constant(-1, dtype=tf.int32)
      if self._use_regional_aggregation:
        aggregated_descriptors = self._ComputeRvlad(
            features,
            num_features_per_region,
            self._codebook,
            use_l2_normalization=self._use_l2_normalization,
            num_assignments=self._num_assignments)
      else:
        aggregated_descriptors = self._ComputeVlad(
            features,
            self._codebook,
            use_l2_normalization=self._use_l2_normalization,
            num_assignments=self._num_assignments)
    elif (self._aggregation_type == _ASMK or
          self._aggregation_type == _ASMK_STAR):
      if self._use_regional_aggregation:
        (aggregated_descriptors,
         feature_visual_words) = self._ComputeRasmk(
             features,
             num_features_per_region,
             self._codebook,
             num_assignments=self._num_assignments)
      else:
        (aggregated_descriptors,
         feature_visual_words) = self._ComputeAsmk(
             features,
             self._codebook,
             num_assignments=self._num_assignments)

    feature_visual_words_output = feature_visual_words.numpy()

    # If using ASMK*/RASMK*, binarize the aggregated descriptors.
    if self._aggregation_type == _ASMK_STAR:
      reshaped_aggregated_descriptors = np.reshape(
          aggregated_descriptors, [-1, self._feature_dimensionality])
      packed_descriptors = np.packbits(
          reshaped_aggregated_descriptors > 0, axis=1)
      aggregated_descriptors_output = np.reshape(packed_descriptors, [-1])
    else:
      aggregated_descriptors_output = aggregated_descriptors.numpy()

    return aggregated_descriptors_output, feature_visual_words_output

  def _ComputeVlad(self,
                   features,
                   codebook,
                   use_l2_normalization=True,
                   num_assignments=1):
    """Compute VLAD representation.

    Args:
      features: [N, D] float tensor.
      codebook: [K, D] float tensor.
      use_l2_normalization: If False, does not L2-normalize after aggregation.
      num_assignments: Number of visual words to assign a feature to.

    Returns:
      vlad: [K*D] float tensor.
    """

    def _ComputeVladEmptyFeatures():
      """Computes VLAD if `features` is empty.

      Returns:
        [K*D] all-zeros tensor.
      """
      return tf.zeros([self._codebook_size * self._feature_dimensionality],
                      dtype=tf.float32)

    def _ComputeVladNonEmptyFeatures():
      """Computes VLAD if `features` is not empty.

      Returns:
        [K*D] tensor with VLAD descriptor.
      """
      num_features = tf.shape(features)[0]

      # Find nearest visual words for each feature. Possibly batch the local
      # features to avoid OOM.
      if self._feature_batch_size <= 0:
        actual_batch_size = num_features
      else:
        actual_batch_size = self._feature_batch_size

      def _BatchNearestVisualWords(ind, selected_visual_words):
        """Compute nearest neighbor visual words for a batch of features.

        Args:
          ind: Integer index denoting feature.
          selected_visual_words: Partial set of visual words.

        Returns:
          output_ind: Next index.
          output_selected_visual_words: Updated set of visual words, including
            the visual words for the new batch.
        """
        # Handle case of last batch, where there may be fewer than
        # `actual_batch_size` features.
        batch_size_to_use = tf.cond(
            tf.greater(ind + actual_batch_size, num_features),
            true_fn=lambda: num_features - ind,
            false_fn=lambda: actual_batch_size)

        # Denote B = batch_size_to_use.
        # K*B x D.
        tiled_features = tf.reshape(
            tf.tile(
                tf.slice(features, [ind, 0],
                         [batch_size_to_use, self._feature_dimensionality]),
                [1, self._codebook_size]), [-1, self._feature_dimensionality])
        # K*B x D.
        tiled_codebook = tf.reshape(
            tf.tile(tf.reshape(codebook, [1, -1]), [batch_size_to_use, 1]),
            [-1, self._feature_dimensionality])
        # B x K.
        squared_distances = tf.reshape(
            tf.reduce_sum(
                tf.math.squared_difference(tiled_features, tiled_codebook),
                axis=1), [batch_size_to_use, self._codebook_size])
        # B x K.
        nearest_visual_words = tf.argsort(squared_distances)
        # B x num_assignments.
        batch_selected_visual_words = tf.slice(
            nearest_visual_words, [0, 0], [batch_size_to_use, num_assignments])
        selected_visual_words = tf.concat(
            [selected_visual_words, batch_selected_visual_words], axis=0)

        return ind + batch_size_to_use, selected_visual_words

      ind_batch = tf.constant(0, dtype=tf.int32)
      keep_going = lambda j, selected_visual_words: tf.less(j, num_features)
      selected_visual_words = tf.zeros([0, num_assignments], dtype=tf.int32)
      _, selected_visual_words = tf.while_loop(
          cond=keep_going,
          body=_BatchNearestVisualWords,
          loop_vars=[ind_batch, selected_visual_words],
          shape_invariants=[
              ind_batch.get_shape(),
              tf.TensorShape([None, num_assignments])
          ],
          parallel_iterations=1,
          back_prop=False)

      # Helper function to collect residuals for relevant visual words.
      def _ConstructVladFromAssignments(ind, vlad):
        """Add contributions of a feature to a VLAD descriptor.

        Args:
          ind: Integer index denoting feature.
          vlad: Partial VLAD descriptor.

        Returns:
          output_ind: Next index (ie, ind+1).
          output_vlad: VLAD descriptor updated to take into account contribution
            from ind-th feature.
        """
        diff = tf.tile(
            tf.expand_dims(features[ind],
                           axis=0), [num_assignments, 1]) - tf.gather(
                               codebook, selected_visual_words[ind])
        return ind + 1, tf.tensor_scatter_nd_add(
            vlad, tf.expand_dims(selected_visual_words[ind], axis=1), diff)

      ind_vlad = tf.constant(0, dtype=tf.int32)
      keep_going = lambda j, vlad: tf.less(j, num_features)
      vlad = tf.zeros([self._codebook_size, self._feature_dimensionality],
                      dtype=tf.float32)
      _, vlad = tf.while_loop(
          cond=keep_going,
          body=_ConstructVladFromAssignments,
          loop_vars=[ind_vlad, vlad],
          back_prop=False)

      vlad = tf.reshape(vlad,
                        [self._codebook_size * self._feature_dimensionality])
      if use_l2_normalization:
        vlad = tf.math.l2_normalize(vlad, epsilon=_NORM_SQUARED_TOLERANCE)

      return vlad

    return tf.cond(
        tf.greater(tf.size(features), 0),
        true_fn=_ComputeVladNonEmptyFeatures,
        false_fn=_ComputeVladEmptyFeatures)

  def _ComputeRvlad(self,
                    features,
                    num_features_per_region,
                    codebook,
                    use_l2_normalization=False,
                    num_assignments=1):
    """Compute R-VLAD representation.

    Args:
      features: [N, D] float tensor.
      num_features_per_region: [R] int tensor. Contains number of features per
        region, such that sum(num_features_per_region) = N. It indicates which
        features correspond to each region.
      codebook: [K, D] float tensor.
      use_l2_normalization: If True, performs L2-normalization after regional
        aggregation; if False (default), performs componentwise division by R
        after regional aggregation.
      num_assignments: Number of visual words to assign a feature to.

    Returns:
      rvlad: [K*D] float tensor.
    """

    def _ComputeRvladEmptyRegions():
      """Computes R-VLAD if `num_features_per_region` is empty.

      Returns:
        [K*D] all-zeros tensor.
      """
      return tf.zeros([self._codebook_size * self._feature_dimensionality],
                      dtype=tf.float32)

    def _ComputeRvladNonEmptyRegions():
      """Computes R-VLAD if `num_features_per_region` is not empty.

      Returns:
        [K*D] tensor with R-VLAD descriptor.
      """

      # Helper function to compose initial R-VLAD from image regions.
      def _ConstructRvladFromVlad(ind, rvlad):
        """Add contributions from different regions into R-VLAD.

        Args:
          ind: Integer index denoting region.
          rvlad: Partial R-VLAD descriptor.

        Returns:
          output_ind: Next index (ie, ind+1).
          output_rvlad: R-VLAD descriptor updated to take into account
            contribution from ind-th region.
        """
        return ind + 1, rvlad + self._ComputeVlad(
            tf.slice(
                features, [tf.reduce_sum(num_features_per_region[:ind]), 0],
                [num_features_per_region[ind], self._feature_dimensionality]),
            codebook,
            num_assignments=num_assignments)

      i = tf.constant(0, dtype=tf.int32)
      num_regions = tf.shape(num_features_per_region)[0]
      keep_going = lambda j, rvlad: tf.less(j, num_regions)
      rvlad = tf.zeros([self._codebook_size * self._feature_dimensionality],
                       dtype=tf.float32)
      _, rvlad = tf.while_loop(
          cond=keep_going,
          body=_ConstructRvladFromVlad,
          loop_vars=[i, rvlad],
          back_prop=False,
          parallel_iterations=1)

      if use_l2_normalization:
        rvlad = tf.math.l2_normalize(rvlad, epsilon=_NORM_SQUARED_TOLERANCE)
      else:
        rvlad /= tf.cast(num_regions, dtype=tf.float32)

      return rvlad

    return tf.cond(
        tf.greater(tf.size(num_features_per_region), 0),
        true_fn=_ComputeRvladNonEmptyRegions,
        false_fn=_ComputeRvladEmptyRegions)

  def _PerCentroidNormalization(self, unnormalized_vector):
    """Perform per-centroid normalization.

    Args:
      unnormalized_vector: [KxD] float tensor.

    Returns:
      per_centroid_normalized_vector: [KxD] float tensor, with normalized
        aggregated residuals. Some residuals may be all-zero.
      visual_words: Int tensor containing indices of visual words which are
        present for the set of features.
    """
    unnormalized_vector = tf.reshape(
        unnormalized_vector,
        [self._codebook_size, self._feature_dimensionality])
    per_centroid_norms = tf.norm(unnormalized_vector, axis=1)

    visual_words = tf.reshape(
        tf.where(
            tf.greater(per_centroid_norms, tf.sqrt(_NORM_SQUARED_TOLERANCE))),
        [-1])

    per_centroid_normalized_vector = tf.math.l2_normalize(
        unnormalized_vector, axis=1, epsilon=_NORM_SQUARED_TOLERANCE)

    return per_centroid_normalized_vector, visual_words

  def _ComputeAsmk(self, features, codebook, num_assignments=1):
    """Compute ASMK representation.

    Args:
      features: [N, D] float tensor.
      codebook: [K, D] float tensor.
      num_assignments: Number of visual words to assign a feature to.

    Returns:
      normalized_residuals: 1-dimensional float tensor with concatenated
        residuals which are non-zero. Note that the dimensionality is
        input-dependent.
      visual_words: 1-dimensional int tensor of sorted visual word ids.
        Dimensionality is shape(normalized_residuals)[0] / D.
    """
    unnormalized_vlad = self._ComputeVlad(
        features,
        codebook,
        use_l2_normalization=False,
        num_assignments=num_assignments)

    per_centroid_normalized_vlad, visual_words = self._PerCentroidNormalization(
        unnormalized_vlad)

    normalized_residuals = tf.reshape(
        tf.gather(per_centroid_normalized_vlad, visual_words),
        [tf.shape(visual_words)[0] * self._feature_dimensionality])

    return normalized_residuals, visual_words

  def _ComputeRasmk(self,
                    features,
                    num_features_per_region,
                    codebook,
                    num_assignments=1):
    """Compute R-ASMK representation.

    Args:
      features: [N, D] float tensor.
      num_features_per_region: [R] int tensor. Contains number of features per
        region, such that sum(num_features_per_region) = N. It indicates which
        features correspond to each region.
      codebook: [K, D] float tensor.
      num_assignments: Number of visual words to assign a feature to.

    Returns:
      normalized_residuals: 1-dimensional float tensor with concatenated
        residuals which are non-zero. Note that the dimensionality is
        input-dependent.
      visual_words: 1-dimensional int tensor of sorted visual word ids.
        Dimensionality is shape(normalized_residuals)[0] / D.
    """
    unnormalized_rvlad = self._ComputeRvlad(
        features,
        num_features_per_region,
        codebook,
        use_l2_normalization=False,
        num_assignments=num_assignments)

    (per_centroid_normalized_rvlad,
     visual_words) = self._PerCentroidNormalization(unnormalized_rvlad)

    normalized_residuals = tf.reshape(
        tf.gather(per_centroid_normalized_rvlad, visual_words),
        [tf.shape(visual_words)[0] * self._feature_dimensionality])

    return normalized_residuals, visual_words
