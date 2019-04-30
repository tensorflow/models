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

import tensorflow as tf

from delf import aggregation_config_pb2


class ExtractAggregatedRepresentation(object):
  """Class for extraction of aggregated local feature representation.

  Args:
    sess: TensorFlow session to use.
    aggregation_config: AggregationConfig object defining type of aggregation to
      use.

  Raises:
    ValueError: If aggregation type is invalid.
  """

  def __init__(self, sess, aggregation_config):
    self._sess = sess
    self._codebook_size = aggregation_config.codebook_size
    self._feature_dimensionality = aggregation_config.feature_dimensionality

    # Inputs to extraction function.
    self._features = tf.compat.v1.placeholder(tf.float32, [None, None])
    self._num_features_per_region = tf.compat.v1.placeholder(tf.int32, [None])

    # Load codebook into graph.
    codebook = tf.compat.v1.get_variable(
        "codebook",
        shape=[
            aggregation_config.codebook_size,
            aggregation_config.feature_dimensionality
        ])
    tf.compat.v1.train.init_from_checkpoint(
        aggregation_config.codebook_path, {
            tf.contrib.factorization.KMeansClustering.CLUSTER_CENTERS_VAR_NAME:
                codebook
        })

    # Construct extraction graph based on desired options.
    # TODO(andrearaujo): Add support for other aggregation options.
    if (aggregation_config.aggregation_type ==
        aggregation_config_pb2.AggregationConfig.VLAD):
      # Feature visual words are unused in the case of VLAD, so just return
      # dummy constant.
      self._feature_visual_words = tf.constant(-1, dtype=tf.int32)
      if aggregation_config.use_regional_aggregation:
        self._aggregated_descriptors = self._ComputeRvlad(
            self._features,
            self._num_features_per_region,
            codebook,
            use_l2_normalization=aggregation_config.use_l2_normalization,
            num_assignments=aggregation_config.num_assignments)
      else:
        self._aggregated_descriptors = self._ComputeVlad(
            self._features,
            codebook,
            use_l2_normalization=aggregation_config.use_l2_normalization,
            num_assignments=aggregation_config.num_assignments)
    else:
      raise ValueError("Invalid aggregation type: %d" %
                       aggregation_config.aggregation_type)

    # Initialize variables in the TF graph.
    sess.run(tf.compat.v1.global_variables_initializer())

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
    if num_features_per_region is None:
      # Use dummy value since it is unused.
      num_features_per_region = []
    else:
      if num_features_per_region.size and sum(
          num_features_per_region) != features.shape[0]:
        raise ValueError(
            "Incorrect arguments: sum(num_features_per_region) and "
            "features.shape[0] are different: %d vs %d" %
            (sum(num_features_per_region), features.shape[0]))

    return self._sess.run(
        [self._aggregated_descriptors, self._feature_visual_words],
        feed_dict={
            self._features: features,
            self._num_features_per_region: num_features_per_region
        })

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

      # Find nearest visual words for each feature.
      # K*N x D.
      tiled_features = tf.reshape(
          tf.tile(features, [1, self._codebook_size]),
          [-1, self._feature_dimensionality])
      # K*N x D.
      tiled_codebook = tf.reshape(
          tf.tile(tf.reshape(codebook, [1, -1]), [num_features, 1]),
          [-1, self._feature_dimensionality])
      # N x K.
      squared_distances = tf.reshape(
          tf.reduce_sum(
              tf.math.squared_difference(tiled_features, tiled_codebook),
              axis=1), [num_features, self._codebook_size])
      # N x K.
      nearest_visual_words = tf.argsort(squared_distances)
      # N x num_assignments.
      selected_visual_words = tf.slice(nearest_visual_words, [0, 0],
                                       [num_features, num_assignments])

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
        return ind + 1, tf.tensor_scatter_nd_add(
            vlad, tf.expand_dims(selected_visual_words[ind], axis=1),
            tf.tile(
                tf.expand_dims(features[ind], axis=0), [num_assignments, 1]) -
            tf.gather(codebook, selected_visual_words[ind]))

      i = tf.constant(0, dtype=tf.int32)
      keep_going = lambda j, vlad: tf.less(j, num_features)
      vlad = tf.zeros([self._codebook_size, self._feature_dimensionality],
                      dtype=tf.float32)
      _, vlad = tf.while_loop(
          cond=keep_going,
          body=_ConstructVladFromAssignments,
          loop_vars=[i, vlad],
          back_prop=False)

      vlad = tf.reshape(vlad,
                        [self._codebook_size * self._feature_dimensionality])
      if use_l2_normalization:
        vlad = tf.math.l2_normalize(vlad)

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
          back_prop=False)

      if use_l2_normalization:
        rvlad = tf.math.l2_normalize(rvlad)
      else:
        rvlad /= tf.cast(num_regions, dtype=tf.float32)

      return rvlad

    return tf.cond(
        tf.greater(tf.size(num_features_per_region), 0),
        true_fn=_ComputeRvladNonEmptyRegions,
        false_fn=_ComputeRvladEmptyRegions)
