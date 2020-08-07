# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

"""Hungarian bipartite matcher implementation."""

import numpy as np
from scipy.optimize import linear_sum_assignment

import tensorflow.compat.v1 as tf
from object_detection.core import matcher


class HungarianBipartiteMatcher(matcher.Matcher):
  """Wraps a Hungarian bipartite matcher into TensorFlow."""

  def _match(self, similarity_matrix, valid_rows):
    """Optimally bipartite matches a collection rows and columns.

    Args:
      similarity_matrix: Float tensor of shape [N, M] with pairwise similarity
        where higher values mean more similar.
      valid_rows: A boolean tensor of shape [N] indicating the rows that are
        valid.

    Returns:
      match_results: int32 tensor of shape [M] with match_results[i]=-1
        meaning that column i is not matched and otherwise that it is matched to
        row match_results[i].
    """
    valid_row_sim_matrix = tf.gather(similarity_matrix,
                                     tf.squeeze(tf.where(valid_rows), axis=-1))
    distance_matrix = -1 * valid_row_sim_matrix

    def numpy_wrapper(inputs):
      def numpy_matching(input_matrix):
        row_indices, col_indices = linear_sum_assignment(input_matrix)
        match_results = np.full(input_matrix.shape[1], -1)
        match_results[col_indices] = row_indices
        return match_results.astype(np.int32)

      return tf.numpy_function(numpy_matching, inputs, Tout=[tf.int32])

    matching_result = tf.autograph.experimental.do_not_convert(
        numpy_wrapper)([distance_matrix])

    return tf.reshape(matching_result, [-1])
