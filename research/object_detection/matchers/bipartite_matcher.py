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

"""Bipartite matcher implementation."""

import tensorflow as tf

from tensorflow.contrib.image.python.ops import image_ops
from object_detection.core import matcher


class GreedyBipartiteMatcher(matcher.Matcher):
  """Wraps a Tensorflow greedy bipartite matcher."""

  def _match(self, similarity_matrix, num_valid_rows=-1):
    """Bipartite matches a collection rows and columns. A greedy bi-partite.

    TODO: Add num_valid_columns options to match only that many columns with
        all the rows.

    Args:
      similarity_matrix: Float tensor of shape [N, M] with pairwise similarity
        where higher values mean more similar.
      num_valid_rows: A scalar or a 1-D tensor with one element describing the
        number of valid rows of similarity_matrix to consider for the bipartite
        matching. If set to be negative, then all rows from similarity_matrix
        are used.

    Returns:
      match_results: int32 tensor of shape [M] with match_results[i]=-1
        meaning that column i is not matched and otherwise that it is matched to
        row match_results[i].
    """
    # Convert similarity matrix to distance matrix as tf.image.bipartite tries
    # to find minimum distance matches.
    distance_matrix = -1 * similarity_matrix
    _, match_results = image_ops.bipartite_match(
        distance_matrix, num_valid_rows)
    match_results = tf.reshape(match_results, [-1])
    match_results = tf.cast(match_results, tf.int32)
    return match_results
