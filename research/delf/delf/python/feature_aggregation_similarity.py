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
"""Local feature aggregation similarity computation.

For more details, please refer to the paper:
"Detect-to-Retrieve: Efficient Regional Aggregation for Image Search",
Proc. CVPR'19 (https://arxiv.org/abs/1812.01584).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from delf import aggregation_config_pb2

# Aliases for aggregation types.
_VLAD = aggregation_config_pb2.AggregationConfig.VLAD
_ASMK = aggregation_config_pb2.AggregationConfig.ASMK
_ASMK_STAR = aggregation_config_pb2.AggregationConfig.ASMK_STAR


class SimilarityAggregatedRepresentation(object):
  """Class for computing similarity of aggregated local feature representations.

  Args:
    aggregation_config: AggregationConfig object defining type of aggregation to
      use.

  Raises:
    ValueError: If aggregation type is invalid.
  """

  def __init__(self, aggregation_config):
    self._feature_dimensionality = aggregation_config.feature_dimensionality
    self._aggregation_type = aggregation_config.aggregation_type

    # Only relevant if using ASMK/ASMK*. Otherwise, ignored.
    self._use_l2_normalization = aggregation_config.use_l2_normalization
    self._alpha = aggregation_config.alpha
    self._tau = aggregation_config.tau

    # Only relevant if using ASMK*. Otherwise, ignored.
    self._number_bits = np.array([bin(n).count('1') for n in range(256)])

  def ComputeSimilarity(self,
                        aggregated_descriptors_1,
                        aggregated_descriptors_2,
                        feature_visual_words_1=None,
                        feature_visual_words_2=None):
    """Computes similarity between aggregated descriptors.

    Args:
      aggregated_descriptors_1: 1-D NumPy array.
      aggregated_descriptors_2: 1-D NumPy array.
      feature_visual_words_1: Used only for ASMK/ASMK* aggregation type. 1-D
        sorted NumPy integer array denoting visual words corresponding to
        `aggregated_descriptors_1`.
      feature_visual_words_2: Used only for ASMK/ASMK* aggregation type. 1-D
        sorted NumPy integer array denoting visual words corresponding to
        `aggregated_descriptors_2`.

    Returns:
      similarity: Float. The larger, the more similar.

    Raises:
      ValueError: If aggregation type is invalid.
    """
    if self._aggregation_type == _VLAD:
      similarity = np.dot(aggregated_descriptors_1, aggregated_descriptors_2)
    elif self._aggregation_type == _ASMK:
      similarity = self._AsmkSimilarity(
          aggregated_descriptors_1,
          aggregated_descriptors_2,
          feature_visual_words_1,
          feature_visual_words_2,
          binarized=False)
    elif self._aggregation_type == _ASMK_STAR:
      similarity = self._AsmkSimilarity(
          aggregated_descriptors_1,
          aggregated_descriptors_2,
          feature_visual_words_1,
          feature_visual_words_2,
          binarized=True)
    else:
      raise ValueError('Invalid aggregation type: %d' % self._aggregation_type)

    return similarity

  def _CheckAsmkDimensionality(self, aggregated_descriptors, num_visual_words,
                               descriptor_name):
    """Checks that ASMK dimensionality is as expected.

    Args:
      aggregated_descriptors: 1-D NumPy array.
      num_visual_words: Integer.
      descriptor_name: String.

    Raises:
      ValueError: If descriptor dimensionality is incorrect.
    """
    if len(aggregated_descriptors
          ) / num_visual_words != self._feature_dimensionality:
      raise ValueError(
          'Feature dimensionality for aggregated descriptor %s is invalid: %d;'
          ' expected %d.' % (descriptor_name, len(aggregated_descriptors) /
                             num_visual_words, self._feature_dimensionality))

  def _SigmaFn(self, x):
    """Selectivity ASMK/ASMK* similarity function.

    Args:
      x: Scalar or 1-D NumPy array.

    Returns:
      result: Same type as input, with output of selectivity function.
    """
    if np.isscalar(x):
      if x > self._tau:
        result = np.sign(x) * np.power(np.absolute(x), self._alpha)
      else:
        result = 0.0
    else:
      result = np.zeros_like(x)
      above_tau = np.nonzero(x > self._tau)
      result[above_tau] = np.sign(x[above_tau]) * np.power(
          np.absolute(x[above_tau]), self._alpha)

    return result

  def _BinaryNormalizedInnerProduct(self, descriptors_1, descriptors_2):
    """Computes normalized binary inner product.

    Args:
      descriptors_1: 1-D NumPy integer array.
      descriptors_2: 1-D NumPy integer array.

    Returns:
      inner_product: Float.

    Raises:
      ValueError: If the dimensionality of descriptors is different.
    """
    num_descriptors = len(descriptors_1)
    if num_descriptors != len(descriptors_2):
      raise ValueError(
          'Descriptors have incompatible dimensionality: %d vs %d' %
          (len(descriptors_1), len(descriptors_2)))

    h = 0
    for i in range(num_descriptors):
      h += self._number_bits[np.bitwise_xor(descriptors_1[i], descriptors_2[i])]

    # If local feature dimensionality is lower than 8, then use that to compute
    # proper binarized inner product.
    bits_per_descriptor = min(self._feature_dimensionality, 8)

    total_num_bits = bits_per_descriptor * num_descriptors

    return 1.0 - 2.0 * h / total_num_bits

  def _AsmkSimilarity(self,
                      aggregated_descriptors_1,
                      aggregated_descriptors_2,
                      visual_words_1,
                      visual_words_2,
                      binarized=False):
    """Compute ASMK-based similarity.

    If `aggregated_descriptors_1` or `aggregated_descriptors_2` is empty, we
    return a similarity of -1.0.

    If binarized is True, `aggregated_descriptors_1` and
    `aggregated_descriptors_2` must be of type uint8.

    Args:
      aggregated_descriptors_1: 1-D NumPy array.
      aggregated_descriptors_2: 1-D NumPy array.
      visual_words_1: 1-D sorted NumPy integer array denoting visual words
        corresponding to `aggregated_descriptors_1`.
      visual_words_2: 1-D sorted NumPy integer array denoting visual words
        corresponding to `aggregated_descriptors_2`.
      binarized: If True, compute ASMK* similarity.

    Returns:
      similarity: Float. The larger, the more similar.

    Raises:
      ValueError: If input descriptor dimensionality is inconsistent, or if
        descriptor type is unsupported.
    """
    num_visual_words_1 = len(visual_words_1)
    num_visual_words_2 = len(visual_words_2)

    if not num_visual_words_1 or not num_visual_words_2:
      return -1.0

    # Parse dimensionality used per visual word. They must be the same for both
    # aggregated descriptors. If using ASMK, they also must be equal to
    # self._feature_dimensionality.
    if binarized:
      if aggregated_descriptors_1.dtype != 'uint8':
        raise ValueError('Incorrect input descriptor type: %s' %
                         aggregated_descriptors_1.dtype)
      if aggregated_descriptors_2.dtype != 'uint8':
        raise ValueError('Incorrect input descriptor type: %s' %
                         aggregated_descriptors_2.dtype)

      per_visual_word_dimensionality = int(
          len(aggregated_descriptors_1) / num_visual_words_1)
      if len(aggregated_descriptors_2
            ) / num_visual_words_2 != per_visual_word_dimensionality:
        raise ValueError('ASMK* dimensionality is inconsistent.')
    else:
      per_visual_word_dimensionality = self._feature_dimensionality
      self._CheckAsmkDimensionality(aggregated_descriptors_1,
                                    num_visual_words_1, '1')
      self._CheckAsmkDimensionality(aggregated_descriptors_2,
                                    num_visual_words_2, '2')

    aggregated_descriptors_1_reshape = np.reshape(
        aggregated_descriptors_1,
        [num_visual_words_1, per_visual_word_dimensionality])
    aggregated_descriptors_2_reshape = np.reshape(
        aggregated_descriptors_2,
        [num_visual_words_2, per_visual_word_dimensionality])

    # Loop over visual words, compute similarity.
    unnormalized_similarity = 0.0
    ind_1 = 0
    ind_2 = 0
    while ind_1 < num_visual_words_1 and ind_2 < num_visual_words_2:
      if visual_words_1[ind_1] == visual_words_2[ind_2]:
        if binarized:
          inner_product = self._BinaryNormalizedInnerProduct(
              aggregated_descriptors_1_reshape[ind_1],
              aggregated_descriptors_2_reshape[ind_2])
        else:
          inner_product = np.dot(aggregated_descriptors_1_reshape[ind_1],
                                 aggregated_descriptors_2_reshape[ind_2])
        unnormalized_similarity += self._SigmaFn(inner_product)
        ind_1 += 1
        ind_2 += 1
      elif visual_words_1[ind_1] > visual_words_2[ind_2]:
        ind_2 += 1
      else:
        ind_1 += 1

    final_similarity = unnormalized_similarity
    if self._use_l2_normalization:
      final_similarity /= np.sqrt(num_visual_words_1 * num_visual_words_2)

    return final_similarity
