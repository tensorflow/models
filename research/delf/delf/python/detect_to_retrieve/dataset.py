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
"""Python library to parse ground-truth/evaluate on Revisited datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.io import matlab
import tensorflow as tf

_GROUND_TRUTH_KEYS = ['easy', 'hard', 'junk']


def ReadDatasetFile(dataset_file_path):
  """Reads dataset file in Revisited Oxford/Paris ".mat" format.

  Args:
    dataset_file_path: Path to dataset file, in .mat format.

  Returns:
    query_list: List of query image names.
    index_list: List of index image names.
    ground_truth: List containing ground-truth information for dataset. Each
      entry is a dict corresponding to the ground-truth information for a query.
      The dict may have keys 'easy', 'hard', or 'junk', mapping to a NumPy
      array of integers; additionally, it has a key 'bbx' mapping to a NumPy
      array of floats with bounding box coordinates.
  """
  with tf.gfile.GFile(dataset_file_path, 'rb') as f:
    cfg = matlab.loadmat(f)

  # Parse outputs according to the specificities of the dataset file.
  query_list = [str(im_array[0]) for im_array in np.squeeze(cfg['qimlist'])]
  index_list = [str(im_array[0]) for im_array in np.squeeze(cfg['imlist'])]
  ground_truth_raw = np.squeeze(cfg['gnd'])
  ground_truth = []
  for query_ground_truth_raw in ground_truth_raw:
    query_ground_truth = {}
    for ground_truth_key in _GROUND_TRUTH_KEYS:
      if ground_truth_key in query_ground_truth_raw.dtype.names:
        adjusted_labels = query_ground_truth_raw[ground_truth_key] - 1
        query_ground_truth[ground_truth_key] = adjusted_labels.flatten()

    query_ground_truth['bbx'] = np.squeeze(query_ground_truth_raw['bbx'])
    ground_truth.append(query_ground_truth)

  return query_list, index_list, ground_truth


def _ParseGroundTruth(ok_list, junk_list):
  """Constructs dictionary of ok/junk indices for a data subset and query.

  Args:
    ok_list: List of NumPy arrays containing true positive indices for query.
    junk_list: List of NumPy arrays containing ignored indices for query.

  Returns:
    ok_junk_dict: Dict mapping 'ok' and 'junk' strings to NumPy array of
      indices.
  """
  ok_junk_dict = {}
  ok_junk_dict['ok'] = np.concatenate(ok_list)
  ok_junk_dict['junk'] = np.concatenate(junk_list)
  return ok_junk_dict


def ParseEasyMediumHardGroundTruth(ground_truth):
  """Parses easy/medium/hard ground-truth from Revisited datasets.

  Args:
    ground_truth: Usually the output from ReadDatasetFile(). List containing
      ground-truth information for dataset. Each entry is a dict corresponding
      to the ground-truth information for a query. The dict must have keys
      'easy', 'hard', and 'junk', mapping to a NumPy array of integers.

  Returns:
    easy_ground_truth: List containing ground-truth information for easy subset
      of dataset. Each entry is a dict corresponding to the ground-truth
      information for a query. The dict has keys 'ok' and 'junk', mapping to a
      NumPy array of integers.
    medium_ground_truth: Same as `easy_ground_truth`, but for the medium subset.
    hard_ground_truth: Same as `easy_ground_truth`, but for the hard subset.
  """
  num_queries = len(ground_truth)

  easy_ground_truth = []
  medium_ground_truth = []
  hard_ground_truth = []
  for i in range(num_queries):
    easy_ground_truth.append(
        _ParseGroundTruth([ground_truth[i]['easy']],
                          [ground_truth[i]['junk'], ground_truth[i]['hard']]))
    medium_ground_truth.append(
        _ParseGroundTruth([ground_truth[i]['easy'], ground_truth[i]['hard']],
                          [ground_truth[i]['junk']]))
    hard_ground_truth.append(
        _ParseGroundTruth([ground_truth[i]['hard']],
                          [ground_truth[i]['junk'], ground_truth[i]['easy']]))

  return easy_ground_truth, medium_ground_truth, hard_ground_truth


def AdjustPositiveRanks(positive_ranks, junk_ranks):
  """Adjusts positive ranks based on junk ranks.

  Args:
    positive_ranks: Sorted 1D NumPy integer array.
    junk_ranks: Sorted 1D NumPy integer array.

  Returns:
    adjusted_positive_ranks: Sorted 1D NumPy array.
  """
  if not junk_ranks.size:
    return positive_ranks

  adjusted_positive_ranks = positive_ranks
  j = 0
  for i, positive_index in enumerate(positive_ranks):
    while (j < len(junk_ranks) and positive_index > junk_ranks[j]):
      j += 1

    adjusted_positive_ranks[i] -= j

  return adjusted_positive_ranks


def ComputeAveragePrecision(positive_ranks):
  """Computes average precision according to dataset convention.

  It assumes that `positive_ranks` contains the ranks for all expected positive
  index images to be retrieved. If `positive_ranks` is empty, returns
  `average_precision` = 0.

  Note that average precision computation here does NOT use the finite sum
  method (see
  https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision)
  which is common in information retrieval literature. Instead, the method
  implemented here integrates over the precision-recall curve by averaging two
  adjacent precision points, then multiplying by the recall step. This is the
  convention for the Revisited Oxford/Paris datasets.

  Args:
    positive_ranks: Sorted 1D NumPy integer array, zero-indexed.

  Returns:
    average_precision: Float.
  """
  average_precision = 0.0

  num_expected_positives = len(positive_ranks)
  if not num_expected_positives:
    return average_precision

  recall_step = 1.0 / num_expected_positives
  for i, rank in enumerate(positive_ranks):
    if not rank:
      left_precision = 1.0
    else:
      left_precision = i / rank

    right_precision = (i + 1) / (rank + 1)
    average_precision += (left_precision + right_precision) * recall_step / 2

  return average_precision


def ComputePRAtRanks(positive_ranks, desired_pr_ranks):
  """Computes precision/recall at desired ranks.

  It assumes that `positive_ranks` contains the ranks for all expected positive
  index images to be retrieved. If `positive_ranks` is empty, return all-zeros
  `precisions`/`recalls`.

  If a desired rank is larger than the last positive rank, its precision is
  computed based on the last positive rank. For example, if `desired_pr_ranks`
  is [10] and `positive_ranks` = [0, 7] --> `precisions` = [0.25], `recalls` =
  [1.0].

  Args:
    positive_ranks: 1D NumPy integer array, zero-indexed.
    desired_pr_ranks: List of integers containing the desired precision/recall
      ranks to be reported. Eg, if precision@1/recall@1 and
      precision@10/recall@10 are desired, this should be set to [1, 10].

  Returns:
    precisions: Precision @ `desired_pr_ranks` (NumPy array of
      floats, with shape [len(desired_pr_ranks)]).
    recalls: Recall @ `desired_pr_ranks` (NumPy array of floats, with
      shape [len(desired_pr_ranks)]).
  """
  num_desired_pr_ranks = len(desired_pr_ranks)
  precisions = np.zeros([num_desired_pr_ranks])
  recalls = np.zeros([num_desired_pr_ranks])

  num_expected_positives = len(positive_ranks)
  if not num_expected_positives:
    return precisions, recalls

  positive_ranks_one_indexed = positive_ranks + 1
  for i, desired_pr_rank in enumerate(desired_pr_ranks):
    recalls[i] = np.sum(
        positive_ranks_one_indexed <= desired_pr_rank) / num_expected_positives

    # If `desired_pr_rank` is larger than last positive's rank, only compute
    # precision with respect to last positive's position.
    precision_rank = min(max(positive_ranks_one_indexed), desired_pr_rank)
    precisions[i] = np.sum(
        positive_ranks_one_indexed <= precision_rank) / precision_rank

  return precisions, recalls


def ComputeMetrics(sorted_index_ids, ground_truth, desired_pr_ranks):
  """Computes metrics for retrieval results on the Revisited datasets.

  If there are no valid ground-truth index images for a given query, the metric
  results for the given query (`average_precisions`, `precisions` and `recalls`)
  are set to NaN, and they are not taken into account when computing the
  aggregated metrics (`mean_average_precision`, `mean_precisions` and
  `mean_recalls`) over all queries.

  Args:
    sorted_index_ids: Integer NumPy array of shape [#queries, #index_images].
      For each query, contains an array denoting the most relevant index images,
      sorted from most to least relevant.
    ground_truth: List containing ground-truth information for dataset. Each
      entry is a dict corresponding to the ground-truth information for a query.
      The dict has keys 'ok' and 'junk', mapping to a NumPy array of integers.
    desired_pr_ranks: List of integers containing the desired precision/recall
      ranks to be reported. Eg, if precision@1/recall@1 and
      precision@10/recall@10 are desired, this should be set to [1, 10]. The
      largest item should be <= #index_images.

  Returns:
    mean_average_precision: Mean average precision (float).
    mean_precisions: Mean precision @ `desired_pr_ranks` (NumPy array of
      floats, with shape [len(desired_pr_ranks)]).
    mean_recalls: Mean recall @ `desired_pr_ranks` (NumPy array of floats, with
      shape [len(desired_pr_ranks)]).
    average_precisions: Average precision for each query (NumPy array of floats,
      with shape [#queries]).
    precisions: Precision @ `desired_pr_ranks`, for each query (NumPy array of
      floats, with shape [#queries, len(desired_pr_ranks)]).
    recalls: Recall @ `desired_pr_ranks`, for each query (NumPy array of
      floats, with shape [#queries, len(desired_pr_ranks)]).

  Raises:
    ValueError: If largest desired PR rank in `desired_pr_ranks` >
      #index_images.
  """
  num_queries, num_index_images = sorted_index_ids.shape
  num_desired_pr_ranks = len(desired_pr_ranks)

  sorted_desired_pr_ranks = sorted(desired_pr_ranks)

  if sorted_desired_pr_ranks[-1] > num_index_images:
    raise ValueError(
        'Requested PR ranks up to %d, however there are only %d images' %
        (sorted_desired_pr_ranks[-1], num_index_images))

  # Instantiate all outputs, then loop over each query and gather metrics.
  mean_average_precision = 0.0
  mean_precisions = np.zeros([num_desired_pr_ranks])
  mean_recalls = np.zeros([num_desired_pr_ranks])
  average_precisions = np.zeros([num_queries])
  precisions = np.zeros([num_queries, num_desired_pr_ranks])
  recalls = np.zeros([num_queries, num_desired_pr_ranks])
  num_empty_gt_queries = 0
  for i in range(num_queries):
    ok_index_images = ground_truth[i]['ok']
    junk_index_images = ground_truth[i]['junk']

    if not ok_index_images.size:
      average_precisions[i] = float('nan')
      precisions[i, :] = float('nan')
      recalls[i, :] = float('nan')
      num_empty_gt_queries += 1
      continue

    positive_ranks = np.arange(num_index_images)[np.in1d(
        sorted_index_ids[i], ok_index_images)]
    junk_ranks = np.arange(num_index_images)[np.in1d(sorted_index_ids[i],
                                                     junk_index_images)]

    adjusted_positive_ranks = AdjustPositiveRanks(positive_ranks, junk_ranks)

    average_precisions[i] = ComputeAveragePrecision(adjusted_positive_ranks)
    precisions[i, :], recalls[i, :] = ComputePRAtRanks(adjusted_positive_ranks,
                                                       desired_pr_ranks)

    mean_average_precision += average_precisions[i]
    mean_precisions += precisions[i, :]
    mean_recalls += recalls[i, :]

  # Normalize aggregated metrics by number of queries.
  num_valid_queries = num_queries - num_empty_gt_queries
  mean_average_precision /= num_valid_queries
  mean_precisions /= num_valid_queries
  mean_recalls /= num_valid_queries

  return (mean_average_precision, mean_precisions, mean_recalls,
          average_precisions, precisions, recalls)
