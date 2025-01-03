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

import os
import pickle

import numpy as np
from scipy.io import matlab
import tensorflow as tf

_GROUND_TRUTH_KEYS = ['easy', 'hard', 'junk']

DATASET_NAMES = ['roxford5k', 'rparis6k']


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
  with tf.io.gfile.GFile(dataset_file_path, 'rb') as f:
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


def SaveMetricsFile(mean_average_precision, mean_precisions, mean_recalls,
                    pr_ranks, output_path):
  """Saves aggregated retrieval metrics to text file.

  Args:
    mean_average_precision: Dict mapping each dataset protocol to a float.
    mean_precisions: Dict mapping each dataset protocol to a NumPy array of
      floats with shape [len(pr_ranks)].
    mean_recalls: Dict mapping each dataset protocol to a NumPy array of floats
      with shape [len(pr_ranks)].
    pr_ranks: List of integers.
    output_path: Full file path.
  """
  with tf.io.gfile.GFile(output_path, 'w') as f:
    for k in sorted(mean_average_precision.keys()):
      f.write('{}\n  mAP={}\n  mP@k{} {}\n  mR@k{} {}\n'.format(
          k, np.around(mean_average_precision[k] * 100, decimals=2),
          np.array(pr_ranks), np.around(mean_precisions[k] * 100, decimals=2),
          np.array(pr_ranks), np.around(mean_recalls[k] * 100, decimals=2)))


def _ParseSpaceSeparatedStringsInBrackets(line, prefixes, ind):
  """Parses line containing space-separated strings in brackets.

  Args:
    line: String, containing line in metrics file with mP@k or mR@k figures.
    prefixes: Tuple/list of strings, containing valid prefixes.
    ind: Integer indicating which field within brackets is parsed.

  Yields:
    entry: String format entry.

  Raises:
    ValueError: If input line does not contain a valid prefix.
  """
  for prefix in prefixes:
    if line.startswith(prefix):
      line = line[len(prefix):]
      break
  else:
    raise ValueError('Line %s is malformed, cannot find valid prefixes' % line)

  for entry in line.split('[')[ind].split(']')[0].split():
    yield entry


def _ParsePrRanks(line):
  """Parses PR ranks from mP@k line in metrics file.

  Args:
    line: String, containing line in metrics file with mP@k figures.

  Returns:
    pr_ranks: List of integers, containing used ranks.

  Raises:
    ValueError: If input line is malformed.
  """
  return [
      int(pr_rank) for pr_rank in _ParseSpaceSeparatedStringsInBrackets(
          line, ['  mP@k['], 0) if pr_rank
  ]


def _ParsePrScores(line, num_pr_ranks):
  """Parses PR scores from line in metrics file.

  Args:
    line: String, containing line in metrics file with mP@k or mR@k figures.
    num_pr_ranks: Integer, number of scores that should be in output list.

  Returns:
    pr_scores: List of floats, containing scores.

  Raises:
    ValueError: If input line is malformed.
  """
  pr_scores = [
      float(pr_score) for pr_score in _ParseSpaceSeparatedStringsInBrackets(
          line, ('  mP@k[', '  mR@k['), 1) if pr_score
  ]

  if len(pr_scores) != num_pr_ranks:
    raise ValueError('Line %s is malformed, expected %d scores but found %d' %
                     (line, num_pr_ranks, len(pr_scores)))

  return pr_scores


def ReadMetricsFile(metrics_path):
  """Reads aggregated retrieval metrics from text file.

  Args:
    metrics_path: Full file path, containing aggregated retrieval metrics.

  Returns:
    mean_average_precision: Dict mapping each dataset protocol to a float.
    pr_ranks: List of integer ranks used in aggregated recall/precision metrics.
    mean_precisions: Dict mapping each dataset protocol to a NumPy array of
      floats with shape [len(`pr_ranks`)].
    mean_recalls: Dict mapping each dataset protocol to a NumPy array of floats
      with shape [len(`pr_ranks`)].

  Raises:
    ValueError: If input file is malformed.
  """
  with tf.io.gfile.GFile(metrics_path, 'r') as f:
    file_contents_stripped = [l.rstrip() for l in f]

  if len(file_contents_stripped) % 4:
    raise ValueError(
        'Malformed input %s: number of lines must be a multiple of 4, '
        'but it is %d' % (metrics_path, len(file_contents_stripped)))

  mean_average_precision = {}
  pr_ranks = []
  mean_precisions = {}
  mean_recalls = {}
  protocols = set()
  for i in range(0, len(file_contents_stripped), 4):
    protocol = file_contents_stripped[i]
    if protocol in protocols:
      raise ValueError(
          'Malformed input %s: protocol %s is found a second time' %
          (metrics_path, protocol))
    protocols.add(protocol)

    # Parse mAP.
    mean_average_precision[protocol] = float(
        file_contents_stripped[i + 1].split('=')[1]) / 100.0

    # Parse (or check consistency of) pr_ranks.
    parsed_pr_ranks = _ParsePrRanks(file_contents_stripped[i + 2])
    if not pr_ranks:
      pr_ranks = parsed_pr_ranks
    else:
      if parsed_pr_ranks != pr_ranks:
        raise ValueError('Malformed input %s: inconsistent PR ranks' %
                         metrics_path)

    # Parse mean precisions.
    mean_precisions[protocol] = np.array(
        _ParsePrScores(file_contents_stripped[i + 2], len(pr_ranks)),
        dtype=float) / 100.0

    # Parse mean recalls.
    mean_recalls[protocol] = np.array(
        _ParsePrScores(file_contents_stripped[i + 3], len(pr_ranks)),
        dtype=float) / 100.0

  return mean_average_precision, pr_ranks, mean_precisions, mean_recalls


def CreateConfigForTestDataset(dataset, dir_main):
  """Creates the configuration dictionary for the test dataset.

  Args:
    dataset: String, dataset name: either 'roxford5k' or 'rparis6k'.
    dir_main: String, path to the folder containing ground truth files.

  Returns:
    cfg: Dataset configuration in a form of dictionary. The configuration
      includes:
      `gnd_fname` - path to the ground truth file for the dataset,
      `ext` and `qext` - image extensions for the images in the test dataset
      and the query images,
      `dir_data` - path to the folder containing ground truth files,
      `dir_images` - path to the folder containing images,
      `n` and `nq` - number of images and query images in the dataset
      respectively,
      `im_fname` and `qim_fname` - functions providing paths for the dataset
      and query images respectively,
      `dataset` - test dataset name.

  Raises:
    ValueError: If an unknown dataset name is provided as an argument.
  """
  dataset = dataset.lower()

  def _ConfigImname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['imlist'][i] + cfg['ext'])

  def _ConfigQimname(cfg, i):
    return os.path.join(cfg['dir_images'], cfg['qimlist'][i] + cfg['qext'])

  if dataset not in DATASET_NAMES:
    raise ValueError('Unknown dataset: {}!'.format(dataset))

  # Loading imlist, qimlist, and gnd in configuration as a dictionary.
  gnd_fname = os.path.join(dir_main, 'gnd_{}.pkl'.format(dataset))
  with tf.io.gfile.GFile(gnd_fname, 'rb') as f:
    cfg = pickle.load(f)
  cfg['gnd_fname'] = gnd_fname
  if dataset == 'rparis6k':
    dir_images = 'paris6k_images'
  elif dataset == 'roxford5k':
    dir_images = 'oxford5k_images'

  cfg['ext'] = '.jpg'
  cfg['qext'] = '.jpg'
  cfg['dir_data'] = os.path.join(dir_main)
  cfg['dir_images'] = os.path.join(cfg['dir_data'], dir_images)

  cfg['n'] = len(cfg['imlist'])
  cfg['nq'] = len(cfg['qimlist'])

  cfg['im_fname'] = _ConfigImname
  cfg['qim_fname'] = _ConfigQimname

  cfg['dataset'] = dataset

  return cfg
