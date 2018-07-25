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
"""Functions for computing metrics like precision, recall, CorLoc and etc."""
from __future__ import division

import numpy as np


def compute_precision_recall(scores, labels, num_gt):
  """Compute precision and recall.

  Args:
    scores: A float numpy array representing detection score
    labels: A float numpy array representing weighted true/false positive labels
    num_gt: Number of ground truth instances

  Raises:
    ValueError: if the input is not of the correct format

  Returns:
    precision: Fraction of positive instances over detected ones. This value is
      None if no ground truth labels are present.
    recall: Fraction of detected positive instance over all positive instances.
      This value is None if no ground truth labels are present.

  """
  if not isinstance(labels, np.ndarray) or len(labels.shape) != 1:
    raise ValueError("labels must be single dimension numpy array")

  if labels.dtype != np.float and labels.dtype != np.bool:
    raise ValueError("labels type must be either bool or float")

  if not isinstance(scores, np.ndarray) or len(scores.shape) != 1:
    raise ValueError("scores must be single dimension numpy array")

  if num_gt < np.sum(labels):
    raise ValueError("Number of true positives must be smaller than num_gt.")

  if len(scores) != len(labels):
    raise ValueError("scores and labels must be of the same size.")

  if num_gt == 0:
    return None, None

  sorted_indices = np.argsort(scores)
  sorted_indices = sorted_indices[::-1]
  true_positive_labels = labels[sorted_indices]
  false_positive_labels = (true_positive_labels <= 0).astype(float)
  cum_true_positives = np.cumsum(true_positive_labels)
  cum_false_positives = np.cumsum(false_positive_labels)
  precision = cum_true_positives.astype(float) / (
      cum_true_positives + cum_false_positives)
  recall = cum_true_positives.astype(float) / num_gt
  return precision, recall


def compute_average_precision(precision, recall):
  """Compute Average Precision according to the definition in VOCdevkit.

  Precision is modified to ensure that it does not decrease as recall
  decrease.

  Args:
    precision: A float [N, 1] numpy array of precisions
    recall: A float [N, 1] numpy array of recalls

  Raises:
    ValueError: if the input is not of the correct format

  Returns:
    average_precison: The area under the precision recall curve. NaN if
      precision and recall are None.

  """
  if precision is None:
    if recall is not None:
      raise ValueError("If precision is None, recall must also be None")
    return np.NAN

  if not isinstance(precision, np.ndarray) or not isinstance(
      recall, np.ndarray):
    raise ValueError("precision and recall must be numpy array")
  if precision.dtype != np.float or recall.dtype != np.float:
    raise ValueError("input must be float numpy array.")
  if len(precision) != len(recall):
    raise ValueError("precision and recall must be of the same size.")
  if not precision.size:
    return 0.0
  if np.amin(precision) < 0 or np.amax(precision) > 1:
    raise ValueError("Precision must be in the range of [0, 1].")
  if np.amin(recall) < 0 or np.amax(recall) > 1:
    raise ValueError("recall must be in the range of [0, 1].")
  if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
    raise ValueError("recall must be a non-decreasing array")

  recall = np.concatenate([[0], recall, [1]])
  precision = np.concatenate([[0], precision, [0]])

  # Preprocess precision to be a non-decreasing array
  for i in range(len(precision) - 2, -1, -1):
    precision[i] = np.maximum(precision[i], precision[i + 1])

  indices = np.where(recall[1:] != recall[:-1])[0] + 1
  average_precision = np.sum(
      (recall[indices] - recall[indices - 1]) * precision[indices])
  return average_precision


def compute_cor_loc(num_gt_imgs_per_class,
                    num_images_correctly_detected_per_class):
  """Compute CorLoc according to the definition in the following paper.

  https://www.robots.ox.ac.uk/~vgg/rg/papers/deselaers-eccv10.pdf

  Returns nans if there are no ground truth images for a class.

  Args:
    num_gt_imgs_per_class: 1D array, representing number of images containing
        at least one object instance of a particular class
    num_images_correctly_detected_per_class: 1D array, representing number of
        images that are correctly detected at least one object instance of a
        particular class

  Returns:
    corloc_per_class: A float numpy array represents the corloc score of each
      class
  """
  return np.where(
      num_gt_imgs_per_class == 0, np.nan,
      num_images_correctly_detected_per_class / num_gt_imgs_per_class)


def compute_median_rank_at_k(tp_fp_list, k):
  """Computes MedianRank@k, where k is the top-scoring labels.

  Args:
    tp_fp_list: a list of numpy arrays; each numpy array corresponds to the all
        detection on a single image, where the detections are sorted by score in
        descending order. Further, each numpy array element can have boolean or
        float values. True positive elements have either value >0.0 or True;
        any other value is considered false positive.
    k: number of top-scoring proposals to take.

  Returns:
    median_rank: median rank of all true positive proposals among top k by
      score.
  """
  ranks = []
  for i in range(len(tp_fp_list)):
    ranks.append(
        np.where(tp_fp_list[i][0:min(k, tp_fp_list[i].shape[0])] > 0)[0])
  concatenated_ranks = np.concatenate(ranks)
  return np.median(concatenated_ranks)


def compute_recall_at_k(tp_fp_list, num_gt, k):
  """Computes Recall@k, MedianRank@k, where k is the top-scoring labels.

  Args:
    tp_fp_list: a list of numpy arrays; each numpy array corresponds to the all
        detection on a single image, where the detections are sorted by score in
        descending order. Further, each numpy array element can have boolean or
        float values. True positive elements have either value >0.0 or True;
        any other value is considered false positive.
    num_gt: number of groundtruth anotations.
    k: number of top-scoring proposals to take.

  Returns:
    recall: recall evaluated on the top k by score detections.
  """

  tp_fp_eval = []
  for i in range(len(tp_fp_list)):
    tp_fp_eval.append(tp_fp_list[i][0:min(k, tp_fp_list[i].shape[0])])

  tp_fp_eval = np.concatenate(tp_fp_eval)

  return np.sum(tp_fp_eval) / num_gt
