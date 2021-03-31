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
"""Python module to compute metrics for Google Landmarks dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def _CountPositives(solution):
  """Counts number of test images with non-empty ground-truth in `solution`.

  Args:
    solution: Dict mapping test image ID to list of ground-truth IDs.

  Returns:
    count: Number of test images with non-empty ground-truth.
  """
  count = 0
  for v in solution.values():
    if v:
      count += 1

  return count


def GlobalAveragePrecision(predictions,
                           recognition_solution,
                           ignore_non_gt_test_images=False):
  """Computes global average precision for recognition prediction.

  Args:
    predictions: Dict mapping test image ID to a dict with keys 'class'
      (integer) and 'score' (float).
    recognition_solution: Dict mapping test image ID to list of ground-truth
      landmark IDs.
    ignore_non_gt_test_images: If True, ignore test images which do not have
      associated ground-truth landmark IDs. For the Google Landmark Recognition
      challenge, this should be set to False.

  Returns:
    gap: Global average precision score (float).
  """
  # Compute number of expected results.
  num_positives = _CountPositives(recognition_solution)

  gap = 0.0
  total_predictions = 0
  correct_predictions = 0

  # Sort predictions according to Kaggle's convention:
  # - first by score (descending);
  # - then by key (ascending);
  # - then by class (ascending).
  sorted_predictions_by_key_class = sorted(
      predictions.items(), key=lambda item: (item[0], item[1]['class']))
  sorted_predictions = sorted(
      sorted_predictions_by_key_class,
      key=lambda item: item[1]['score'],
      reverse=True)

  # Loop over sorted predictions (descending order) and compute GAPs.
  for key, prediction in sorted_predictions:
    if ignore_non_gt_test_images and not recognition_solution[key]:
      continue

    total_predictions += 1
    if prediction['class'] in recognition_solution[key]:
      correct_predictions += 1
      gap += correct_predictions / total_predictions

  gap /= num_positives

  return gap


def Top1Accuracy(predictions, recognition_solution):
  """Computes top-1 accuracy for recognition prediction.

  Note that test images without ground-truth are ignored.

  Args:
    predictions: Dict mapping test image ID to a dict with keys 'class'
      (integer) and 'score' (float).
    recognition_solution: Dict mapping test image ID to list of ground-truth
      landmark IDs.

  Returns:
    accuracy: Top-1 accuracy (float).
  """
  # Loop over test images in solution. If it has at least one class label, we
  # check if the predicion is correct.
  num_correct_predictions = 0
  num_test_images_with_ground_truth = 0
  for key, ground_truth in recognition_solution.items():
    if ground_truth:
      num_test_images_with_ground_truth += 1
      if key in predictions:
        if predictions[key]['class'] in ground_truth:
          num_correct_predictions += 1

  return num_correct_predictions / num_test_images_with_ground_truth


def MeanAveragePrecision(predictions, retrieval_solution, max_predictions=100):
  """Computes mean average precision for retrieval prediction.

  Args:
    predictions: Dict mapping test image ID to a list of strings corresponding
      to index image IDs.
    retrieval_solution: Dict mapping test image ID to list of ground-truth image
      IDs.
    max_predictions: Maximum number of predictions per query to take into
      account. For the Google Landmark Retrieval challenge, this should be set
      to 100.

  Returns:
    mean_ap: Mean average precision score (float).

  Raises:
    ValueError: If a test image in `predictions` is not included in
      `retrieval_solutions`.
  """
  # Compute number of test images.
  num_test_images = len(retrieval_solution.keys())

  # Loop over predictions for each query and compute mAP.
  mean_ap = 0.0
  for key, prediction in predictions.items():
    if key not in retrieval_solution:
      raise ValueError('Test image %s is not part of retrieval_solution' % key)

    # Loop over predicted images, keeping track of those which were already
    # used (duplicates are skipped).
    ap = 0.0
    already_predicted = set()
    num_expected_retrieved = min(len(retrieval_solution[key]), max_predictions)
    num_correct = 0
    for i in range(min(len(prediction), max_predictions)):
      if prediction[i] not in already_predicted:
        if prediction[i] in retrieval_solution[key]:
          num_correct += 1
          ap += num_correct / (i + 1)
        already_predicted.add(prediction[i])

    ap /= num_expected_retrieved
    mean_ap += ap

  mean_ap /= num_test_images

  return mean_ap


def MeanPrecisions(predictions, retrieval_solution, max_predictions=100):
  """Computes mean precisions for retrieval prediction.

  Args:
    predictions: Dict mapping test image ID to a list of strings corresponding
      to index image IDs.
    retrieval_solution: Dict mapping test image ID to list of ground-truth image
      IDs.
    max_predictions: Maximum number of predictions per query to take into
      account.

  Returns:
    mean_precisions: NumPy array with mean precisions at ranks 1 through
      `max_predictions`.

  Raises:
    ValueError: If a test image in `predictions` is not included in
      `retrieval_solutions`.
  """
  # Compute number of test images.
  num_test_images = len(retrieval_solution.keys())

  # Loop over predictions for each query and compute precisions@k.
  precisions = np.zeros((num_test_images, max_predictions))
  count_test_images = 0
  for key, prediction in predictions.items():
    if key not in retrieval_solution:
      raise ValueError('Test image %s is not part of retrieval_solution' % key)

    # Loop over predicted images, keeping track of those which were already
    # used (duplicates are skipped).
    already_predicted = set()
    num_correct = 0
    for i in range(max_predictions):
      if i < len(prediction):
        if prediction[i] not in already_predicted:
          if prediction[i] in retrieval_solution[key]:
            num_correct += 1
          already_predicted.add(prediction[i])
      precisions[count_test_images, i] = num_correct / (i + 1)
    count_test_images += 1

  mean_precisions = np.mean(precisions, axis=0)

  return mean_precisions


def MeanMedianPosition(predictions, retrieval_solution, max_predictions=100):
  """Computes mean and median positions of first correct image.

  Args:
    predictions: Dict mapping test image ID to a list of strings corresponding
      to index image IDs.
    retrieval_solution: Dict mapping test image ID to list of ground-truth image
      IDs.
    max_predictions: Maximum number of predictions per query to take into
      account.

  Returns:
    mean_position: Float.
    median_position: Float.

  Raises:
    ValueError: If a test image in `predictions` is not included in
      `retrieval_solutions`.
  """
  # Compute number of test images.
  num_test_images = len(retrieval_solution.keys())

  # Loop over predictions for each query to find first correct ranked image.
  positions = (max_predictions + 1) * np.ones((num_test_images))
  count_test_images = 0
  for key, prediction in predictions.items():
    if key not in retrieval_solution:
      raise ValueError('Test image %s is not part of retrieval_solution' % key)

    for i in range(min(len(prediction), max_predictions)):
      if prediction[i] in retrieval_solution[key]:
        positions[count_test_images] = i + 1
        break

    count_test_images += 1

  mean_position = np.mean(positions)
  median_position = np.median(positions)

  return mean_position, median_position
