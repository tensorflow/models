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
"""Tests for Google Landmarks dataset metric computation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from delf.python.datasets.google_landmarks_dataset import metrics


def _CreateRecognitionSolution():
  """Creates recognition solution to be used in tests.

  Returns:
    solution: Dict mapping test image ID to list of ground-truth landmark IDs.
  """
  return {
      '0123456789abcdef': [0, 12],
      '0223456789abcdef': [100, 200, 300],
      '0323456789abcdef': [1],
      '0423456789abcdef': [],
      '0523456789abcdef': [],
  }


def _CreateRecognitionPredictions():
  """Creates recognition predictions to be used in tests.

  Returns:
    predictions: Dict mapping test image ID to a dict with keys 'class'
      (integer) and 'score' (float).
  """
  return {
      '0223456789abcdef': {
          'class': 0,
          'score': 0.01
      },
      '0323456789abcdef': {
          'class': 1,
          'score': 10.0
      },
      '0423456789abcdef': {
          'class': 150,
          'score': 15.0
      },
  }


def _CreateRetrievalSolution():
  """Creates retrieval solution to be used in tests.

  Returns:
    solution: Dict mapping test image ID to list of ground-truth image IDs.
  """
  return {
      '0123456789abcdef': ['fedcba9876543210', 'fedcba9876543220'],
      '0223456789abcdef': ['fedcba9876543210'],
      '0323456789abcdef': [
          'fedcba9876543230', 'fedcba9876543240', 'fedcba9876543250'
      ],
      '0423456789abcdef': ['fedcba9876543230'],
  }


def _CreateRetrievalPredictions():
  """Creates retrieval predictions to be used in tests.

  Returns:
    predictions: Dict mapping test image ID to a list with predicted index image
    ids.
  """
  return {
      '0223456789abcdef': ['fedcba9876543200', 'fedcba9876543210'],
      '0323456789abcdef': ['fedcba9876543240'],
      '0423456789abcdef': ['fedcba9876543230', 'fedcba9876543240'],
  }


class MetricsTest(tf.test.TestCase):

  def testGlobalAveragePrecisionWorks(self):
    # Define input.
    predictions = _CreateRecognitionPredictions()
    solution = _CreateRecognitionSolution()

    # Run tested function.
    gap = metrics.GlobalAveragePrecision(predictions, solution)

    # Define expected results.
    expected_gap = 0.166667

    # Compare actual and expected results.
    self.assertAllClose(gap, expected_gap)

  def testGlobalAveragePrecisionIgnoreNonGroundTruthWorks(self):
    # Define input.
    predictions = _CreateRecognitionPredictions()
    solution = _CreateRecognitionSolution()

    # Run tested function.
    gap = metrics.GlobalAveragePrecision(
        predictions, solution, ignore_non_gt_test_images=True)

    # Define expected results.
    expected_gap = 0.333333

    # Compare actual and expected results.
    self.assertAllClose(gap, expected_gap)

  def testTop1AccuracyWorks(self):
    # Define input.
    predictions = _CreateRecognitionPredictions()
    solution = _CreateRecognitionSolution()

    # Run tested function.
    accuracy = metrics.Top1Accuracy(predictions, solution)

    # Define expected results.
    expected_accuracy = 0.333333

    # Compare actual and expected results.
    self.assertAllClose(accuracy, expected_accuracy)

  def testMeanAveragePrecisionWorks(self):
    # Define input.
    predictions = _CreateRetrievalPredictions()
    solution = _CreateRetrievalSolution()

    # Run tested function.
    mean_ap = metrics.MeanAveragePrecision(predictions, solution)

    # Define expected results.
    expected_mean_ap = 0.458333

    # Compare actual and expected results.
    self.assertAllClose(mean_ap, expected_mean_ap)

  def testMeanAveragePrecisionMaxPredictionsWorks(self):
    # Define input.
    predictions = _CreateRetrievalPredictions()
    solution = _CreateRetrievalSolution()

    # Run tested function.
    mean_ap = metrics.MeanAveragePrecision(
        predictions, solution, max_predictions=1)

    # Define expected results.
    expected_mean_ap = 0.5

    # Compare actual and expected results.
    self.assertAllClose(mean_ap, expected_mean_ap)

  def testMeanPrecisionsWorks(self):
    # Define input.
    predictions = _CreateRetrievalPredictions()
    solution = _CreateRetrievalSolution()

    # Run tested function.
    mean_precisions = metrics.MeanPrecisions(
        predictions, solution, max_predictions=2)

    # Define expected results.
    expected_mean_precisions = [0.5, 0.375]

    # Compare actual and expected results.
    self.assertAllClose(mean_precisions, expected_mean_precisions)

  def testMeanMedianPositionWorks(self):
    # Define input.
    predictions = _CreateRetrievalPredictions()
    solution = _CreateRetrievalSolution()

    # Run tested function.
    mean_position, median_position = metrics.MeanMedianPosition(
        predictions, solution)

    # Define expected results.
    expected_mean_position = 26.25
    expected_median_position = 1.5

    # Compare actual and expected results.
    self.assertAllClose(mean_position, expected_mean_position)
    self.assertAllClose(median_position, expected_median_position)

  def testMeanMedianPositionMaxPredictionsWorks(self):
    # Define input.
    predictions = _CreateRetrievalPredictions()
    solution = _CreateRetrievalSolution()

    # Run tested function.
    mean_position, median_position = metrics.MeanMedianPosition(
        predictions, solution, max_predictions=1)

    # Define expected results.
    expected_mean_position = 1.5
    expected_median_position = 1.5

    # Compare actual and expected results.
    self.assertAllClose(mean_position, expected_mean_position)
    self.assertAllClose(median_position, expected_median_position)


if __name__ == '__main__':
  tf.test.main()
