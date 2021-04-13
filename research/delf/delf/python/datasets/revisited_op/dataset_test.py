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
"""Tests for the python library parsing Revisited Oxford/Paris datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import numpy as np
import tensorflow as tf

from delf.python.detect_to_retrieve import dataset

FLAGS = flags.FLAGS


class DatasetTest(tf.test.TestCase):

  def testParseEasyMediumHardGroundTruth(self):
    # Define input.
    ground_truth = [{
        'easy': np.array([10, 56, 100]),
        'hard': np.array([0]),
        'junk': np.array([6, 90])
    }, {
        'easy': np.array([], dtype='int64'),
        'hard': [5],
        'junk': [99, 100]
    }, {
        'easy': [33],
        'hard': [66, 99],
        'junk': np.array([], dtype='int64')
    }]

    # Run tested function.
    (easy_ground_truth, medium_ground_truth,
     hard_ground_truth) = dataset.ParseEasyMediumHardGroundTruth(ground_truth)

    # Define expected outputs.
    expected_easy_ground_truth = [{
        'ok': np.array([10, 56, 100]),
        'junk': np.array([6, 90, 0])
    }, {
        'ok': np.array([], dtype='int64'),
        'junk': np.array([99, 100, 5])
    }, {
        'ok': np.array([33]),
        'junk': np.array([66, 99])
    }]
    expected_medium_ground_truth = [{
        'ok': np.array([10, 56, 100, 0]),
        'junk': np.array([6, 90])
    }, {
        'ok': np.array([5]),
        'junk': np.array([99, 100])
    }, {
        'ok': np.array([33, 66, 99]),
        'junk': np.array([], dtype='int64')
    }]
    expected_hard_ground_truth = [{
        'ok': np.array([0]),
        'junk': np.array([6, 90, 10, 56, 100])
    }, {
        'ok': np.array([5]),
        'junk': np.array([99, 100])
    }, {
        'ok': np.array([66, 99]),
        'junk': np.array([33])
    }]

    # Compare actual versus expected.
    def _AssertListOfDictsOfArraysAreEqual(ground_truth, expected_ground_truth):
      """Helper function to compare ground-truth data.

      Args:
        ground_truth: List of dicts of arrays.
        expected_ground_truth: List of dicts of arrays.
      """
      self.assertEqual(len(ground_truth), len(expected_ground_truth))

      for i, ground_truth_entry in enumerate(ground_truth):
        self.assertEqual(sorted(ground_truth_entry.keys()), ['junk', 'ok'])
        self.assertAllEqual(ground_truth_entry['junk'],
                            expected_ground_truth[i]['junk'])
        self.assertAllEqual(ground_truth_entry['ok'],
                            expected_ground_truth[i]['ok'])

    _AssertListOfDictsOfArraysAreEqual(easy_ground_truth,
                                       expected_easy_ground_truth)
    _AssertListOfDictsOfArraysAreEqual(medium_ground_truth,
                                       expected_medium_ground_truth)
    _AssertListOfDictsOfArraysAreEqual(hard_ground_truth,
                                       expected_hard_ground_truth)

  def testAdjustPositiveRanksWorks(self):
    # Define inputs.
    positive_ranks = np.array([0, 2, 6, 10, 20])
    junk_ranks = np.array([1, 8, 9, 30])

    # Run tested function.
    adjusted_positive_ranks = dataset.AdjustPositiveRanks(
        positive_ranks, junk_ranks)

    # Define expected output.
    expected_adjusted_positive_ranks = [0, 1, 5, 7, 17]

    # Compare actual versus expected.
    self.assertAllEqual(adjusted_positive_ranks,
                        expected_adjusted_positive_ranks)

  def testComputeAveragePrecisionWorks(self):
    # Define input.
    positive_ranks = [0, 2, 5]

    # Run tested function.
    average_precision = dataset.ComputeAveragePrecision(positive_ranks)

    # Define expected output.
    expected_average_precision = 0.677778

    # Compare actual versus expected.
    self.assertAllClose(average_precision, expected_average_precision)

  def testComputePRAtRanksWorks(self):
    # Define inputs.
    positive_ranks = np.array([0, 2, 5])
    desired_pr_ranks = np.array([1, 5, 10])

    # Run tested function.
    precisions, recalls = dataset.ComputePRAtRanks(positive_ranks,
                                                   desired_pr_ranks)

    # Define expected outputs.
    expected_precisions = [1.0, 0.4, 0.5]
    expected_recalls = [0.333333, 0.666667, 1.0]

    # Compare actual versus expected.
    self.assertAllClose(precisions, expected_precisions)
    self.assertAllClose(recalls, expected_recalls)

  def testComputeMetricsWorks(self):
    # Define inputs: 3 queries. For the last one, there are no expected images
    # to be retrieved
    sorted_index_ids = np.array([[4, 2, 0, 1, 3], [0, 2, 4, 1, 3],
                                 [0, 1, 2, 3, 4]])
    ground_truth = [{
        'ok': np.array([0, 1]),
        'junk': np.array([2])
    }, {
        'ok': np.array([0, 4]),
        'junk': np.array([], dtype='int64')
    }, {
        'ok': np.array([], dtype='int64'),
        'junk': np.array([], dtype='int64')
    }]
    desired_pr_ranks = [1, 2, 5]

    # Run tested function.
    (mean_average_precision, mean_precisions, mean_recalls, average_precisions,
     precisions, recalls) = dataset.ComputeMetrics(sorted_index_ids,
                                                   ground_truth,
                                                   desired_pr_ranks)

    # Define expected outputs.
    expected_mean_average_precision = 0.604167
    expected_mean_precisions = [0.5, 0.5, 0.666667]
    expected_mean_recalls = [0.25, 0.5, 1.0]
    expected_average_precisions = [0.416667, 0.791667, float('nan')]
    expected_precisions = [[0.0, 0.5, 0.666667], [1.0, 0.5, 0.666667],
                           [float('nan'),
                            float('nan'),
                            float('nan')]]
    expected_recalls = [[0.0, 0.5, 1.0], [0.5, 0.5, 1.0],
                        [float('nan'), float('nan'),
                         float('nan')]]

    # Compare actual versus expected.
    self.assertAllClose(mean_average_precision, expected_mean_average_precision)
    self.assertAllClose(mean_precisions, expected_mean_precisions)
    self.assertAllClose(mean_recalls, expected_mean_recalls)
    self.assertAllClose(average_precisions, expected_average_precisions)
    self.assertAllClose(precisions, expected_precisions)
    self.assertAllClose(recalls, expected_recalls)

  def testSaveMetricsFileWorks(self):
    # Define inputs.
    mean_average_precision = {'hard': 0.7, 'medium': 0.9}
    mean_precisions = {
        'hard': np.array([1.0, 0.8]),
        'medium': np.array([1.0, 1.0])
    }
    mean_recalls = {
        'hard': np.array([0.5, 0.8]),
        'medium': np.array([0.5, 1.0])
    }
    pr_ranks = [1, 5]
    output_path = os.path.join(FLAGS.test_tmpdir, 'metrics.txt')

    # Run tested function.
    dataset.SaveMetricsFile(mean_average_precision, mean_precisions,
                            mean_recalls, pr_ranks, output_path)

    # Define expected results.
    expected_metrics = ('hard\n'
                        '  mAP=70.0\n'
                        '  mP@k[1 5] [100.  80.]\n'
                        '  mR@k[1 5] [50. 80.]\n'
                        'medium\n'
                        '  mAP=90.0\n'
                        '  mP@k[1 5] [100. 100.]\n'
                        '  mR@k[1 5] [ 50. 100.]\n')

    # Parse actual results, and compare to expected.
    with tf.io.gfile.GFile(output_path) as f:
      metrics = f.read()

    self.assertEqual(metrics, expected_metrics)

  def testSaveAndReadMetricsWorks(self):
    # Define inputs.
    mean_average_precision = {'hard': 0.7, 'medium': 0.9}
    mean_precisions = {
        'hard': np.array([1.0, 0.8]),
        'medium': np.array([1.0, 1.0])
    }
    mean_recalls = {
        'hard': np.array([0.5, 0.8]),
        'medium': np.array([0.5, 1.0])
    }
    pr_ranks = [1, 5]
    output_path = os.path.join(FLAGS.test_tmpdir, 'metrics.txt')

    # Run tested functions.
    dataset.SaveMetricsFile(mean_average_precision, mean_precisions,
                            mean_recalls, pr_ranks, output_path)
    (read_mean_average_precision, read_pr_ranks, read_mean_precisions,
     read_mean_recalls) = dataset.ReadMetricsFile(output_path)

    # Compares actual and expected metrics.
    self.assertEqual(read_mean_average_precision, mean_average_precision)
    self.assertEqual(read_pr_ranks, pr_ranks)
    self.assertEqual(read_mean_precisions.keys(), mean_precisions.keys())
    self.assertAllEqual(read_mean_precisions['hard'], mean_precisions['hard'])
    self.assertAllEqual(read_mean_precisions['medium'],
                        mean_precisions['medium'])
    self.assertEqual(read_mean_recalls.keys(), mean_recalls.keys())
    self.assertAllEqual(read_mean_recalls['hard'], mean_recalls['hard'])
    self.assertAllEqual(read_mean_recalls['medium'], mean_recalls['medium'])

  def testReadMetricsWithRepeatedProtocolFails(self):
    # Define inputs.
    input_path = os.path.join(FLAGS.test_tmpdir, 'metrics.txt')
    with tf.io.gfile.GFile(input_path, 'w') as f:
      f.write('hard\n'
              '  mAP=70.0\n'
              '  mP@k[1 5] [ 100.   80.]\n'
              '  mR@k[1 5] [ 50.  80.]\n'
              'medium\n'
              '  mAP=90.0\n'
              '  mP@k[1 5] [ 100.  100.]\n'
              '  mR@k[1 5] [  50.  100.]\n'
              'medium\n'
              '  mAP=90.0\n'
              '  mP@k[1 5] [ 100.  100.]\n'
              '  mR@k[1 5] [  50.  100.]\n')

    # Run tested functions.
    with self.assertRaisesRegex(ValueError, 'Malformed input'):
      dataset.ReadMetricsFile(input_path)


if __name__ == '__main__':
  tf.test.main()
