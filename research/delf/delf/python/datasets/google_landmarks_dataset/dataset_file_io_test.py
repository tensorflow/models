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
"""Tests for dataset file IO module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import flags
import tensorflow as tf

from delf.python.datasets.google_landmarks_dataset import dataset_file_io

FLAGS = flags.FLAGS


class DatasetFileIoTest(tf.test.TestCase):

  def testReadRecognitionSolutionWorks(self):
    # Define inputs.
    file_path = os.path.join(FLAGS.test_tmpdir, 'recognition_solution.csv')
    with tf.io.gfile.GFile(file_path, 'w') as f:
      f.write('id,landmarks,Usage\n')
      f.write('0123456789abcdef,0 12,Public\n')
      f.write('0223456789abcdef,,Public\n')
      f.write('0323456789abcdef,100,Ignored\n')
      f.write('0423456789abcdef,1,Private\n')
      f.write('0523456789abcdef,,Ignored\n')

    # Run tested function.
    (public_solution, private_solution,
     ignored_ids) = dataset_file_io.ReadSolution(
         file_path, dataset_file_io.RECOGNITION_TASK_ID)

    # Define expected results.
    expected_public_solution = {
        '0123456789abcdef': [0, 12],
        '0223456789abcdef': []
    }
    expected_private_solution = {
        '0423456789abcdef': [1],
    }
    expected_ignored_ids = ['0323456789abcdef', '0523456789abcdef']

    # Compare actual and expected results.
    self.assertEqual(public_solution, expected_public_solution)
    self.assertEqual(private_solution, expected_private_solution)
    self.assertEqual(ignored_ids, expected_ignored_ids)

  def testReadRetrievalSolutionWorks(self):
    # Define inputs.
    file_path = os.path.join(FLAGS.test_tmpdir, 'retrieval_solution.csv')
    with tf.io.gfile.GFile(file_path, 'w') as f:
      f.write('id,images,Usage\n')
      f.write('0123456789abcdef,None,Ignored\n')
      f.write('0223456789abcdef,fedcba9876543210 fedcba9876543200,Public\n')
      f.write('0323456789abcdef,fedcba9876543200,Private\n')
      f.write('0423456789abcdef,fedcba9876543220,Private\n')
      f.write('0523456789abcdef,None,Ignored\n')

    # Run tested function.
    (public_solution, private_solution,
     ignored_ids) = dataset_file_io.ReadSolution(
         file_path, dataset_file_io.RETRIEVAL_TASK_ID)

    # Define expected results.
    expected_public_solution = {
        '0223456789abcdef': ['fedcba9876543210', 'fedcba9876543200'],
    }
    expected_private_solution = {
        '0323456789abcdef': ['fedcba9876543200'],
        '0423456789abcdef': ['fedcba9876543220'],
    }
    expected_ignored_ids = ['0123456789abcdef', '0523456789abcdef']

    # Compare actual and expected results.
    self.assertEqual(public_solution, expected_public_solution)
    self.assertEqual(private_solution, expected_private_solution)
    self.assertEqual(ignored_ids, expected_ignored_ids)

  def testReadRecognitionPredictionsWorks(self):
    # Define inputs.
    file_path = os.path.join(FLAGS.test_tmpdir, 'recognition_predictions.csv')
    with tf.io.gfile.GFile(file_path, 'w') as f:
      f.write('id,landmarks\n')
      f.write('0123456789abcdef,12 0.1 \n')
      f.write('0423456789abcdef,0 19.0\n')
      f.write('0223456789abcdef,\n')
      f.write('\n')
      f.write('0523456789abcdef,14 0.01\n')
    public_ids = ['0123456789abcdef', '0223456789abcdef']
    private_ids = ['0423456789abcdef']
    ignored_ids = ['0323456789abcdef', '0523456789abcdef']

    # Run tested function.
    public_predictions, private_predictions = dataset_file_io.ReadPredictions(
        file_path, public_ids, private_ids, ignored_ids,
        dataset_file_io.RECOGNITION_TASK_ID)

    # Define expected results.
    expected_public_predictions = {
        '0123456789abcdef': {
            'class': 12,
            'score': 0.1
        }
    }
    expected_private_predictions = {
        '0423456789abcdef': {
            'class': 0,
            'score': 19.0
        }
    }

    # Compare actual and expected results.
    self.assertEqual(public_predictions, expected_public_predictions)
    self.assertEqual(private_predictions, expected_private_predictions)

  def testReadRetrievalPredictionsWorks(self):
    # Define inputs.
    file_path = os.path.join(FLAGS.test_tmpdir, 'retrieval_predictions.csv')
    with tf.io.gfile.GFile(file_path, 'w') as f:
      f.write('id,images\n')
      f.write('0123456789abcdef,fedcba9876543250 \n')
      f.write('0423456789abcdef,fedcba9876543260\n')
      f.write('0223456789abcdef,fedcba9876543210 fedcba9876543200 '
              'fedcba9876543220\n')
      f.write('\n')
      f.write('0523456789abcdef,\n')
    public_ids = ['0223456789abcdef']
    private_ids = ['0323456789abcdef', '0423456789abcdef']
    ignored_ids = ['0123456789abcdef', '0523456789abcdef']

    # Run tested function.
    public_predictions, private_predictions = dataset_file_io.ReadPredictions(
        file_path, public_ids, private_ids, ignored_ids,
        dataset_file_io.RETRIEVAL_TASK_ID)

    # Define expected results.
    expected_public_predictions = {
        '0223456789abcdef': [
            'fedcba9876543210', 'fedcba9876543200', 'fedcba9876543220'
        ]
    }
    expected_private_predictions = {'0423456789abcdef': ['fedcba9876543260']}

    # Compare actual and expected results.
    self.assertEqual(public_predictions, expected_public_predictions)
    self.assertEqual(private_predictions, expected_private_predictions)


if __name__ == '__main__':
  tf.test.main()
