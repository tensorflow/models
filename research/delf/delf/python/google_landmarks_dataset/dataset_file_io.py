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
"""IO module for files from Landmark recognition/retrieval challenges."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv

import tensorflow as tf

RECOGNITION_TASK_ID = 'recognition'
RETRIEVAL_TASK_ID = 'retrieval'


def ReadSolution(file_path, task):
  """Reads solution from file, for a given task.

  Args:
    file_path: Path to CSV file with solution. File contains a header.
    task: Type of challenge task. Supported values: 'recognition', 'retrieval'.

  Returns:
    public_solution: Dict mapping test image ID to list of ground-truth IDs, for
      the Public subset of test images. If `task` == 'recognition', the IDs are
      integers corresponding to landmark IDs. If `task` == 'retrieval', the IDs
      are strings corresponding to index image IDs.
    private_solution: Same as `public_solution`, but for the private subset of
      test images.
    ignored_ids: List of test images that are ignored in scoring.

  Raises:
    ValueError: If Usage field is not Public, Private or Ignored; or if `task`
      is not supported.
  """
  public_solution = {}
  private_solution = {}
  ignored_ids = []
  with tf.gfile.GFile(file_path, 'r') as csv_file:
    reader = csv.reader(csv_file)
    next(reader, None)  # Skip header.
    for row in reader:
      test_id = row[0]
      if row[2] == 'Ignored':
        ignored_ids.append(test_id)
      else:
        ground_truth_ids = []
        if task == RECOGNITION_TASK_ID:
          if row[1]:
            for landmark_id in row[1].split(' '):
              ground_truth_ids.append(int(landmark_id))
        elif task == RETRIEVAL_TASK_ID:
          for image_id in row[1].split(' '):
            ground_truth_ids.append(image_id)
        else:
          raise ValueError('Unrecognized task: %s' % task)

        if row[2] == 'Public':
          public_solution[test_id] = ground_truth_ids
        elif row[2] == 'Private':
          private_solution[test_id] = ground_truth_ids
        else:
          raise ValueError('Test image %s has unrecognized Usage tag %s' %
                           (row[0], row[2]))

  return public_solution, private_solution, ignored_ids


def ReadPredictions(file_path, public_ids, private_ids, ignored_ids, task):
  """Reads predictions from file, for a given task.

  Args:
    file_path: Path to CSV file with predictions. File contains a header.
    public_ids: Set (or list) of test image IDs in Public subset of test images.
    private_ids: Same as `public_ids`, but for the private subset of test
      images.
    ignored_ids: Set (or list) of test image IDs that are ignored in scoring and
      are associated to no ground-truth.
    task: Type of challenge task. Supported values: 'recognition', 'retrieval'.

  Returns:
    public_predictions: Dict mapping test image ID to prediction, for the Public
      subset of test images. If `task` == 'recognition', the prediction is a
      dict with keys 'class' (integer) and 'score' (float). If `task` ==
      'retrieval', the prediction is a list of strings corresponding to index
      image IDs.
    private_predictions: Same as `public_predictions`, but for the private
      subset of test images.

  Raises:
    ValueError:
      - If test image ID is unrecognized/repeated;
      - If `task` is not supported;
      - If prediction is malformed.
  """
  public_predictions = {}
  private_predictions = {}
  with tf.gfile.GFile(file_path, 'r') as csv_file:
    reader = csv.reader(csv_file)
    next(reader, None)  # Skip header.
    for row in reader:
      # Skip row if empty.
      if not row:
        continue

      test_id = row[0]

      # Makes sure this query has not yet been seen.
      if test_id in public_predictions:
        raise ValueError('Test image %s is repeated.' % test_id)
      if test_id in private_predictions:
        raise ValueError('Test image %s is repeated' % test_id)

      # If ignored, skip it.
      if test_id in ignored_ids:
        continue

      # Only parse result if there is a prediction.
      if row[1]:
        prediction_split = row[1].split(' ')
        # Remove empty spaces at end (if any).
        if not prediction_split[-1]:
          prediction_split = prediction_split[:-1]

        if task == RECOGNITION_TASK_ID:
          if len(prediction_split) != 2:
            raise ValueError('Prediction is malformed: there should only be 2 '
                             'elements in second column, but found %d for test '
                             'image %s' % (len(prediction_split), test_id))

          landmark_id = int(prediction_split[0])
          score = float(prediction_split[1])
          prediction_entry = {'class': landmark_id, 'score': score}
        elif task == RETRIEVAL_TASK_ID:
          prediction_entry = prediction_split
        else:
          raise ValueError('Unrecognized task: %s' % task)

        if test_id in public_ids:
          public_predictions[test_id] = prediction_entry
        elif test_id in private_ids:
          private_predictions[test_id] = prediction_entry
        else:
          raise ValueError('test_id %s is unrecognized' % test_id)

  return public_predictions, private_predictions
