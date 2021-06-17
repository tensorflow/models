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
"""Computes metrics for Google Landmarks Recognition dataset predictions.

Metrics are written to stdout.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from absl import app
from delf.python.datasets.google_landmarks_dataset import dataset_file_io
from delf.python.datasets.google_landmarks_dataset import metrics

cmd_args = None


def main(argv):
  if len(argv) > 1:
    raise RuntimeError('Too many command-line arguments.')

  # Read solution.
  print('Reading solution...')
  public_solution, private_solution, ignored_ids = dataset_file_io.ReadSolution(
      cmd_args.solution_path, dataset_file_io.RECOGNITION_TASK_ID)
  print('done!')

  # Read predictions.
  print('Reading predictions...')
  public_predictions, private_predictions = dataset_file_io.ReadPredictions(
      cmd_args.predictions_path, set(public_solution.keys()),
      set(private_solution.keys()), set(ignored_ids),
      dataset_file_io.RECOGNITION_TASK_ID)
  print('done!')

  # Global Average Precision.
  print('**********************************************')
  print('(Public)  Global Average Precision: %f' %
        metrics.GlobalAveragePrecision(public_predictions, public_solution))
  print('(Private) Global Average Precision: %f' %
        metrics.GlobalAveragePrecision(private_predictions, private_solution))

  # Global Average Precision ignoring non-landmark queries.
  print('**********************************************')
  print(
      '(Public)  Global Average Precision ignoring non-landmark queries: %f' %
      metrics.GlobalAveragePrecision(
          public_predictions, public_solution, ignore_non_gt_test_images=True))
  print(
      '(Private) Global Average Precision ignoring non-landmark queries: %f' %
      metrics.GlobalAveragePrecision(
          private_predictions, private_solution,
          ignore_non_gt_test_images=True))

  # Top-1 accuracy.
  print('**********************************************')
  print('(Public)  Top-1 accuracy: %.2f' %
        (100.0 * metrics.Top1Accuracy(public_predictions, public_solution)))
  print('(Private) Top-1 accuracy: %.2f' %
        (100.0 * metrics.Top1Accuracy(private_predictions, private_solution)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--predictions_path',
      type=str,
      default='/tmp/predictions.csv',
      help="""
      Path to CSV predictions file, formatted with columns 'id,landmarks' (the
      file should include a header).
      """)
  parser.add_argument(
      '--solution_path',
      type=str,
      default='/tmp/solution.csv',
      help="""
      Path to CSV solution file, formatted with columns 'id,landmarks,Usage'
      (the file should include a header).
      """)
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
