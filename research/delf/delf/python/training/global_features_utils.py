# Copyright 2021 The TensorFlow Authors All Rights Reserved.
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
"""Utilities for the global model training."""

from absl import logging

from tensorboard import program
import numpy as np

from delf.python.datasets.revisited_op import dataset


class AverageMeter():
  """Computes and stores the average and current value of loss."""

  def __init__(self):
    """Initialization of the AverageMeter."""
    self.reset()

  def reset(self):
    """Resets all the values."""
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    """Updates values in the AverageMeter.

    Args:
      val: Float, loss value.
      n: Integer, number of instances.
    """
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def compute_metrics_and_print(dataset_name, sorted_index_ids, ground_truth,
                              desired_pr_ranks=[1, 5, 10], log=True):
  """Computes and loggs easy/medium/hard ground-truth metrics for Revisited
  datasets.

  Args:
    dataset_name: String, name of the dataset.
    sorted_index_ids: Integer NumPy array of shape [#queries, #index_images].
      For each query, contains an array denoting the most relevant index images,
      sorted from most to least relevant.
    ground_truth: List containing ground-truth information for dataset. Each
      entry is a dict corresponding to the ground-truth information for a query.
      The dict has keys 'ok' and 'junk', mapping to a NumPy array of integers.
    desired_pr_ranks: List of integers containing the desired precision/recall
      ranks to be reported. E.g., if precision@1/recall@1 and
      precision@10/recall@10 are desired, this should be set to [1, 10]. The
      largest item should be <= #sorted_index_ids.

  Returns:
    mAP: [metricsE, metricsM, metricsH] List of the metrics for different
      levels of complexity. Each metrics is a list containing:
      mean_average_precision (float), mean_precisions (NumPy array of
      floats, with shape [len(desired_pr_ranks)]), mean_recalls (NumPy array
      of floats, with shape [len(desired_pr_ranks)]), average_precisions
      (NumPy array of floats, with shape [#queries]), precisions (NumPy array of
      floats, with shape [#queries, len(desired_pr_ranks)]), recalls (NumPy
      array of floats, with shape [#queries, len(desired_pr_ranks)]).
  """

  if dataset_name.startswith('roxford5k') or dataset_name.startswith(
          'rparis6k'):
    (easy_ground_truth, medium_ground_truth,
     hard_ground_truth) = dataset.ParseEasyMediumHardGroundTruth(ground_truth)

    metrics_easy = dataset.ComputeMetrics(sorted_index_ids, easy_ground_truth,
                                          desired_pr_ranks)
    metrics_medium = dataset.ComputeMetrics(sorted_index_ids,
                                            medium_ground_truth,
                                            desired_pr_ranks)
    metrics_hard = dataset.ComputeMetrics(sorted_index_ids, hard_ground_truth,
                                          desired_pr_ranks)

    debug_and_log(
      '>> {}: mAP E: {}, M: {}, H: {}'.format(
        dataset_name, np.around(metrics_easy[0] * 100, decimals=2),
        np.around(metrics_medium[0] * 100, decimals=2),
        np.around(metrics_hard[0] * 100, decimals=2)), log=log)

    debug_and_log(
      '>> {}: mP@k{} E: {}, M: {}, H: {}'.format(
        dataset_name, desired_pr_ranks,
        np.around(metrics_easy[1] * 100, decimals=2),
        np.around(metrics_medium[1] * 100, decimals=2),
        np.around(metrics_hard[1] * 100, decimals=2)), log=log)

    return metrics_easy, metrics_medium, metrics_hard


def htime(time_difference):
  """Time formatting function.

  Depending on the value of `time_difference` outputs time in an appropriate
  time format.

  Args:
    time_difference: Float, time difference between the two events.

  Returns:
    time: String representing time in an appropriate time format.
  """
  time_difference = round(time_difference)

  days = time_difference // 86400
  hours = time_difference // 3600 % 24
  minutes = time_difference // 60 % 60
  seconds = time_difference % 60

  if days > 0:
    return '{:d}d {:d}h {:d}m {:d}s'.format(days, hours, minutes, seconds)
  if hours > 0:
    return '{:d}h {:d}m {:d}s'.format(hours, minutes, seconds)
  if minutes > 0:
    return '{:d}m {:d}s'.format(minutes, seconds)
  return '{:d}s'.format(seconds)


def debug_and_log(msg, debug=True, log=True, debug_on_the_same_line=False):
  """Outputs `msg` to both stdout (if in the debug mode) and the log file.

  Args:
    msg: String, message to be logged.
    debug: Bool, if True, will print `msg` to stdout.
    log: Bool, if True, will redirect `msg` to the logfile.
    debug_on_the_same_line: Bool, if True, will print `msg` to stdout without
      a new line. When using this mode, logging to a logfile is disabled.
  """
  if debug_on_the_same_line:
    print(msg, end='')
    return
  if debug:
    print(msg)
  if log:
    logging.info(msg)


def launch_tensorboard(log_dir):
  """Runs tensorboard with the given `log_dir` and waits for user input to kill
  the app.
  
  Args:
    log_dir: String, directory to start tensorboard in.
  """
  tb = program.TensorBoard()
  tb.configure(argv=[None, '--logdir', log_dir])
  url = tb.launch()
  debug_and_log("Launching Tensorboard: {}".format(url))
