# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""Logging utilities for benchmark."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import json
import numbers
import os

import tensorflow as tf

_METRIC_LOG_FILE_NAME = "metric.log"
_DATE_TIME_FORMAT_PATTERN = "%Y-%m-%dT%H:%M:%S.%fZ"


class BenchmarkLogger(object):
  """Class to log the benchmark information to local disk."""

  def __init__(self, logging_dir):
    self._logging_dir = logging_dir
    if not tf.gfile.IsDirectory(self._logging_dir):
      tf.gfile.MakeDirs(self._logging_dir)

  def log_metric(self, name, value, unit=None, global_step=None, extras=None):
    """Log the benchmark metric information to local file.

    Currently the logging is done in a synchronized way. This should be updated
    to log asynchronously.

    Args:
      name: string, the name of the metric to log.
      value: number, the value of the metric. The value will not be logged if it
        is not a number type.
      unit: string, the unit of the metric, E.g "image per second".
      global_step: int, the global_step when the metric is logged.
      extras: map of string:string, the extra information about the metric.
    """
    if not isinstance(value, numbers.Number):
      tf.logging.warning(
          "Metric value to log should be a number. Got %s", type(value))
      return

    with tf.gfile.GFile(
        os.path.join(self._logging_dir, _METRIC_LOG_FILE_NAME), "a") as f:
      metric = {
          "name": name,
          "value": float(value),
          "unit": unit,
          "global_step": global_step,
          "timestamp": datetime.datetime.now().strftime(
              _DATE_TIME_FORMAT_PATTERN),
          "extras": extras}
      try:
        json.dump(metric, f)
        f.write("\n")
      except (TypeError, ValueError) as e:
        tf.logging.warning("Failed to dump metric to log file: "
                           "name %s, value %s, error %s", name, value, e)
