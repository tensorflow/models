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

"""Logging utilities for benchmark.

For collecting local environment metrics like CPU and memory, certain python
packages need be installed. See README for details.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import json
import multiprocessing
import numbers
import os
import threading

import tensorflow as tf
from tensorflow.python.client import device_lib

METRIC_LOG_FILE_NAME = "metric.log"
BENCHMARK_RUN_LOG_FILE_NAME = "benchmark_run.log"
_DATE_TIME_FORMAT_PATTERN = "%Y-%m-%dT%H:%M:%S.%fZ"


# Don't use it directly. Use get_benchmark_logger to access a logger.
_benchmark_logger = None
_logger_lock = threading.Lock()


def config_benchmark_logger(logging_dir):
  """Config the global benchmark logger"""
  _logger_lock.acquire()
  try:
    global _benchmark_logger
    if logging_dir:
      _benchmark_logger = BenchmarkFileLogger(logging_dir)
    else:
      _benchmark_logger = BaseBenchmarkLogger()
  finally:
    _logger_lock.release()
  return _benchmark_logger


def get_benchmark_logger():
  if not _benchmark_logger:
    config_benchmark_logger(None)

  return _benchmark_logger


class BaseBenchmarkLogger(object):
  """Class to log the benchmark information to STDOUT."""

  def log_evaluation_result(self, eval_results):
    """Log the evaluation result.

    The evaluate result is a dictionary that contains metrics defined in
    model_fn. It also contains a entry for global_step which contains the value
    of the global step when evaluation was performed.

    Args:
      eval_results: dict, the result of evaluate.
    """
    if not isinstance(eval_results, dict):
      tf.logging.warning("eval_results should be dictionary for logging. "
                         "Got %s", type(eval_results))
      return
    global_step = eval_results[tf.GraphKeys.GLOBAL_STEP]
    for key in sorted(eval_results):
      if key != tf.GraphKeys.GLOBAL_STEP:
        self.log_metric(key, eval_results[key], global_step=global_step)

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
    extras = _convert_to_json_dict(extras)

    tf.logging.info("Benchmark metric: "
                    "Name %s, value %d, unit %s, global_step %d, extras %s",
                    name, value, unit, global_step, extras)

  def log_run_info(self, model_name):
    tf.logging.info("Benchmark run: %s", _gather_run_info(model_name))


class BenchmarkFileLogger(BaseBenchmarkLogger):
  """Class to log the benchmark information to local disk."""

  def __init__(self, logging_dir):
    super(BenchmarkFileLogger, self).__init__()
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
    extras = _convert_to_json_dict(extras)

    with tf.gfile.GFile(
        os.path.join(self._logging_dir, METRIC_LOG_FILE_NAME), "a") as f:
      metric = {
          "name": name,
          "value": float(value),
          "unit": unit,
          "global_step": global_step,
          "timestamp": datetime.datetime.utcnow().strftime(
              _DATE_TIME_FORMAT_PATTERN),
          "extras": extras}
      try:
        json.dump(metric, f)
        f.write("\n")
      except (TypeError, ValueError) as e:
        tf.logging.warning("Failed to dump metric to log file: "
                           "name %s, value %s, error %s", name, value, e)

  def log_run_info(self, model_name):
    """Collect most of the TF runtime information for the local env.

    The schema of the run info follows official/benchmark/datastore/schema.

    Args:
      model_name: string, the name of the model.
    """
    run_info = _gather_run_info(model_name)

    with tf.gfile.GFile(os.path.join(
        self._logging_dir, BENCHMARK_RUN_LOG_FILE_NAME), "w") as f:
      try:
        json.dump(run_info, f)
        f.write("\n")
      except (TypeError, ValueError) as e:
        tf.logging.warning("Failed to dump benchmark run info to log file: %s",
                           e)


def _gather_run_info(model_name):
  """Collect the benchmark run information for the local environment."""
  run_info = {
      "model_name": model_name,
      "machine_config": {},
      "run_date": datetime.datetime.utcnow().strftime(
          _DATE_TIME_FORMAT_PATTERN)}
  _collect_tensorflow_info(run_info)
  _collect_tensorflow_environment_variables(run_info)
  _collect_cpu_info(run_info)
  _collect_gpu_info(run_info)
  _collect_memory_info(run_info)
  return run_info


def _collect_tensorflow_info(run_info):
  run_info["tensorflow_version"] = {
      "version": tf.VERSION, "git_hash": tf.GIT_VERSION}


def _collect_tensorflow_environment_variables(run_info):
  run_info["tensorflow_environment_variables"] = [
      {"name": k, "value": v}
      for k, v in sorted(os.environ.items()) if k.startswith("TF_")]


# The following code is mirrored from tensorflow/tools/test/system_info_lib
# which is not exposed for import.
def _collect_cpu_info(run_info):
  """Collect the CPU information for the local environment."""
  cpu_info = {}

  cpu_info["num_cores"] = multiprocessing.cpu_count()

  # Note: cpuinfo is not installed in the TensorFlow OSS tree.
  # It is installable via pip.
  import cpuinfo    # pylint: disable=g-import-not-at-top

  info = cpuinfo.get_cpu_info()
  cpu_info["cpu_info"] = info["brand"]
  cpu_info["mhz_per_cpu"] = info["hz_advertised_raw"][0] / 1.0e6

  run_info["machine_config"]["cpu_info"] = cpu_info


def _collect_gpu_info(run_info):
  """Collect local GPU information by TF device library."""
  gpu_info = {}
  local_device_protos = device_lib.list_local_devices()

  gpu_info["count"] = len([d for d in local_device_protos
                           if d.device_type == "GPU"])
  # The device description usually is a JSON string, which contains the GPU
  # model info, eg:
  # "device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:00:04.0"
  for d in local_device_protos:
    if d.device_type == "GPU":
      gpu_info["model"] = _parse_gpu_model(d.physical_device_desc)
      # Assume all the GPU connected are same model
      break
  run_info["machine_config"]["gpu_info"] = gpu_info


def _collect_memory_info(run_info):
  # Note: psutil is not installed in the TensorFlow OSS tree.
  # It is installable via pip.
  import psutil   # pylint: disable=g-import-not-at-top
  vmem = psutil.virtual_memory()
  run_info["machine_config"]["memory_total"] = vmem.total
  run_info["machine_config"]["memory_available"] = vmem.available


def _parse_gpu_model(physical_device_desc):
  # Assume all the GPU connected are same model
  for kv in physical_device_desc.split(","):
    k, _, v = kv.partition(":")
    if k.strip() == "name":
      return v.strip()
  return None


def _convert_to_json_dict(input_dict):
  if input_dict:
    return [{"name": k, "value": v} for k, v in sorted(input_dict.items())]
  else:
    return []
