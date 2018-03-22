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
import glob
import json
import multiprocessing
import numbers
import os
import re

# pylint: disable=g-bad-import-order
# Note: cpuinfo and psutil are not installed in the TensorFlow OSS tree.
# They are installable via pip.
import cpuinfo
import psutil
# pylint: enable=g-bad-import-order

import tensorflow as tf
from tensorflow.python.client import device_lib

_METRIC_LOG_FILE_NAME = "metric.log"
_BENCHMARK_RUN_LOG_FILE_NAME = "benchmark_run.log"
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

  def log_run_info(self, model_name):
    """Collect most of the TF runtime information for the local env.

    The schema of the run info follows official/benchmark/datastore/schema.
    """
    run_info = {"model_name": model_name}
    _collect_tensorflow_info(run_info)
    _collect_environment_variable(run_info)
    _collect_cpu_info(run_info)
    _collect_gpu_info(run_info)
    _collect_memory_info(run_info)

    with tf.gfile.GFile(os.path.join(
        self._logging_dir, _BENCHMARK_RUN_LOG_FILE_NAME), "w") as f:
      try:
        json.dump(run_info, f)
        f.write("\n")
      except (TypeError, ValueError) as e:
        tf.logging.warning("Failed to dump benchmark run info to log file: %s",
                           e)

def _collect_tensorflow_info(run_info):
  run_info["tensorflow_version"] = {
      "version": tf.VERSION, "git_hash": tf.GIT_VERSION}

def _collect_environment_variable(run_info):
  run_info["environment_variable"] = {
      k:v for k, v in os.environ.items() if k.startswith("TF_")}

# The following code is mirrored from tensorflow/tools/test/system_info_lib
# which is not exposed for import.
def _collect_cpu_info(run_info):
  cpu_info = {}

  cpu_info["num_cores"] = multiprocessing.cpu_count()
  # Gather num_cores_allowed
  try:
    with tf.gfile.GFile("/proc/self/status", "rb") as fh:
      nc = re.search(r"(?m)^Cpus_allowed:\s*(.*)$", fh.read())
    if nc:  # e.g. "ff" => 8, "fff" => 12
      cpu_info["num_cores_allowed"] = (
        bin(int(nc.group(1).replace(",", ""), 16)).count("1"))
  except tf.OpError:
    pass
  finally:
    if "num_cores_allowed" not in cpu_info:
      cpu_info["num_cores_allowed"]= cpu_info["num_cores"]

  info = cpuinfo.get_cpu_info()
  cpu_info["cpu_info"] = info["brand"]
  cpu_info["num_cores"] = info["count"]
  cpu_info["mhz_per_cpu"] = info["hz_advertised_raw"][0] / 1.0e6
  l2_cache_size = re.match(r"(\d+)", str(info.get("l2_cache_size", "")))
  if l2_cache_size:
    # If a value is returned, it"s in KB
    cpu_info["cache_size"] = {"L2": int(l2_cache_size.group(0)) * 1024}

  # Try to get the CPU governor
  try:
    cpu_governors = set([
      tf.gfile.GFile(f, "r").readline().rstrip()
      for f in glob.glob(
          "/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor")
    ])
    if cpu_governors:
      if len(cpu_governors) > 1:
        cpu_info["cpu_governor"] = "mixed"
      else:
        cpu_info["cpu_governor"] = list(cpu_governors)[0]
  except tf.OpError:
    pass

  run_info["cpu_info"] = cpu_info

def _collect_gpu_info(run_info):
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
  run_info["gpu_info"] = gpu_info

def _collect_memory_info(run_info):
  vmem = psutil.virtual_memory()
  run_info["memory_total"] = vmem.total
  run_info["memory_available"] = vmem.available

def _parse_gpu_model(physical_device_desc):
  # Assume all the GPU connected are same model
  for kv in physical_device_desc.split(","):
    k, _, v = kv.partition(":")
    if k.strip() == "name":
      return v.strip()
  return None
