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

"""Wrapper for the mlperf logging utils.

MLPerf compliance logging is only desired under a limited set of circumstances.
This module is intended to keep users from needing to consider logging (or
install the module) unless they are performing mlperf runs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import json
import os
import re
import subprocess
import sys
import typing

import tensorflow as tf

_MIN_VERSION = (0, 0, 6)
_STACK_OFFSET = 2

SUDO = "sudo" if os.geteuid() else ""

# This indirection is used in docker.
DROP_CACHE_LOC = os.getenv("DROP_CACHE_LOC", "/proc/sys/vm/drop_caches")

_NCF_PREFIX = "NCF_RAW_"

# TODO(robieta): move line parsing to mlperf util
_PREFIX = r"(?:{})?:::MLPv([0-9]+).([0-9]+).([0-9]+)".format(_NCF_PREFIX)
_BENCHMARK = r"([a-zA-Z0-9_]+)"
_TIMESTAMP = r"([0-9]+\.[0-9]+)"
_CALLSITE = r"\((.+):([0-9]+)\)"
_TAG = r"([a-zA-Z0-9_]+)"
_VALUE = r"(.*)"

ParsedLine = namedtuple("ParsedLine", ["version", "benchmark", "timestamp",
                                       "callsite", "tag", "value"])

LINE_PATTERN = re.compile(
    "^{prefix} {benchmark} {timestamp} {callsite} {tag}(: |$){value}?$".format(
        prefix=_PREFIX, benchmark=_BENCHMARK, timestamp=_TIMESTAMP,
        callsite=_CALLSITE, tag=_TAG, value=_VALUE))


def parse_line(line): # type: (str) -> typing.Optional[ParsedLine]
  match = LINE_PATTERN.match(line.strip())
  if not match:
    return

  major, minor, micro, benchmark, timestamp = match.groups()[:5]
  call_file, call_line, tag, _, value = match.groups()[5:]

  return ParsedLine(version=(int(major), int(minor), int(micro)),
                    benchmark=benchmark, timestamp=timestamp,
                    callsite=(call_file, call_line), tag=tag, value=value)


def unparse_line(parsed_line): # type: (ParsedLine) -> str
  version_str = "{}.{}.{}".format(*parsed_line.version)
  callsite_str = "({}:{})".format(*parsed_line.callsite)
  value_str = ": {}".format(parsed_line.value) if parsed_line.value else ""
  return ":::MLPv{} {} {} {} {} {}".format(
      version_str, parsed_line.benchmark, parsed_line.timestamp, callsite_str,
      parsed_line.tag, value_str)


def get_mlperf_log():
  """Shielded import of mlperf_log module."""
  try:
    import mlperf_compliance

    def test_mlperf_log_pip_version():
      """Check that mlperf_compliance is up to date."""
      import pkg_resources
      version = pkg_resources.get_distribution("mlperf_compliance")
      version = tuple(int(i) for i in version.version.split("."))
      if version < _MIN_VERSION:
        tf.logging.warning(
            "mlperf_compliance is version {}, must be >= {}".format(
                ".".join([str(i) for i in version]),
                ".".join([str(i) for i in _MIN_VERSION])))
        raise ImportError
      return mlperf_compliance.mlperf_log

    mlperf_log = test_mlperf_log_pip_version()

  except ImportError:
    mlperf_log = None

  return mlperf_log


class Logger(object):
  """MLPerf logger indirection class.

  This logger only logs for MLPerf runs, and prevents various errors associated
  with not having the mlperf_compliance package installed.
  """
  class Tags(object):
    def __init__(self, mlperf_log):
      self._enabled = False
      self._mlperf_log = mlperf_log

    def __getattr__(self, item):
      if self._mlperf_log is None or not self._enabled:
        return
      return getattr(self._mlperf_log, item)

  def __init__(self):
    self._enabled = False
    self._mlperf_log = get_mlperf_log()
    self.tags = self.Tags(self._mlperf_log)

  def __call__(self, enable=False):
    if enable and self._mlperf_log is None:
      raise ImportError("MLPerf logging was requested, but mlperf_compliance "
                        "module could not be loaded.")

    self._enabled = enable
    self.tags._enabled = enable
    return self

  def __enter__(self):
    pass

  def __exit__(self, exc_type, exc_val, exc_tb):
    self._enabled = False
    self.tags._enabled = False

  @property
  def log_file(self):
    if self._mlperf_log is None:
      return
    return self._mlperf_log.LOG_FILE

  @property
  def enabled(self):
    return self._enabled

  def ncf_print(self, key, value=None, stack_offset=_STACK_OFFSET,
                deferred=False, extra_print=False, prefix=_NCF_PREFIX):
    if self._mlperf_log is None or not self.enabled:
      return
    self._mlperf_log.ncf_print(key=key, value=value, stack_offset=stack_offset,
                               deferred=deferred, extra_print=extra_print,
                               prefix=prefix)

  def set_ncf_root(self, path):
    if self._mlperf_log is None:
      return
    self._mlperf_log.ROOT_DIR_NCF = path


LOGGER = Logger()
ncf_print, set_ncf_root = LOGGER.ncf_print, LOGGER.set_ncf_root
TAGS = LOGGER.tags


def clear_system_caches():
  if not LOGGER.enabled:
    return
  ret_code = subprocess.call(
      ["sync && echo 3 | {} tee {}".format(SUDO, DROP_CACHE_LOC)],
      shell=True)

  if ret_code:
    raise ValueError("Failed to clear caches")


def stitch_ncf():
  """Format NCF logs for MLPerf compliance."""
  if not LOGGER.enabled:
    return

  if LOGGER.log_file is None or not tf.gfile.Exists(LOGGER.log_file):
    tf.logging.warning("Could not find log file to stitch.")
    return

  log_lines = []
  num_eval_users = None
  start_time = None
  stop_time = None
  with tf.gfile.Open(LOGGER.log_file, "r") as f:
    for line in f:
      parsed_line = parse_line(line)
      if not parsed_line:
        tf.logging.warning("Failed to parse line: {}".format(line))
        continue
      log_lines.append(parsed_line)

      if parsed_line.tag == TAGS.RUN_START:
        assert start_time is None
        start_time = float(parsed_line.timestamp)

      if parsed_line.tag == TAGS.RUN_STOP:
        assert stop_time is None
        stop_time = float(parsed_line.timestamp)

      if (parsed_line.tag == TAGS.EVAL_HP_NUM_USERS and parsed_line.value
          is not None and "DEFERRED" not in parsed_line.value):
        assert num_eval_users is None or num_eval_users == parsed_line.value
        num_eval_users = parsed_line.value
        log_lines.pop()

  for i, parsed_line in enumerate(log_lines):
    if parsed_line.tag == TAGS.EVAL_HP_NUM_USERS:
      log_lines[i] = ParsedLine(*parsed_line[:-1], value=num_eval_users)

  log_lines = sorted([unparse_line(i) for i in log_lines])

  output_path = os.getenv("STITCHED_COMPLIANCE_FILE", None)
  if output_path:
    with tf.gfile.Open(output_path, "w") as f:
      for line in log_lines:
        f.write(line + "\n")
  else:
    for line in log_lines:
      print(line)
    sys.stdout.flush()

  if start_time is not None and stop_time is not None:
    tf.logging.info("MLPerf time: {:.1f} sec.".format(stop_time - start_time))

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  with LOGGER(True):
    ncf_print(key=TAGS.RUN_START)
