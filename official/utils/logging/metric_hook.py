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
"""Session hook for logging benchmark metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from official.utils.logging import logger
import six
import tensorflow as tf
from tensorflow.python.framework import ops

GLOBAL_STEP_TENSOR_NAME = "global_step"


class LoggingMetricHook(tf.train.SessionRunHook):
  """Hook to log benchmark metric information.

  This hook is very similar as tf.train.LoggingTensorHook, which logs given
  tensors every N local steps, every N seconds, or at the end.

  Note that if `at_end` is True, `tensors` should not include any tensor
  whose evaluation produces a side effect such as consuming additional inputs.
  """

  def __init__(self, tensors, log_dir=None, metric_logger=None,
               every_n_iter=None, every_n_secs=None, at_end=False):
    """Initializer for LoggingMetricHook.

    Args:
      tensors: `dict` that maps string-valued tags to tensors/tensor names,
          or `iterable` of tensors/tensor names.
      log_dir: `string`, directory path that metric hook should write log to.
      metric_logger: `BenchmarkLogger`, the benchmark logger that hook should
          use to write the log. Exactly one of the `log_dir` and `metric_logger`
          should be provided.
      every_n_iter: `int`, print the values of `tensors` once every N local
          steps taken on the current worker.
      every_n_secs: `int` or `float`, print the values of `tensors` once every N
          seconds. Exactly one of `every_n_iter` and `every_n_secs` should be
          provided.
      at_end: `bool` specifying whether to print the values of `tensors` at the
          end of the run.

    Raises:
      ValueError:
        1. `every_n_iter` is non-positive, or
        2. Exactly one of every_n_iter and every_n_secs should be provided.
        3. Exactly one of log_dir and metric_logger should be provided.
    """
    only_log_at_end = (
        at_end and (every_n_iter is None) and (every_n_secs is None))
    if (not only_log_at_end and
        (every_n_iter is None) == (every_n_secs is None)):
      raise ValueError(
        "either at_end and/or exactly one of every_n_iter and every_n_secs "
        "must be provided.")
    if every_n_iter is not None and every_n_iter <= 0:
      raise ValueError("invalid every_n_iter=%s." % every_n_iter)
    if (log_dir is None) == (metric_logger is None):
      raise ValueError(
          "exactly one of log_dir and metric_logger should be provided.")
    if not isinstance(tensors, dict):
      self._tag_order = tensors
      tensors = {item: item for item in tensors}
    else:
      self._tag_order = tensors.keys()

    if log_dir is not None:
      self._logger = logger.BenchmarkLogger(log_dir)
    else:
      self._logger = metric_logger
    self._tensors = tensors
    self._timer = (
        _NeverTriggerTimer() if only_log_at_end else
        tf.train.SecondOrStepTimer(
            every_secs=every_n_secs, every_steps=every_n_iter))
    self._log_at_end = at_end

  def begin(self):
    self._timer.reset()
    self._iter_count = 0
    if tf.train.get_global_step() is None:
      raise RuntimeError(
          "Global step should be created to use LoggingMetricHook.")

    # Convert names to tensors if given
    self._current_tensors = {tag: _as_graph_element(tensor)
                             for (tag, tensor) in self._tensors.items()}
    self._current_tensors[GLOBAL_STEP_TENSOR_NAME] = tf.train.get_global_step()

  def before_run(self, unused_run_context):
    self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
    if self._should_trigger:
      return tf.train.SessionRunArgs(self._current_tensors)
    else:
      return None

  def after_run(self, unused_run_context, run_values):
    if self._should_trigger:
      self._log_metric(run_values.results)

    self._iter_count += 1

  def end(self, session):
    if self._log_at_end:
      values = session.run(self._current_tensors)
      self._log_metric(values)

  def _log_metric(self, tensor_values):
    self._timer.update_last_triggered_step(self._iter_count)
    global_step = tensor_values[GLOBAL_STEP_TENSOR_NAME]
    for tag in self._tag_order:
      self._logger.log_metric(tag, tensor_values[tag], global_step=global_step)


class _NeverTriggerTimer(object):
  """Timer that never triggers.

  This code is shamelessly copied from
  tensorflow/python/training/basic_session_run_hooks.py, which should be
  deduplicate when the original code is exposed.
  """

  def __init__(self):
    pass

  def reset(self):
    """Resets the timer."""
    pass

  def should_trigger_for_step(self, step):
    _ = step
    return False

  def update_last_triggered_step(self, step):
    _ = step
    return (None, None)

  def last_triggered_step(self):
    return None


def _as_graph_element(obj):
  """Retrieves Graph element.

  This code is shamelessly copied from
  tensorflow/python/training/basic_session_run_hooks.py, which should be
  deduplicate when the original function is exposed.
  """
  graph = ops.get_default_graph()
  if not isinstance(obj, six.string_types):
    if not hasattr(obj, "graph") or obj.graph != graph:
      raise ValueError("Passed %s should have graph attribute that is equal "
                       "to current graph %s." % (obj, graph))
    return obj
  if ":" in obj:
    element = graph.as_graph_element(obj)
  else:
    element = graph.as_graph_element(obj + ":0")
    # Check that there is no :1 (e.g. it's single output).
    try:
      graph.as_graph_element(obj + ":1")
    except (KeyError, ValueError):
      pass
    else:
      raise ValueError("Name %s is ambiguous, "
                       "as this `Operation` has multiple outputs "
                       "(at least 2)." % obj)
  return element
