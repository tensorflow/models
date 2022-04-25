# Copyright 2022 The Orbit Authors. All Rights Reserved.
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

"""Provides the `NewBestMetric` condition and associated helper classes."""

import json
import os
import sys
from typing import Any, Callable, Optional, Union
import uuid

from orbit import runner
from orbit import utils

import tensorflow as tf

MetricFn = Callable[[runner.Output], Union[float, tf.Tensor]]


class NewBestMetric:
  """Condition that is satisfied when a new best metric is achieved.

  This class keeps track of the best metric value seen so far, optionally in a
  persistent (preemption-safe) way.

  Two methods are provided, which each satisfy the `Action` protocol: `test` for
  only testing whether a new best metric is achieved by a given train/eval
  output, and `commit`, which both tests and records the new best metric value
  if it is achieved. These separate methods enable the same `NewBestMetric`
  instance to be reused as a condition multiple times, and can also provide
  additional preemption/failure safety. For example, to avoid updating the best
  metric if a model export fails or is pre-empted:

      new_best_metric = orbit.actions.NewBestMetric(
        'accuracy', filename='/model/dir/best_metric')
      action = orbit.actions.ConditionalAction(
          condition=new_best_metric.test,
          action=[
            orbit.actions.ExportSavedModel(...),
            new_best_metric.commit
          ])

  The default `__call__` implementation is equivalent to `commit`.

  This class is safe to use in multi-client settings if all clients can be
  guaranteed to compute the same metric. However when saving metrics it may be
  helpful to avoid unnecessary writes by setting the `write_value` parameter to
  `False` for most clients.

  Attributes:
    metric: The metric passed to __init__ (may be a string key or a callable
      that can be applied to train/eval output).
    higher_is_better: Whether higher metric values are better.
  """

  def __init__(self,
               metric: Union[str, MetricFn],
               higher_is_better: bool = True,
               filename: Optional[str] = None,
               write_metric=True):
    """Initializes the instance.

    Args:
      metric: Either a string key name to use to look up a metric (assuming the
        train/eval output is a dictionary), or a callable that accepts the
        train/eval output and returns a metric value.
      higher_is_better: Whether higher metric values are better. If `True`, a
        new best metric is achieved when the metric value is strictly greater
        than the previous best metric. If `False`, a new best metric is achieved
        when the metric value is strictly less than the previous best metric.
      filename: A filename to use for storage of the best metric value seen so
        far, to allow peristence of the value across preemptions. If `None`
        (default), values aren't persisted.
      write_metric: If `filename` is set, this controls whether this instance
        will write new best metric values to the file, or just read from the
        file to obtain the initial value. Setting this to `False` for most
        clients in some multi-client setups can avoid unnecessary file writes.
        Has no effect if `filename` is `None`.
    """
    self.metric = metric
    self.higher_is_better = higher_is_better
    float_max = sys.float_info.max
    self._best_value = JSONPersistedValue(
        initial_value=-float_max if higher_is_better else float_max,
        filename=filename,
        write_value=write_metric)

  def __call__(self, output: runner.Output) -> bool:
    """Tests `output` and updates the current best value if necessary.

    This is equivalent to `commit` below.

    Args:
      output: The train or eval output to test.

    Returns:
      `True` if `output` contains a new best metric value, `False` otherwise.
    """
    return self.commit(output)

  def metric_value(self, output: runner.Output) -> float:
    """Computes the metric value for the given `output`."""
    if callable(self.metric):
      value = self.metric(output)
    else:
      value = output[self.metric]
    return float(utils.get_value(value))

  @property
  def best_value(self) -> float:
    """Returns the best metric value seen so far."""
    return self._best_value.read()

  def test(self, output: runner.Output) -> bool:
    """Tests `output` to see if it contains a new best metric value.

    If `output` does contain a new best metric value, this method does *not*
    save it (i.e., calling this method multiple times in a row with the same
    `output` will continue to return `True`).

    Args:
      output: The train or eval output to test.

    Returns:
      `True` if `output` contains a new best metric value, `False` otherwise.
    """
    metric_value = self.metric_value(output)
    if self.higher_is_better:
      if metric_value > self.best_value:
        return True
    else:  # Lower is better.
      if metric_value < self.best_value:
        return True
    return False

  def commit(self, output: runner.Output) -> bool:
    """Tests `output` and updates the current best value if necessary.

    Unlike `test` above, if `output` does contain a new best metric value, this
    method *does* save it (i.e., subsequent calls to this method with the same
    `output` will return `False`).

    Args:
      output: The train or eval output to test.

    Returns:
      `True` if `output` contains a new best metric value, `False` otherwise.
    """

    if self.test(output):
      self._best_value.write(self.metric_value(output))
      return True
    return False


class JSONPersistedValue:
  """Represents a value that is persisted via a file-based backing store.

  The value must be JSON-serializable. Each time the value is updated, it will
  be written to the backing file. It is only read from the file at
  initialization.
  """

  def __init__(self,
               initial_value: Any,
               filename: str,
               write_value: bool = True):
    """Initializes the instance.

    Args:
      initial_value: The initial value to use if no backing file exists or was
        given. This must be a JSON-serializable value (possibly nested
        combination of lists, dicts, and primitive values).
      filename: The path to use for persistent storage of the value. This may be
        `None`, in which case the value is not stable across preemptions.
      write_value: If `True`, new values will be written to `filename` on calls
        to `write()`. If `False`, `filename` is only read once to restore any
        persisted value, and new values will not be written to it. This can be
        useful in certain multi-client settings to avoid race conditions or
        excessive file writes. If `filename` is `None`, this parameter has no
        effect.
    """
    self._value = None
    self._filename = filename
    self._write_value = write_value

    if self._filename is not None:
      if tf.io.gfile.exists(self._filename):
        if tf.io.gfile.stat(self._filename).length > 0:
          with tf.io.gfile.GFile(self._filename, 'r') as f:
            self._value = json.load(f)
      elif self._write_value:
        tf.io.gfile.makedirs(os.path.dirname(self._filename))

    if self._value is None:
      self.write(initial_value)

  def read(self):
    """Returns the value."""
    return self._value

  def write(self, value):
    """Writes the value, updating the backing store if one was provided."""
    self._value = value
    if self._filename is not None and self._write_value:
      # To achieve atomic writes, we first write to a temporary file, and then
      # rename it to `self._filename`.
      tmp_filename = f'{self._filename}.tmp.{uuid.uuid4().hex}'
      with tf.io.gfile.GFile(tmp_filename, 'w') as f:
        json.dump(self._value, f)
      tf.io.gfile.rename(tmp_filename, self._filename, overwrite=True)
