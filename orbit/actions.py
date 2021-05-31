# Copyright 2021 The Orbit Authors. All Rights Reserved.
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

"""Defines an "action" abstraction for use with `orbit.Controller`.

"Actions" are simply arbitrary callables that are applied by the `Controller`
to the output of train steps (after each inner loop of `steps_per_loop` steps)
or an evaluation. This provides a hook mechanism, enabling things like reporting
metrics to Vizier, model exporting, additional logging, etc.

The basic `Action` abstraction (just a type alias) is defined in the
`controller` module. This `actions` module adds a `ConditionalAction` utility
class to make it easy to trigger actions conditionally based on reusable
predicates, as well as a small handful of predefined conditions/actions (in
particular, a `NewBestMetric` condition and an `ExportSavedModel` action).

One example of using actions to do metric-conditional export:

    new_best_metric = orbit.actions.NewBestMetric('accuracy')
    export_action = orbit.actions.ConditionalAction(
        condition=lambda x: x['accuracy'] > 0.9 and new_best_metric(x),
        action=orbit.actions.ExportSavedModel(
            model,
            orbit.actions.ExportFileManager(
                base_name=f'{FLAGS.model_dir}/saved_model',
                next_id_fn=trainer.global_step.numpy),
            signatures=model.infer))

    controller = orbit.Controller(
        strategy=strategy,
        trainer=trainer,
        evaluator=evaluator,
        eval_actions=[export_action],
        global_step=trainer.global_step,
        steps_per_loop=FLAGS.steps_per_loop,
        checkpoint_manager=checkpoint_manager,
        summary_interval=1000)

Note: In multi-client settings where each client runs its own `Controller`
instance, some care should be taken in deciding which clients should run certain
actions. Isolating actions to an individual client (say client 0) can be
achieved using `ConditionalAction` as follows:

    client_0_actions = orbit.actions.ConditionalAction(
        condition=lambda _: client_id() == 0,
        action=[
            ...
        ])

In particular, the `NewBestMetric` condition may be used in multi-client
settings if all clients are guaranteed to compute the same metric (ensuring this
is up to client code, not Orbit). However, when saving metrics it may be helpful
to avoid unnecessary writes by setting the `write_value` parameter to `False`
for most clients.
"""

import json
import os
import sys
from typing import Any, Callable, Optional, Sequence, Union
import uuid

from orbit import controller
from orbit import runner
from orbit import utils

import tensorflow as tf

Condition = Callable[[runner.Output], Union[bool, tf.Tensor]]


def _as_sequence(maybe_sequence: Union[Any, Sequence[Any]]) -> Sequence[Any]:
  if isinstance(maybe_sequence, Sequence):
    return maybe_sequence
  return [maybe_sequence]


class ConditionalAction:
  """Represents an action that is only taken when a given condition is met.

  This class is itself an `Action` (a callable that can be applied to train or
  eval outputs), but is intended to make it easier to write modular and reusable
  conditions by decoupling "when" something whappens (the condition) from "what"
  happens (the action).
  """

  def __init__(
      self,
      condition: Condition,
      action: Union[controller.Action, Sequence[controller.Action]],
  ):
    """Initializes the instance.

    Args:
      condition: A callable accepting train or eval outputs and returing a bool.
      action: The action (or optionally sequence of actions) to perform when
        `condition` is met.
    """
    self.condition = condition
    self.action = action

  def __call__(self, output: runner.Output) -> None:
    if self.condition(output):
      for action in _as_sequence(self.action):
        action(output)


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
            self._value = json.loads(f.read())
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


class _CounterIdFn:
  """Implements a counter-based ID function for `ExportFileManager`."""

  def __init__(self, base_name: str):
    filenames = tf.io.gfile.glob(f'{base_name}-*')
    max_counter = -1
    for filename in filenames:
      try:
        _, file_number = filename.rsplit('-', maxsplit=1)
        max_counter = max(max_counter, int(file_number))
      except ValueError:
        continue
    self.value = max_counter + 1

  def __call__(self):
    output = self.value
    self.value += 1
    return output


class ExportFileManager:
  """Utility class that manages a group of files with a shared base name.

  For actions like SavedModel exporting, there are potentially many different
  file naming and cleanup strategies that may be desirable. This class provides
  a basic interface allowing SavedModel export to be decoupled from these
  details, and a default implementation that should work for many basic
  scenarios. Users may subclass this class to alter behavior and define more
  customized naming and cleanup strategies.
  """

  def __init__(self,
               base_name: str,
               max_to_keep: int = 5,
               next_id_fn: Optional[Callable[[], int]] = None):
    """Initializes the instance.

    Args:
      base_name: A shared base name for file names generated by this class.
      max_to_keep: The maximum number of files matching `base_name` to keep
        after each call to `cleanup`. The most recent (as determined by file
        modification time) `max_to_keep` files are preserved; the rest are
        deleted. If < 0, all files are preserved.
      next_id_fn: An optional callable that returns integer IDs to append to
        base name (formatted as `'{base_name}-{id}'`). The order of integers is
        used to sort files to determine the oldest ones deleted by `clean_up`.
        If not supplied, a default ID based on an incrementing counter is used.
        One common alternative maybe be to use the current global step count,
        for instance passing `next_id_fn=global_step.numpy`.
    """
    self._base_name = base_name
    self._max_to_keep = max_to_keep
    self._next_id_fn = next_id_fn or _CounterIdFn(base_name)

  @property
  def managed_files(self):
    """Returns all files managed by this instance, in sorted order.

    Returns:
      The list of files matching the `base_name` provided when constructing this
      `ExportFileManager` instance, sorted in increasing integer order of the
      IDs returned by `next_id_fn`.
    """

    def id_key(name):
      _, id_num = name.rsplit('-', maxsplit=1)
      return int(id_num)

    filenames = tf.io.gfile.glob(f'{self._base_name}-*')
    return sorted(filenames, key=id_key)

  def clean_up(self):
    """Cleans up old files matching `{base_name}-*`.

    The most recent `max_to_keep` files are preserved.
    """
    if self._max_to_keep < 0:
      return

    for filename in self.managed_files[:-self._max_to_keep]:
      tf.io.gfile.rmtree(filename)

  def next_name(self) -> str:
    """Returns a new file name based on `base_name` and `next_id_fn()`."""
    return f'{self._base_name}-{self._next_id_fn()}'


class ExportSavedModel:
  """Action that exports the given model as a SavedModel."""

  def __init__(self,
               model: tf.Module,
               file_manager: ExportFileManager,
               signatures,
               options: Optional[tf.saved_model.SaveOptions] = None):
    """Initializes the instance.

    Args:
      model: The model to export.
      file_manager: An instance of `ExportFileManager` (or a subclass), that
        provides file naming and cleanup functionality.
      signatures: The signatures to forward to `tf.saved_model.save()`.
      options: Optional options to forward to `tf.saved_model.save()`.
    """
    self.model = model
    self.file_manager = file_manager
    self.signatures = signatures
    self.options = options

  def __call__(self, _):
    """Exports the SavedModel."""
    export_dir = self.file_manager.next_name()
    tf.saved_model.save(self.model, export_dir, self.signatures, self.options)
    self.file_manager.clean_up()
