# Copyright 2020 The Orbit Authors. All Rights Reserved.
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
"""Some layered modules/functions to help users writing custom training loop."""

import abc
import contextlib
import functools
import inspect

import os
import numpy as np
import tensorflow as tf


def create_loop_fn(step_fn):
  """Creates a multiple steps function driven by the python while loop.

  Args:
    step_fn: A function which takes `iterator` as input.

  Returns:
    A callable defined as the `loop_fn` defination below.
  """

  def loop_fn(iterator, num_steps, state=None, reduce_fn=None):
    """A loop function with multiple steps.

    Args:
      iterator: A nested structure of tf.data `Iterator` or
        `DistributedIterator`.
      num_steps: The number of steps in the loop. If `num_steps==-1`, will
        iterate until exausting the iterator.
      state: An optional initial state before running the loop.
      reduce_fn: a callable defined as `def reduce_fn(state, value)`, where
        `value` is the outputs from `step_fn`.

    Returns:
      The updated state.
    """
    try:
      step = 0
      # To make sure the OutOfRangeError exception can be handled well with
      # async remote eager, we need to wrap the loop body in a `async_scope`.
      with tf.experimental.async_scope():
        while (num_steps == -1 or step < num_steps):
          outputs = step_fn(iterator)
          if reduce_fn is not None:
            state = reduce_fn(state, outputs)
          step += 1
        return state
    except (StopIteration, tf.errors.OutOfRangeError):
      tf.experimental.async_clear_error()
      return state

  return loop_fn


def create_tf_while_loop_fn(step_fn):
  """Create a multiple steps function driven by tf.while_loop on the host.

  Args:
    step_fn: A function which takes `iterator` as input.

  Returns:
    A callable defined as the `loop_fn` defination below.
  """

  def loop_fn(iterator, num_steps):
    """A loop function with multiple steps.

    Args:
      iterator: A nested structure of tf.data `Iterator` or
        `DistributedIterator`.
      num_steps: The number of steps in the loop. Must be a tf.Tensor.
    """
    if not isinstance(num_steps, tf.Tensor):
      raise ValueError("`num_steps` should be an `tf.Tensor`. Python object "
                       "may cause retracing.")

    for _ in tf.range(num_steps):
      step_fn(iterator)

  return loop_fn


def create_global_step() -> tf.Variable:
  """Creates a `tf.Variable` suitable for use as a global step counter.

  Creating and managing a global step variable may be necessary for
  `AbstractTrainer` subclasses that perform multiple parameter updates per
  `Controller` "step", or use different optimizers on different steps.

  In these cases, an `optimizer.iterations` property generally can't be used
  directly, since it would correspond to parameter updates instead of iterations
  in the `Controller`'s training loop. Such use cases should simply call
  `step.assign_add(1)` at the end of each step.

  Returns:
    A non-trainable scalar `tf.Variable` of dtype `tf.int64`, with only the
    first replica's value retained when synchronizing across replicas in
    a distributed setting.
  """
  return tf.Variable(
      0,
      dtype=tf.int64,
      trainable=False,
      aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)


def make_distributed_dataset(strategy, dataset_or_fn, *args, **kwargs):
  """A helper function to create distributed dataset.

  Args:
    strategy: An instance of `tf.distribute.Strategy`.
    dataset_or_fn: A instance of `tf.data.Dataset` or a function which takes an
      `tf.distribute.InputContext` as input and returns a `tf.data.Dataset`. If
      it is a function, it could optionally have an argument named
      `input_context` which is `tf.distribute.InputContext` argument type.
    *args: The list of arguments to be passed to dataset_or_fn.
    **kwargs: Any keyword arguments to be passed.

  Returns:
    A distributed Dataset.
  """
  if strategy is None:
    strategy = tf.distribute.get_strategy()

  if isinstance(dataset_or_fn, tf.data.Dataset):
    return strategy.experimental_distribute_dataset(dataset_or_fn)

  if not callable(dataset_or_fn):
    raise ValueError("`dataset_or_fn` should be either callable or an instance "
                     "of `tf.data.Dataset`")

  def dataset_fn(ctx):
    """Wrapped dataset function for creating distributed dataset.."""

    # If `dataset_or_fn` is a function and has `input_context` as argument
    # names, pass `ctx` as the value of `input_context` when calling
    # `dataset_or_fn`. Otherwise `ctx` will not be used when calling
    # `dataset_or_fn`.
    argspec = inspect.getfullargspec(dataset_or_fn)
    args_names = argspec.args

    if "input_context" in args_names:
      kwargs["input_context"] = ctx
    ds = dataset_or_fn(*args, **kwargs)
    return ds

  return strategy.experimental_distribute_datasets_from_function(dataset_fn)


class SummaryManager:
  """A class manages writing summaries."""

  def __init__(self, summary_dir, summary_fn, global_step=None):
    """Construct a summary manager object.

    Args:
      summary_dir: the directory to write summaries.
      summary_fn: A callable defined as `def summary_fn(name, tensor,
        step=None)`, which describes the summary operation.
      global_step: A `tf.Variable` instance for the global step.
    """
    self._enabled = (summary_dir is not None)
    self._summary_dir = summary_dir
    self._summary_fn = summary_fn
    self._summary_writers = {}

    if global_step is None:
      self._global_step = tf.summary.experimental.get_step()
    else:
      self._global_step = global_step

  def summary_writer(self, relative_path=""):
    """Returns the underlying summary writer.

    Args:
      relative_path: The current path in which to write summaries, relative to
        the summary directory. By default it is empty, which specifies the root
        directory.
    """
    if self._summary_writers and relative_path in self._summary_writers:
      return self._summary_writers[relative_path]
    if self._enabled:
      self._summary_writers[relative_path] = tf.summary.create_file_writer(
          os.path.join(self._summary_dir, relative_path))
    else:
      self._summary_writers[relative_path] = tf.summary.create_noop_writer()
    return self._summary_writers[relative_path]

  def flush(self):
    """Flush the underlying summary writers."""
    if self._enabled:
      tf.nest.map_structure(tf.summary.flush, self._summary_writers)

  def write_summaries(self, summary_dict):
    """Write summaries for the given values.

    This recursively creates subdirectories for any nested dictionaries
    provided in `summary_dict`, yielding a hierarchy of directories which will
    then be reflected in the TensorBoard UI as different colored curves.

    E.g. users may evaluate on muliple datasets and return `summary_dict` as a
    nested dictionary.

    ```
    {
        "dataset": {
            "loss": loss,
            "accuracy": accuracy
        },
        "dataset2": {
            "loss": loss2,
            "accuracy": accuracy2
        },
    }
    ```

    This will create two subdirectories "dataset" and "dataset2" inside the
    summary root directory. Each directory will contain event files including
    both "loss" and "accuracy" summaries.

    Args:
      summary_dict: A dictionary of values. If any value in `summary_dict` is
        itself a dictionary, then the function will recursively create
        subdirectories with names given by the keys in the dictionary. The
        Tensor values are summarized using the summary writer instance specific
        to the parent relative path.
    """
    if not self._enabled:
      return
    self._write_summaries(summary_dict)

  def _write_summaries(self, summary_dict, relative_path=""):
    for name, value in summary_dict.items():
      if isinstance(value, dict):
        self._write_summaries(
            value, relative_path=os.path.join(relative_path, name))
      else:
        with self.summary_writer(relative_path).as_default():
          self._summary_fn(name, value, step=self._global_step)


class Trigger(metaclass=abc.ABCMeta):
  """An abstract class representing a "trigger" for some event."""

  @abc.abstractmethod
  def __call__(self, value: float, force_trigger=False):
    """Maybe trigger the event based on the given value.

    Args:
      value: the value for triggering.
      force_trigger: Whether the trigger is forced triggered.

    Returns:
      `True` if the trigger is triggered on the given `value`, and
      `False` otherwise.
    """

  @abc.abstractmethod
  def reset(self):
    """Reset states in the trigger."""


class IntervalTrigger(Trigger):
  """Triggers on every fixed interval."""

  def __init__(self, interval, start=0):
    """Constructs the IntervalTrigger.

    Args:
      interval: The triggering interval.
      start: An initial value for the trigger.
    """
    self._interval = interval
    self._last_trigger_value = start

  def __call__(self, value, force_trigger=False):
    """Maybe trigger the event based on the given value.

    Args:
      value: the value for triggering.
      force_trigger: If True, the trigger will be forced triggered unless the
        last trigger value is equal to `value`.

    Returns:
      `True` if the trigger is triggered on the given `value`, and
      `False` otherwise.
    """
    if force_trigger and value != self._last_trigger_value:
      self._last_trigger_value = value
      return True

    if self._interval and self._interval > 0:
      if value >= self._last_trigger_value + self._interval:
        self._last_trigger_value = value
        return True
    return False

  def reset(self):
    """See base class."""
    self._last_trigger_value = 0


class EpochHelper:
  """A Helper class to handle epochs in Customized Training Loop."""

  def __init__(self, epoch_steps, global_step):
    """Constructs the EpochHelper.

    Args:
      epoch_steps: An integer indicates how many steps in an epoch.
      global_step: A `tf.Variable` instance indicates the current global step.
    """
    self._epoch_steps = epoch_steps
    self._global_step = global_step
    self._current_epoch = None
    self._epoch_start_step = None
    self._in_epoch = False

  def epoch_begin(self):
    """Returns whether a new epoch should begin."""
    if self._in_epoch:
      return False
    current_step = self._global_step.numpy()
    self._epoch_start_step = current_step
    self._current_epoch = current_step // self._epoch_steps
    self._in_epoch = True
    return True

  def epoch_end(self):
    """Returns whether the current epoch should end."""
    if not self._in_epoch:
      raise ValueError("`epoch_end` can only be called inside an epoch")
    current_step = self._global_step.numpy()
    epoch = current_step // self._epoch_steps

    if epoch > self._current_epoch:
      self._in_epoch = False
      return True
    return False

  @property
  def batch_index(self):
    """Index of the next batch within the current epoch."""
    return self._global_step.numpy() - self._epoch_start_step

  @property
  def current_epoch(self):
    return self._current_epoch


@contextlib.contextmanager
def _soft_device_placement():
  """Context manager for soft device placement, allowing summaries on CPU."""
  original_setting = tf.config.get_soft_device_placement()
  try:
    tf.config.set_soft_device_placement(True)
    yield
  finally:
    tf.config.set_soft_device_placement(original_setting)


def train_function_with_summaries(*args, **kwargs):
  """Utility function to support TPU summaries via multiple `tf.function`s.

  This permits interleaving summaries inside TPU-compatible code, but without
  any performance impact on steps that do not write summaries.

  Usage is as a decorator, similar to `tf.function`, and any `tf.function`
  arguments will be passed through if supplied:

      @trainer.train_function_with_summaries
      def train(self, num_steps):
        ...

  The decorated function is assumed to be a loop method accepting a `num_steps`
  parameter, as for instance would be called within the `Controller`'s outer
  train loop. The implementation here assumes that `summary_frequency` is
  divisible by `steps_per_loop`. The decorated method should accept two
  arguments, `self` and `num_steps`.

  Two `tf.function` versions of `train_fn` are created: one inside a summary
  writer scope with soft device placement enabled (used on steps that require
  summary writing), and one with no summary writer present and soft device
  placement disabled (used on all other steps).

  Args:
    *args: Arguments to pass through to `tf.function`.
    **kwargs: Keyword arguments to pass through to `tf.function`.

  Returns:
    If the first argument is a callable, returns the decorated callable.
    Otherwise, returns a decorator.
  """

  def decorator(train_fn):
    # TODO(dhr): Validate the signature of train_fn?

    train_fn_with_summaries = tf.function(train_fn, *args, **kwargs)
    train_fn_without_summaries = tf.function(train_fn, *args, **kwargs)

    @functools.wraps(train_fn)
    def wrapper(self, num_steps):
      if tf.summary.should_record_summaries():
        with _soft_device_placement():
          output = train_fn_with_summaries(self, tf.constant(1))
          num_steps -= 1
      if num_steps >= 1:
        with tf.summary.record_if(False):
          output = train_fn_without_summaries(self, num_steps)
      return output

    return wrapper

  if args and callable(args[0]):
    train_fn, args = args[0], args[1:]
    return decorator(train_fn)
  return decorator


def get_value(x) -> np.ndarray:
  """Returns the value of a variable/tensor.

  Args:
      x: input variable.

  Returns:
      A Numpy array or number.
  """
  if not tf.is_tensor(x):
    return x
  return x.numpy()
