# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

import abc
import inspect
import six

import tensorflow.compat.v2 as tf


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

  @tf.function
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
    if six.PY3:
      argspec = inspect.getfullargspec(dataset_or_fn)
    else:
      argspec = inspect.getargspec(dataset_or_fn)
    args_names = argspec.args

    if "input_context" in args_names:
      kwargs["input_context"] = ctx
    ds = dataset_or_fn(*args, **kwargs)
    return ds

  return strategy.experimental_distribute_datasets_from_function(dataset_fn)


class SummaryManager(object):
  """A class manages writing summaries."""

  def __init__(self,
               summary_writer,
               summary_fn,
               global_step=None,
               summary_interval=None):
    """Construct a summary manager object.

    Args:
      summary_writer: A `tf.summary.SummaryWriter` instance for writing
        summaries.
      summary_fn: A callable defined as `def summary_fn(name, tensor,
        step=None)`, which describes the summary operation.
      global_step: A `tf.Variable` instance for checking the current global step
        value, in case users want to save summaries every N steps.
      summary_interval: An integer, indicates the minimum step interval between
        two summaries.
    """
    if summary_writer is not None:
      self._summary_writer = summary_writer
      self._enabled = True
    else:
      self._summary_writer = tf.summary.create_noop_writer()
      self._enabled = False
    self._summary_fn = summary_fn

    if global_step is None:
      self._global_step = tf.summary.experimental.get_step()
    else:
      self._global_step = global_step

    if summary_interval is not None:
      if self._global_step is None:
        raise ValueError("`summary_interval` is not None, but no `global_step` "
                         "can be obtained ")
      self._last_summary_step = self._global_step.numpy()
    self._summary_interval = summary_interval

  @property
  def summary_interval(self):
    return self._summary_interval

  @property
  def summary_writer(self):
    """Returns the underlying summary writer."""
    return self._summary_writer

  def flush(self):
    """Flush the underlying summary writer."""
    if self._enabled:
      tf.summary.flush(self._summary_writer)

  def write_summaries(self, items, always_write=True):
    """Write a bulk of summaries.

    Args:
      items: a dictionary of `Tensors` for writing summaries.
      always_write: An optional boolean. If `True`, the manager will always
        write summaries unless the summaries have been written for the same
        step. Otherwise the manager will only write the summaries if the
        interval between summaries are larger than `summary_interval`.

    Returns:
      A boolean indicates whether the summaries are written or not.
    """
    # TODO(rxsang): Support writing summaries with nested structure, so users
    # can split the summaries into different directories for nicer visualization
    # in Tensorboard, like train and eval metrics.
    if not self._enabled:
      return False

    if self._summary_interval is not None:
      current_step = self._global_step.numpy()
      if current_step == self._last_summary_step:
        return False
      if not always_write and current_step < (self._last_summary_step +
                                              self._summary_interval):
        return False
      self._last_summary_step = current_step

    with self._summary_writer.as_default():
      for name, tensor in items.items():
        self._summary_fn(name, tensor, step=self._global_step)
    return True


@six.add_metaclass(abc.ABCMeta)
class Trigger(object):
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


class EpochHelper(object):
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
