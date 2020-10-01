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
"""AbstractTrainer/Evaluator implementations for standard settings."""

import abc

from typing import Any, Dict, Optional, Text

import dataclasses

from orbit import runner
from orbit.utils import loop_fns

import tensorflow as tf


@dataclasses.dataclass(frozen=True)
class StandardTrainerOptions:
  """Advanced options for `orbit.StandardTrainer`.

  Attributes:
    use_tf_while_loop: A boolean indicating whether to run the training loop
      using a `tf.while_loop`. If `True`, `use_tf_function` must also be `True`.
    use_tf_function: A boolean indicating whether to apply `tf.function` to the
      training loop. This will only affect the body of the loop (involving
      `train_step`); `train_loop_begin` and `train_loop_end` will always be run
      in eager mode.
    use_tpu_summary_optimization: A boolean indicating whether to enable a
      performance optimization for summaries in TPUs. Writing summaries
      conditionally with outside compilation on TPUs can be extremely slow. If
      `True`, this optimization creates two `tf.function`s with two XLA programs
      (one with summary calls, and one without). The program with summaries runs
      only for one step when summaries should be recorded.
  """
  use_tf_while_loop: bool = True
  use_tf_function: bool = True
  use_tpu_summary_optimization: bool = False


def _create_train_loop_fn(train_step_fn, options: StandardTrainerOptions):
  """Creates a training loop from the given step function and options."""
  if options.use_tf_while_loop:
    loop_fn = loop_fns.create_tf_while_loop_fn(train_step_fn)
    if options.use_tpu_summary_optimization:
      loop_fn = loop_fns.LoopFnWithSummaries(loop_fn)
    else:
      loop_fn = tf.function(loop_fn)
  else:
    if options.use_tf_function:
      train_step_fn = tf.function(train_step_fn)
    loop_fn = loop_fns.create_loop_fn(train_step_fn)
  return loop_fn


class StandardTrainer(runner.AbstractTrainer, metaclass=abc.ABCMeta):
  """Implements the standard functionality of AbstractTrainer APIs."""

  def __init__(self, train_dataset, options: StandardTrainerOptions = None):
    """Construct a `StandardTrainer` object.

    Args:
      train_dataset: A tf.nest-compatible structure of tf.data.Dataset or
        DistributedDataset.
      options: An `orbit.StandardTrainerOptions` instance.
    """
    options = options or StandardTrainerOptions()
    if options.use_tf_while_loop and not options.use_tf_function:
      raise ValueError("`use_tf_while_loop=True` and `use_tf_function=False` "
                       "is not supported")
    if options.use_tpu_summary_optimization and not options.use_tf_while_loop:
      raise ValueError("`use_tpu_summary_optimization=True` and "
                       "`use_tf_while_loop=False` is not supported")

    self._train_options = options
    self._train_dataset = train_dataset
    self._train_iter = None
    self._train_loop_fn = None

  def train(
      self,
      num_steps: Optional[tf.Tensor],
  ) -> Optional[Dict[Text, tf.Tensor]]:
    """See base class."""
    self.train_loop_begin()

    if self._train_loop_fn is None:
      self._train_loop_fn = _create_train_loop_fn(
          self.train_step, options=self._train_options)

    if self._train_iter is None:
      self._train_iter = tf.nest.map_structure(iter, self.train_dataset)

    self._train_loop_fn(self._train_iter, num_steps)
    return self.train_loop_end()

  def train_loop_begin(self):
    """Called once at the beginning of the training loop.

    This method is called before dataset iterators creation.
    This is a good place to reset metrics that accumulate values over multiple
    steps of training.
    """
    pass

  @abc.abstractmethod
  def train_step(self, iterator):
    """Implements one step of training.

    What a "step" consists of is up to the implementer. If using distribution
    strategies, the call to this method should take place in the "cross-replica
    context" for generality, to allow e.g. multiple iterator dequeues and calls
    to `strategy.run`.

    Note that if `use_tf_function=True`, all the code inside `train_step` should
    be tf.function compatible, as they will be traced with tf.function. This
    means you cannot put arbitrary python code in this function. If users have
    any numpy operations, they should be put in `train_loop_begin` or
    `train_loop_end` functions.

    Args:
      iterator: A tf.nest-compatible structure of tf.data Iterator or
        DistributedIterator.
    """
    pass

  def train_loop_end(self) -> Optional[Dict[Text, tf.Tensor]]:
    """Called at the end of the training loop.

    This is a good place to get metric results. The value returned from this
    function will be returned as-is from the train() method.

    Returns:
      The function may return a dictionary of `Tensors`, which will be
      written to logs and as TensorBoard summaries. It can also be a
      nested dictionary, yielding a hierarchy of summary directories.
    """
    pass

  @property
  def train_dataset(self):
    """Returns the train_dataset instance."""
    return self._train_dataset

  @train_dataset.setter
  def train_dataset(self, train_dataset):
    """Set a new train dataset and replace with the existing one.

    Any unfinished work in the previous dataset will be discarded.

    Args:
      train_dataset: A tf.nest-compatible structure of tf.data.Dataset or
        DistributedDataset.
    """
    self._train_dataset = train_dataset
    self._train_iter = None


@dataclasses.dataclass(frozen=True)
class StandardEvaluatorOptions:
  """Advanced options for the `orbit.StandardEvaluator`.

  Attributes:
    use_tf_function: A boolean indicating whether to apply `tf.function` to the
      training loop. This will only affect the body of the loop (involving
      `train_step`); `train_loop_begin` and `train_loop_end` will always be run
      in eager mode.
  """
  use_tf_function: bool = True


def _create_eval_loop_fn(eval_step_fn, options: StandardEvaluatorOptions):
  if options.use_tf_function:
    eval_step_fn = tf.function(eval_step_fn)
  return loop_fns.create_loop_fn(eval_step_fn)


class StandardEvaluator(runner.AbstractEvaluator, metaclass=abc.ABCMeta):
  """Implements the standard functionality of AbstractEvaluator APIs."""

  def __init__(self, eval_dataset, options: StandardEvaluatorOptions = None):
    """Construct a `StandardEvaluator` object.

    Args:
      eval_dataset: A tf.nest-compatible structure of tf.data.Dataset or
        DistributedDataset.
      options: An `orbit.StandardEvaluatorOptions` instance.
    """
    self._eval_options = options or StandardEvaluatorOptions()
    self._eval_dataset = eval_dataset
    self._eval_loop_fn = None

  def evaluate(
      self,
      num_steps: Optional[tf.Tensor],
  ) -> Optional[Dict[Text, tf.Tensor]]:
    """See base class."""
    outputs = self.eval_begin()  # pylint: disable=assignment-from-no-return

    if self._eval_loop_fn is None:
      self._eval_loop_fn = _create_eval_loop_fn(
          self.eval_step, options=self._eval_options)

    eval_iter = tf.nest.map_structure(iter, self.eval_dataset)
    outputs = self._eval_loop_fn(
        eval_iter, num_steps, state=outputs, reduce_fn=self.eval_reduce)

    if outputs is None:
      return self.eval_end()
    else:
      return self.eval_end(outputs)

  def eval_begin(self) -> Any:
    """Called once at the beginning of the evaluation.

    This method is called before dataset iterators creation.
    This is a good place to reset metrics that accumulate values over the entire
    evaluation.

    Returns:
      An output which is passed as `state` argument into `eval_reduce` function.
    """
    pass

  @abc.abstractmethod
  def eval_step(self, iterator) -> Any:
    """Implements one step of evaluation.

    What a "step" consists of is up to the implementer. If using distribution
    strategies, the call to this method should take place in the "cross-replica
    context" for generality, to allow e.g. multiple iterator dequeues and calls
    to `strategy.run`.

    Note that if `use_tf_function=True`, all the code inside `eval_step` should
    be tf.function compatible, as they will be traced with tf.function. This
    means you cannot put arbitrary python code in this function. If users have
    any numpy operations, they should be put in `eval_begin`, `eval_end` or
    `eval_reduce` functions.

    Args:
      iterator: A tf.nest-compatible structure of tf.data Iterator or
        DistributedIterator.

    Returns:
      An output which is passed as `step_outputs` argument into `eval_reduce`
      function.
    """
    pass

  def eval_end(self, *args) -> Optional[Dict[Text, tf.Tensor]]:
    """Called at the end of the evaluation.

    This is a good place to get metric results. The value returned from this
    function will be returned as-is from the evaluate() method.

    Args:
      *args: the outputs from `eval_reduce` for the last eval step.

    Returns:
      The function may return a dictionary of `Tensors`, which will be
      written to logs and as TensorBoard summaries. It can also be a
      nested dictionary, yielding a hierarchy of summary directories.
    """
    pass

  def eval_reduce(self, state=None, step_outputs=None) -> Any:
    """A function to do the reduction on the evaluation outputs per step.

    This is useful for passing states throughout evaluation. E.g. it can be used
    to maintain the output losses from all the evaluation steps, and compute the
    mean loss in `eval_end` function.

    Args:
      state: A maintained state throughout the evaluation.
      step_outputs: Outputs from the current evaluation step.

    Returns:
      An output which is passed as `state` argument into `eval_reduce` function
      for the next step. After evaluation is finished, the output from last step
      will be passed into `eval_end` function.
    """
    pass

  @property
  def eval_dataset(self):
    """Returns the train_datase instance."""
    return self._eval_dataset

  @eval_dataset.setter
  def eval_dataset(self, eval_dataset):
    """Set a new eval dataset and replace with the existing one.

    Args:
      eval_dataset: A tf.nest-compatible structure of tf.data.Dataset or
        DistributedDataset.
    """
    self._eval_dataset = eval_dataset
