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

import inspect

import numpy as np
import tensorflow as tf


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
      name="global_step",
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


def get_value(x) -> np.number:
  """Returns the value of a variable/tensor.

  Args:
      x: input variable.

  Returns:
      A Numpy array or number.
  """
  if not tf.is_tensor(x):
    return x
  return x.numpy()
