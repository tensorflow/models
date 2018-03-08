# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Hooks helper to return a list of TensorFlow hooks for training by name."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import hooks


def get_train_hooks(name_list, **kwargs):
  """Factory for getting a list of TensorFlow hooks for training by name.

  Args:
    name_list: a list of strings to name desired hook classes. Allowed:
    LoggingTensorHook, ProfilerHook, ExamplesPerSecondHook, which are defined
    as keys in HOOKS
    kwargs: a dictionary of arguments to the hooks.

  Returns:
    list of instantiated hooks, ready to be used in a classifier.train call.

  Raises:
    ValueError: if an unrecognized name is passed.
  """

  if not name_list:
    return []

  if not isinstance(name_list, list):
    raise ValueError('Hook names should be in a list')

  train_hooks = []
  for name in name_list:
    hook_name = HOOKS.get(name.strip().lower())
    if hook_name is None:
      raise ValueError('Unrecognized training hook requested: {}'.format(name))
    else:
      train_hooks.append(hook_name(**kwargs))

  return train_hooks


def get_logging_tensor_hook(every_n_iter=100, **kwargs):  # pylint: disable=unused-argument
  """Function to get LoggingTensorHook.

  Args:
    every_n_iter: `int`, print the values of `tensors` once every N local
      steps taken on the current worker.
    kwargs: a dictionary of arguments to LoggingTensorHook.

  Returns:
    Returns a LoggingTensorHook with a standard set of tensors that will be
      printed to stdout.
  """
  tensors_to_log = {
      'learning_rate': 'learning_rate',
      'cross_entropy': 'cross_entropy',
      'train_accuracy': 'train_accuracy'
  }

  return tf.train.LoggingTensorHook(
      tensors=tensors_to_log,
      every_n_iter=every_n_iter)


def get_profiler_hook(save_steps=1000, **kwargs):  # pylint: disable=unused-argument
  """Function to get ProfilerHook.

  Args:
    save_steps: `int`, print profile traces every N steps.
    kwargs: a dictionary of arguments to ProfilerHook.

  Returns:
    Returns a ProfilerHook that writes out timelines that can be loaded into
      profiling tools like chrome://tracing.
  """
  return tf.train.ProfilerHook(save_steps=save_steps)


def get_examples_per_second_hook(every_n_steps=100, **kwargs):
  """Function to get ExamplesPerSecondHook.

  Args:
    every_n_steps: `int`, print current and average examples per second every
      N steps.
    kwargs: a dictionary of arguments to ExamplesPerSecondHook.

  Returns:
    Returns a ProfilerHook that writes out timelines that can be loaded into
      profiling tools like chrome://tracing.
  """
  return hooks.ExamplesPerSecondHook(
      every_n_steps=every_n_steps,
      batch_size=kwargs.get('batch_size', 128),
      warm_steps=kwargs.get('warm_steps', 10))


HOOKS = {
    'loggingtensorhook': get_logging_tensor_hook,
    'profilerhook': get_profiler_hook,
    'examplespersecondhook': get_profiler_hook,
}
