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

import perf_hooks
import tensorflow as tf


def get_train_hooks(names, **kwargs):
  """Factory for getting a list of TensorFlow hooks for training by name.

  Args:
    names: case-insensitive, comma-separated string with names of desired hook
      classes. Allowed: LoggingTensorHook, ProfilerHook, ExamplesPerSecondHook
    **kwargs: a dictionary of arguments to the hooks

  Returns:
    list of instantiated hooks, ready to be used in a classifier.train call

  Raises:
    ValueError: if an unrecognized name is passed
  """

  if not names:
    return []

  if not isinstance(names, str):
    raise ValueError('Hook names should be a string')

  name_list = [name.strip().lower() for name in names.split(',')]

  hooks = []
  for name in name_list:
    if name == 'loggingtensorhook':
      hooks.append(get_logging_tensor_hook(**kwargs))
    elif name == 'profilerhook':
      hooks.append(get_profiler_hook())
    elif name == 'examplespersecondhook':
      hooks.append(get_examples_per_second_hook())
    else:
      raise ValueError('Unrecognized training hook requested: {}'.format(name))

  return hooks


def get_logging_tensor_hook(**kwargs):
  # Returns a LoggingTensorHook with a standard set of tensors that will be
  # printed to stdout.

  tensors_to_log = {
      'learning_rate': 'learning_rate',
      'cross_entropy': 'cross_entropy',
      'train_accuracy': 'train_accuracy'
  }

  return tf.train.LoggingTensorHook(
      tensors=tensors_to_log,
      every_n_iter=kwargs.get('every_n_iter', 100))


def get_profiler_hook(**kwargs):
  # Returns a ProfilerHook that writes out timelines that can be loaded into
  # profiling tools like chrome://tracing.

  return tf.train.ProfilerHook(save_steps=kwargs.get('save_steps', 1000))


def get_examples_per_second_hook(**kwargs):
  return perf_hooks.ExamplesPerSecondHook(
      batch_size=kwargs.get('batch_size', 128),
      warm_steps=kwargs.get('warm_steps', 1000))
