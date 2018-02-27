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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_train_hooks(names):
  """Factory for getting a list of TensorFlow hooks for training by name.

  Args:
    names: case-insensitive, comma-separated string with names of desired hook
      classes. Allowed: LoggingTensorHook, ProfilerHook, ExamplesPerSecondHook

  Returns:
    list of instantiated hooks, ready to be used in a classifier.train call

  Raises:
    ValueError: if an unrecognized name is passed
  """

  name_list = [name.strip().lower() for name in names.split(',')]

  hooks = []
  for name in name_list:
    if name == 'loggingtensorhook':
      hooks.append(get_logging_tensor_hook())
    elif name == 'profilerhook':
      hooks.append(get_profiler_hook())
    elif name == 'examplepersecondhook':
      hooks.append(get_examples_per_second_hook())
    else:
      raise ValueError('Unrecognized training hook requested: {}'.format(name))

  return hooks


def get_logging_tensor_hook():
  """Returns a LoggingTensorHook with a standard set of tensors that will be
  printed to stdout.
  """
  tensors_to_log = {
      'learning_rate': 'learning_rate',
      'cross_entropy': 'cross_entropy',
      'train_accuracy': 'train_accuracy'
  }

  return tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)


def get_profiler_hook():
  """Returns a ProfilerHook that writes out timelines that can be loaded into
  profiling tools like chrome://tracing.
  """
  return tf.train.ProfilerHook(save_steps=1000, output_dir='')


def get_examples_per_second_hook():
  pass
