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
"""Flags related to distributed execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import tensorflow as tf

from official.utils.flags._conventions import help_wrap


def define_distribution(worker_hosts=True, task_index=True):
  """Register distributed execution flags.

  Args:
    worker_hosts: Create a flag for specifying comma-separated list of workers.
    task_index: Create a flag for specifying index of task.

  Returns:
    A list of flags for core.py to marks as key flags.
  """
  key_flags = []

  if worker_hosts:
    flags.DEFINE_string(
        name='worker_hosts', default=None,
        help=help_wrap(
            'Comma-separated list of worker ip:port pairs for running '
            'multi-worker models with DistributionStrategy.  The user would '
            'start the program on each host with identical value for this '
            'flag.'))

  if task_index:
    flags.DEFINE_integer(
        name='task_index', default=-1,
        help=help_wrap('If multi-worker training, the task_index of this '
                       'worker.'))

  return key_flags
