# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Common util functions and classes used by both keras cifar and imagenet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os

import numpy as np

# pylint: disable=g-bad-import-order
from absl import flags
import tensorflow as tf


FLAGS = flags.FLAGS


def define_ctl_flags():
  """Define flags for CTL."""

  flags.DEFINE_boolean(name='enable_function', default=True,
      help='Enable tf.function')
  flags.DEFINE_boolean(name='skip_eval', default=False, help='Skip evaluation?')
  flags.DEFINE_boolean(name='report_accuracy_metrics', default=True,
                       help='Report metrics during training and evaluation.')
  flags.DEFINE_integer(
      name='train_steps', default=None,
      help='The number of steps to run for training. If it is larger than '
      '# batches per epoch, then use # batches per epoch. When this flag is '
      'set, only one epoch is going to run for training.')
  # This is used when we create a synth dataset.
  flags.DEFINE_boolean(
      name='enable_xla', default=False,
      help='Whether to enable XLA auto jit compilation. This is still an '
      'experimental feature, and is not yet effective with TF 2.0.')

def is_v2_0():
  """Returns true if using tf 2.0."""
  return tf.__version__.startswith('2')
