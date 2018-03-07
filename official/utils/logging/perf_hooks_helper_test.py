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

"""Tests for perf_hooks_helper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import perf_hooks_helper
import tensorflow as tf

_ALLOWED_HOOKS = ['loggingtensorhook', 'profilerhook', 'examplespersecondhook']
tf.logging.set_verbosity(tf.logging.ERROR)


class BaseTest(tf.test.TestCase):

  def test_raise_in_non_string_names(self):
    with self.assertRaises(ValueError):
      perf_hooks_helper.get_train_hooks(
          ['LoggingTensorHook', 'ProfilerHook'], batch_size=256)

  def test_raise_in_invalid_names(self):
    invalid_names = 'StepCounterHook, StopAtStepHook'
    with self.assertRaises(ValueError):
      perf_hooks_helper.get_train_hooks(invalid_names, batch_size=256)

  def get_train_hooks_valid_names_helper(self, hook_names, **kwargs):
    returned_hooks = perf_hooks_helper.get_train_hooks(hook_names, **kwargs)
    for returned_hook in returned_hooks:
      self.assertIsInstance(returned_hook, tf.train.SessionRunHook)
      self.assertIn(returned_hook.__class__.__name__.lower(), _ALLOWED_HOOKS)

  def test_get_train_hooks_one_valid_names(self):
    valid_names = 'LoggingTensorHook'
    self.get_train_hooks_valid_names_helper(valid_names, batch_size=256)

  def test_get_train_hooks_three_valid_names(self):
    valid_names = 'LoggingTensorHook, profilerhook, ExamplesPerSecondHook'
    self.get_train_hooks_valid_names_helper(valid_names, batch_size=256)

if __name__ == '__main__':
  tf.test.main()
