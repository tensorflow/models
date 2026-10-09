# Copyright 2025 The Orbit Authors. All Rights Reserved.
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

"""Provides the `PeriodicAction` abstraction."""

from typing import Callable

from orbit import runner
import tensorflow as tf


class PeriodicAction:
  """Wraps an action to be executed only at a specific step interval."""

  def __init__(self,
               action: Callable[[runner.Output], None],
               interval: int,
               global_step: tf.Variable):
    """Initializes the instance.

    Args:
      action: The action (callable) to wrap.
      interval: The interval (in global steps) at which to execute the action.
      global_step: The global step variable.
    """
    self._action = action
    self._interval = interval
    self._global_step = global_step

  def __call__(self, output: runner.Output) -> None:
    # Execute action only if the current step is divisible by the interval.
    if self._global_step.numpy() % self._interval == 0:
      self._action(output)