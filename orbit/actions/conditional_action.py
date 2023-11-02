# Copyright 2023 The Orbit Authors. All Rights Reserved.
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

"""Provides a `ConditionalAction` abstraction."""

from typing import Any, Callable, Sequence, Union

from orbit import controller
from orbit import runner

import tensorflow as tf, tf_keras

Condition = Callable[[runner.Output], Union[bool, tf.Tensor]]


def _as_sequence(maybe_sequence: Union[Any, Sequence[Any]]) -> Sequence[Any]:
  if isinstance(maybe_sequence, Sequence):
    return maybe_sequence
  return [maybe_sequence]


class ConditionalAction:
  """Represents an action that is only taken when a given condition is met.

  This class is itself an `Action` (a callable that can be applied to train or
  eval outputs), but is intended to make it easier to write modular and reusable
  conditions by decoupling "when" something whappens (the condition) from "what"
  happens (the action).
  """

  def __init__(
      self,
      condition: Condition,
      action: Union[controller.Action, Sequence[controller.Action]],
  ):
    """Initializes the instance.

    Args:
      condition: A callable accepting train or eval outputs and returing a bool.
      action: The action (or optionally sequence of actions) to perform when
        `condition` is met.
    """
    self.condition = condition
    self.action = action

  def __call__(self, output: runner.Output) -> None:
    if self.condition(output):
      for action in _as_sequence(self.action):
        action(output)
