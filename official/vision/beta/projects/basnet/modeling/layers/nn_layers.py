# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Contains common building blocks for neural networks."""

from typing import Callable, Dict, List, Optional, Tuple, Union

from absl import logging
import tensorflow as tf

from official.modeling import tf_utils


# Type annotations.
States = Dict[str, tf.Tensor]
Activation = Union[str, Callable]


def make_divisible(value: float,
                   divisor: int,
                   min_value: Optional[float] = None
                   ) -> int:
  """This is to ensure that all layers have channels that are divisible by 8.

  Args:
    value: A `float` of original value.
    divisor: An `int` off the divisor that need to be checked upon.
    min_value: A `float` of  minimum value threshold.

  Returns:
    The adjusted value in `int` that is divisible against divisor.
  """
  if min_value is None:
    min_value = divisor
  new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_value < 0.9 * value:
    new_value += divisor
  return new_value

def round_filters(filters: int,
                  multiplier: float,
                  divisor: int = 8,
                  min_depth: Optional[int] = None,
                  skip: bool = False):
  """Rounds number of filters based on width multiplier."""
  orig_f = filters
  if skip or not multiplier:
    return filters
