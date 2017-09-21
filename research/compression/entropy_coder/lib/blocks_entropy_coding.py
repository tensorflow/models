# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Set of blocks related to entropy coding."""

import math

import tensorflow as tf

import block_base

# pylint does not recognize block_base.BlockBase.__call__().
# pylint: disable=not-callable


class CodeLength(block_base.BlockBase):
  """Theoretical bound for a code length given a probability distribution.
  """

  def __init__(self, name=None):
    super(CodeLength, self).__init__(name)

  def _Apply(self, c, p):
    """Theoretical bound of the coded length given a probability distribution.

    Args:
      c: The binary codes. Belong to {0, 1}.
      p: The probability of: P(code==+1)

    Returns:
      The average code length.
      Note: the average code length can be greater than 1 bit (e.g. when
          encoding the least likely symbol).
    """
    entropy = ((1.0 - c) * tf.log(1.0 - p) + c * tf.log(p)) / (-math.log(2))
    entropy = tf.reduce_mean(entropy)
    return entropy
