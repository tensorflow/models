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

"""Common blocks which work as operators on other blocks."""

import tensorflow as tf

import block_base

# pylint: disable=not-callable


class CompositionOperator(block_base.BlockBase):
  """Composition of several blocks."""

  def __init__(self, block_list, name=None):
    """Initialization of the composition operator.

    Args:
      block_list: List of blocks.BlockBase that are chained to create
        a new blocks.BlockBase.
      name: Name of this block.
    """
    super(CompositionOperator, self).__init__(name)
    self._blocks = block_list

  def _Apply(self, x):
    """Apply successively all the blocks on the given input tensor."""
    h = x
    for layer in self._blocks:
      h = layer(h)
    return h


class LineOperator(block_base.BlockBase):
  """Repeat the same block over all the lines of an input tensor."""

  def __init__(self, block, name=None):
    super(LineOperator, self).__init__(name)
    self._block = block

  def _Apply(self, x):
    height = x.get_shape()[1].value
    if height is None:
      raise ValueError('Unknown tensor height')
    all_line_x = tf.split(value=x, num_or_size_splits=height, axis=1)

    y = []
    for line_x in all_line_x:
      y.append(self._block(line_x))
    y = tf.concat(values=y, axis=1)

    return y


class TowerOperator(block_base.BlockBase):
  """Parallel execution with concatenation of several blocks."""

  def __init__(self, block_list, dim=3, name=None):
    """Initialization of the parallel exec + concat (Tower).

    Args:
      block_list: List of blocks.BlockBase that are chained to create
        a new blocks.BlockBase.
      dim: the dimension on which to concat.
      name: Name of this block.
    """
    super(TowerOperator, self).__init__(name)
    self._blocks = block_list
    self._concat_dim = dim

  def _Apply(self, x):
    """Apply successively all the blocks on the given input tensor."""
    outputs = [layer(x) for layer in self._blocks]
    return tf.concat(outputs, self._concat_dim)
