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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow as tf

flags = tf.flags
FLAGS = tf.app.flags.FLAGS
LSTMTuple = collections.namedtuple('LSTMTuple', ['c', 'h'])


def cell_depth(num):
  num /= 2
  val = np.log2(1 + num)
  assert abs(val - int(val)) == 0
  return int(val)


class GenericMultiRNNCell(tf.contrib.rnn.RNNCell):
  """More generic version of MultiRNNCell that allows you to pass in a dropout mask"""

  def __init__(self, cells):
    """Create a RNN cell composed sequentially of a number of RNNCells.

    Args:
      cells: list of RNNCells that will be composed in this order.
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  If False, the states are all
        concatenated along the column axis.  This latter behavior will soon be
        deprecated.

    Raises:
      ValueError: if cells is empty (not allowed), or at least one of the cells
        returns a state tuple but the flag `state_is_tuple` is `False`.
    """
    self._cells = cells

  @property
  def state_size(self):
    return tuple(cell.state_size for cell in self._cells)

  @property
  def output_size(self):
    return self._cells[-1].output_size

  def __call__(self, inputs, state, input_masks=None, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    with tf.variable_scope(scope or type(self).__name__):
      cur_inp = inputs
      new_states = []
      for i, cell in enumerate(self._cells):
        with tf.variable_scope('Cell%d' % i):
          cur_state = state[i]
          if input_masks is not None:
            cur_inp *= input_masks[i]
          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)
    new_states = tuple(new_states)
    return cur_inp, new_states


class AlienRNNBuilder(tf.contrib.rnn.RNNCell):

  def __init__(self, num_units, params, additional_params, base_size):
    self.num_units = num_units
    self.cell_create_index = additional_params[0]
    self.cell_inject_index = additional_params[1]
    self.base_size = base_size
    self.cell_params = params[
        -2:]  # Cell injection parameters are always the last two
    params = params[:-2]
    self.depth = cell_depth(len(params))
    self.params = params
    self.units_per_layer = [2**i for i in range(self.depth)
                           ][::-1]  # start with the biggest layer

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      definition1 = ['add', 'elem_mult', 'max']
      definition2 = [tf.identity, tf.tanh, tf.sigmoid, tf.nn.relu, tf.sin]
      layer_outputs = [[] for _ in range(self.depth)]
      with tf.variable_scope('rnn_builder'):
        curr_index = 0
        c, h = state

        # Run all dense matrix multiplications at once
        big_h_mat = tf.get_variable(
            'big_h_mat', [self.num_units,
                          self.base_size * self.num_units], tf.float32)
        big_inputs_mat = tf.get_variable(
            'big_inputs_mat', [self.num_units,
                               self.base_size * self.num_units], tf.float32)
        big_h_output = tf.matmul(h, big_h_mat)
        big_inputs_output = tf.matmul(inputs, big_inputs_mat)
        h_splits = tf.split(big_h_output, self.base_size, axis=1)
        inputs_splits = tf.split(big_inputs_output, self.base_size, axis=1)

        for layer_num, units in enumerate(self.units_per_layer):
          for unit_num in range(units):
            with tf.variable_scope(
                'layer_{}_unit_{}'.format(layer_num, unit_num)):
              if layer_num == 0:
                prev1_mat = h_splits[unit_num]
                prev2_mat = inputs_splits[unit_num]
              else:
                prev1_mat = layer_outputs[layer_num - 1][2 * unit_num]
                prev2_mat = layer_outputs[layer_num - 1][2 * unit_num + 1]
              if definition1[self.params[curr_index]] == 'add':
                output = prev1_mat + prev2_mat
              elif definition1[self.params[curr_index]] == 'elem_mult':
                output = prev1_mat * prev2_mat
              elif definition1[self.params[curr_index]] == 'max':
                output = tf.maximum(prev1_mat, prev2_mat)
              if curr_index / 2 == self.cell_create_index:  # Take the new cell before the activation
                new_c = tf.identity(output)
              output = definition2[self.params[curr_index + 1]](output)
              if curr_index / 2 == self.cell_inject_index:
                if definition1[self.cell_params[0]] == 'add':
                  output += c
                elif definition1[self.cell_params[0]] == 'elem_mult':
                  output *= c
                elif definition1[self.cell_params[0]] == 'max':
                  output = tf.maximum(output, c)
                output = definition2[self.cell_params[1]](output)
              layer_outputs[layer_num].append(output)
              curr_index += 2
        new_h = layer_outputs[-1][-1]
        return new_h, LSTMTuple(new_c, new_h)

  @property
  def state_size(self):
    return LSTMTuple(self.num_units, self.num_units)

  @property
  def output_size(self):
    return self.num_units


class Alien(AlienRNNBuilder):
  """Base 8 Cell."""

  def __init__(self, num_units):
    params = [
        0, 2, 0, 3, 0, 2, 1, 3, 0, 1, 0, 2, 0, 1, 0, 2, 1, 1, 0, 1, 1, 1, 0, 2,
        1, 0, 0, 1, 1, 1, 0, 1
    ]
    additional_params = [12, 8]
    base_size = 8
    super(Alien, self).__init__(num_units, params, additional_params, base_size)
