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

"""Blocks of LSTM and its variants."""

import numpy as np
import tensorflow as tf

import block_base
import block_util
import blocks_std

# pylint does not recognize block_base.BlockBase.__call__().
# pylint: disable=not-callable


def LSTMBiasInit(shape, dtype):
  """Returns ones for forget-gate, and zeros for the others."""
  shape = np.array(shape)

  # Check internal consistencies.
  assert shape.shape == (1,), shape
  assert shape[0] % 4 == 0, shape

  n = shape[0] // 4
  ones = tf.fill([n], tf.constant(1, dtype=dtype))
  zeros = tf.fill([3 * n], tf.constant(0, dtype=dtype))
  return tf.concat([ones, zeros], 0)


class LSTMBase(block_base.BlockBase):
  """Base class for LSTM implementations.

  These LSTM implementations use the pattern found in [1]. No peephole
  connection, i.e., cell content is not used in recurrence computation.
  Hidden units are also output units.

  [1] Zaremba, Sutskever, Vinyals. Recurrent Neural Network Regularization,
  2015. arxiv:1409.2329.
  """

  def __init__(self, output_shape, name):
    """Initializes LSTMBase class object.

    Args:
      output_shape: List representing the LSTM output shape. This argument
        does not include batch dimension. For example, if the LSTM output has
        shape [batch, depth], then pass [depth].
      name: Name of this block.
    """
    super(LSTMBase, self).__init__(name)

    with self._BlockScope():
      self._output_shape = [None] + list(output_shape)
      self._hidden = None
      self._cell = None

  @property
  def hidden(self):
    """Returns the hidden units of this LSTM."""
    return self._hidden

  @hidden.setter
  def hidden(self, value):
    """Assigns to the hidden units of this LSTM.

    Args:
      value: The new value for the hidden units. If None, the hidden units are
        considered to be filled with zeros.
    """
    if value is not None:
      value.get_shape().assert_is_compatible_with(self._output_shape)
    self._hidden = value

  @property
  def cell(self):
    """Returns the cell units of this LSTM."""
    return self._cell

  @cell.setter
  def cell(self, value):
    """Assigns to the cell units of this LSTM.

    Args:
      value: The new value for the cell units. If None, the cell units are
        considered to be filled with zeros.
    """
    if value is not None:
      value.get_shape().assert_is_compatible_with(self._output_shape)
    self._cell = value

  # Consider moving bias terms to the base, and require this method to be
  # linear.
  def _TransformInputs(self, _):
    """Transforms the input units to (4 * depth) units.

    The forget-gate, input-gate, output-gate, and cell update is computed as
      f, i, j, o = T(h) + R(x)
    where h is hidden units, x is input units, and T, R are transforms of
    h, x, respectively.

    This method implements R. Note that T is strictly linear, so if LSTM is
    going to use bias, this method must include the bias to the transformation.

    Subclasses must implement this method. See _Apply() for more details.
    """
    raise NotImplementedError()

  def _TransformHidden(self, _):
    """Transforms the hidden units to (4 * depth) units.

    The forget-gate, input-gate, output-gate, and cell update is computed as
      f, i, j, o = T(h) + R(x)
    where h is hidden units, x is input units, and T, R are transforms of
    h, x, respectively.

    This method implements T in the equation. The method must implement a
    strictly linear transformation. For example, it may use MatMul or Conv2D,
    but must not add bias. This is because when hidden units are zeros, then
    the LSTM implementation will skip calling this method, instead of passing
    zeros to this function.

    Subclasses must implement this method. See _Apply() for more details.
    """
    raise NotImplementedError()

  def _Apply(self, *args):
    xtransform = self._TransformInputs(*args)
    depth_axis = len(self._output_shape) - 1

    if self.hidden is not None:
      htransform = self._TransformHidden(self.hidden)
      f, i, j, o = tf.split(
          value=htransform + xtransform, num_or_size_splits=4, axis=depth_axis)
    else:
      f, i, j, o = tf.split(
          value=xtransform, num_or_size_splits=4, axis=depth_axis)

    if self.cell is not None:
      self.cell = tf.sigmoid(f) * self.cell + tf.sigmoid(i) * tf.tanh(j)
    else:
      self.cell = tf.sigmoid(i) * tf.tanh(j)

    self.hidden = tf.sigmoid(o) * tf.tanh(self.cell)
    return self.hidden


class LSTM(LSTMBase):
  """Efficient LSTM implementation used in [1].

  [1] Zaremba, Sutskever, Vinyals. Recurrent Neural Network Regularization,
  2015. arxiv:1409.2329.
  """

  def __init__(self,
               depth,
               bias=LSTMBiasInit,
               initializer=block_util.RsqrtInitializer(),
               name=None):
    super(LSTM, self).__init__([depth], name)

    with self._BlockScope():
      self._depth = depth
      self._nn = blocks_std.NN(
          4 * depth, bias=bias, act=None, initializer=initializer)
      self._hidden_linear = blocks_std.Linear(
          4 * depth, initializer=initializer)

  def _TransformInputs(self, *args):
    return self._nn(*args)

  def _TransformHidden(self, h):
    return self._hidden_linear(h)


class Conv2DLSTM(LSTMBase):
  """Convolutional LSTM implementation with optimizations inspired by [1].

  Note that when using the batch normalization feature, the bias initializer
  will not be used, since BN effectively cancels its effect out.

  [1] Zaremba, Sutskever, Vinyals. Recurrent Neural Network Regularization,
  2015. arxiv:1409.2329.
  """

  def __init__(self,
               depth,
               filter_size,
               hidden_filter_size,
               strides,
               padding,
               bias=LSTMBiasInit,
               initializer=block_util.RsqrtInitializer(dims=(0, 1, 2)),
               use_moving_average=False,
               name=None):
    super(Conv2DLSTM, self).__init__([None, None, depth], name)
    self._iter = 0

    with self._BlockScope():
      self._input_conv = blocks_std.Conv2D(
          4 * depth,
          filter_size,
          strides,
          padding,
          bias=None,
          act=None,
          initializer=initializer,
          name='input_conv2d')

      self._hidden_conv = blocks_std.Conv2D(
          4 * depth,
          hidden_filter_size,
          [1, 1],
          'SAME',
          bias=None,
          act=None,
          initializer=initializer,
          name='hidden_conv2d')

      if bias is not None:
        self._bias = blocks_std.BiasAdd(bias, name='biases')
      else:
        self._bias = blocks_std.PassThrough()

  def _TransformInputs(self, x):
    return self._bias(self._input_conv(x))

  def _TransformHidden(self, h):
    return self._hidden_conv(h)

  def _Apply(self, *args):
    xtransform = self._TransformInputs(*args)
    depth_axis = len(self._output_shape) - 1

    if self.hidden is not None:
      htransform = self._TransformHidden(self.hidden)
      f, i, j, o = tf.split(
          value=htransform + xtransform, num_or_size_splits=4, axis=depth_axis)
    else:
      f, i, j, o = tf.split(
          value=xtransform, num_or_size_splits=4, axis=depth_axis)

    if self.cell is not None:
      self.cell = tf.sigmoid(f) * self.cell + tf.sigmoid(i) * tf.tanh(j)
    else:
      self.cell = tf.sigmoid(i) * tf.tanh(j)

    self.hidden = tf.sigmoid(o) * tf.tanh(self.cell)

    self._iter += 1
    return self.hidden
