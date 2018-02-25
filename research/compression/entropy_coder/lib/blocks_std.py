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

"""Basic blocks for building tensorflow models."""

import numpy as np
import tensorflow as tf

import block_base
import block_util

# pylint does not recognize block_base.BlockBase.__call__().
# pylint: disable=not-callable


def HandleConvPaddingModes(x, padding, kernel_shape, strides):
  """Returns an updated tensor and padding type for REFLECT and SYMMETRIC.

  Args:
    x: A 4D tensor with shape [batch_size, height, width, depth].
    padding: Padding mode (SAME, VALID, REFLECT, or SYMMETRIC).
    kernel_shape: Shape of convolution kernel that will be applied.
    strides: Convolution stride that will be used.

  Returns:
    x and padding after adjustments for REFLECT and SYMMETRIC.
  """
  # For 1x1 convolution, all padding modes are the same.
  if np.all(kernel_shape[:2] == 1):
    return x, 'VALID'

  if padding == 'REFLECT' or padding == 'SYMMETRIC':
    # We manually compute the number of paddings as if 'SAME'.
    # From Tensorflow kernel, the formulas are as follows.
    #   output_shape = ceil(input_shape / strides)
    #   paddings = (output_shape - 1) * strides + filter_size - input_shape
    # Let x, y, s be a shorthand notations for input_shape, output_shape, and
    # strides, respectively. Let (x - 1) = sn + r where 0 <= r < s. Note that
    #   y - 1 = ceil(x / s) - 1 = floor((x - 1) / s) = n
    # provided that x > 0. Therefore
    #   paddings = n * s + filter_size - (sn + r + 1)
    #            = filter_size - r - 1.
    input_shape = x.get_shape()  # shape at graph construction time
    img_shape = tf.shape(x)[1:3]  # image shape (no batch) at run time
    remainder = tf.mod(img_shape - 1, strides[1:3])
    pad_sizes = kernel_shape[:2] - remainder - 1

    pad_rows = pad_sizes[0]
    pad_cols = pad_sizes[1]
    pad = tf.stack([[0, 0], tf.stack([pad_rows // 2, (pad_rows + 1) // 2]),
                    tf.stack([pad_cols // 2, (pad_cols + 1) // 2]), [0, 0]])

    # Manually pad the input and switch the padding mode to 'VALID'.
    x = tf.pad(x, pad, mode=padding)
    x.set_shape([input_shape[0], x.get_shape()[1],
                 x.get_shape()[2], input_shape[3]])
    padding = 'VALID'

  return x, padding


class PassThrough(block_base.BlockBase):
  """A dummy transform block that does nothing."""

  def __init__(self):
    # Pass an empty string to disable name scoping.
    super(PassThrough, self).__init__(name='')

  def _Apply(self, inp):
    return inp

  @property
  def initialized(self):
    """Always returns True."""
    return True


class Bias(object):
  """An initialization helper class for BiasAdd block below."""

  def __init__(self, value=0):
    self.value = value


class BiasAdd(block_base.BlockBase):
  """A tf.nn.bias_add wrapper.

  This wrapper may act as a PassThrough block depending on the initializer
  provided, to make easier optional bias applications in NN blocks, etc.
  See __init__() for the details.
  """

  def __init__(self, initializer=Bias(0), name=None):
    """Initializes Bias block.

    |initializer| parameter have two special cases.

    1. If initializer is None, then this block works as a PassThrough.
    2. If initializer is a Bias class object, then tf.constant_initializer is
       used with the stored value.

    Args:
      initializer: An initializer for the bias variable.
      name: Name of this block.
    """
    super(BiasAdd, self).__init__(name)

    with self._BlockScope():
      if isinstance(initializer, Bias):
        self._initializer = tf.constant_initializer(value=initializer.value)
      else:
        self._initializer = initializer

      self._bias = None

  def _Apply(self, x):
    if not self._bias:
      init = self._initializer([int(x.get_shape()[-1])], x.dtype)
      self._bias = self.NewVar(init)

    return tf.nn.bias_add(x, self._bias)

  def CreateWeightLoss(self):
    return []


class LinearBase(block_base.BlockBase):
  """A matmul wrapper.

  Returns input * W, where matrix W can be customized through derivation.
  """

  def __init__(self, depth, name=None):
    super(LinearBase, self).__init__(name)

    with self._BlockScope():
      self._depth = depth
      self._matrix = None

  def _CreateKernel(self, shape, dtype):
    raise NotImplementedError('This method must be sub-classed.')

  def _Apply(self, x):
    if not self._matrix:
      shape = [int(x.get_shape()[-1]), self._depth]
      self._matrix = self._CreateKernel(shape, x.dtype)

    return tf.matmul(x, self._matrix)


class Linear(LinearBase):
  """A matmul wrapper.

  Returns input * W, where matrix W is learned.
  """

  def __init__(self,
               depth,
               initializer=block_util.RsqrtInitializer(),
               name=None):
    super(Linear, self).__init__(depth, name)

    with self._BlockScope():
      self._initializer = initializer

  def _CreateKernel(self, shape, dtype):
    init = self._initializer(shape, dtype)
    return self.NewVar(init)


class NN(block_base.BlockBase):
  """A neural network layer wrapper.

  Returns act(input * W + b), where matrix W, bias b are learned, and act is an
  optional activation function (i.e., nonlinearity).

  This transform block can handle multiple inputs. If x_1, x_2, ..., x_m are
  the inputs, then returns act(x_1 * W_1 + ... + x_m * W_m + b).

  Attributes:
    nunits: The dimension of the output.
  """

  def __init__(self,
               depth,
               bias=Bias(0),
               act=None,  # e.g., tf.nn.relu
               initializer=block_util.RsqrtInitializer(),
               linear_block_factory=(lambda d, i: Linear(d, initializer=i)),
               name=None):
    """Initializes NN block.

    Args:
      depth: The depth of the output.
      bias: An initializer for the bias, or a Bias class object. If None, there
        will be no bias term for this NN block. See BiasAdd block.
      act: Optional activation function. If None, no activation is applied.
      initializer: The initialization method for the matrix weights.
      linear_block_factory: A function used to create a linear block.
      name: The name of this block.
    """
    super(NN, self).__init__(name)

    with self._BlockScope():
      self._linear_block_factory = linear_block_factory
      self._depth = depth
      self._initializer = initializer
      self._matrices = None

      self._bias = BiasAdd(bias) if bias else PassThrough()
      self._act = act if act else PassThrough()

  def _Apply(self, *args):
    if not self._matrices:
      self._matrices = [
          self._linear_block_factory(self._depth, self._initializer)
          for _ in args]

    if len(self._matrices) != len(args):
      raise ValueError('{} expected {} inputs, but observed {} inputs'.format(
          self.name, len(self._matrices), len(args)))

    if len(args) > 1:
      y = tf.add_n([m(x) for m, x in zip(self._matrices, args)])
    else:
      y = self._matrices[0](args[0])

    return self._act(self._bias(y))


class Conv2DBase(block_base.BlockBase):
  """A tf.nn.conv2d operator."""

  def __init__(self, depth, filter_size, strides, padding,
               bias=None, act=None, atrous_rate=None, conv=tf.nn.conv2d,
               name=None):
    """Initializes a Conv2DBase block.

    Arguments:
      depth: The output depth of the block (i.e. #filters); if negative, the
        output depth will be set to be the same as the input depth.
      filter_size: The size of the 2D filter. If it's specified as an integer,
        it's going to create a square filter. Otherwise, this is a tuple
        specifying the height x width of the filter.
      strides: A tuple specifying the y and x stride.
      padding: One of the valid padding modes allowed by tf.nn.conv2d, or
        'REFLECT'/'SYMMETRIC' for mirror padding.
      bias: An initializer for the bias, or a Bias class object. If None, there
          will be no bias in this block. See BiasAdd block.
      act: Optional activation function applied to the output.
      atrous_rate: optional input rate for ATrous convolution. If not None, this
          will be used and the strides will be ignored.
      conv: The convolution function to use (e.g. tf.nn.conv2d).
      name: The name for this conv2d op.
    """
    super(Conv2DBase, self).__init__(name)

    with self._BlockScope():
      self._act = act if act else PassThrough()
      self._bias = BiasAdd(bias) if bias else PassThrough()

      self._kernel_shape = np.zeros((4,), dtype=np.int32)
      self._kernel_shape[:2] = filter_size
      self._kernel_shape[3] = depth

      self._strides = np.ones((4,), dtype=np.int32)
      self._strides[1:3] = strides
      self._strides = list(self._strides)

      self._padding = padding

      self._kernel = None
      self._conv = conv

      self._atrous_rate = atrous_rate

  def _CreateKernel(self, shape, dtype):
    raise NotImplementedError('This method must be sub-classed')

  def _Apply(self, x):
    """Apply the self._conv op.

    Arguments:
      x: input tensor. It needs to be a 4D tensor of the form
          [batch, height, width, channels].
    Returns:
      The output of the convolution of x with the current convolutional
      kernel.
    Raises:
      ValueError: if number of channels is not defined at graph construction.
    """
    input_shape = x.get_shape().with_rank(4)
    input_shape[3:].assert_is_fully_defined()  # channels must be defined
    if self._kernel is None:
      assert self._kernel_shape[2] == 0, self._kernel_shape
      self._kernel_shape[2] = input_shape[3].value
      if self._kernel_shape[3] < 0:
        # Make output depth be the same as input depth.
        self._kernel_shape[3] = self._kernel_shape[2]
      self._kernel = self._CreateKernel(self._kernel_shape, x.dtype)

    x, padding = HandleConvPaddingModes(
        x, self._padding, self._kernel_shape, self._strides)
    if self._atrous_rate is None:
      x = self._conv(x, self._kernel, strides=self._strides, padding=padding)
    else:
      x = self._conv(x, self._kernel, rate=self._atrous_rate, padding=padding)

    if self._padding != 'VALID':
      # Manually update shape. Known shape information can be lost by tf.pad().
      height = (1 + (input_shape[1].value - 1) // self._strides[1]
                if input_shape[1].value else None)
      width = (1 + (input_shape[2].value - 1) // self._strides[2]
               if input_shape[2].value else None)
      shape = x.get_shape()
      x.set_shape([shape[0], height, width, shape[3]])

    return self._act(self._bias(x))


class Conv2D(Conv2DBase):
  """A tf.nn.conv2d operator."""

  def __init__(self, depth, filter_size, strides, padding,
               bias=None, act=None, initializer=None, name=None):
    """Initializes a Conv2D block.

    Arguments:
      depth: The output depth of the block (i.e., #filters)
      filter_size: The size of the 2D filter. If it's specified as an integer,
        it's going to create a square filter. Otherwise, this is a tuple
        specifying the height x width of the filter.
      strides: A tuple specifying the y and x stride.
      padding: One of the valid padding modes allowed by tf.nn.conv2d, or
        'REFLECT'/'SYMMETRIC' for mirror padding.
      bias: An initializer for the bias, or a Bias class object. If None, there
          will be no bias in this block. See BiasAdd block.
      act: Optional activation function applied to the output.
      initializer: Optional initializer for weights.
      name: The name for this conv2d op.
    """
    super(Conv2D, self).__init__(depth, filter_size, strides, padding, bias,
                                 act, conv=tf.nn.conv2d, name=name)

    with self._BlockScope():
      if initializer is None:
        initializer = block_util.RsqrtInitializer(dims=(0, 1, 2))
      self._initializer = initializer

  def _CreateKernel(self, shape, dtype):
    return self.NewVar(self._initializer(shape, dtype))
