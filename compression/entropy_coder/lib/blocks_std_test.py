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

"""Tests for basic tensorflow blocks_std."""

from __future__ import division
from __future__ import unicode_literals

import math
import os

import numpy as np
import tensorflow as tf

import blocks_std


def _NumpyConv2D(x, f, strides, padding, rate=1):
  assert strides[0] == 1 and strides[3] == 1, strides

  if rate > 1:
    f_shape = f.shape
    expand_f = np.zeros([f_shape[0], ((f_shape[1] - 1) * rate + 1),
                         f_shape[2], f_shape[3]])
    expand_f[:, [y * rate for y in range(f_shape[1])], :, :] = f
    f = np.zeros([((f_shape[0] - 1) * rate + 1), expand_f.shape[1],
                  f_shape[2], f_shape[3]])
    f[[y * rate for y in range(f_shape[0])], :, :, :] = expand_f

  if padding != 'VALID':
    assert x.shape[1] > 0 and x.shape[2] > 0, x.shape
    # Compute the number of padded rows and cols.
    # See Conv2D block comments for a math explanation.
    remainder = ((x.shape[1] - 1) % strides[1], (x.shape[2] - 1) % strides[2])
    pad_rows = f.shape[0] - remainder[0] - 1
    pad_cols = f.shape[1] - remainder[1] - 1
    pad = ((0, 0),
           (pad_rows // 2, (pad_rows + 1) // 2),
           (pad_cols // 2, (pad_cols + 1) // 2),
           (0, 0))

    # Pad the input using numpy.pad().
    mode = None
    if padding == 'SAME':
      mode = str('constant')
    if padding == 'REFLECT':
      mode = str('reflect')
    if padding == 'SYMMETRIC':
      mode = str('symmetric')
    x = np.pad(x, pad, mode=mode)

  # Since x is now properly padded, proceed as if padding mode is VALID.
  x_window = np.empty(
      (x.shape[0],
       int(math.ceil((x.shape[1] - f.shape[0] + 1) / strides[1])),
       int(math.ceil((x.shape[2] - f.shape[1] + 1) / strides[2])),
       np.prod(f.shape[:3])))

  # The output at pixel location (i, j) is the result of linear transformation
  # applied to the window whose top-left corner is at
  # (i * row_stride, j * col_stride).
  for i in xrange(x_window.shape[1]):
    k = i * strides[1]
    for j in xrange(x_window.shape[2]):
      l = j * strides[2]
      x_window[:, i, j, :] = x[:,
                               k:(k + f.shape[0]),
                               l:(l + f.shape[1]),
                               :].reshape((x_window.shape[0], -1))

  y = np.tensordot(x_window, f.reshape((-1, f.shape[3])), axes=1)
  return y


class BlocksStdTest(tf.test.TestCase):

  def CheckUnary(self, y, op_type):
    self.assertEqual(op_type, y.op.type)
    self.assertEqual(1, len(y.op.inputs))
    return y.op.inputs[0]

  def CheckBinary(self, y, op_type):
    self.assertEqual(op_type, y.op.type)
    self.assertEqual(2, len(y.op.inputs))
    return y.op.inputs

  def testPassThrough(self):
    p = blocks_std.PassThrough()
    x = tf.placeholder(dtype=tf.float32, shape=[1])
    self.assertIs(p(x), x)

  def CheckBiasAdd(self, y, b):
    x, u = self.CheckBinary(y, 'BiasAdd')
    self.assertIs(u, b._bias.value())
    self.assertEqual(x.dtype, u.dtype.base_dtype)
    return x

  def testBiasAdd(self):
    b = blocks_std.BiasAdd()
    x = tf.placeholder(dtype=tf.float32, shape=[4, 8])
    y = b(x)
    self.assertEqual(b._bias.get_shape(), x.get_shape()[-1:])
    self.assertIs(x, self.CheckBiasAdd(y, b))

  def testBiasRankTest(self):
    b = blocks_std.BiasAdd()
    x = tf.placeholder(dtype=tf.float32, shape=[10])
    with self.assertRaises(ValueError):
      b(x)

  def CheckLinear(self, y, m):
    x, w = self.CheckBinary(y, 'MatMul')
    self.assertIs(w, m._matrix.value())
    self.assertEqual(x.dtype, w.dtype.base_dtype)
    return x

  def testLinear(self):
    m = blocks_std.Linear(10)
    x = tf.placeholder(dtype=tf.float32, shape=[8, 9])
    y = m(x)
    self.assertEqual(m._matrix.get_shape(), [9, 10])
    self.assertIs(x, self.CheckLinear(y, m))

  def testLinearShared(self):
    # Create a linear map which is applied twice on different inputs
    # (i.e. the weights of the map are shared).
    linear_map = blocks_std.Linear(6)
    x1 = tf.random_normal(shape=[1, 5])
    x2 = tf.random_normal(shape=[1, 5])
    xs = x1 + x2

    # Apply the transform with the same weights.
    y1 = linear_map(x1)
    y2 = linear_map(x2)
    ys = linear_map(xs)

    with self.test_session() as sess:
      # Initialize all the variables of the graph.
      tf.global_variables_initializer().run()

      y1_res, y2_res, ys_res = sess.run([y1, y2, ys])
      self.assertAllClose(y1_res + y2_res, ys_res)

  def CheckNN(self, y, nn, act=None):
    if act:
      pre_act = self.CheckUnary(y, act)
    else:
      pre_act = y

    if not isinstance(nn._bias, blocks_std.PassThrough):
      pre_bias = self.CheckBiasAdd(pre_act, nn._bias)
    else:
      pre_bias = pre_act

    if len(nn._matrices) > 1:
      self.assertEqual('AddN', pre_bias.op.type)
      pre_bias = pre_bias.op.inputs
    else:
      pre_bias = [pre_bias]

    self.assertEqual(len(pre_bias), len(nn._matrices))
    return [self.CheckLinear(u, m) for u, m in zip(pre_bias, nn._matrices)]

  def testNNWithoutActWithoutBias(self):
    nn = blocks_std.NN(10, act=None, bias=None)
    x = tf.placeholder(dtype=tf.float32, shape=[5, 7])
    y = nn(x)
    self.assertIs(x, self.CheckNN(y, nn)[0])

  def testNNWithoutBiasWithAct(self):
    nn = blocks_std.NN(10, act=tf.nn.relu, bias=None)
    x = tf.placeholder(dtype=tf.float32, shape=[5, 7])
    y = nn(x)
    self.assertIs(x, self.CheckNN(y, nn, 'Relu')[0])

  def testNNWithBiasWithoutAct(self):
    nn = blocks_std.NN(10, bias=blocks_std.Bias(0), act=None)
    x = tf.placeholder(dtype=tf.float32, shape=[5, 7])
    y = nn(x)
    self.assertIs(x, self.CheckNN(y, nn)[0])

  def testNNWithBiasWithAct(self):
    nn = blocks_std.NN(10, bias=blocks_std.Bias(0), act=tf.square)
    x = tf.placeholder(dtype=tf.float32, shape=[5, 7])
    y = nn(x)
    self.assertIs(x, self.CheckNN(y, nn, 'Square')[0])

  def testNNMultipleInputs(self):
    nn = blocks_std.NN(10, bias=blocks_std.Bias(0), act=tf.tanh)
    x = [tf.placeholder(dtype=tf.float32, shape=[5, 7]),
         tf.placeholder(dtype=tf.float32, shape=[5, 3]),
         tf.placeholder(dtype=tf.float32, shape=[5, 5])]
    y = nn(*x)
    xs = self.CheckNN(y, nn, 'Tanh')
    self.assertEqual(len(x), len(xs))
    for u, v in zip(x, xs):
      self.assertIs(u, v)

  def testConv2DSAME(self):
    np.random.seed(142536)

    x_shape = [4, 16, 11, 5]
    f_shape = [4, 3, 5, 6]
    strides = [1, 2, 2, 1]
    padding = 'SAME'

    conv = blocks_std.Conv2D(depth=f_shape[-1],
                             filter_size=f_shape[0:2],
                             strides=strides[1:3],
                             padding=padding,
                             act=None,
                             bias=None)
    x_value = np.random.normal(size=x_shape)
    x = tf.convert_to_tensor(x_value, dtype=tf.float32)
    y = conv(x)

    with self.test_session():
      tf.global_variables_initializer().run()
      f_value = conv._kernel.eval()
      y_value = y.eval()

    y_expected = _NumpyConv2D(x_value, f_value,
                              strides=strides, padding=padding)
    self.assertAllClose(y_expected, y_value)

  def testConv2DValid(self):
    np.random.seed(253647)

    x_shape = [4, 11, 12, 5]
    f_shape = [5, 2, 5, 5]
    strides = [1, 2, 2, 1]
    padding = 'VALID'

    conv = blocks_std.Conv2D(depth=f_shape[-1],
                             filter_size=f_shape[0:2],
                             strides=strides[1:3],
                             padding=padding,
                             act=None,
                             bias=None)
    x_value = np.random.normal(size=x_shape)
    x = tf.convert_to_tensor(x_value, dtype=tf.float32)
    y = conv(x)

    with self.test_session():
      tf.global_variables_initializer().run()
      f_value = conv._kernel.eval()
      y_value = y.eval()

    y_expected = _NumpyConv2D(x_value, f_value,
                              strides=strides, padding=padding)
    self.assertAllClose(y_expected, y_value)

  def testConv2DSymmetric(self):
    np.random.seed(364758)

    x_shape = [4, 10, 12, 6]
    f_shape = [3, 4, 6, 5]
    strides = [1, 1, 1, 1]
    padding = 'SYMMETRIC'

    conv = blocks_std.Conv2D(depth=f_shape[-1],
                             filter_size=f_shape[0:2],
                             strides=strides[1:3],
                             padding=padding,
                             act=None,
                             bias=None)
    x_value = np.random.normal(size=x_shape)
    x = tf.convert_to_tensor(x_value, dtype=tf.float32)
    y = conv(x)

    with self.test_session():
      tf.global_variables_initializer().run()
      f_value = conv._kernel.eval()
      y_value = y.eval()

    y_expected = _NumpyConv2D(x_value, f_value,
                              strides=strides, padding=padding)
    self.assertAllClose(y_expected, y_value)

  def testConv2DReflect(self):
    np.random.seed(768798)

    x_shape = [4, 10, 12, 6]
    f_shape = [3, 4, 6, 5]
    strides = [1, 2, 2, 1]
    padding = 'REFLECT'

    conv = blocks_std.Conv2D(depth=f_shape[-1],
                             filter_size=f_shape[0:2],
                             strides=strides[1:3],
                             padding=padding,
                             act=None,
                             bias=None)
    x_value = np.random.normal(size=x_shape)
    x = tf.convert_to_tensor(x_value, dtype=tf.float32)
    y = conv(x)

    with self.test_session():
      tf.global_variables_initializer().run()
      f_value = conv._kernel.eval()
      y_value = y.eval()

    y_expected = _NumpyConv2D(x_value, f_value,
                              strides=strides, padding=padding)
    self.assertAllClose(y_expected, y_value)

  def testConv2DBias(self):
    input_shape = [19, 14, 14, 64]
    filter_shape = [3, 7, 64, 128]
    strides = [1, 2, 2, 1]
    output_shape = [19, 6, 4, 128]

    conv = blocks_std.Conv2D(depth=filter_shape[-1],
                             filter_size=filter_shape[0:2],
                             strides=strides[1:3],
                             padding='VALID',
                             act=None,
                             bias=blocks_std.Bias(1))
    x = tf.placeholder(dtype=tf.float32, shape=input_shape)

    y = conv(x)
    self.CheckBiasAdd(y, conv._bias)
    self.assertEqual(output_shape, y.get_shape().as_list())


if __name__ == '__main__':
  tf.test.main()
