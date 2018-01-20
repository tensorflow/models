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

"""Tests of the 2D masked convolution blocks."""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
from six.moves import xrange
import tensorflow as tf

import blocks_masked_conv2d


class MaskedConv2DTest(tf.test.TestCase):

  def testRasterScanKernel(self):
    kernel_size = 5
    input_depth = 1
    output_depth = 1
    kernel_shape = [kernel_size, kernel_size, input_depth, output_depth]

    # pylint: disable=bad-whitespace
    kernel_feed = [[ 1.0,  2.0,  3.0,  4.0,  5.0],
                   [ 6.0,  7.0,  8.0,  9.0, 10.0],
                   [11.0, 12.0, 13.0, 14.0, 15.0],
                   [16.0, 17.0, 18.0, 19.0, 20.0],
                   [21.0, 22.0, 23.0, 24.0, 25.0]]
    kernel_feed = np.reshape(kernel_feed, kernel_shape)
    kernel_expected = [[ 1.0,  2.0, 3.0, 4.0,  5.0],
                       [ 6.0,  7.0, 8.0, 9.0, 10.0],
                       [11.0, 12.0, 0.0, 0.0,  0.0],
                       [ 0.0,  0.0, 0.0, 0.0,  0.0],
                       [ 0.0,  0.0, 0.0, 0.0,  0.0]]
    kernel_expected = np.reshape(kernel_expected, kernel_shape)
    # pylint: enable=bad-whitespace

    init_kernel = lambda s, t: tf.constant(kernel_feed, dtype=t, shape=s)
    masked_conv2d = blocks_masked_conv2d.RasterScanConv2D(
        output_depth, [kernel_size] * 2, [1] * 2, 'SAME',
        initializer=init_kernel)
    x = tf.placeholder(dtype=tf.float32, shape=[10] * 3 + [input_depth])
    _ = masked_conv2d(x)

    with self.test_session():
      tf.global_variables_initializer().run()
      kernel_value = masked_conv2d._kernel.eval()

    self.assertAllEqual(kernel_expected, kernel_value)

  def testDepthOrderKernel(self):
    kernel_size = 1
    input_depth = 7
    output_depth = input_depth
    kernel_shape = [kernel_size, kernel_size, input_depth, output_depth]

    kernel_feed = np.ones(kernel_shape)
    x_shape = [5] * 3 + [input_depth]
    x_feed = np.ones(x_shape)
    y_expected = np.zeros(x_shape[0:3] + [output_depth])
    y_expected[:, :, :] = np.arange(output_depth)

    init_kernel = lambda s, t: tf.constant(kernel_feed, dtype=t, shape=s)
    masked_conv2d = blocks_masked_conv2d.DepthOrderConv2D(
        output_depth, [kernel_size] * 2, [1] * 2, 'SAME',
        strict_order=True,
        initializer=init_kernel)
    x = tf.placeholder(dtype=tf.float32, shape=x_shape)
    y = masked_conv2d(x)

    with self.test_session():
      tf.global_variables_initializer().run()
      y_value = y.eval(feed_dict={x: x_feed})

    self.assertAllEqual(y_expected, y_value)

  def testGroupRasterScanKernel(self):
    kernel_size = 3
    input_depth = 4
    input_group_size = 2
    output_depth = 2
    output_group_size = 1
    kernel_shape = [kernel_size, kernel_size, input_depth, output_depth]
    kernel_feed = np.ones(shape=kernel_shape)

    height = 5
    width = 5
    x_shape = [1, height, width, input_depth]
    x_feed = np.ones(shape=x_shape)

    # pylint: disable=bad-whitespace
    y_expected = [
        [[ 0,  2], [ 4,  6], [ 4,  6], [ 4,  6], [ 4,  6]],
        [[ 8, 10], [16, 18], [16, 18], [16, 18], [12, 14]],
        [[ 8, 10], [16, 18], [16, 18], [16, 18], [12, 14]],
        [[ 8, 10], [16, 18], [16, 18], [16, 18], [12, 14]],
        [[ 8, 10], [16, 18], [16, 18], [16, 18], [12, 14]],
    ]
    y_expected = np.reshape(y_expected, [1, height, width, output_depth])
    # pylint: enable=bad-whitespace

    init_kernel = lambda s, t: tf.constant(kernel_feed, dtype=t, shape=s)
    masked_conv2d = blocks_masked_conv2d.GroupRasterScanConv2D(
        output_depth, [kernel_size] * 2, [1] * 2, 'SAME',
        strict_order=True,
        input_group_size=input_group_size,
        output_group_size=output_group_size,
        initializer=init_kernel)
    x = tf.placeholder(dtype=tf.float32, shape=x_shape)
    y = masked_conv2d(x)

    with self.test_session():
      tf.global_variables_initializer().run()
      y_value = y.eval(feed_dict={x: x_feed})

    self.assertAllEqual(y_expected, y_value)

  def testInFillingKernel(self):
    kernel_size = 5
    input_depth = 1
    output_depth = 1
    kernel_shape = [kernel_size, kernel_size, input_depth, output_depth]

    # pylint: disable=bad-whitespace
    kernel_feed = [[ 1.0,  2.0,  3.0,  4.0,  5.0],
                   [ 6.0,  7.0,  8.0,  9.0, 10.0],
                   [11.0, 12.0, 13.0, 14.0, 15.0],
                   [16.0, 17.0, 18.0, 19.0, 20.0],
                   [21.0, 22.0, 23.0, 24.0, 25.0]]
    kernel_feed = np.reshape(kernel_feed, kernel_shape)
    kernel_expected = [[ 1.0,  2.0,  3.0,  4.0,  5.0],
                       [ 6.0,  7.0,  8.0,  9.0, 10.0],
                       [11.0, 12.0,  0.0, 14.0, 15.0],
                       [16.0, 17.0, 18.0, 19.0, 20.0],
                       [21.0, 22.0, 23.0, 24.0, 25.0]]
    kernel_expected = np.reshape(kernel_expected, kernel_shape)
    # pylint: enable=bad-whitespace

    init_kernel = lambda s, t: tf.constant(kernel_feed, dtype=t, shape=s)
    masked_conv2d = blocks_masked_conv2d.InFillingConv2D(
        output_depth, [kernel_size] * 2, [1] * 2, 'SAME',
        initializer=init_kernel)
    x = tf.placeholder(dtype=tf.float32, shape=[10] * 3 + [input_depth])
    _ = masked_conv2d(x)

    with self.test_session():
      tf.global_variables_initializer().run()
      kernel_value = masked_conv2d._kernel.eval()

    self.assertAllEqual(kernel_expected, kernel_value)

  def testConv2DMaskedNumerics(self):
    kernel_size = 5
    input_shape = [1, 10, 10, 1]
    filter_shape = [kernel_size, kernel_size, 1, 1]
    strides = [1, 1, 1, 1]
    output_shape = [1, 10, 10, 1]

    conv = blocks_masked_conv2d.RasterScanConv2D(
        depth=filter_shape[-1],
        filter_size=filter_shape[0:2],
        strides=strides[1:3],
        padding='SAME',
        initializer=tf.constant_initializer(value=1.0))
    x = tf.placeholder(dtype=tf.float32, shape=input_shape)
    y = conv(x)

    x_feed = - np.ones(input_shape, dtype=float)
    y_expected = np.ones(output_shape, dtype=float)
    for i in xrange(input_shape[1]):
      for j in xrange(input_shape[2]):
        x_feed[0, i, j, 0] = 10 * (j + 1) + i
        v = 0
        ki_start = max(i - kernel_size // 2, 0)
        kj_start = max(j - kernel_size // 2, 0)
        kj_end = min(j + kernel_size // 2, input_shape[2] - 1)
        for ki in range(ki_start, i + 1):
          for kj in range(kj_start, kj_end + 1):
            if ki > i:
              continue
            if ki == i and kj >= j:
              continue
            v += 10 * (kj + 1) + ki
        y_expected[0, i, j, 0] = v

    with self.test_session():
      tf.global_variables_initializer().run()
      y_value = y.eval(feed_dict={x: x_feed})

    self.assertAllEqual(y_expected, y_value)


if __name__ == '__main__':
  tf.test.main()
