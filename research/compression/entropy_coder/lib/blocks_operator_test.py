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

"""Tests of the block operators."""

import numpy as np
import tensorflow as tf

import block_base
import blocks_operator


class AddOneBlock(block_base.BlockBase):

  def __init__(self, name=None):
    super(AddOneBlock, self).__init__(name)

  def _Apply(self, x):
    return x + 1.0


class SquareBlock(block_base.BlockBase):

  def __init__(self, name=None):
    super(SquareBlock, self).__init__(name)

  def _Apply(self, x):
    return x * x


class BlocksOperatorTest(tf.test.TestCase):

  def testComposition(self):
    x_value = np.array([[1.0, 2.0, 3.0],
                        [-1.0, -2.0, -3.0]])
    y_expected_value = np.array([[4.0, 9.0, 16.0],
                                 [0.0, 1.0, 4.0]])

    x = tf.placeholder(dtype=tf.float32, shape=[2, 3])
    complex_block = blocks_operator.CompositionOperator(
        [AddOneBlock(),
         SquareBlock()])
    y = complex_block(x)

    with self.test_session():
      y_value = y.eval(feed_dict={x: x_value})

    self.assertAllClose(y_expected_value, y_value)


if __name__ == '__main__':
  tf.test.main()
