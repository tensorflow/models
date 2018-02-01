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

"""Tests for basic tensorflow blocks_entropy_coding."""

from __future__ import division
from __future__ import unicode_literals

import math

import numpy as np
import tensorflow as tf

import blocks_entropy_coding


class BlocksEntropyCodingTest(tf.test.TestCase):

  def testCodeLength(self):
    shape = [2, 4]
    proba_feed = [[0.65, 0.25, 0.70, 0.10],
                  [0.28, 0.20, 0.44, 0.54]]
    symbol_feed = [[1.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]]
    mean_code_length = - (
        (math.log(0.65) + math.log(0.75) + math.log(0.70) + math.log(0.90) +
         math.log(0.72) + math.log(0.80) + math.log(0.56) + math.log(0.54)) /
        math.log(2.0)) / (shape[0] * shape[1])

    symbol = tf.placeholder(dtype=tf.float32, shape=shape)
    proba = tf.placeholder(dtype=tf.float32, shape=shape)
    code_length_calculator = blocks_entropy_coding.CodeLength()
    code_length = code_length_calculator(symbol, proba)

    with self.test_session():
      tf.global_variables_initializer().run()
      code_length_eval = code_length.eval(
          feed_dict={symbol: symbol_feed, proba: proba_feed})

    self.assertAllClose(mean_code_length, code_length_eval)


if __name__ == '__main__':
  tf.test.main()
