# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Test Transformer model helper methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from official.nlp.transformer import model_utils

NEG_INF = -1e9


class ModelUtilsTest(tf.test.TestCase):

  def test_get_padding(self):
    x = tf.constant([[1, 0, 0, 0, 2], [3, 4, 0, 0, 0], [0, 5, 6, 0, 7]])
    padding = model_utils.get_padding(x, padding_value=0)

    self.assertAllEqual([[0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [1, 0, 0, 1, 0]],
                        padding)

  def test_get_padding_bias(self):
    x = tf.constant([[1, 0, 0, 0, 2], [3, 4, 0, 0, 0], [0, 5, 6, 0, 7]])
    bias = model_utils.get_padding_bias(x)
    bias_shape = tf.shape(bias)
    flattened_bias = tf.reshape(bias, [3, 5])

    self.assertAllEqual([[0, NEG_INF, NEG_INF, NEG_INF, 0],
                         [0, 0, NEG_INF, NEG_INF, NEG_INF],
                         [NEG_INF, 0, 0, NEG_INF, 0]],
                        flattened_bias)
    self.assertAllEqual([3, 1, 1, 5], bias_shape)

  def test_get_decoder_self_attention_bias(self):
    length = 5
    bias = model_utils.get_decoder_self_attention_bias(length)

    self.assertAllEqual([[[[0, NEG_INF, NEG_INF, NEG_INF, NEG_INF],
                           [0, 0, NEG_INF, NEG_INF, NEG_INF],
                           [0, 0, 0, NEG_INF, NEG_INF],
                           [0, 0, 0, 0, NEG_INF],
                           [0, 0, 0, 0, 0]]]],
                        bias)


if __name__ == "__main__":
  tf.test.main()
