# Copyright 2017 Google Inc. All Rights Reserved.
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

"""Tests for dragnn.python.transformer_units."""


import numpy as np
import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

from dragnn.python import transformer_units


class TransformerTest(test_util.TensorFlowTestCase):

  def testComputePadding(self):
    with tf.Graph().as_default(), self.test_session() as session:
      lengths = [5, 1, 2, 0]
      expected = [[[[0, 0, 0, 0, 0]]],
                  [[[0, -1e9, -1e9, -1e9, -1e9]]],
                  [[[0, 0, -1e9, -1e9, -1e9]]],
                  [[[-1e9, -1e9, -1e9, -1e9, -1e9]]]]
      tensor = transformer_units.compute_padding_mask(lengths)
      session.run(tf.global_variables_initializer())
      actual = session.run(tensor)
      self.assertAllEqual(actual, expected)

  def testDotProductAttention(self):
    with tf.Graph().as_default(), self.test_session() as session:
      padding = [[[[0, 0, 0, 0, 0]]],
                 [[[0, -1e9, -1e9, -1e9, -1e9]]]]
      # batch x heads x length x d
      np.random.seed(4)
      q = np.random.random((2, 2, 5, 2)).astype(np.float32)
      k = np.random.random((2, 2, 5, 2)).astype(np.float32)
      v = np.random.random((2, 2, 5, 2)).astype(np.float32)

      # Should have shape: 2x2x5x5. Computed as follows:
      # r = np.einsum('hijk,hilk->hijl', q, k) + padding_bias
      # r = r - np.expand_dims(np.max(r, axis=-1), -1)
      # r = np.exp(r)
      # ax_sum = np.expand_dims(np.sum(r, axis=-1), -1)
      # r = r / ax_sum
      # for i in range(2):
      #   for j in range(2):
      #     np.dot(r[i,j], v[i,j])
      expected = [[[[0.46580601, 0.64643575],
                    [0.46182397, 0.64578158],
                    [0.46866544, 0.64562998],
                    [0.47930001, 0.64838011],
                    [0.45466267, 0.64061598]],
                   [[0.50887558, 0.39900422],
                    [0.51721343, 0.39245871],
                    [0.50348963, 0.40090425],
                    [0.49889359, 0.4035989],
                    [0.50523872, 0.39916877]]],
                  [[[0.26092216, 0.41247222],
                    [0.26092216, 0.41247222],
                    [0.26092216, 0.41247222],
                    [0.26092216, 0.41247222],
                    [0.26092216, 0.41247222]],
                   [[0.34745133, 0.05888009],
                    [0.34745133, 0.05888009],
                    [0.34745133, 0.05888009],
                    [0.34745133, 0.05888009],
                    [0.34745133, 0.05888009]]]]

      tensor = transformer_units.dot_product_attention(q, k, v, 1.0, padding)
      session.run(tf.global_variables_initializer())
      actual = session.run(tensor)

      self.assertAllClose(actual, expected, 1e-6, 1e-6)


if __name__ == '__main__':
  googletest.main()
