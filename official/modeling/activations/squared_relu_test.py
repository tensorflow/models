# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for the customized Squared ReLU activation."""

import numpy as np
import tensorflow as tf, tf_keras

from official.modeling import activations


class CustomizedSquaredReluTest(tf.test.TestCase):

  def _squared_relu_nn(self, x):
    x = np.float32(x)
    return tf.math.square(tf.nn.relu(x))

  def test_squared_relu(self):
    features = [[0.25, 0, -0.25], [-1, -2, 3]]
    customized_squared_relu_data = activations.squared_relu(features)
    squared_relu_data = self._squared_relu_nn(features)
    self.assertAllClose(customized_squared_relu_data, squared_relu_data)


if __name__ == '__main__':
  tf.test.main()
