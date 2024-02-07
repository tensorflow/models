# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for the customized Sigmoid activation."""

import numpy as np
import tensorflow as tf, tf_keras

from official.modeling import activations


class CustomizedSigmoidTest(tf.test.TestCase):

  def _hard_sigmoid_nn(self, x):
    x = np.float32(x)
    return tf.nn.relu6(x + 3.) * 0.16667

  def test_hard_sigmoid(self):
    features = [[.25, 0, -.25], [-1, -2, 3]]
    customized_hard_sigmoid_data = activations.hard_sigmoid(features)
    sigmoid_data = self._hard_sigmoid_nn(features)
    self.assertAllClose(customized_hard_sigmoid_data, sigmoid_data)


if __name__ == '__main__':
  tf.test.main()
