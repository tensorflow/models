# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for the customized Swish activation."""
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.modeling import activations


@keras_parameterized.run_all_keras_modes
class CustomizedSwishTest(keras_parameterized.TestCase):

  def _hard_swish_np(self, x):
    x = np.float32(x)
    return x * np.clip(x + 3, 0, 6) / 6

  def test_simple_swish(self):
    features = [[.25, 0, -.25], [-1, -2, 3]]
    customized_swish_data = activations.simple_swish(features)
    swish_data = tf.nn.swish(features)
    self.assertAllClose(customized_swish_data, swish_data)

  def test_hard_swish(self):
    features = [[.25, 0, -.25], [-1, -2, 3]]
    customized_swish_data = activations.hard_swish(features)
    swish_data = self._hard_swish_np(features)
    self.assertAllClose(customized_swish_data, swish_data)


if __name__ == '__main__':
  tf.test.main()
