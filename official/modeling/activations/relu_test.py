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

"""Tests for the customized Relu activation."""

import tensorflow as tf

from tensorflow.python.keras import \
  keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.modeling import activations


@keras_parameterized.run_all_keras_modes
class CustomizedReluTest(keras_parameterized.TestCase):

  def test_relu6(self):
    features = [[.25, 0, -.25], [-1, -2, 3]]
    customized_relu6_data = activations.relu6(features)
    relu6_data = tf.nn.relu6(features)
    self.assertAllClose(customized_relu6_data, relu6_data)


if __name__ == '__main__':
  tf.test.main()
