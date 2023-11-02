# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for the Gaussian error linear unit."""

import tensorflow as tf, tf_keras

from official.modeling import activations


class GeluTest(tf.test.TestCase):

  def test_gelu(self):
    expected_data = [[0.14967535, 0., -0.10032465],
                     [-0.15880796, -0.04540223, 2.9963627]]
    gelu_data = activations.gelu([[.25, 0, -.25], [-1, -2, 3]])
    self.assertAllClose(expected_data, gelu_data)


if __name__ == '__main__':
  tf.test.main()
