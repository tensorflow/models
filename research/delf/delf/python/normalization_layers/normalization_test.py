# Copyright 2021 The TensorFlow Authors All Rights Reserved.
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
"""Tests for normalization layers."""

import tensorflow as tf

from delf.python.normalization_layers import normalization


class NormalizationsTest(tf.test.TestCase):

  def testL2Normalization(self):
    x = tf.constant([-4.0, 0.0, 4.0])
    layer = normalization.L2Normalization()
    # Run tested function.
    result = layer(x, axis=0)
    # Define expected result.
    exp_output = [-0.70710677, 0.0, 0.70710677]
    # Compare actual and expected.
    self.assertAllClose(exp_output, result)


if __name__ == '__main__':
  tf.test.main()
