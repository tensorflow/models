# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the customized Swish activation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.modeling import activations


@keras_parameterized.run_all_keras_modes
class CustomizedSwishTest(keras_parameterized.TestCase):

  def test_gelu(self):
    customized_swish_data = activations.swish([[.25, 0, -.25], [-1, -2, 3]])
    swish_data = tf.nn.swish([[.25, 0, -.25], [-1, -2, 3]])
    self.assertAllClose(customized_swish_data, swish_data)


if __name__ == '__main__':
  tf.test.main()
