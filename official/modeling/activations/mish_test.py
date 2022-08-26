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

"""Tests for the customized Mish activation."""

import tensorflow as tf

from tensorflow.python.keras import keras_parameterized  # pylint: disable=g-direct-tensorflow-import
from official.modeling import activations


@keras_parameterized.run_all_keras_modes
class MishTest(keras_parameterized.TestCase):

  def test_mish(self):
    x = tf.constant([1.0, 0.0])
    self.assertAllClose([0.86509839, 0.0], activations.mish(x))


if __name__ == '__main__':
  tf.test.main()
