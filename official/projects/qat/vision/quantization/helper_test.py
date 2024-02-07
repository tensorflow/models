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

"""Tests for helper."""
import numpy as np
import tensorflow as tf, tf_keras

import tensorflow_model_optimization as tfmot
from official.projects.qat.vision.quantization import helper


class HelperTest(tf.test.TestCase):

  def create_simple_model(self):
    return tf_keras.models.Sequential([
        tf_keras.layers.Dense(8, input_shape=(16,)),
    ])

  def test_copy_original_weights_for_simple_model_with_custom_weights(self):
    one_model = self.create_simple_model()
    one_weights = [np.ones_like(weight) for weight in one_model.get_weights()]
    one_model.set_weights(one_weights)

    qat_model = tfmot.quantization.keras.quantize_model(
        self.create_simple_model())
    zero_weights = [np.zeros_like(weight) for weight in qat_model.get_weights()]
    qat_model.set_weights(zero_weights)

    helper.copy_original_weights(one_model, qat_model)

    qat_model_weights = qat_model.get_weights()
    count = 0
    for idx, weight in enumerate(qat_model.weights):
      if not helper.is_quantization_weight_name(weight.name):
        self.assertAllEqual(
            qat_model_weights[idx], np.ones_like(qat_model_weights[idx]))
        count += 1
    self.assertLen(one_model.weights, count)
    self.assertGreater(len(qat_model.weights), len(one_model.weights))


if __name__ == '__main__':
  tf.test.main()
