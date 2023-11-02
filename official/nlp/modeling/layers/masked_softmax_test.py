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

"""Tests for Keras-based masked softmax layer."""

import numpy as np
import tensorflow as tf, tf_keras

from official.nlp.modeling.layers import masked_softmax


class MaskedSoftmaxLayerTest(tf.test.TestCase):

  def test_non_masked_softmax(self):
    test_layer = masked_softmax.MaskedSoftmax()
    input_tensor = tf_keras.Input(shape=(4, 8))
    output = test_layer(input_tensor)
    model = tf_keras.Model(input_tensor, output)

    input_data = 10 * np.random.random_sample((3, 4, 8))
    output_data = model.predict(input_data)
    expected_data = tf.nn.softmax(input_data)
    self.assertAllClose(expected_data, output_data)

  def test_masked_softmax(self):
    test_layer = masked_softmax.MaskedSoftmax()
    input_tensor = tf_keras.Input(shape=(4, 8))
    mask_tensor = tf_keras.Input(shape=(4, 8))
    output = test_layer(input_tensor, mask_tensor)
    model = tf_keras.Model([input_tensor, mask_tensor], output)

    input_data = 10 * np.random.random_sample((3, 4, 8))
    mask_data = np.random.randint(2, size=(3, 4, 8))

    output_data = model.predict([input_data, mask_data])
    expected_zeros = np.greater(mask_data, 0)
    is_zeros = np.greater(output_data, 0)
    self.assertAllEqual(expected_zeros, is_zeros)

  def test_masked_softmax_with_none_mask(self):
    test_layer = masked_softmax.MaskedSoftmax()
    input_tensor = tf_keras.Input(shape=(4, 8))
    output = test_layer(input_tensor, None)
    model = tf_keras.Model(input_tensor, output)

    input_data = 10 * np.random.random_sample((3, 4, 8))
    output_data = model.predict(input_data)
    expected_data = tf.nn.softmax(input_data)
    self.assertAllClose(expected_data, output_data)

  def test_softmax_with_axes_expansion(self):
    test_layer = masked_softmax.MaskedSoftmax(mask_expansion_axes=[1])
    input_tensor = tf_keras.Input(shape=(4, 8))
    mask_tensor = tf_keras.Input(shape=(8))
    output = test_layer(input_tensor, mask_tensor)
    model = tf_keras.Model([input_tensor, mask_tensor], output)

    input_data = 10 * np.random.random_sample((3, 4, 8))
    mask_data = np.random.randint(2, size=(3, 8))

    output_data = model.predict([input_data, mask_data])
    expanded_mask = np.expand_dims(mask_data, axis=1) * np.ones_like(input_data)
    expected_zeros = np.greater(expanded_mask, 0)
    is_zeros = np.greater(output_data, 0)
    self.assertAllEqual(expected_zeros, is_zeros)

  def test_masked_softmax_high_dims(self):
    test_layer = masked_softmax.MaskedSoftmax(
        mask_expansion_axes=[1], normalization_axes=[6, 7])
    input_shape = [2, 3, 4, 5, 6, 7, 8]
    mask_shape = [5, 6, 7, 8]
    input_tensor = tf_keras.Input(shape=input_shape)
    mask_tensor = tf_keras.Input(shape=mask_shape)
    output = test_layer(input_tensor, mask_tensor)
    model = tf_keras.Model([input_tensor, mask_tensor], output)

    input_data = 10 * np.random.random_sample([3] + input_shape)
    mask_data = np.random.randint(2, size=[3] + mask_shape)

    output_data = model.predict([input_data, mask_data])
    expanded_mask = np.expand_dims(mask_data, axis=1)
    expanded_mask = np.expand_dims(expanded_mask, axis=1)
    expanded_mask = np.expand_dims(
        expanded_mask, axis=1) * np.ones_like(input_data)
    expected_zeros = np.greater(expanded_mask, 0)
    is_zeros = np.greater(output_data, 0)
    self.assertAllEqual(expected_zeros, is_zeros)

  def test_serialize_deserialize(self):
    test_layer = masked_softmax.MaskedSoftmax(
        mask_expansion_axes=[1], normalization_axes=[6, 7])
    new_layer = masked_softmax.MaskedSoftmax.from_config(
        test_layer.get_config())

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(test_layer.get_config(), new_layer.get_config())


if __name__ == '__main__':
  tf.test.main()
