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

"""Tests for ExpandCondense tensor network layer."""

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras
from official.nlp.modeling.layers.tn_expand_condense import TNExpandCondense


class TNLayerTest(tf.test.TestCase, parameterized.TestCase):
  """Unit tests for ExpandCondense TN layer.
  """

  def setUp(self):
    super().setUp()
    self.labels = np.concatenate((np.ones((50, 1)), np.zeros((50, 1))), axis=0)

  def _build_model(self, data, proj_multiple=2):
    model = tf_keras.models.Sequential()
    model.add(
        TNExpandCondense(
            proj_multiplier=proj_multiple,
            use_bias=True,
            activation='relu',
            input_shape=(data.shape[-1],)))
    model.add(tf_keras.layers.Dense(1, activation='sigmoid'))
    return model

  @parameterized.parameters((768, 6), (1024, 2))
  def test_train(self, input_dim, proj_multiple):
    tf_keras.utils.set_random_seed(0)
    data = np.random.randint(10, size=(100, input_dim))
    model = self._build_model(data, proj_multiple)

    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model for 5 epochs
    history = model.fit(data, self.labels, epochs=5, batch_size=32)

    # Check that loss decreases and accuracy increases
    self.assertGreater(history.history['loss'][0], history.history['loss'][-1])
    self.assertLess(
        history.history['accuracy'][0], history.history['accuracy'][-1])

  @parameterized.parameters((768, 6), (1024, 2))
  def test_weights_change(self, input_dim, proj_multiple):
    tf_keras.utils.set_random_seed(0)
    data = np.random.randint(10, size=(100, input_dim))
    model = self._build_model(data, proj_multiple)
    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    before = model.get_weights()

    model.fit(data, self.labels, epochs=5, batch_size=32)

    after = model.get_weights()
    # Make sure every layer's weights changed
    for i, _ in enumerate(before):
      self.assertTrue((after[i] != before[i]).any())

  @parameterized.parameters((768, 6), (1024, 2))
  def test_output_shape(self, input_dim, proj_multiple):
    data = np.random.randint(10, size=(100, input_dim))
    model = self._build_model(data, proj_multiple)
    input_shape = data.shape

    actual_output_shape = model(data).shape
    expected_output_shape = model.compute_output_shape(input_shape)

    self.assertEqual(expected_output_shape, actual_output_shape)

  @parameterized.parameters((768, 6), (1024, 2))
  def test_expandcondense_num_parameters(self, input_dim, proj_multiple):
    data = np.random.randint(10, size=(100, input_dim))
    proj_size = proj_multiple * data.shape[-1]
    model = tf_keras.models.Sequential()
    model.add(
        TNExpandCondense(
            proj_multiplier=proj_multiple,
            use_bias=True,
            activation='relu',
            input_shape=(data.shape[-1],)))

    w1_params = data.shape[-1]**2
    w2_params = 128 * 128 * (proj_size // data.shape[-1])
    w3_params = 128 * 128 * (proj_size // data.shape[-1])
    w4_params = (data.shape[-1] // 128) * 128 * data.shape[-1]
    bias_params = ((data.shape[-1] // 128) * 128 *
                   (proj_size // data.shape[-1]))

    expected_num_parameters = (w1_params + w2_params + w3_params +
                               w4_params) + bias_params

    self.assertEqual(expected_num_parameters, model.count_params())

  @parameterized.parameters((912, 6), (200, 2))
  def test_incorrect_sizes(self, input_dim, proj_multiple):
    data = np.random.randint(10, size=(100, input_dim))

    with self.assertRaises(AssertionError):
      model = self._build_model(data, proj_multiple)
      model.compile(optimizer='adam', loss='binary_crossentropy')

  @parameterized.parameters((768, 6), (1024, 2))
  def test_config(self, input_dim, proj_multiple):
    data = np.random.randint(10, size=(100, input_dim))
    model = self._build_model(data, proj_multiple)

    expected_num_parameters = model.layers[0].count_params()

    # Serialize model and use config to create new layer
    model_config = model.get_config()
    layer_config = model_config['layers'][1]['config']

    new_model = TNExpandCondense.from_config(layer_config)

    # Build the layer so we can count params below
    new_model.build(layer_config['batch_input_shape'])

    # Check that original layer had same num params as layer built from config
    self.assertEqual(expected_num_parameters, new_model.count_params())

  @parameterized.parameters((768, 6), (1024, 2))
  def test_model_save(self, input_dim, proj_multiple):
    data = np.random.randint(10, size=(100, input_dim))
    model = self._build_model(data, proj_multiple)

    model.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model for 5 epochs
    model.fit(data, self.labels, epochs=5, batch_size=32)

    save_path = os.path.join(self.get_temp_dir(), 'test_model')
    model.save(save_path)
    loaded_model = tf_keras.models.load_model(save_path)

    # Compare model predictions and loaded_model predictions
    self.assertAllEqual(model.predict(data), loaded_model.predict(data))

if __name__ == '__main__':
  tf.test.main()
