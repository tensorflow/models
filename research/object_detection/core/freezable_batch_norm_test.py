# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for object_detection.core.freezable_batch_norm."""
import numpy as np
import tensorflow as tf

from object_detection.core import freezable_batch_norm


class FreezableBatchNormTest(tf.test.TestCase):
  """Tests for FreezableBatchNorm operations."""

  def _build_model(self, training=None):
    model = tf.keras.models.Sequential()
    norm = freezable_batch_norm.FreezableBatchNorm(training=training,
                                                   input_shape=(10,),
                                                   momentum=0.8)
    model.add(norm)
    return model, norm

  def _train_freezable_batch_norm(self, training_mean, training_var):
    model, _ = self._build_model()
    model.compile(loss='mse', optimizer='sgd')

    # centered on training_mean, variance training_var
    train_data = np.random.normal(
        loc=training_mean,
        scale=training_var,
        size=(1000, 10))
    model.fit(train_data, train_data, epochs=4, verbose=0)
    return model.weights

  def test_batchnorm_freezing_training_true(self):
    with self.test_session():
      training_mean = 5.0
      training_var = 10.0

      testing_mean = -10.0
      testing_var = 5.0

      # Initially train the batch norm, and save the weights
      trained_weights = self._train_freezable_batch_norm(training_mean,
                                                         training_var)

      # Load the batch norm weights, freezing training to True.
      # Apply the batch norm layer to testing data and ensure it is normalized
      # according to the batch statistics.
      model, norm = self._build_model(training=True)
      for trained_weight, blank_weight in zip(trained_weights, model.weights):
        weight_copy = blank_weight.assign(tf.keras.backend.eval(trained_weight))
        tf.keras.backend.eval(weight_copy)

      # centered on testing_mean, variance testing_var
      test_data = np.random.normal(
          loc=testing_mean,
          scale=testing_var,
          size=(1000, 10))

      out_tensor = norm(tf.convert_to_tensor(test_data, dtype=tf.float32))
      out = tf.keras.backend.eval(out_tensor)

      out -= tf.keras.backend.eval(norm.beta)
      out /= tf.keras.backend.eval(norm.gamma)

      np.testing.assert_allclose(out.mean(), 0.0, atol=1.5e-1)
      np.testing.assert_allclose(out.std(), 1.0, atol=1.5e-1)

  def test_batchnorm_freezing_training_false(self):
    with self.test_session():
      training_mean = 5.0
      training_var = 10.0

      testing_mean = -10.0
      testing_var = 5.0

      # Initially train the batch norm, and save the weights
      trained_weights = self._train_freezable_batch_norm(training_mean,
                                                         training_var)

      # Load the batch norm back up, freezing training to False.
      # Apply the batch norm layer to testing data and ensure it is normalized
      # according to the training data's statistics.
      model, norm = self._build_model(training=False)
      for trained_weight, blank_weight in zip(trained_weights, model.weights):
        weight_copy = blank_weight.assign(tf.keras.backend.eval(trained_weight))
        tf.keras.backend.eval(weight_copy)

      # centered on testing_mean, variance testing_var
      test_data = np.random.normal(
          loc=testing_mean,
          scale=testing_var,
          size=(1000, 10))

      out_tensor = norm(tf.convert_to_tensor(test_data, dtype=tf.float32))
      out = tf.keras.backend.eval(out_tensor)

      out -= tf.keras.backend.eval(norm.beta)
      out /= tf.keras.backend.eval(norm.gamma)

      out *= training_var
      out += (training_mean - testing_mean)
      out /= testing_var

      np.testing.assert_allclose(out.mean(), 0.0, atol=1.5e-1)
      np.testing.assert_allclose(out.std(), 1.0, atol=1.5e-1)

if __name__ == '__main__':
  tf.test.main()
