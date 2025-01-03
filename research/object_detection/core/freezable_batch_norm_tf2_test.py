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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from absl.testing import parameterized
import numpy as np
from six.moves import zip
import tensorflow as tf


from object_detection.core import freezable_batch_norm
from object_detection.utils import tf_version

# pylint: disable=g-import-not-at-top
if tf_version.is_tf2():
  from object_detection.core import freezable_sync_batch_norm
# pylint: enable=g-import-not-at-top


@unittest.skipIf(tf_version.is_tf1(), 'Skipping TF2.X only test.')
class FreezableBatchNormTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for FreezableBatchNorm operations."""

  def _build_model(self, use_sync_batch_norm, training=None):
    model = tf.keras.models.Sequential()
    norm = None
    if use_sync_batch_norm:
      norm = freezable_sync_batch_norm.FreezableSyncBatchNorm(training=training,
                                                              input_shape=(10,),
                                                              momentum=0.8)
    else:
      norm = freezable_batch_norm.FreezableBatchNorm(training=training,
                                                     input_shape=(10,),
                                                     momentum=0.8)

    model.add(norm)
    return model, norm

  def _copy_weights(self, source_weights, target_weights):
    for source, target in zip(source_weights, target_weights):
      target.assign(source)

  def _train_freezable_batch_norm(self, training_mean, training_var,
                                  use_sync_batch_norm):
    model, _ = self._build_model(use_sync_batch_norm=use_sync_batch_norm)
    model.compile(loss='mse', optimizer='sgd')

    # centered on training_mean, variance training_var
    train_data = np.random.normal(
        loc=training_mean,
        scale=training_var,
        size=(1000, 10))
    model.fit(train_data, train_data, epochs=4, verbose=0)
    return model.weights

  def _test_batchnorm_layer(
      self, norm, should_be_training, test_data,
      testing_mean, testing_var, training_arg, training_mean, training_var):
    out_tensor = norm(tf.convert_to_tensor(test_data, dtype=tf.float32),
                      training=training_arg)
    out = out_tensor
    out -= norm.beta
    out /= norm.gamma

    if not should_be_training:
      out *= training_var
      out += (training_mean - testing_mean)
      out /= testing_var

    np.testing.assert_allclose(out.numpy().mean(), 0.0, atol=1.5e-1)
    np.testing.assert_allclose(out.numpy().std(), 1.0, atol=1.5e-1)

  @parameterized.parameters(True, False)
  def test_batchnorm_freezing_training_none(self, use_sync_batch_norm):
    training_mean = 5.0
    training_var = 10.0

    testing_mean = -10.0
    testing_var = 5.0

    # Initially train the batch norm, and save the weights
    trained_weights = self._train_freezable_batch_norm(training_mean,
                                                       training_var,
                                                       use_sync_batch_norm)

    # Load the batch norm weights, freezing training to True.
    # Apply the batch norm layer to testing data and ensure it is normalized
    # according to the batch statistics.
    model, norm = self._build_model(use_sync_batch_norm, training=True)
    self._copy_weights(trained_weights, model.weights)

    # centered on testing_mean, variance testing_var
    test_data = np.random.normal(
        loc=testing_mean,
        scale=testing_var,
        size=(1000, 10))

    # Test with training=True passed to the call method:
    training_arg = True
    should_be_training = True
    self._test_batchnorm_layer(norm, should_be_training, test_data,
                               testing_mean, testing_var, training_arg,
                               training_mean, training_var)

    # Reset the weights, because they may have been updating by
    # running with training=True
    self._copy_weights(trained_weights, model.weights)

    # Test with training=False passed to the call method:
    training_arg = False
    should_be_training = False
    self._test_batchnorm_layer(norm, should_be_training, test_data,
                               testing_mean, testing_var, training_arg,
                               training_mean, training_var)

    # Test the layer in various Keras learning phase scopes:
    training_arg = None
    should_be_training = False
    self._test_batchnorm_layer(norm, should_be_training, test_data,
                               testing_mean, testing_var, training_arg,
                               training_mean, training_var)

    tf.keras.backend.set_learning_phase(True)
    should_be_training = True
    self._test_batchnorm_layer(norm, should_be_training, test_data,
                               testing_mean, testing_var, training_arg,
                               training_mean, training_var)

    # Reset the weights, because they may have been updating by
    # running with training=True
    self._copy_weights(trained_weights, model.weights)

    tf.keras.backend.set_learning_phase(False)
    should_be_training = False
    self._test_batchnorm_layer(norm, should_be_training, test_data,
                               testing_mean, testing_var, training_arg,
                               training_mean, training_var)

  @parameterized.parameters(True, False)
  def test_batchnorm_freezing_training_false(self, use_sync_batch_norm):
    training_mean = 5.0
    training_var = 10.0

    testing_mean = -10.0
    testing_var = 5.0

    # Initially train the batch norm, and save the weights
    trained_weights = self._train_freezable_batch_norm(training_mean,
                                                       training_var,
                                                       use_sync_batch_norm)

    # Load the batch norm back up, freezing training to False.
    # Apply the batch norm layer to testing data and ensure it is normalized
    # according to the training data's statistics.
    model, norm = self._build_model(use_sync_batch_norm, training=False)
    self._copy_weights(trained_weights, model.weights)

    # centered on testing_mean, variance testing_var
    test_data = np.random.normal(
        loc=testing_mean,
        scale=testing_var,
        size=(1000, 10))

    # Make sure that the layer is never training
    # Test with training=True passed to the call method:
    training_arg = True
    should_be_training = False
    self._test_batchnorm_layer(norm, should_be_training, test_data,
                               testing_mean, testing_var, training_arg,
                               training_mean, training_var)

    # Test with training=False passed to the call method:
    training_arg = False
    should_be_training = False
    self._test_batchnorm_layer(norm, should_be_training, test_data,
                               testing_mean, testing_var, training_arg,
                               training_mean, training_var)

    # Test the layer in various Keras learning phase scopes:
    training_arg = None
    should_be_training = False
    self._test_batchnorm_layer(norm, should_be_training, test_data,
                               testing_mean, testing_var, training_arg,
                               training_mean, training_var)

    tf.keras.backend.set_learning_phase(True)
    should_be_training = False
    self._test_batchnorm_layer(norm, should_be_training, test_data,
                               testing_mean, testing_var, training_arg,
                               training_mean, training_var)

    tf.keras.backend.set_learning_phase(False)
    should_be_training = False
    self._test_batchnorm_layer(norm, should_be_training, test_data,
                               testing_mean, testing_var, training_arg,
                               training_mean, training_var)


if __name__ == '__main__':
  tf.test.main()
