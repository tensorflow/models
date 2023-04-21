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

"""Tests for optimizer_factory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import tensorflow as tf
from official.legacy.image_classification import optimizer_factory
from official.legacy.image_classification.configs import base_configs


class OptimizerFactoryTest(tf.test.TestCase, parameterized.TestCase):

  def build_toy_model(self) -> tf.keras.Model:
    """Creates a toy `tf.Keras.Model`."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
    return model

  @parameterized.named_parameters(
      ('sgd', 'sgd', 0., False), ('momentum', 'momentum', 0., False),
      ('rmsprop', 'rmsprop', 0., False), ('adam', 'adam', 0., False),
      ('adamw', 'adamw', 0., False),
      ('momentum_lookahead', 'momentum', 0., True),
      ('sgd_ema', 'sgd', 0.999, False),
      ('momentum_ema', 'momentum', 0.999, False),
      ('rmsprop_ema', 'rmsprop', 0.999, False))
  def test_optimizer(self, optimizer_name, moving_average_decay, lookahead):
    """Smoke test to be sure no syntax errors."""
    model = self.build_toy_model()
    params = {
        'learning_rate': 0.001,
        'rho': 0.09,
        'momentum': 0.,
        'epsilon': 1e-07,
        'moving_average_decay': moving_average_decay,
        'lookahead': lookahead,
    }
    optimizer = optimizer_factory.build_optimizer(
        optimizer_name=optimizer_name,
        base_learning_rate=params['learning_rate'],
        params=params,
        model=model)
    self.assertTrue(
        issubclass(type(optimizer), tf.keras.optimizers.legacy.Optimizer)
    )

  def test_unknown_optimizer(self):
    with self.assertRaises(ValueError):
      optimizer_factory.build_optimizer(
          optimizer_name='this_optimizer_does_not_exist',
          base_learning_rate=None,
          params=None)

  def test_learning_rate_without_decay_or_warmups(self):
    params = base_configs.LearningRateConfig(
        name='exponential',
        initial_lr=0.01,
        decay_rate=0.01,
        decay_epochs=None,
        warmup_epochs=None,
        scale_by_batch_size=0.01,
        examples_per_epoch=1,
        boundaries=[0],
        multipliers=[0, 1])
    batch_size = 1
    train_steps = 1

    lr = optimizer_factory.build_learning_rate(
        params=params, batch_size=batch_size, train_steps=train_steps)
    self.assertTrue(
        issubclass(
            type(lr), tf.keras.optimizers.schedules.LearningRateSchedule))

  @parameterized.named_parameters(('exponential', 'exponential'),
                                  ('cosine_with_warmup', 'cosine_with_warmup'))
  def test_learning_rate_with_decay_and_warmup(self, lr_decay_type):
    """Basic smoke test for syntax."""
    params = base_configs.LearningRateConfig(
        name=lr_decay_type,
        initial_lr=0.01,
        decay_rate=0.01,
        decay_epochs=1,
        warmup_epochs=1,
        scale_by_batch_size=0.01,
        examples_per_epoch=1,
        boundaries=[0],
        multipliers=[0, 1])
    batch_size = 1
    train_epochs = 1
    train_steps = 1

    lr = optimizer_factory.build_learning_rate(
        params=params,
        batch_size=batch_size,
        train_epochs=train_epochs,
        train_steps=train_steps)
    self.assertTrue(
        issubclass(
            type(lr), tf.keras.optimizers.schedules.LearningRateSchedule))


if __name__ == '__main__':
  tf.test.main()
