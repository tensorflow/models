# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for TFM actions."""

import os

from absl.testing import parameterized
import numpy as np
import orbit
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.core import actions
from official.modeling import optimization


class TestModel(tf.Module):

  def __init__(self):
    self.value = tf.Variable(0)

  @tf.function(input_signature=[])
  def __call__(self):
    return self.value


class ActionsTest(tf.test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.cloud_tpu_strategy,
              strategy_combinations.one_device_strategy_gpu,
          ],))
  def test_ema_checkpointing(self, distribution):
    with distribution.scope():
      directory = self.create_tempdir()
      model = TestModel()
      optimizer = tf.keras.optimizers.SGD()
      optimizer = optimization.ExponentialMovingAverage(
          optimizer, trainable_weights_only=False)

      # Creats average weights for the model variables. Average weights are
      # initialized to zero.
      optimizer.shadow_copy(model)
      checkpoint = tf.train.Checkpoint(model=model)

      # Changes model.value to 3, average value is still 0.
      model.value.assign(3)

      # Checks model.value is 3
      self.assertEqual(model(), 3)
      ema_action = actions.EMACheckpointing(directory, optimizer, checkpoint)

      ema_action({})
      self.assertNotEmpty(
          tf.io.gfile.glob(os.path.join(directory, 'ema_checkpoints')))

      checkpoint.read(tf.train.latest_checkpoint(
          os.path.join(directory, 'ema_checkpoints')))

      # Checks model.value is 0 after swapping.
      self.assertEqual(model(), 0)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.default_strategy,
              strategy_combinations.cloud_tpu_strategy,
              strategy_combinations.one_device_strategy_gpu,
          ],))
  def test_recovery_condition(self, distribution):
    with distribution.scope():
      global_step = orbit.utils.create_global_step()
      recover_condition = actions.RecoveryCondition(
          global_step, loss_upper_bound=0.5, recovery_max_trials=2)
      outputs = {'training_loss': 0.6}
      self.assertTrue(recover_condition(outputs))
      self.assertTrue(recover_condition(outputs))
      with self.assertRaises(RuntimeError):
        recover_condition(outputs)

      global_step = orbit.utils.create_global_step()
      recover_condition = actions.RecoveryCondition(
          global_step, loss_upper_bound=0.5, recovery_max_trials=2)
      outputs = {'training_loss': tf.constant([np.nan], tf.float32)}
      self.assertTrue(recover_condition(outputs))
      self.assertTrue(recover_condition(outputs))
      with self.assertRaises(RuntimeError):
        recover_condition(outputs)


if __name__ == '__main__':
  tf.test.main()
