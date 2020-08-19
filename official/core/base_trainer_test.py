# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow_models.core.trainers.trainer."""
# pylint: disable=g-direct-tensorflow-import

import os
from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.core import base_trainer as trainer_lib
from official.core import train_lib
from official.modeling.hyperparams import config_definitions as cfg
from official.utils.testing import mock_task


def all_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.tpu_strategy,
          strategy_combinations.one_device_strategy_gpu,
      ],
      mode='eager',
  )


class TrainerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._config = cfg.ExperimentConfig(
        trainer=cfg.TrainerConfig(
            optimizer_config=cfg.OptimizationConfig({
                'optimizer': {
                    'type': 'sgd'
                },
                'learning_rate': {
                    'type': 'constant'
                }
            })))

  def create_test_trainer(self):
    task = mock_task.MockTask()
    trainer = trainer_lib.Trainer(self._config, task)
    return trainer

  @combinations.generate(all_strategy_combinations())
  def test_trainer_train(self, distribution):
    with distribution.scope():
      trainer = self.create_test_trainer()
      logs = trainer.train(tf.convert_to_tensor(5, dtype=tf.int32))
      self.assertIn('training_loss', logs)
      self.assertIn('learning_rate', logs)

  @combinations.generate(all_strategy_combinations())
  def test_trainer_validate(self, distribution):
    with distribution.scope():
      trainer = self.create_test_trainer()
      logs = trainer.evaluate(tf.convert_to_tensor(5, dtype=tf.int32))
      self.assertIn('validation_loss', logs)
      self.assertEqual(logs['acc'], 5. * distribution.num_replicas_in_sync)

  @combinations.generate(
      combinations.combine(
          mixed_precision_dtype=['float32', 'bfloat16', 'float16'],
          loss_scale=[None, 'dynamic', 128, 256],
      ))
  def test_configure_optimizer(self, mixed_precision_dtype, loss_scale):
    config = cfg.ExperimentConfig(
        runtime=cfg.RuntimeConfig(
            mixed_precision_dtype=mixed_precision_dtype, loss_scale=loss_scale),
        trainer=cfg.TrainerConfig(
            optimizer_config=cfg.OptimizationConfig({
                'optimizer': {
                    'type': 'sgd'
                },
                'learning_rate': {
                    'type': 'constant'
                }
            })))
    task = mock_task.MockTask()
    trainer = trainer_lib.Trainer(config, task)
    if mixed_precision_dtype != 'float16':
      self.assertIsInstance(trainer.optimizer, tf.keras.optimizers.SGD)
    elif mixed_precision_dtype == 'float16' and loss_scale is None:
      self.assertIsInstance(trainer.optimizer, tf.keras.optimizers.SGD)
    else:
      self.assertIsInstance(
          trainer.optimizer,
          tf.keras.mixed_precision.experimental.LossScaleOptimizer)

    metrics = trainer.train(tf.convert_to_tensor(5, dtype=tf.int32))
    self.assertIn('training_loss', metrics)

  @combinations.generate(all_strategy_combinations())
  def test_export_best_ckpt(self, distribution):
    config = cfg.ExperimentConfig(
        trainer=cfg.TrainerConfig(
            best_checkpoint_export_subdir='best_ckpt',
            best_checkpoint_eval_metric='acc',
            optimizer_config=cfg.OptimizationConfig({
                'optimizer': {
                    'type': 'sgd'
                },
                'learning_rate': {
                    'type': 'constant'
                }
            })))
    model_dir = self.get_temp_dir()
    task = mock_task.MockTask(config.task, logging_dir=model_dir)
    ckpt_exporter = train_lib.maybe_create_best_ckpt_exporter(config, model_dir)
    trainer = trainer_lib.Trainer(
        config, task, checkpoint_exporter=ckpt_exporter)
    trainer.train(tf.convert_to_tensor(1, dtype=tf.int32))
    trainer.evaluate(tf.convert_to_tensor(1, dtype=tf.int32))
    self.assertTrue(tf.io.gfile.exists(
        os.path.join(model_dir, 'best_ckpt', 'info.json')))


if __name__ == '__main__':
  tf.test.main()
