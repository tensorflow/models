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

"""Tests for the progressive trainer."""
# pylint: disable=g-direct-tensorflow-import
import os

from absl.testing import parameterized
import orbit
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.core import config_definitions as cfg
from official.modeling import optimization
from official.modeling.fast_training.progressive import policies
from official.modeling.fast_training.progressive import trainer as trainer_lib
from official.nlp.configs import bert
from official.utils.testing import mock_task


def all_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.cloud_tpu_strategy,
          strategy_combinations.one_device_strategy_gpu,
      ],)


def get_exp_config():
  return cfg.ExperimentConfig(
      task=cfg.TaskConfig(
          model=bert.PretrainerConfig()),
      trainer=trainer_lib.ProgressiveTrainerConfig(
          export_checkpoint=True,
          export_checkpoint_interval=1,
          export_only_final_stage_ckpt=False))


class TestPolicy(policies.ProgressivePolicy, mock_task.MockTask):
  """Just for testing purposes."""

  def __init__(self, strategy, task_config, change_train_dataset=True):
    self._strategy = strategy
    self._change_train_dataset = change_train_dataset
    self._my_train_dataset = None
    mock_task.MockTask.__init__(self, params=task_config, logging_dir=None)
    policies.ProgressivePolicy.__init__(self)

  def num_stages(self) -> int:
    return 2

  def num_steps(self, stage_id: int) -> int:
    return 2 if stage_id == 0 else 4

  def get_model(self,
                stage_id: int,
                old_model: tf.keras.Model) -> tf.keras.Model:
    del stage_id, old_model
    return self.build_model()

  def get_optimizer(self, stage_id: int) -> tf.keras.optimizers.Optimizer:
    optimizer_type = 'sgd' if stage_id == 0 else 'adamw'
    optimizer_config = cfg.OptimizationConfig({
        'optimizer': {'type': optimizer_type},
        'learning_rate': {'type': 'constant'}})
    opt_factory = optimization.OptimizerFactory(optimizer_config)
    return opt_factory.build_optimizer(opt_factory.build_learning_rate())

  def get_train_dataset(self, stage_id: int) -> tf.data.Dataset:
    if not self._change_train_dataset and self._my_train_dataset:
      return self._my_train_dataset
    if self._strategy:
      self._my_train_dataset = orbit.utils.make_distributed_dataset(
          self._strategy,
          self._build_inputs,
          stage_id)
    else:
      self._my_train_dataset = self._build_inputs(stage_id)
    return self._my_train_dataset

  def get_eval_dataset(self, stage_id: int) -> tf.data.Dataset:
    if self._strategy:
      return orbit.utils.make_distributed_dataset(
          self._strategy,
          self._build_inputs,
          stage_id)
    return self._build_inputs(stage_id)

  def _build_inputs(self, stage_id):
    def dummy_data(_):
      batch_size = 2 if stage_id == 0 else 1
      x = tf.zeros(shape=(batch_size, 2), dtype=tf.float32)
      label = tf.zeros(shape=(batch_size, 1), dtype=tf.float32)
      return x, label
    dataset = tf.data.Dataset.range(1)
    dataset = dataset.repeat()
    return dataset.map(
        dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)


class TrainerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(TrainerTest, self).setUp()
    self._config = get_exp_config()

  def create_test_trainer(self, distribution, model_dir, change_train_dataset):
    trainer = trainer_lib.ProgressiveTrainer(
        self._config,
        prog_task=TestPolicy(
            distribution, self._config.task, change_train_dataset),
        ckpt_dir=model_dir)
    return trainer

  @combinations.generate(all_strategy_combinations())
  def test_checkpointing(self, distribution):
    model_dir = self.get_temp_dir()
    ckpt_file = os.path.join(model_dir, 'ckpt')
    with distribution.scope():
      trainer = self.create_test_trainer(distribution, model_dir, True)
      self.assertFalse(trainer._task.is_last_stage)
      trainer.train(tf.convert_to_tensor(4, dtype=tf.int32))
      self.assertTrue(trainer._task.is_last_stage)
      trainer.checkpoint.save(ckpt_file)

      trainer = self.create_test_trainer(distribution, model_dir, True)
      self.assertFalse(trainer._task.is_last_stage)
      trainer.checkpoint.restore(ckpt_file + '-1')
      self.assertTrue(trainer._task.is_last_stage)

  @combinations.generate(all_strategy_combinations())
  def test_train_dataset(self, distribution):
    model_dir = self.get_temp_dir()
    with distribution.scope():
      trainer = self.create_test_trainer(distribution, model_dir, True)
      # Using dataset of stage == 0
      train_iter = tf.nest.map_structure(iter, trainer.train_dataset)
      train_data = train_iter.next()[0]
      if distribution.num_replicas_in_sync > 1:
        train_data = train_data.values[0]
      self.assertEqual(train_data.shape[0], 2)

      trainer.train(tf.convert_to_tensor(4, dtype=tf.int32))
      # Using dataset of stage == 1
      train_iter = tf.nest.map_structure(iter, trainer.train_dataset)
      train_data = train_iter.next()[0]
      if distribution.num_replicas_in_sync > 1:
        train_data = train_data.values[0]
      self.assertEqual(train_data.shape[0], 1)

      with self.assertRaises(SyntaxError):
        trainer.train_dataset = None

  @combinations.generate(all_strategy_combinations())
  def test_train_dataset_no_switch(self, distribution):
    model_dir = self.get_temp_dir()
    with distribution.scope():
      trainer = self.create_test_trainer(distribution, model_dir, False)
      trainer.train(tf.convert_to_tensor(2, dtype=tf.int32))
      # _train_iter is not reset since the dataset is not changed.
      self.assertIsNotNone(trainer._train_iter)
    with distribution.scope():
      trainer = self.create_test_trainer(distribution, model_dir, True)
      trainer.train(tf.convert_to_tensor(2, dtype=tf.int32))
      # _train_iter is reset since the dataset changed.
      self.assertIsNone(trainer._train_iter)


class TrainerWithMaskedLMTaskTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(TrainerWithMaskedLMTaskTest, self).setUp()
    self._config = get_exp_config()

  def create_test_trainer(self, distribution):
    trainer = trainer_lib.ProgressiveTrainer(
        self._config,
        prog_task=TestPolicy(distribution, self._config.task),
        ckpt_dir=self.get_temp_dir())
    return trainer

  @combinations.generate(all_strategy_combinations())
  def test_trainer_train(self, distribution):
    with distribution.scope():
      trainer = self.create_test_trainer(distribution)
      logs = trainer.train(tf.convert_to_tensor(5, dtype=tf.int32))
      self.assertIn('training_loss', logs)
      self.assertIn('learning_rate', logs)

  @combinations.generate(all_strategy_combinations())
  def test_trainer_validate(self, distribution):
    with distribution.scope():
      trainer = self.create_test_trainer(distribution)
      logs = trainer.evaluate(tf.convert_to_tensor(5, dtype=tf.int32))
      self.assertIn('validation_loss', logs)
      self.assertEqual(logs['counter'], 5. * distribution.num_replicas_in_sync)

  @combinations.generate(
      combinations.combine(
          mixed_precision_dtype=['float32', 'bfloat16', 'float16'],
          loss_scale=[None, 'dynamic', 128, 256],
      ))
  def test_configure_optimizer(self, mixed_precision_dtype, loss_scale):
    config = cfg.ExperimentConfig(
        task=cfg.TaskConfig(
            model=bert.PretrainerConfig()),
        runtime=cfg.RuntimeConfig(
            mixed_precision_dtype=mixed_precision_dtype, loss_scale=loss_scale),
        trainer=trainer_lib.ProgressiveTrainerConfig(
            export_checkpoint=True,
            export_checkpoint_interval=1,
            export_only_final_stage_ckpt=False))
    task = TestPolicy(None, config.task)
    trainer = trainer_lib.ProgressiveTrainer(config, task, self.get_temp_dir())
    if mixed_precision_dtype != 'float16':
      self.assertIsInstance(
          trainer.optimizer,
          (tf.keras.optimizers.SGD, tf.keras.optimizers.legacy.SGD))
    elif mixed_precision_dtype == 'float16' and loss_scale is None:
      self.assertIsInstance(
          trainer.optimizer,
          (tf.keras.optimizers.SGD, tf.keras.optimizers.legacy.SGD))

    metrics = trainer.train(tf.convert_to_tensor(5, dtype=tf.int32))
    self.assertIn('training_loss', metrics)


if __name__ == '__main__':
  tf.test.main()
