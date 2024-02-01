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

"""Tests for tensorflow_models.core.trainers.trainer."""
# pylint: disable=g-direct-tensorflow-import
import gc
import multiprocessing
import os
import sys

from absl.testing import parameterized
import orbit
import portpicker
import tensorflow as tf, tf_keras

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.core import base_trainer as trainer_lib
from official.core import config_definitions as cfg
from official.core import train_lib
from official.utils.testing import mock_task

TPU_TEST = 'test_tpu' in sys.argv[0]
GPU_TEST = 'test_gpu' in sys.argv[0]


def all_strategy_combinations():
  return combinations.combine(
      distribution=[
          strategy_combinations.default_strategy,
          strategy_combinations.cloud_tpu_strategy,
          strategy_combinations.one_device_strategy_gpu,
      ],)


def create_in_process_cluster(num_workers, num_ps):
  """Creates and starts local servers and returns the cluster_resolver."""
  worker_ports = [portpicker.pick_unused_port() for _ in range(num_workers)]
  ps_ports = [portpicker.pick_unused_port() for _ in range(num_ps)]

  cluster_dict = {}
  cluster_dict['worker'] = ['localhost:%s' % port for port in worker_ports]
  if num_ps > 0:
    cluster_dict['ps'] = ['localhost:%s' % port for port in ps_ports]

  cluster_spec = tf.train.ClusterSpec(cluster_dict)

  # Workers need some inter_ops threads to work properly.
  worker_config = tf.compat.v1.ConfigProto()
  if multiprocessing.cpu_count() < num_workers + 1:
    worker_config.inter_op_parallelism_threads = num_workers + 1

  for i in range(num_workers):
    tf.distribute.Server(
        cluster_spec,
        job_name='worker',
        task_index=i,
        config=worker_config,
        protocol='grpc')

  for i in range(num_ps):
    tf.distribute.Server(
        cluster_spec, job_name='ps', task_index=i, protocol='grpc')

  cluster_resolver = tf.distribute.cluster_resolver.SimpleClusterResolver(
      cluster_spec, rpc_layer='grpc')
  return cluster_resolver


def dataset_fn(input_context=None):
  del input_context

  def dummy_data(_):
    return tf.zeros((1, 1), dtype=tf.float32)

  dataset = tf.data.Dataset.range(1)
  dataset = dataset.repeat()
  dataset = dataset.map(
      dummy_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset


class MockAsyncTrainer(trainer_lib._AsyncTrainer):
  """Mock AsyncTrainer to test the _AsyncTrainer class."""

  def __init__(self):
    self._strategy = tf.distribute.get_strategy()
    self.init_async()

    self.global_step = tf.Variable(
        0,
        dtype=tf.int64,
        name='global_step',
        trainable=False,
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
    self.eval_global_step = tf.Variable(
        0,
        dtype=tf.int64,
        name='eval_global_step',
        trainable=False,
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

    train_dataset = self.distribute_dataset(dataset_fn)
    orbit.StandardTrainer.__init__(
        self, train_dataset, options=orbit.StandardTrainerOptions())

    validation_dataset = self.distribute_dataset(dataset_fn)
    orbit.StandardEvaluator.__init__(
        self,
        validation_dataset,
        options=orbit.StandardEvaluatorOptions(use_tf_while_loop=True))

  def train_loop_begin(self):
    self.global_step.assign(0)

  def train_step(self, iterator):

    def replica_step(_):
      self.global_step.assign_add(1)

    self._strategy.run(replica_step, args=(next(iterator),))

  def train_loop_end(self):
    self.join()
    return self.global_step.numpy()

  def eval_begin(self):
    self.eval_global_step.assign(0)

  def eval_step(self, iterator):

    def replica_step(_):
      self.eval_global_step.assign_add(1)

    self._strategy.run(replica_step, args=(next(iterator),))

  def eval_end(self):
    self.join()
    return self.eval_global_step.numpy()


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

  def tearDown(self):
    gc.collect()
    # This will only contain uncollectable garbage, i.e. reference cycles
    # involving objects with __del__ defined.
    self.assertEmpty(gc.garbage)
    super().tearDown()

  def create_test_trainer(self, config, model_dir=None, task=None):
    task = task or mock_task.MockTask(config.task, logging_dir=model_dir)
    ckpt_exporter = train_lib.maybe_create_best_ckpt_exporter(config, model_dir)
    trainer = trainer_lib.Trainer(
        config,
        task,
        model=task.build_model(),
        optimizer=task.create_optimizer(config.trainer.optimizer_config,
                                        config.runtime),
        checkpoint_exporter=ckpt_exporter)
    return trainer

  @combinations.generate(all_strategy_combinations())
  def test_trainer_train(self, distribution):
    with distribution.scope():
      trainer = self.create_test_trainer(self._config)
      logs = trainer.train(tf.convert_to_tensor(5, dtype=tf.int32))
      self.assertIn('training_loss', logs)
      self.assertIn('learning_rate', logs)

  @combinations.generate(all_strategy_combinations())
  def test_trainer_passing_datasets(self, distribution):
    with distribution.scope():
      task = mock_task.MockTask(self._config)
      train_dataset = orbit.utils.make_distributed_dataset(
          distribution, task.build_inputs, self._config.task.train_data)
      validation_dataset = orbit.utils.make_distributed_dataset(
          distribution, task.build_inputs, self._config.task.validation_data)
      self._config.task.train_data = None
      self._config.task.validation_data = None
      trainer = trainer_lib.Trainer(
          self._config,
          task,
          model=task.build_model(),
          optimizer=task.create_optimizer(self._config.trainer.optimizer_config,
                                          self._config.runtime),
          train_dataset=train_dataset,
          validation_dataset=validation_dataset)
    logs = trainer.train(tf.convert_to_tensor(5, dtype=tf.int32))
    self.assertIn('training_loss', logs)
    self.assertIn('learning_rate', logs)
    logs = trainer.evaluate(tf.convert_to_tensor(5, dtype=tf.int32))
    self.assertIn('validation_loss', logs)

  def test_base_async_trainer(self):
    if TPU_TEST or GPU_TEST:
      self.skipTest('Aysnc training is not available on GPU/GPU.')
    num_workers = 3
    num_ps = 2
    cluster_resolver = create_in_process_cluster(num_workers, num_ps)
    distribution = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver)
    with distribution.scope():
      trainer = MockAsyncTrainer()
      trainer.init_async()
      self.assertIsInstance(
          trainer._coordinator,
          tf.distribute.experimental.coordinator.ClusterCoordinator)
      self.assertEqual(trainer.train(tf.constant(10)), 10)
      self.assertEqual(trainer.evaluate(tf.constant(11)), 11)

  def test_async_trainer_train(self):
    if TPU_TEST or GPU_TEST:
      self.skipTest('Aysnc training is not available on GPU/TPU.')
    num_workers = 3
    num_ps = 2
    cluster_resolver = create_in_process_cluster(num_workers, num_ps)
    distribution = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver)
    with distribution.scope():
      config = cfg.ExperimentConfig(**self._config.as_dict())
      config.trainer.eval_tf_while_loop = True
      trainer = self.create_test_trainer(config)
      logs = trainer.train(tf.convert_to_tensor(5, dtype=tf.int32))
      self.assertIn('training_loss', logs)
      self.assertIn('learning_rate', logs)

  def test_async_trainer_validate(self):
    if TPU_TEST or GPU_TEST:
      self.skipTest('Aysnc training is not available on GPU/GPU.')
    num_workers = 3
    num_ps = 2
    cluster_resolver = create_in_process_cluster(num_workers, num_ps)
    distribution = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver)
    with distribution.scope():
      config = cfg.ExperimentConfig(**self._config.as_dict())
      config.trainer.eval_tf_while_loop = True
      trainer = self.create_test_trainer(config)
      logs = trainer.evaluate(tf.convert_to_tensor(5, dtype=tf.int32))
      self.assertIn('acc', logs)
      self.assertIn('validation_loss', logs)

  @combinations.generate(all_strategy_combinations())
  def test_trainer_validate(self, distribution):
    with distribution.scope():
      trainer = self.create_test_trainer(self._config)
      logs = trainer.evaluate(tf.convert_to_tensor(5, dtype=tf.int32))
      self.assertEqual(logs['counter'], 5. * distribution.num_replicas_in_sync)
      self.assertIn('validation_loss', logs)

  @combinations.generate(all_strategy_combinations())
  def test_trainer_validate_without_loss(self, distribution):

    class MockTaskWithoutValidationLoss(mock_task.MockTask):

      def validation_step(self, inputs, model, metrics=None):
        # Disable validation loss.
        logs = super().validation_step(inputs, model)
        del logs[self.loss]
        return logs

    with distribution.scope():
      task = MockTaskWithoutValidationLoss()
      trainer = self.create_test_trainer(self._config, task=task)
      logs = trainer.evaluate(tf.convert_to_tensor(5, dtype=tf.int32))
      self.assertEqual(logs['counter'], 5. * distribution.num_replicas_in_sync)
      self.assertNotIn('validation_loss', logs)

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
                },
            })))
    trainer = self.create_test_trainer(config)
    if mixed_precision_dtype == 'float16':
      self.assertIsInstance(trainer.optimizer,
                            tf_keras.mixed_precision.LossScaleOptimizer)
      if loss_scale in (None, 'dynamic'):
        self.assertTrue(trainer.optimizer.dynamic)
      else:
        self.assertFalse(trainer.optimizer.dynamic)
        self.assertEqual(trainer.optimizer.initial_scale, loss_scale)
    else:
      self.assertIsInstance(
          trainer.optimizer,
          (tf_keras.optimizers.SGD, tf_keras.optimizers.legacy.SGD))

    metrics = trainer.train(tf.convert_to_tensor(5, dtype=tf.int32))
    self.assertIn('training_loss', metrics)

  def test_export_best_ckpt(self):
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
    trainer = self.create_test_trainer(config, model_dir=model_dir)
    trainer.train(tf.convert_to_tensor(1, dtype=tf.int32))
    trainer.evaluate(tf.convert_to_tensor(1, dtype=tf.int32))
    self.assertTrue(
        tf.io.gfile.exists(os.path.join(model_dir, 'best_ckpt', 'info.json')))

  def test_model_with_compiled_loss(self):
    task = mock_task.MockTask()
    model = task.build_model()
    model.compile(loss=tf_keras.losses.CategoricalCrossentropy())
    trainer = trainer_lib.Trainer(
        self._config,
        task,
        model=model,
        optimizer=task.create_optimizer(self._config.trainer.optimizer_config))
    logs = trainer.train(tf.convert_to_tensor(5, dtype=tf.int32))
    self.assertIn('training_loss', logs)


if __name__ == '__main__':
  tf.test.main()
