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
"""Standard Trainer implementation.

The base trainer implements the Orbit `StandardTrainable` and
`StandardEvaluable` interfaces. Trainers inside this project should be
interchangable and independent on model architectures and tasks.
"""

import gin
import orbit
import tensorflow as tf

from official.core import base_task
from official.modeling import optimization
from official.modeling import performance
from official.modeling.hyperparams import config_definitions

ExperimentConfig = config_definitions.ExperimentConfig


@gin.configurable
class Trainer(orbit.StandardTrainer, orbit.StandardEvaluator):
  """Implements the common trainer shared for TensorFlow models."""

  def __init__(self,
               config: ExperimentConfig,
               task: base_task.Task,
               model: tf.keras.Model,
               train: bool = True,
               evaluate: bool = True,
               optimizer=None,
               checkpoint_exporter=None):
    """Initialize common trainer for TensorFlow models.

    Args:
      config: An `ExperimentConfig` instance specifying experiment config.
      task: A base_task.Task instance.
      model: tf.keras.Model instance. If provided, it will be used instead of
        building model using task.build_model(). Default to None.
      train: bool, whether or not this trainer will be used for training.
        default to True.
      evaluate: bool, whether or not this trainer will be used for evaluation.
        default to True.
      optimizer: tf.keras.optimizers.Optimizer instance. If provided, it will
        used instead of the optimizer from config. Default to None.
      checkpoint_exporter: an object that has the `maybe_export_checkpoint`
        interface.
    """
    # Gets the current distribution strategy. If not inside any strategy scope,
    # it gets a single-replica no-op strategy.
    self._strategy = tf.distribute.get_strategy()
    self._config = config
    self._task = task
    self._model = model

    if optimizer is None:
      opt_factory = optimization.OptimizerFactory(
          config.trainer.optimizer_config)
      self._optimizer = opt_factory.build_optimizer(
          opt_factory.build_learning_rate())
    else:
      self._optimizer = optimizer

    self._checkpoint_exporter = checkpoint_exporter

    # Configuring optimizer when loss_scale is set in runtime config. This helps
    # avoiding overflow/underflow for float16 computations.
    if config.runtime.loss_scale:
      self._optimizer = performance.configure_optimizer(
          self._optimizer,
          use_float16=config.runtime.mixed_precision_dtype == 'float16',
          loss_scale=config.runtime.loss_scale)

    # global_step increases by 1 after each training iteration.
    # We should have global_step.numpy() == self.optimizer.iterations.numpy()
    # when there is only 1 optimizer.
    self._global_step = orbit.utils.create_global_step()
    if hasattr(self.model, 'checkpoint_items'):
      checkpoint_items = self.model.checkpoint_items
    else:
      checkpoint_items = {}
    self._checkpoint = tf.train.Checkpoint(
        global_step=self.global_step,
        model=self.model,
        optimizer=self.optimizer,
        **checkpoint_items)

    self._train_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
    self._validation_loss = tf.keras.metrics.Mean(
        'validation_loss', dtype=tf.float32)
    self._train_metrics = self.task.build_metrics(
        training=True) + self.model.metrics
    self._validation_metrics = self.task.build_metrics(
        training=False) + self.model.metrics

    if train:
      train_dataset = orbit.utils.make_distributed_dataset(
          self.strategy, self.task.build_inputs, self.config.task.train_data)
      orbit.StandardTrainer.__init__(
          self,
          train_dataset,
          options=orbit.StandardTrainerOptions(
              use_tf_while_loop=config.trainer.train_tf_while_loop,
              use_tf_function=config.trainer.train_tf_function,
              use_tpu_summary_optimization=config.trainer.allow_tpu_summary))

    if evaluate:
      eval_dataset = orbit.utils.make_distributed_dataset(
          self.strategy, self.task.build_inputs,
          self.config.task.validation_data)
      orbit.StandardEvaluator.__init__(
          self,
          eval_dataset,
          options=orbit.StandardEvaluatorOptions(
              use_tf_function=config.trainer.eval_tf_function))

  @property
  def strategy(self):
    return self._strategy

  @property
  def config(self):
    return self._config

  @property
  def task(self):
    return self._task

  @property
  def model(self):
    return self._model

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def global_step(self):
    return self._global_step

  @property
  def train_loss(self):
    """Accesses the training loss metric object."""
    return self._train_loss

  @property
  def validation_loss(self):
    """Accesses the validation loss metric object."""
    return self._validation_loss

  @property
  def train_metrics(self):
    """Accesses all training metric objects."""
    return self._train_metrics

  @property
  def validation_metrics(self):
    """Accesses all validation metric metric objects."""
    return self._validation_metrics

  def initialize(self):
    """A callback function.

    This function will be called when no checkpoint found for the model.
    If there is a checkpoint, the checkpoint will be loaded and this function
    will not be called. Tasks may use this callback function to load a
    pretrained checkpoint, saved under a directory other than the model_dir.
    """
    self.task.initialize(self.model)

  @property
  def checkpoint(self):
    """Accesses the training checkpoint."""
    return self._checkpoint

  def train_loop_end(self):
    """See base class."""
    logs = {}
    for metric in self.train_metrics + [self.train_loss]:
      logs[metric.name] = metric.result()
      metric.reset_states()
    if callable(self.optimizer.learning_rate):
      logs['learning_rate'] = self.optimizer.learning_rate(self.global_step)
    else:
      logs['learning_rate'] = self.optimizer.learning_rate
    return logs

  def train_step(self, iterator):
    """See base class."""

    def step_fn(inputs):
      logs = self.task.train_step(
          inputs,
          model=self.model,
          optimizer=self.optimizer,
          metrics=self.train_metrics)
      self._train_loss.update_state(logs[self.task.loss])
      self.global_step.assign_add(1)

    self.strategy.run(step_fn, args=(next(iterator),))

  def eval_begin(self):
    """Sets up metrics."""
    for metric in self.validation_metrics + [self.validation_loss]:
      metric.reset_states()

  def eval_step(self, iterator):
    """See base class."""

    def step_fn(inputs):
      logs = self.task.validation_step(
          inputs, model=self.model, metrics=self.validation_metrics)
      self._validation_loss.update_state(logs[self.task.loss])
      return logs

    distributed_outputs = self.strategy.run(step_fn, args=(next(iterator),))
    return tf.nest.map_structure(self.strategy.experimental_local_results,
                                 distributed_outputs)

  def eval_end(self, aggregated_logs=None):
    """Processes evaluation results."""
    logs = {}
    for metric in self.validation_metrics + [self.validation_loss]:
      logs[metric.name] = metric.result()
    if aggregated_logs:
      metrics = self.task.reduce_aggregated_logs(aggregated_logs)
      logs.update(metrics)

    if self._checkpoint_exporter:
      self._checkpoint_exporter.maybe_export_checkpoint(
          self.checkpoint, logs, self.global_step.numpy())
      metric_name = self.config.trainer.best_checkpoint_eval_metric
      logs['best_' + metric_name] = self._checkpoint_exporter.best_ckpt_logs[
          metric_name]

    return logs

  def eval_reduce(self, state=None, step_outputs=None):
    return self.task.aggregate_logs(state, step_outputs)
