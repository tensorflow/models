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

"""Progressive Trainer implementation.

The trainer implements the Orbit `StandardTrainable` and
`StandardEvaluable` interfaces. Trainers inside this project should be
interchangable and independent on model architectures and tasks.
"""

import dataclasses
import os
from typing import Any, Optional

# Import libraries
from absl import logging
import gin
import orbit
import tensorflow as tf, tf_keras
from official.core import base_task
from official.core import base_trainer as trainer_lib
from official.core import config_definitions
from official.modeling.fast_training.progressive import policies
from official.modeling.fast_training.progressive import utils

ExperimentConfig = config_definitions.ExperimentConfig


@dataclasses.dataclass
class ProgressiveTrainerConfig(config_definitions.TrainerConfig):
  """Configuration for progressive trainer.

  Attributes:
    progressive: A task-specific config. Users can subclass ProgressiveConfig
      and define any task-specific settings in their subclass.
    export_checkpoint: A bool. Whether to export checkpoints in non-progressive
      manner (without the volatiles wrapper) such that your down-stream tasks
      can load checkpoints from a progressive trainer as if it is a regular
      checkpoint.
    export_checkpoint_interval: A bool. The number of steps between exporting
      checkpoints. If None (by default), will use the same value as
      TrainerConfig.checkpoint_interval.
    export_max_to_keep: The maximum number of exported checkpoints to keep.
      If None (by default), will use the same value as
      TrainerConfig.max_to_keep.
    export_only_final_stage_ckpt: A bool. Whether to just export checkpoints
      during the final progressive training stage. In other words, whether to
      not export small, partial models. In many cases, it is not meaningful to
      finetune a small, partial model in down-stream tasks.
  """
  progressive: Optional[policies.ProgressiveConfig] = None
  export_checkpoint: bool = True
  export_checkpoint_interval: Optional[int] = None
  export_max_to_keep: Optional[int] = None
  export_only_final_stage_ckpt: bool = True


@gin.configurable
class ProgressiveTrainer(trainer_lib.Trainer):
  """Implements the progressive trainer shared for TensorFlow models."""

  def __init__(
      self,
      config: ExperimentConfig,
      prog_task: base_task.Task,  # also implemented ProgressivePolicy.
      ckpt_dir: str = '',
      train: bool = True,
      evaluate: bool = True,
      checkpoint_exporter: Any = None):
    """Initialize common trainer for TensorFlow models.

    Args:
      config: An `ExperimentConfig` instance specifying experiment config.
      prog_task: An instance both implemented policies.ProgressivePolicy and
        base_task.Task.
      ckpt_dir: Checkpoint directory.
      train: bool, whether or not this trainer will be used for training.
        default to True.
      evaluate: bool, whether or not this trainer will be used for evaluation.
        default to True.
      checkpoint_exporter: an object that has the `maybe_export_checkpoint`
        interface.
    """
    # Gets the current distribution strategy. If not inside any strategy scope,
    # it gets a single-replica no-op strategy.
    self._strategy = tf.distribute.get_strategy()
    self._config = config
    self._runtime_options = trainer_lib.get_runtime_options(config)
    self._task = prog_task

    # Directory for non-progressive checkpoint
    self._export_ckpt_dir = os.path.join(ckpt_dir, 'exported_ckpts')
    tf.io.gfile.makedirs(self._export_ckpt_dir)
    self._export_ckpt_manager = None

    # Receive other checkpoint export, e.g, best checkpoint exporter.
    # TODO(lehou): unify the checkpoint exporting logic, although the default
    # setting does not use checkpoint_exporter.
    self._checkpoint_exporter = checkpoint_exporter

    self._global_step = orbit.utils.create_global_step()

    self._checkpoint = utils.CheckpointWithHooks(
        before_load_hook=self._update_pt_stage_from_ckpt,
        global_step=self.global_step,
        **self._task.cur_checkpoint_items)

    self._train_loss = tf_keras.metrics.Mean('training_loss', dtype=tf.float32)
    self._validation_loss = tf_keras.metrics.Mean(
        'validation_loss', dtype=tf.float32)
    self._train_metrics = self.task.build_metrics(
        training=True) + self.model.metrics
    self._validation_metrics = self.task.build_metrics(
        training=False) + self.model.metrics

    if train:
      orbit.StandardTrainer.__init__(
          self,
          None,  # Manage train_dataset by ourselves, not by StandardTrainer.
          options=orbit.StandardTrainerOptions(
              use_tf_while_loop=config.trainer.train_tf_while_loop,
              use_tf_function=config.trainer.train_tf_function))

    if evaluate:
      orbit.StandardEvaluator.__init__(
          self,
          None,  # Manage train_dataset by ourselves, not by StandardEvaluator.
          options=orbit.StandardEvaluatorOptions(
              use_tf_function=config.trainer.eval_tf_function))

  @property
  def model(self):
    return self._task.cur_model

  @property
  def optimizer(self):
    return self._task.cur_optimizer

  # override
  @property
  def train_dataset(self):
    """Overriding StandardTrainer.train_dataset."""
    return self._task.cur_train_dataset

  # override
  @train_dataset.setter
  def train_dataset(self, _):
    raise SyntaxError('Please do not set train_dataset. Progressive training '
                      'relies on progressive policy to manager train dataset.')

  # override
  @property
  def eval_dataset(self):
    """Overriding StandardEvaluator.eval_dataset."""
    return self._task.cur_eval_dataset

  # override
  @eval_dataset.setter
  def eval_dataset(self, _):
    raise SyntaxError('Please do not set eval_dataset. Progressive training '
                      'relies on progressive policy to manager eval dataset.')

  def train_loop_end(self):
    """See base class."""
    logs = {}
    for metric in self.train_metrics + [self.train_loss]:
      logs[metric.name] = metric.result()
      metric.reset_states()
    if callable(self.optimizer.learning_rate):
      logs['learning_rate'] = self.optimizer.learning_rate(
          self.optimizer.iterations)
    else:
      logs['learning_rate'] = self.optimizer.learning_rate

    self._maybe_export_non_progressive_checkpoint(self._export_ckpt_dir)
    if self._task.is_stage_advancing(self.global_step.numpy()):
      old_train_dataset = self.train_dataset

      # Update progressive properties
      self._task.update_pt_stage(self.global_step.numpy())

      # Setting `self._train_loop_fn` and `self._eval_loop_fn` to None will
      # rebuild the train and eval functions with the updated model.
      self._train_loop_fn = None
      self._eval_loop_fn = None

      if self.train_dataset != old_train_dataset:
        # Setting `self._train_iter` to None will rebuild the dataset iterator.
        self._train_iter = None

      # Setting `self._export_ckpt_manager` to None will rebuild the checkpoint
      # for exporting.
      self._export_ckpt_manager = None

    return logs

  def _update_pt_stage_from_ckpt(self, ckpt_file):
    """Update stage properties based on the global_step variable in a ckpt file.

    Before loading variables from a checkpoint file, we need to go to the
    correct stage and build corresponding model and optimizer, to make sure that
    we retore variables of the right model and optimizer.

    Args:
      ckpt_file: Checkpoint file that will be restored/read from.
    """
    if not ckpt_file:
      return
    ckpt = tf.train.Checkpoint(global_step=self.global_step)
    ckpt.read(ckpt_file).expect_partial().assert_existing_objects_matched()

    if self._task.is_stage_advancing(self.global_step.numpy()):
      old_train_dataset = self.train_dataset

      # Update progressive properties
      self._task.update_pt_stage(self.global_step.numpy(), pass_old_model=False)

      # Setting `self._train_loop_fn` and `self._eval_loop_fn` to None will
      # rebuild the train and eval functions with the updated model.
      self._train_loop_fn = None
      self._eval_loop_fn = None

      if self.train_dataset != old_train_dataset:
        # Setting `self._train_iter` to None will rebuild the dataset iterator.
        self._train_iter = None

      # Setting `self._export_ckpt_manager` to None will rebuild the checkpoint
      # for exporting.
      self._export_ckpt_manager = None

  def _maybe_export_non_progressive_checkpoint(self, export_ckpt_dir):
    """Export checkpoints in non-progressive format.

    This basically removes the wrapping of self._task.cur_checkpoint_items
    -- just save the model, optimizer, etc., directly.
    The purpose is to let your down-stream tasks to use these checkpoints.

    Args:
      export_ckpt_dir: A str. folder of exported checkpoints.
    """
    if not self.config.trainer.export_checkpoint:
      logging.info('Not exporting checkpoints.')
      return
    if not self._task.is_last_stage and (
        self.config.trainer.export_only_final_stage_ckpt):
      logging.info('Not exporting checkpoints until the last stage.')
      return

    if self._export_ckpt_manager is None:
      # Create a checkpoint object just now, to make sure we use
      # progressive_policy.cur_model and progressive_policy.cur_optimizer of the
      # current stage.
      if hasattr(self.model, 'checkpoint_items'):
        checkpoint_items = self.model.checkpoint_items
      else:
        checkpoint_items = {}
      checkpoint = tf.train.Checkpoint(
          global_step=self.global_step,
          model=self.model,
          optimizer=self.optimizer,
          **checkpoint_items)

      max_to_keep = self.config.trainer.export_max_to_keep or (
          self.config.trainer.max_to_keep)
      checkpoint_interval = self.config.trainer.export_checkpoint_interval or (
          self.config.trainer.checkpoint_interval)
      self._export_ckpt_manager = tf.train.CheckpointManager(
          checkpoint,
          directory=export_ckpt_dir,
          checkpoint_name='ckpt',
          step_counter=self.global_step,
          max_to_keep=max_to_keep,
          checkpoint_interval=checkpoint_interval,
      )

    # Make sure we export the last checkpoint.
    last_checkpoint = (
        self.global_step.numpy() == self._config.trainer.train_steps)
    checkpoint_path = self._export_ckpt_manager.save(
        checkpoint_number=self.global_step.numpy(),
        check_interval=not last_checkpoint)
    if checkpoint_path:
      logging.info('Checkpoints exported: %s.', checkpoint_path)
