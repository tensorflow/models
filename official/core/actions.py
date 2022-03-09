# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Provides TFM orbit actions and associated helper functions/classes."""

import os
from typing import List
from absl import logging

import gin
import orbit
import tensorflow as tf

from official.core import base_trainer
from official.core import config_definitions
from official.modeling import optimization


class PruningAction:
  """Train action to updates pruning related information.

  This action updates pruning steps at the end of trainig loop, and log
    pruning metrics to tensorboard.

  This action must be used when training a pruned model to avoid pruning error.
  """

  def __init__(
      self,
      export_dir: str,
      model: tf.keras.Model,
      optimizer: tf.keras.optimizers.Optimizer,
  ):
    """Initializes the instance.

    Args:
      export_dir: `str` for the export directory of the pruning summaries.
      model: `tf.keras.Model` model instance used for training. This will be
        used to assign a pruning step to each prunable weight.
      optimizer: `tf.keras.optimizers.Optimizer` optimizer instance used for
        training. This will be used to find the current training steps.
    """
    # TODO(b/221490190): Avoid local import when the bug is fixed.
    import tensorflow_model_optimization as tfmot  # pylint: disable=g-import-not-at-top
    self._optimizer = optimizer
    self.update_pruning_step = tfmot.sparsity.keras.UpdatePruningStep()
    self.update_pruning_step.set_model(model)
    self.update_pruning_step.on_train_begin()

    self.pruning_summaries = tfmot.sparsity.keras.PruningSummaries(
        log_dir=export_dir)
    model.optimizer = optimizer
    self.pruning_summaries.set_model(model)

  def __call__(self, output: orbit.runner.Output):
    """Update pruning step and log pruning summaries.

    Args:
      output: The train output.
    """
    self.update_pruning_step.on_epoch_end(batch=None)
    self.pruning_summaries.on_epoch_begin(epoch=None)


class EMACheckpointing:
  """Eval action to save checkpoint with average weights when EMA is used.

  This action swaps the weights of the model with the average weights, then it
  saves the checkpoint under export_dir/ema_checkpoints. Checkpointing is
  expensive for large models, so doing this action in eval is more efficient
  than training.
  """

  def __init__(self,
               export_dir: str,
               optimizer: tf.keras.optimizers.Optimizer,
               checkpoint: tf.train.Checkpoint,
               max_to_keep: int = 1):
    """Initializes the instance.

    Args:
      export_dir: `str` for the export directory of the EMA average weights.
      optimizer: `tf.keras.optimizers.Optimizer` optimizer instance used for
        training. This will be used to swap the model weights with the average
        weigths.
      checkpoint: `tf.train.Checkpoint` instance.
      max_to_keep: `int` for max checkpoints to keep in ema_checkpoints subdir.
    """
    if not isinstance(optimizer, optimization.ExponentialMovingAverage):
      raise ValueError('Optimizer has to be instance of'
                       'optimization.ExponentialMovingAverage for'
                       'EMACheckpointing action')

    export_dir = os.path.join(export_dir, 'ema_checkpoints')
    tf.io.gfile.makedirs(os.path.dirname(export_dir))
    self._optimizer = optimizer
    self._checkpoint = checkpoint
    self._checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=export_dir,
        max_to_keep=max_to_keep,
        checkpoint_name='average_weights')

  def __call__(self, output: orbit.runner.Output):
    """Swaps model weights, and saves the checkpoint.

    Args:
      output: The train or eval output.
    """
    self._optimizer.swap_weights()
    self._checkpoint_manager.save(checkpoint_number=self._optimizer.iterations)
    self._optimizer.swap_weights()


class RecoveryAction:
  """Train action to recover from loss blowup.

  Checks the loss value by the given threshold. If applicable, recover the
  model by reading the checkpoint on disk.
  """

  def __init__(self, checkpoint_manager: tf.train.CheckpointManager):
    self.checkpoint_manager = checkpoint_manager

  def __call__(self, _):
    """Recovers the training by triggering checkpoint restoration."""
    # Loads the previous good checkpoint.
    checkpoint_path = self.checkpoint_manager.restore_or_initialize()
    logging.warning('Recovering the model from checkpoint: %s.',
                    checkpoint_path)


class RecoveryCondition:
  """Recovery Condition."""

  def __init__(self,
               global_step: tf.Variable,
               loss_upper_bound: float,
               recovery_begin_steps: int = 0,
               recovery_max_trials: int = 3):
    self.recover_counter = 0
    self.recovery_begin_steps = recovery_begin_steps
    self.recovery_max_trials = recovery_max_trials
    self.loss_upper_bound = loss_upper_bound
    self.global_step = global_step

  def __call__(self, outputs: orbit.runner.Output):
    loss_value = outputs['training_loss']
    if tf.math.is_nan(loss_value):
      self.recover_counter += 1
      if self.recover_counter > self.recovery_max_trials:
        raise RuntimeError(
            'The loss value is NaN after training loop and it happens %d times.'
            % self.recover_counter)
      return True
    if (self.global_step >= self.recovery_begin_steps and
        loss_value > self.loss_upper_bound):
      self.recover_counter += 1
      if self.recover_counter > self.recovery_max_trials:
        raise RuntimeError(
            f'The loss value is {loss_value}, which is larger than the bound {self.loss_upper_bound}, happens {self.recover_counter} times.'
        )
      return True
    return False


@gin.configurable
def get_eval_actions(params: config_definitions.ExperimentConfig,
                     trainer: base_trainer.Trainer,
                     model_dir: str) -> List[orbit.Action]:
  """Gets eval actions for TFM trainer."""
  eval_actions = []
  # Adds ema checkpointing action to save the average weights under
  # ema_checkpoints subdir.
  if isinstance(trainer.optimizer, optimization.ExponentialMovingAverage):
    eval_actions.append(
        EMACheckpointing(
            export_dir=model_dir,
            optimizer=trainer.optimizer,
            checkpoint=trainer.checkpoint,
            max_to_keep=params.trainer.max_to_keep))

  return eval_actions


@gin.configurable
def get_train_actions(
    params: config_definitions.ExperimentConfig, trainer: base_trainer.Trainer,
    model_dir: str,
    checkpoint_manager: tf.train.CheckpointManager) -> List[orbit.Action]:
  """Gets train actions for TFM trainer."""
  train_actions = []
  # Adds pruning callback actions.
  if hasattr(params.task, 'pruning') and params.task.pruning:
    train_actions.append(
        PruningAction(
            export_dir=model_dir,
            model=trainer.model,
            optimizer=trainer.optimizer))

  if params.trainer.recovery_max_trials >= 0:
    recovery_condition = RecoveryCondition(
        global_step=trainer.global_step,
        loss_upper_bound=params.trainer.loss_upper_bound,
        recovery_begin_steps=params.trainer.recovery_begin_steps,
        recovery_max_trials=params.trainer.recovery_max_trials,
    )
    recover_action = orbit.actions.ConditionalAction(
        condition=recovery_condition,
        action=RecoveryAction(checkpoint_manager),
    )
    train_actions.append(recover_action)
  return train_actions
