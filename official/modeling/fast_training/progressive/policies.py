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

"""Base ProgressivePolicy definition for progressive training.

To write a progressive model, subclass ProgressivePolicy and implement its
abstract methods to handle each training stage.
"""

import abc
import dataclasses
from typing import Any, Mapping
from absl import logging
import six
import tensorflow as tf, tf_keras

from official.common import streamz_counters
from official.modeling.fast_training.progressive import utils
from official.modeling.hyperparams import base_config


@dataclasses.dataclass
class ProgressiveConfig(base_config.Config):
  pass


@six.add_metaclass(abc.ABCMeta)
class ProgressivePolicy:
  """The APIs for handling progressive training stages.

  Attributes:
    cur_model: The model for the current progressive training stage.
    cur_train_dataset: The train dataset function for the current stage.
    cur_eval_dataset: The eval dataset function for the current stage.
    cur_optimizer: The optimizer for the current stage.
    cur_checkpoint_items: Items to be saved in and restored from checkpoints,
      for the progressive trainer.
    is_last_stage: Whether it is currently in the last stage.

  Interfaces:
    is_stage_advancing: Returns if progressive training is advancing to the
      next stage.
    update_pt_stage: Update progressive training stage.
  """

  def __init__(self):
    """Initialize stage policy."""
    self._cur_train_dataset = None
    self._cur_eval_dataset = None
    self._volatiles = utils.VolatileTrackable(optimizer=None, model=None)

    stage_id = 0
    self._stage_id = tf.Variable(
        stage_id,
        trainable=False,
        dtype=tf.int64,
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        shape=[])
    self._volatiles.reassign_trackable(
        optimizer=self.get_optimizer(stage_id),
        model=self.get_model(stage_id, old_model=None))  # pytype: disable=wrong-arg-types  # typed-keras

    streamz_counters.progressive_policy_creation_counter.get_cell(
        ).increase_by(1)

  def compute_stage_id(self, global_step: int) -> int:
    for stage_id in range(self.num_stages()):
      global_step -= self.num_steps(stage_id)
      if global_step < 0:
        return stage_id
    logging.error('Global step %d found no matching progressive stages. '
                  'Default to the last stage.', global_step)
    return self.num_stages() - 1

  @abc.abstractmethod
  def num_stages(self) -> int:
    """Return the total number of progressive stages."""
    pass

  @abc.abstractmethod
  def num_steps(self, stage_id: int) -> int:
    """Return the total number of steps in this stage."""
    pass

  @abc.abstractmethod
  def get_model(self,
                stage_id: int,
                old_model: tf_keras.Model = None) -> tf_keras.Model:  # pytype: disable=annotation-type-mismatch  # typed-keras
    """Return model for this stage. For initialization, `old_model` = None."""
    pass

  @abc.abstractmethod
  def get_optimizer(self, stage_id: int) -> tf_keras.optimizers.Optimizer:
    """Return optimizer for this stage."""
    pass

  @abc.abstractmethod
  def get_train_dataset(self, stage_id: int) -> tf.data.Dataset:
    """Return training Dataset for this stage."""
    pass

  @abc.abstractmethod
  def get_eval_dataset(self, stage_id: int) -> tf.data.Dataset:
    """Return evaluation Dataset for this stage."""
    pass

  @property
  def cur_model(self) -> tf_keras.Model:
    return self._volatiles.model

  @property
  def cur_train_dataset(self) -> tf.data.Dataset:
    if self._cur_train_dataset is None:
      self._cur_train_dataset = self.get_train_dataset(self._stage_id.numpy())
    return self._cur_train_dataset

  @property
  def cur_eval_dataset(self) -> tf.data.Dataset:
    if self._cur_eval_dataset is None:
      self._cur_eval_dataset = self.get_eval_dataset(self._stage_id.numpy())
    return self._cur_eval_dataset

  @property
  def cur_optimizer(self) -> tf_keras.optimizers.Optimizer:
    return self._volatiles.optimizer

  @property
  def is_last_stage(self) -> bool:
    stage_id = self._stage_id.numpy()
    return stage_id >= self.num_stages() - 1

  @property
  def cur_checkpoint_items(self) -> Mapping[str, Any]:
    return dict(stage_id=self._stage_id, volatiles=self._volatiles)

  def is_stage_advancing(self, global_step: int) -> bool:
    old_stage_id = self._stage_id.numpy()
    new_stage_id = self.compute_stage_id(global_step)
    return old_stage_id != new_stage_id

  def update_pt_stage(self, global_step: int, pass_old_model=True) -> None:
    """Update progressive training internal status.

    Call this after a training loop ends.

    Args:
      global_step: an integer scalar of the current global step.
      pass_old_model: whether to pass the old_model to get_model() function.
        This is set to False if the old_model is irrelevant (e.g, just a default
        model from stage 0).
    """
    old_stage_id = self._stage_id.numpy()
    new_stage_id = self.compute_stage_id(global_step)
    logging.info('Switching stage from %d to %d', old_stage_id, new_stage_id)

    # Update stage id.
    self._stage_id.assign(new_stage_id)
    # Update dataset function.
    self._cur_train_dataset = None
    self._cur_eval_dataset = None

    # Update optimizer and model.
    new_optimizer = self.get_optimizer(new_stage_id)
    self._volatiles.reassign_trackable(optimizer=new_optimizer)
    new_model = self.get_model(
        new_stage_id, old_model=self.cur_model if pass_old_model else None)
    self._volatiles.reassign_trackable(model=new_model)
