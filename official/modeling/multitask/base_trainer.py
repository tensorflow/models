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

"""Multitask base trainer implementation.

The trainer derives from the Orbit `StandardTrainer` class.
"""
from typing import Union

import gin
import orbit
import tensorflow as tf

from official.modeling import optimization
from official.modeling.multitask import base_model
from official.modeling.multitask import multitask


@gin.configurable
class MultiTaskBaseTrainer(orbit.StandardTrainer):
  """Multitask base trainer."""

  def __init__(self,
               multi_task: multitask.MultiTask,
               multi_task_model: Union[tf.keras.Model,
                                       base_model.MultiTaskBaseModel],
               optimizer: tf.optimizers.Optimizer,
               trainer_options=None,
               train_datasets=None):
    self._strategy = tf.distribute.get_strategy()
    self._multi_task = multi_task
    self._multi_task_model = multi_task_model
    self._optimizer = optimizer

    self._training_losses = None
    self._training_metrics = None
    self._global_step = orbit.utils.create_global_step()

    # Creates a shadow copy of the weights to store weights moving average.
    if isinstance(self._optimizer, optimization.ExponentialMovingAverage
                 ) and not self._optimizer.has_shadow_copy:
      self._optimizer.shadow_copy(multi_task_model)

    if hasattr(self.multi_task_model, "checkpoint_items"):
      checkpoint_items = self.multi_task_model.checkpoint_items
    else:
      checkpoint_items = {}

    self._checkpoint = tf.train.Checkpoint(
        model=self.multi_task_model,
        optimizer=self.optimizer,
        global_step=self.global_step,
        **checkpoint_items)

    if train_datasets is None:
      train_datasets = {}
      for name, task in self.multi_task.tasks.items():
        train_datasets[name] = orbit.utils.make_distributed_dataset(
            self.strategy, task.build_inputs, task.task_config.train_data)

    super().__init__(
        train_dataset=train_datasets,
        options=trainer_options or orbit.StandardTrainerOptions())

  def train_loop_begin(self):
    """Clean up states that hold losses and metrics."""
    for _, train_loss_metric in self.training_losses.items():
      train_loss_metric.reset_states()

    for _, metrics in self.training_metrics.items():
      for metric in metrics:
        metric.reset_states()

  def train_loop_end(self):
    """Record loss and metric values per task."""
    result = {}
    for task_name, loss in self.training_losses.items():
      result[task_name] = {loss.name: loss.result()}
    for task_name, task_metrics in self.training_metrics.items():
      result[task_name].update(
          {metric.name: metric.result() for metric in task_metrics})
    # Note that, the learning rate schedule is managed by the keras optimizer
    # internally, which respects the number of backward pass as `iterations`.
    # The learning rate schedule does not follow the trainer logical global
    # step of multiple tasks.
    if callable(self.optimizer.learning_rate):
      result["learning_rate"] = self.optimizer.learning_rate(
          self.optimizer.iterations)
    else:
      result["learning_rate"] = self.optimizer.learning_rate
    return result

  @property
  def checkpoint(self):
    """Accesses the training checkpoint."""
    return self._checkpoint

  @property
  def training_losses(self):
    """Access training loss metric objects for all tasks."""
    if self._training_losses is None:
      # Builds the per-task metrics and losses.
      # This the total summed training loss of tasks in the joint training.
      self._training_losses = dict(
          total_loss=tf.keras.metrics.Mean("training_loss", dtype=tf.float32))
      for name in self.multi_task.tasks:
        self._training_losses[name] = tf.keras.metrics.Mean(
            "training_loss", dtype=tf.float32)
    return self._training_losses

  @property
  def training_metrics(self):
    """Access training metric metric objects for all tasks."""
    if self._training_metrics is None:
      # Builds the per-task metrics and losses.
      self._training_metrics = {}
      for name, task in self.multi_task.tasks.items():
        self._training_metrics[name] = task.build_metrics(training=True)
    return self._training_metrics

  @property
  def strategy(self):
    return self._strategy

  @property
  def multi_task(self):
    return self._multi_task

  @property
  def multi_task_model(self):
    return self._multi_task_model

  @property
  def optimizer(self):
    return self._optimizer

  @property
  def global_step(self):
    return self._global_step

  def train_step(self, iterator_map):
    """The default train step calling the multi-task train step.

    Args:
      iterator_map: a dictionary of task names and per-task dataset iterators.
    """

    def step_fn(inputs):
      losses = self.multi_task.joint_train_step(
          inputs,
          multi_task_model=self.multi_task_model,
          optimizer=self.optimizer,
          task_metrics=self.training_metrics)
      for key, loss in losses.items():
        self.training_losses[key].update_state(loss)

    self.strategy.run(
        step_fn, args=(tf.nest.map_structure(next, iterator_map),))
    self.global_step.assign_add(1)
