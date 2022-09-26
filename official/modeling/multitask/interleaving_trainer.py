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

"""Multitask trainer that interleaves each task's train step."""
from typing import Union
import gin
import orbit
import tensorflow as tf
from official.modeling.multitask import base_model
from official.modeling.multitask import base_trainer
from official.modeling.multitask import multitask
from official.modeling.multitask import task_sampler as sampler


@gin.configurable
class MultiTaskInterleavingTrainer(base_trainer.MultiTaskBaseTrainer):
  """MultiTask trainer that interleaves task update."""

  def __init__(self,
               multi_task: multitask.MultiTask,
               multi_task_model: Union[tf.keras.Model,
                                       base_model.MultiTaskBaseModel],
               optimizer: Union[tf.optimizers.Optimizer,
                                tf.keras.optimizers.experimental.Optimizer,
                                tf.keras.optimizers.legacy.Optimizer],
               task_sampler: sampler.TaskSampler,
               trainer_options=None):
    super().__init__(
        multi_task=multi_task,
        multi_task_model=multi_task_model,
        optimizer=optimizer,
        trainer_options=trainer_options)
    self._task_sampler = task_sampler

    # Build per task train step.
    def _get_task_step(task_name, task):

      def step_fn(inputs):
        if isinstance(self.multi_task_model, base_model.MultiTaskBaseModel):
          task_model = self.multi_task_model.sub_tasks[task_name]
        else:
          task_model = self.multi_task_model
        task_logs = task.train_step(
            inputs,
            model=task_model,
            optimizer=self.optimizer,
            metrics=self.training_metrics[task_name])
        self.training_losses[task_name].update_state(task_logs[task.loss])

      return step_fn

    self._task_train_step_map = {
        name: _get_task_step(name, task)
        for name, task in self.multi_task.tasks.items()
    }

    # TODO(haozhangthu): Add taskwise step counter to train_loop_end for logging
    # on TensorBoard.
    self._task_step_counters = {
        name: orbit.utils.create_global_step() for name in self.multi_task.tasks
    }

    # If the new Keras optimizer is used, we require all model variables are
    # created before the training and let the optimizer to create the slot
    # variable all together.
    if isinstance(optimizer, tf.keras.optimizers.experimental.Optimizer):
      multi_task_model.build()
      optimizer.build(multi_task_model.trainable_variables)

  def task_step_counter(self, name):
    return self._task_step_counters[name]

  def train_step(self, iterator_map):
    # Sample one task to train according to a multinomial distribution
    rn = tf.random.stateless_uniform(shape=[], seed=(0, self.global_step))
    cumulative_sample_distribution = self._task_sampler.task_cumulative_distribution(
        self.global_step)
    # Prepend a [0.0] for indexing convenience.
    cumulative_sample_distribution = tf.concat(
        [tf.constant([0.0], dtype=tf.float32), cumulative_sample_distribution],
        axis=0)

    for idx, (name, _) in enumerate(self.multi_task.tasks.items()):
      begin = cumulative_sample_distribution[idx]
      end = cumulative_sample_distribution[idx + 1]
      if rn >= begin and rn < end:
        self._strategy.run(
            self._task_train_step_map[name], args=(next(iterator_map[name]),))
        self.global_step.assign_add(1)
        self.task_step_counter(name).assign_add(1)

  def train_loop_end(self):
    """Record loss and metric values per task."""
    result = super().train_loop_end()
    # Interleaving training does not have a good semantic for `total_loss`. In
    # fact, it is always zero. To avoid confusion, we filter the `total_loss`
    # from the result logs.
    if 'total_loss' in result:
      result.pop('total_loss')
    return result
