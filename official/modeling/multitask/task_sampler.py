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

"""Utils to sample tasks for interleaved optimization."""
import abc
from typing import Union, Dict, Text
import tensorflow as tf

from official.modeling.multitask import configs


class TaskSampler(tf.Module, metaclass=abc.ABCMeta):
  """An abstract class defining task sampling API for interleaving trainer."""

  def __init__(self, task_weights: Dict[Text, Union[float, int]]):
    self._task_weights = task_weights

  @property
  def task_weights(self):
    return self._task_weights

  @abc.abstractmethod
  def task_cumulative_distribution(self, global_step: tf.Tensor) -> tf.Tensor:
    """Compute cumulative distribution to sample tasks.

    It calculates the cumulative distribution of the multinomial task
    distribution with respect to which to be sampled against.

    Args:
      global_step: A tensor indicating current progess of training.

    Returns:
      A float tensor with shape (#(task), 1) that represents the cumulative
        sampling distribution.
    """
    pass


class UniformTaskSampler(TaskSampler):
  """Sample all tasks uniformly."""

  def __init__(self, task_weights: Dict[Text, Union[float, int]]):
    super(UniformTaskSampler, self).__init__(task_weights=task_weights)
    self._uniform_cumulative = tf.math.cumsum(
        tf.constant(
            [1.0 / len(self._task_weights)] * len(self._task_weights),
            dtype=tf.float32))

  def task_cumulative_distribution(self, global_step: tf.Tensor) -> tf.Tensor:
    del global_step
    return self._uniform_cumulative


class ProportionalTaskSampler(TaskSampler):
  """Sample tasks proportional to task weights."""

  def __init__(self,
               task_weights: Dict[Text, Union[float, int]],
               alpha: float = 1.0):
    super(ProportionalTaskSampler, self).__init__(task_weights=task_weights)
    self._alpha = tf.cast(alpha, dtype=tf.float32)
    task_weight_dict_ordered_list = tf.constant(
        [weight for _, weight in self._task_weights.items()], dtype=tf.float32)
    task_sizes = tf.math.pow(task_weight_dict_ordered_list, self._alpha)
    task_distribution = task_sizes / tf.reduce_sum(task_sizes)
    self._porportional_cumulative = tf.math.cumsum(task_distribution)

  def task_cumulative_distribution(self, global_step: tf.Tensor) -> tf.Tensor:
    del global_step
    return self._porportional_cumulative


class AnnealingTaskSampler(TaskSampler):
  """Sample tasks according to task weights as well as training progress.

  See http://proceedings.mlr.press/v97/stickland19a/stickland19a.pdf
  """

  def __init__(self,
               task_weights: Dict[Text, Union[float, int]],
               steps_per_epoch: int,
               total_steps: int):
    super(AnnealingTaskSampler, self).__init__(task_weights=task_weights)
    self._steps_per_epoch = tf.cast(steps_per_epoch, dtype=tf.float32)
    self._total_epochs = tf.cast(
        total_steps / self._steps_per_epoch, dtype=tf.float32)

  def task_cumulative_distribution(self, global_step: tf.Tensor) -> tf.Tensor:
    cur_epoch = tf.math.floor(
        tf.cast(global_step, dtype=tf.float32) / self._steps_per_epoch)
    alpha = 1.0 - 0.8 * (cur_epoch - 1) / (self._total_epochs - 1 + 1e-10)
    task_weight_dict_ordered_list = [
        weight for _, weight in self._task_weights.items()
    ]
    task_sizes = tf.math.pow(
        tf.constant(task_weight_dict_ordered_list, dtype=tf.float32),
        tf.cast(alpha, dtype=tf.float32))
    dynamic_task_distribution = task_sizes / tf.reduce_sum(task_sizes)
    return tf.math.cumsum(dynamic_task_distribution)


def get_task_sampler(config: configs.TaskSamplingConfig,
                     task_weights: Dict[Text, float]) -> TaskSampler:
  """Utils to create task sampler with configuration and task weights."""
  oneof_config = config.get()
  if config.type == 'uniform':
    return UniformTaskSampler(task_weights=task_weights)
  elif config.type == 'proportional':
    return ProportionalTaskSampler(
        task_weights=task_weights, alpha=oneof_config.alpha)
  elif config.type == 'annealing':
    return AnnealingTaskSampler(
        task_weights=task_weights,
        steps_per_epoch=oneof_config.steps_per_epoch,
        total_steps=oneof_config.total_steps)
  else:
    raise RuntimeError('Task sampler type not supported')
