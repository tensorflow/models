# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Multitask Evaluator implementation.

The evaluator implements the Orbit `AbstractEvaluator` interface.
"""
from typing import Dict, List, Optional, Union
import gin
import orbit
import tensorflow as tf, tf_keras

from official.core import base_task
from official.core import train_utils
from official.modeling.multitask import base_model


@gin.configurable
class MultiTaskEvaluator(orbit.AbstractEvaluator):
  """Implements the common trainer shared for TensorFlow models."""

  def __init__(
      self,
      eval_tasks: List[base_task.Task],
      model: Union[tf_keras.Model, base_model.MultiTaskBaseModel],
      global_step: Optional[tf.Variable] = None,
      eval_steps: Optional[Dict[str, int]] = None,
      checkpoint_exporter: Optional[train_utils.BestCheckpointExporter] = None):
    """Initialize common trainer for TensorFlow models.

    Args:
      eval_tasks: A list of tasks to evaluate.
      model: tf_keras.Model instance.
      global_step: the global step variable.
      eval_steps: a dictionary of steps to run eval keyed by task names.
      checkpoint_exporter: an object that has the `maybe_export_checkpoint`
        interface.
    """
    # Gets the current distribution strategy. If not inside any strategy scope,
    # it gets a single-replica no-op strategy.
    self._strategy = tf.distribute.get_strategy()
    self._tasks = eval_tasks
    self._model = model
    self._global_step = global_step or orbit.utils.create_global_step()
    self._checkpoint_exporter = checkpoint_exporter
    if hasattr(self.model, "checkpoint_items"):
      checkpoint_items = self.model.checkpoint_items
    else:
      checkpoint_items = {}

    self._checkpoint = tf.train.Checkpoint(
        model=self.model,
        global_step=self.global_step,
        **checkpoint_items)

    self._validation_losses = None
    self._validation_metrics = None

    # Builds per-task datasets.
    self.eval_datasets = {}
    self.eval_steps = eval_steps or {}
    for task in self.tasks:
      self.eval_datasets[task.name] = orbit.utils.make_distributed_dataset(
          self.strategy, task.build_inputs, task.task_config.validation_data)

    # Builds per-task validation loops.
    def get_function(task_name, task):

      task_metrics = self.validation_metrics[task_name]
      task_loss = self.validation_losses[task_name]
      if isinstance(self.model, base_model.MultiTaskBaseModel):
        model = self.model.sub_tasks[task_name]
      else:
        model = self.model

      def step_fn(inputs):
        logs = task.validation_step(inputs, model=model, metrics=task_metrics)
        task_loss.update_state(logs[task.loss])
        return logs

      @tf.function
      def eval_step_fn(iterator):
        distributed_outputs = self.strategy.run(step_fn, args=(next(iterator),))
        return tf.nest.map_structure(self.strategy.experimental_local_results,
                                     distributed_outputs)

      return orbit.utils.create_loop_fn(eval_step_fn)

    self.task_fns = {
        task.name: get_function(task.name, task) for task in self.tasks
    }

  @property
  def strategy(self):
    return self._strategy

  @property
  def tasks(self):
    return self._tasks

  @property
  def model(self):
    return self._model

  @property
  def global_step(self):
    return self._global_step

  @property
  def validation_losses(self):
    """Accesses the validation loss metric object."""
    if self._validation_losses is None:
      # Builds the per-task metrics and losses.
      self._validation_losses = {}
      for task in self.tasks:
        self._validation_losses[task.name] = tf_keras.metrics.Mean(
            "validation_loss", dtype=tf.float32)
    return self._validation_losses

  @property
  def validation_metrics(self):
    """Accesses all validation metric metric objects."""
    if self._validation_metrics is None:
      # Builds the per-task metrics and losses.
      self._validation_metrics = {}
      for task in self.tasks:
        self._validation_metrics[task.name] = task.build_metrics(training=False)
    return self._validation_metrics

  @property
  def checkpoint(self):
    """Accesses the training checkpoint."""
    return self._checkpoint

  def evaluate(self, num_steps: tf.Tensor):
    """Performs evaluation for each `EvalTask`."""
    for metric in self.validation_losses.values():
      metric.reset_states()
    for metrics in self.validation_metrics.values():
      for metric in metrics:
        metric.reset_states()
    results = {}
    eval_iters = tf.nest.map_structure(iter, self.eval_datasets)

    for task in self.tasks:
      outputs = None
      name = task.name
      eval_iter = eval_iters[name]
      task_eval_steps = self.eval_steps.get(name, None) or num_steps
      outputs = self.task_fns[name](
          eval_iter,
          task_eval_steps,
          state=outputs,
          reduce_fn=task.aggregate_logs)
      task_metrics = self.validation_metrics[name]
      task_loss = self.validation_losses[name]
      logs = {}
      for metric in task_metrics + [task_loss]:
        logs[metric.name] = metric.result()
      if outputs:
        metrics = task.reduce_aggregated_logs(
            outputs, global_step=self.global_step)
        logs.update(metrics)
      results[name] = logs

    if self._checkpoint_exporter:
      self._checkpoint_exporter.maybe_export_checkpoint(
          self.checkpoint, results, self.global_step.numpy())
    return results
