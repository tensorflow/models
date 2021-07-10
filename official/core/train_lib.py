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
"""TFM common training driver library."""

import copy
import json
import os
from typing import Any, Mapping, Tuple

# Import libraries
from absl import logging
import orbit
import tensorflow as tf

from official.core import train_utils
from official.core import base_task
from official.modeling.hyperparams import config_definitions


class BestCheckpointExporter:
  """Keeps track of the best result, and saves its checkpoint.

  Orbit will support an API for checkpoint exporter. This class will be used
  together with orbit once this functionality is ready.
  """

  def __init__(self,
               export_dir: str,
               metric_name: str,
               metric_comp: str):
    """Initialization.

    Arguments:
      export_dir: The directory that will contain exported checkpoints.
      metric_name: Indicates which metric to look at, when determining which
        result is better.
      metric_comp: Indicates how to compare results. Either `lower` or `higher`.
    """
    self._export_dir = export_dir
    self._metric_name = metric_name
    self._metric_comp = metric_comp
    if self._metric_comp not in ('lower', 'higher'):
      raise ValueError(
          'best checkpoint metric comp must be one of '
          'higher, lower. Got: {}'.format(self._metric_comp))
    tf.io.gfile.makedirs(os.path.dirname(self.best_ckpt_logs_path))
    self._best_ckpt_logs = self._maybe_load_best_eval_metric()

  def maybe_export_checkpoint(self, checkpoint, eval_logs, global_step):
    logging.info('[BestCheckpointExporter] received eval_logs: %s, at step: %d',
                 eval_logs, global_step)
    if self._best_ckpt_logs is None or self._new_metric_is_better(
        self._best_ckpt_logs, eval_logs):
      self._best_ckpt_logs = eval_logs
      self._export_best_eval_metric(
          checkpoint, self._best_ckpt_logs, global_step)

  def _maybe_load_best_eval_metric(self):
    if not tf.io.gfile.exists(self.best_ckpt_logs_path):
      return None
    with tf.io.gfile.GFile(self.best_ckpt_logs_path, 'r') as reader:
      return json.loads(reader.read())

  def _new_metric_is_better(self, old_logs, new_logs):
    """Check if the metric in new_logs is better than the metric in old_logs."""
    if self._metric_name not in old_logs or self._metric_name not in new_logs:
      raise KeyError(
          'best checkpoint eval metric name {} is not valid. '
          'old_logs: {}, new_logs: {}'.format(
              self._metric_name, old_logs, new_logs))
    old_value = float(orbit.utils.get_value(old_logs[self._metric_name]))
    new_value = float(orbit.utils.get_value(new_logs[self._metric_name]))

    logging.info('[BestCheckpointExporter] comparing results. old: %f, new: %f',
                 old_value, new_value)
    if self._metric_comp == 'higher':
      if new_value > old_value:
        logging.info('[BestCheckpointExporter] '
                     'the new number is better since it is higher.')
        return True
    else:  # self._metric_comp == 'lower':
      if new_value < old_value:
        logging.info('[BestCheckpointExporter] '
                     'the new number is better since it is lower.')
        return True
    return False

  def _export_best_eval_metric(self, checkpoint, eval_logs, global_step):
    """Export evaluation results of the best checkpoint into a json file."""
    eval_logs_ext = copy.copy(eval_logs)
    eval_logs_ext['best_ckpt_global_step'] = global_step
    for name, value in eval_logs_ext.items():
      eval_logs_ext[name] = str(orbit.utils.get_value(value))
    # Saving json file is very fast.
    with tf.io.gfile.GFile(self.best_ckpt_logs_path, 'w') as writer:
      writer.write(json.dumps(eval_logs_ext, indent=4) + '\n')

    # Saving the best checkpoint might be interrupted if the job got killed.
    for file_to_remove in tf.io.gfile.glob(self.best_ckpt_path + '*'):
      tf.io.gfile.rmtree(file_to_remove)
    checkpoint.save(self.best_ckpt_path)

  @property
  def best_ckpt_logs(self):
    return self._best_ckpt_logs

  @property
  def best_ckpt_logs_path(self):
    return os.path.join(self._export_dir, 'info.json')

  @property
  def best_ckpt_path(self):
    return os.path.join(self._export_dir, 'best_ckpt')


def maybe_create_best_ckpt_exporter(
    params: config_definitions.ExperimentConfig,
    data_dir: str) -> Any:
  """Maybe create a BestCheckpointExporter object, according to the config."""
  export_subdir = params.trainer.best_checkpoint_export_subdir
  metric_name = params.trainer.best_checkpoint_eval_metric
  metric_comp = params.trainer.best_checkpoint_metric_comp
  if data_dir and export_subdir and metric_name:
    best_ckpt_dir = os.path.join(data_dir, export_subdir)
    best_ckpt_exporter = BestCheckpointExporter(
        best_ckpt_dir, metric_name, metric_comp)
  else:
    best_ckpt_exporter = None
    logging.info('Not exporting the best checkpoint. '
                 'data_dir: %s, export_subdir: %s, metric_name: %s',
                 data_dir, export_subdir, metric_name)
  return best_ckpt_exporter


def run_experiment(distribution_strategy: tf.distribute.Strategy,
                   task: base_task.Task,
                   mode: str,
                   params: config_definitions.ExperimentConfig,
                   model_dir: str,
                   run_post_eval: bool = False,
                   save_summary: bool = True) \
-> Tuple[tf.keras.Model, Mapping[str, Any]]:
  """Runs train/eval configured by the experiment params.

  Args:
    distribution_strategy: A distribution distribution_strategy.
    task: A Task instance.
    mode: A 'str', specifying the mode. Can be 'train', 'eval', 'train_and_eval'
      or 'continuous_eval'.
    params: ExperimentConfig instance.
    model_dir: A 'str', a path to store model checkpoints and summaries.
    run_post_eval: Whether to run post eval once after training, metrics logs
      are returned.
    save_summary: Whether to save train and validation summary.

  Returns:
    A 2-tuple of (model, eval_logs).
      model: `tf.keras.Model` instance.
      eval_logs: returns eval metrics logs when run_post_eval is set to True,
        otherwise, returns {}.
  """

  with distribution_strategy.scope():
    trainer = train_utils.create_trainer(
        params,
        task,
        model_dir,
        train='train' in mode,
        evaluate=('eval' in mode) or run_post_eval,
        checkpoint_exporter=maybe_create_best_ckpt_exporter(params, model_dir))

  if trainer.checkpoint:
    checkpoint_manager = tf.train.CheckpointManager(
        trainer.checkpoint,
        directory=model_dir,
        max_to_keep=params.trainer.max_to_keep,
        step_counter=trainer.global_step,
        checkpoint_interval=params.trainer.checkpoint_interval,
        init_fn=trainer.initialize)
  else:
    checkpoint_manager = None

  controller = orbit.Controller(
      distribution_strategy,
      trainer=trainer if 'train' in mode else None,
      evaluator=trainer,
      global_step=trainer.global_step,
      steps_per_loop=params.trainer.steps_per_loop,
      checkpoint_manager=checkpoint_manager,
      summary_dir=os.path.join(model_dir, 'train') if (
          save_summary) else None,
      eval_summary_dir=os.path.join(model_dir, 'validation') if (
          save_summary) else None,
      summary_interval=params.trainer.summary_interval if (
          save_summary) else None)

  logging.info('Starts to execute mode: %s', mode)
  with distribution_strategy.scope():
    if mode == 'train':
      controller.train(steps=params.trainer.train_steps)
    elif mode == 'train_and_eval':
      controller.train_and_evaluate(
          train_steps=params.trainer.train_steps,
          eval_steps=params.trainer.validation_steps,
          eval_interval=params.trainer.validation_interval)
    elif mode == 'eval':
      controller.evaluate(steps=params.trainer.validation_steps)
    elif mode == 'continuous_eval':
      controller.evaluate_continuously(
          steps=params.trainer.validation_steps,
          timeout=params.trainer.continuous_eval_timeout)
    else:
      raise NotImplementedError('The mode is not implemented: %s' % mode)

  if run_post_eval:
    with distribution_strategy.scope():
      return trainer.model, trainer.evaluate(
          tf.convert_to_tensor(params.trainer.validation_steps))
  else:
    return trainer.model, {}
