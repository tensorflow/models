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

"""Training utils."""
import copy
import json
import os
import pprint
from typing import List, Optional

from absl import logging
import dataclasses
import gin
import orbit
import tensorflow as tf

from official.core import base_task
from official.core import base_trainer
from official.core import config_definitions
from official.core import exp_factory
from official.modeling import hyperparams


class BestCheckpointExporter:
  """Keeps track of the best result, and saves its checkpoint.

  Orbit will support an API for checkpoint exporter. This class will be used
  together with orbit once this functionality is ready.
  """

  def __init__(self, export_dir: str, metric_name: str, metric_comp: str):
    """Initialization.

    Args:
      export_dir: The directory that will contain exported checkpoints.
      metric_name: Indicates which metric to look at, when determining which
        result is better.
      metric_comp: Indicates how to compare results. Either `lower` or `higher`.
    """
    self._export_dir = export_dir
    self._metric_name = metric_name
    self._metric_comp = metric_comp
    if self._metric_comp not in ('lower', 'higher'):
      raise ValueError('best checkpoint metric comp must be one of '
                       'higher, lower. Got: {}'.format(self._metric_comp))
    tf.io.gfile.makedirs(os.path.dirname(self.best_ckpt_logs_path))
    self._best_ckpt_logs = self._maybe_load_best_eval_metric()

  def maybe_export_checkpoint(self, checkpoint, eval_logs, global_step):
    logging.info('[BestCheckpointExporter] received eval_logs: %s, at step: %d',
                 eval_logs, global_step)
    if self._best_ckpt_logs is None or self._new_metric_is_better(
        self._best_ckpt_logs, eval_logs):
      self._best_ckpt_logs = eval_logs
      self._export_best_eval_metric(checkpoint, self._best_ckpt_logs,
                                    global_step)

  def _maybe_load_best_eval_metric(self):
    if not tf.io.gfile.exists(self.best_ckpt_logs_path):
      return None
    with tf.io.gfile.GFile(self.best_ckpt_logs_path, 'r') as reader:
      return json.loads(reader.read())

  def _new_metric_is_better(self, old_logs, new_logs):
    """Check if the metric in new_logs is better than the metric in old_logs."""
    if self._metric_name not in old_logs or self._metric_name not in new_logs:
      raise KeyError('best checkpoint eval metric name {} is not valid. '
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
      eval_logs_ext[name] = float(orbit.utils.get_value(value))
    # Saving json file is very fast.
    with tf.io.gfile.GFile(self.best_ckpt_logs_path, 'w') as writer:
      writer.write(json.dumps(eval_logs_ext, indent=4) + '\n')

    # Saving the best checkpoint might be interrupted if the job got killed.
    for file_to_remove in tf.io.gfile.glob(self.best_ckpt_path + '*'):
      tf.io.gfile.remove(file_to_remove)
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


@gin.configurable
def create_trainer(params: config_definitions.ExperimentConfig,
                   task: base_task.Task,
                   train: bool,
                   evaluate: bool,
                   checkpoint_exporter: Optional[BestCheckpointExporter] = None,
                   trainer_cls=base_trainer.Trainer) -> base_trainer.Trainer:
  """Create trainer."""
  logging.info('Running default trainer.')
  model = task.build_model()
  optimizer = task.create_optimizer(params.trainer.optimizer_config,
                                    params.runtime)
  return trainer_cls(
      params,
      task,
      model=model,
      optimizer=optimizer,
      train=train,
      evaluate=evaluate,
      checkpoint_exporter=checkpoint_exporter)


@dataclasses.dataclass
class ParseConfigOptions:
  """Use this dataclass instead of FLAGS to customize parse_configuration()."""
  experiment: str
  config_file: List[str]
  tpu: str = ''
  tf_data_service: str = ''
  params_override: str = ''


def parse_configuration(flags_obj, lock_return=True, print_return=True):
  """Parses ExperimentConfig from flags."""

  # 1. Get the default config from the registered experiment.
  params = exp_factory.get_exp_config(flags_obj.experiment)

  # 2. Get the first level of override from `--config_file`.
  #    `--config_file` is typically used as a template that specifies the common
  #    override for a particular experiment.
  for config_file in flags_obj.config_file or []:
    params = hyperparams.override_params_dict(
        params, config_file, is_strict=True)

  # 3. Override the TPU address and tf.data service address.
  params.override({
      'runtime': {
          'tpu': flags_obj.tpu,
      },
  })
  if flags_obj.tf_data_service and isinstance(params.task,
                                              config_definitions.TaskConfig):
    params.override({
        'task': {
            'train_data': {
                'tf_data_service_address': flags_obj.tf_data_service,
            },
            'validation_data': {
                'tf_data_service_address': flags_obj.tf_data_service,
            }
        }
    })

  # 4. Get the second level of override from `--params_override`.
  #    `--params_override` is typically used as a further override over the
  #    template. For example, one may define a particular template for training
  #    ResNet50 on ImageNet in a config file and pass it via `--config_file`,
  #    then define different learning rates and pass it via `--params_override`.
  if flags_obj.params_override:
    params = hyperparams.override_params_dict(
        params, flags_obj.params_override, is_strict=True)

  params.validate()
  if lock_return:
    params.lock()

  if print_return:
    pp = pprint.PrettyPrinter()
    logging.info('Final experiment parameters: %s',
                 pp.pformat(params.as_dict()))

  return params


def serialize_config(params: config_definitions.ExperimentConfig,
                     model_dir: str):
  """Serializes and saves the experiment config."""
  params_save_path = os.path.join(model_dir, 'params.yaml')
  logging.info('Saving experiment configuration to %s', params_save_path)
  tf.io.gfile.makedirs(model_dir)
  hyperparams.save_params_dict_to_yaml(params, params_save_path)


def read_global_step_from_checkpoint(ckpt_file_path):
  """Read global step from checkpoint, or get global step from its filename."""
  global_step = tf.Variable(-1, dtype=tf.int64)
  ckpt = tf.train.Checkpoint(global_step=global_step)
  try:
    ckpt.restore(ckpt_file_path).expect_partial()
    global_step_maybe_restored = global_step.numpy()
  except tf.errors.InvalidArgumentError:
    global_step_maybe_restored = -1

  if global_step_maybe_restored == -1:
    raise ValueError('global_step not found in checkpoint {}. '
                     'If you want to run finetune eval jobs, you need to '
                     'make sure that your pretrain model writes '
                     'global_step in its checkpoints.'.format(ckpt_file_path))
  global_step_restored = global_step.numpy()
  logging.info('get global_step %d from checkpoint %s', global_step_restored,
               ckpt_file_path)
  return global_step_restored


def write_json_summary(log_dir, global_step, eval_metrics):
  """Dump evaluation metrics to json file."""
  serializable_dict = {}
  for name, value in eval_metrics.items():
    if hasattr(value, 'numpy'):
      serializable_dict[name] = str(value.numpy())
    else:
      serializable_dict[name] = str(value)
  output_json = os.path.join(log_dir, 'metrics-{}.json'.format(global_step))
  logging.info('Evaluation results at pretrain step %d: %s', global_step,
               serializable_dict)
  with tf.io.gfile.GFile(output_json, 'w') as writer:
    writer.write(json.dumps(serializable_dict, indent=4) + '\n')


def write_summary(summary_writer, global_step, eval_metrics):
  """Write evaluation metrics to TF summary."""
  numeric_dict = {}
  for name, value in eval_metrics.items():
    numeric_dict[name] = float(orbit.utils.get_value(value))
  with summary_writer.as_default():
    for name, value in numeric_dict.items():
      tf.summary.scalar(name, value, step=global_step)
    summary_writer.flush()


def remove_ckpts(model_dir):
  """Remove model checkpoints, so we can restart."""
  ckpts = os.path.join(model_dir, 'ckpt-*')
  logging.info('removing checkpoint files %s', ckpts)
  for file_to_remove in tf.io.gfile.glob(ckpts):
    tf.io.gfile.rmtree(file_to_remove)

  file_to_remove = os.path.join(model_dir, 'checkpoint')
  if tf.io.gfile.exists(file_to_remove):
    tf.io.gfile.remove(file_to_remove)
