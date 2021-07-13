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
from typing import Any, Callable, Dict, List, Optional

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


def get_leaf_nested_dict(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
  """Get leaf from a dictionary with arbitrary depth with a list of keys.

  Args:
    d: The dictionary to extract value from.
    keys: The list of keys to extract values recursively.

  Returns:
    The value of the leaf.

  Raises:
    KeyError: If the value of keys extracted is a dictionary.
  """
  leaf = d
  for k in keys:
    if not isinstance(leaf, dict) or k not in leaf:
      raise KeyError(
          'Path not exist while traversing the dictionary: d with keys'
          ': %s.' % keys)
    leaf = leaf[k]

  if isinstance(leaf, dict):
    raise KeyError('The value extracted with keys: %s is not a leaf of the '
                   'dictionary: %s.' % (keys, d))
  return leaf


def cast_leaf_nested_dict(d: Dict[str, Any],
                          cast_fn: Callable[[Any], Any]) -> Dict[str, Any]:
  """Cast the leaves of a dictionary with arbitrary depth in place.

  Args:
    d: The dictionary to extract value from.
    cast_fn: The casting function.

  Returns:
    A dictionray with the same structure as d.
  """
  for key, value in d.items():
    if isinstance(value, dict):
      d[key] = cast_leaf_nested_dict(value, cast_fn)
    else:
      d[key] = cast_fn(value)
  return d


def maybe_create_best_ckpt_exporter(params: config_definitions.ExperimentConfig,
                                    data_dir: str) -> Any:
  """Maybe create a BestCheckpointExporter object, according to the config."""
  export_subdir = params.trainer.best_checkpoint_export_subdir
  metric_name = params.trainer.best_checkpoint_eval_metric
  metric_comp = params.trainer.best_checkpoint_metric_comp
  if data_dir and export_subdir and metric_name:
    best_ckpt_dir = os.path.join(data_dir, export_subdir)
    best_ckpt_exporter = BestCheckpointExporter(best_ckpt_dir, metric_name,
                                                metric_comp)
    logging.info(
        'Created the best checkpoint exporter. '
        'data_dir: %s, export_subdir: %s, metric_name: %s', data_dir,
        export_subdir, metric_name)
  else:
    best_ckpt_exporter = None

  return best_ckpt_exporter


# TODO(b/180147589): Add tests for this module.
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
        result is better. If eval_logs being passed to maybe_export_checkpoint
        is a nested dictionary, use `|` as a seperator for different layers.
      metric_comp: Indicates how to compare results. Either `lower` or `higher`.
    """
    self._export_dir = export_dir
    self._metric_name = metric_name.split('|')
    self._metric_comp = metric_comp
    if self._metric_comp not in ('lower', 'higher'):
      raise ValueError('best checkpoint metric comp must be one of '
                       'higher, lower. Got: {}'.format(self._metric_comp))
    tf.io.gfile.makedirs(os.path.dirname(self.best_ckpt_logs_path))
    self._best_ckpt_logs = self._maybe_load_best_eval_metric()
    self._checkpoint_manager = None

  def _get_checkpoint_manager(self, checkpoint):
    """Gets an existing checkpoint manager or creates a new one."""
    if self._checkpoint_manager is None or (self._checkpoint_manager.checkpoint
                                            != checkpoint):
      logging.info('Creates a new checkpoint manager.')
      self._checkpoint_manager = tf.train.CheckpointManager(
          checkpoint,
          directory=self._export_dir,
          max_to_keep=1,
          checkpoint_name='best_ckpt')

    return self._checkpoint_manager

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
    old_value = float(
        orbit.utils.get_value(
            get_leaf_nested_dict(old_logs, self._metric_name)))
    new_value = float(
        orbit.utils.get_value(
            get_leaf_nested_dict(new_logs, self._metric_name)))

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
    eval_logs_ext = cast_leaf_nested_dict(
        eval_logs_ext, lambda x: float(orbit.utils.get_value(x)))
    # Saving json file is very fast.
    with tf.io.gfile.GFile(self.best_ckpt_logs_path, 'w') as writer:
      writer.write(json.dumps(eval_logs_ext, indent=4) + '\n')

    self._get_checkpoint_manager(checkpoint).save()

  @property
  def best_ckpt_logs(self):
    return self._best_ckpt_logs

  @property
  def best_ckpt_logs_path(self):
    return os.path.join(self._export_dir, 'info.json')

  @property
  def best_ckpt_path(self):
    """Returns the best ckpt path or None if there is no ckpt yet."""
    return tf.train.latest_checkpoint(self._export_dir)


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

  def __contains__(self, name):
    return name in dataclasses.asdict(self)


def parse_configuration(flags_obj, lock_return=True, print_return=True):
  """Parses ExperimentConfig from flags."""

  if flags_obj.experiment is None:
    raise ValueError('The flag --experiment must be specified.')

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
  if ('tf_data_service' in flags_obj and flags_obj.tf_data_service and
      isinstance(params.task, config_definitions.TaskConfig)):
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
    logging.info('Final experiment parameters:\n%s',
                 pp.pformat(params.as_dict()))

  return params


def serialize_config(params: config_definitions.ExperimentConfig,
                     model_dir: str):
  """Serializes and saves the experiment config."""
  if model_dir is None:
    raise ValueError('model_dir must be specified, but got None')
  params_save_path = os.path.join(model_dir, 'params.yaml')
  logging.info('Saving experiment configuration to %s', params_save_path)
  tf.io.gfile.makedirs(model_dir)
  hyperparams.save_params_dict_to_yaml(params, params_save_path)


def save_gin_config(filename_surfix: str, model_dir: str):
  """Serializes and saves the experiment config."""
  gin_save_path = os.path.join(
      model_dir, 'operative_config.{}.gin'.format(filename_surfix))
  logging.info('Saving gin configurations to %s', gin_save_path)
  tf.io.gfile.makedirs(model_dir)
  with tf.io.gfile.GFile(gin_save_path, 'w') as f:
    f.write(gin.operative_config_str())


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


def try_count_params(model: tf.keras.Model):
  """Count the number of parameters if model is possible.

  Args:
    model: Try to count the number of params in this model.

  Returns:
    The number of parameters or None.
  """
  if hasattr(model, 'count_params'):
    try:
      return model.count_params()
    except ValueError:
      logging.info('Number of trainable params unknown, because the build() '
                   'methods in keras layers were not called. This is probably '
                   'because the model was not feed any input, e.g., the max '
                   'train step already reached before this run.')
      return None
  return None
