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

"""Training utils."""

import dataclasses
import inspect
import json
import os
import pprint
from typing import Any, Callable, Dict, List, Optional, Union

from absl import logging
import gin
import numpy as np
import orbit
import tensorflow as tf, tf_keras

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import ops
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
# pylint: enable=g-direct-tensorflow-import
from official.core import base_task
from official.core import base_trainer
from official.core import config_definitions
from official.core import exp_factory
from official.modeling import hyperparams


BEST_CHECKPOINT_NAME = 'best_ckpt'


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


def _filter_leaf_nested_dict(
    d: Dict[str, Any], predicate: Callable[[Any], bool]
) -> Dict[str, Any]:
  """Filters the leaves of a dictionary with arbitrary depth in place.

  Args:
    d: The dictionary to extract value from.
    predicate: A function that will be called on every leave item. When the
      function returns True the leave will be kept. Otherwise the leave will be
      dropped.

  Returns:
    A new dictionray with filtered result.
  """
  result = {}
  for key, value in d.items():
    if isinstance(value, dict):
      result[key] = _filter_leaf_nested_dict(value, predicate)
    elif predicate(value):
      result[key] = value
  return result


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
          checkpoint_name=BEST_CHECKPOINT_NAME)

    return self._checkpoint_manager

  def maybe_export_checkpoint(
      self, checkpoint, eval_logs, global_step, write_logs=True) -> bool:
    """Compare eval_logs with past eval_logs and export checkpoint if better."""
    logging.info('[BestCheckpointExporter] received eval_logs: %s, at step: %d',
                 eval_logs, global_step)
    if self._best_ckpt_logs is None or self._new_metric_is_better(
        self._best_ckpt_logs, eval_logs):
      self._best_ckpt_logs = eval_logs
      if write_logs:
        self.export_best_eval_metric(self._best_ckpt_logs, global_step)
      self._get_checkpoint_manager(checkpoint).save()
      return True
    return False

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

  def export_best_eval_metric(self, eval_logs, global_step):
    """Export evaluation results of the best checkpoint into a json file."""
    # eval_log_ext may contains non-scalar tensors, such as image data when
    # `allow_image_summary` is True. Here we only keep scalar tensors.
    eval_logs_ext = _filter_leaf_nested_dict(
        eval_logs, lambda x: tf.rank(x) <= 1
    )
    eval_logs_ext['best_ckpt_global_step'] = global_step
    eval_logs_ext = cast_leaf_nested_dict(
        eval_logs_ext, lambda x: float(orbit.utils.get_value(x)))
    # Saving json file is very fast.
    with tf.io.gfile.GFile(self.best_ckpt_logs_path, 'w') as writer:
      writer.write(json.dumps(eval_logs_ext, indent=4) + '\n')

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


def create_optimizer(task: base_task.Task,
                     params: config_definitions.ExperimentConfig
                     ) -> tf_keras.optimizers.Optimizer:
  """A create optimizer util to be backward compatability with new args."""
  if 'dp_config' in inspect.signature(task.create_optimizer).parameters:
    dp_config = None
    if hasattr(params.task, 'differential_privacy_config'):
      dp_config = params.task.differential_privacy_config
    optimizer = task.create_optimizer(
        params.trainer.optimizer_config, params.runtime,
        dp_config=dp_config)
  else:
    if hasattr(params.task, 'differential_privacy_config'
              ) and params.task.differential_privacy_config is not None:
      raise ValueError('Differential privacy config is specified but '
                       'task.create_optimizer api does not accept it.')
    optimizer = task.create_optimizer(
        params.trainer.optimizer_config,
        params.runtime)
  return optimizer


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
  optimizer = create_optimizer(task, params)
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


class ExperimentParser:
  """Constructs the Experiment config from Flags or equivalent object.

  Most of the cases, users only need to call the `parse()` function:
  ```
  builder = ExperimentParser(FLAGS)
  params = builder.parse()
  ```

  The advanced users can modify the flow by calling the parse_*() functions
  separately.
  """

  def __init__(self, flags_obj):
    self._flags_obj = flags_obj

  def parse(self):
    """Overrall process of constructing Experiment config."""
    params = self.base_experiment()
    params = self.parse_config_file(params)
    params = self.parse_runtime(params)
    params = self.parse_data_service(params)
    params = self.parse_params_override(params)
    return params

  def base_experiment(self):
    """Get the base experiment config from --experiment field."""
    if self._flags_obj.experiment is None:
      raise ValueError('The flag --experiment must be specified.')
    return exp_factory.get_exp_config(self._flags_obj.experiment)

  def parse_config_file(self, params):
    """Override the configs of params from the config_file."""
    for config_file in self._flags_obj.config_file or []:
      params = hyperparams.override_params_dict(
          params, config_file, is_strict=True)
    return params

  def parse_runtime(self, params):
    """Override the runtime configs of params from flags."""
    # Override the TPU address and tf.data service address.
    params.override({
        'runtime': {
            'tpu': self._flags_obj.tpu,
        },
    })
    return params

  def parse_data_service(self, params):
    """Override the data service configs of params from flags."""
    if ('tf_data_service' in self._flags_obj and
        self._flags_obj.tf_data_service and
        isinstance(params.task, config_definitions.TaskConfig)):
      params.override({
          'task': {
              'train_data': {
                  'tf_data_service_address': self._flags_obj.tf_data_service,
              },
              'validation_data': {
                  'tf_data_service_address': self._flags_obj.tf_data_service,
              }
          }
      })
    return params

  def parse_params_override(self, params):
    # Get the second level of override from `--params_override`.
    # `--params_override` is typically used as a further override over the
    # template. For example, one may define a particular template for training
    # ResNet50 on ImageNet in a config file and pass it via `--config_file`,
    # then define different learning rates and pass it via `--params_override`.
    if self._flags_obj.params_override:
      params = hyperparams.override_params_dict(
          params, self._flags_obj.params_override, is_strict=True)
    return params


def parse_configuration(flags_obj, lock_return=True, print_return=True):
  """Parses ExperimentConfig from flags."""

  params = ExperimentParser(flags_obj).parse()

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


def save_gin_config(filename_suffix: str, model_dir: str):
  """Serializes and saves the experiment config."""
  gin_save_path = os.path.join(
      model_dir, 'operative_config.{}.gin'.format(filename_suffix))
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


def write_model_params(model: Union[tf.Module, tf_keras.Model],
                       output_path: str) -> None:
  """Writes the model parameters and shapes to a file.

  Args:
    model: A model instance.
    output_path: Output file path.
  """
  with tf.io.gfile.GFile(output_path, 'w') as f:
    total_params = 0
    for var in model.variables:
      shape = tf.shape(var)
      total_params += tf.math.reduce_prod(shape).numpy()
      f.write(f'{var.name} {shape.numpy().tolist()}\n')
    f.write(f'\nTotal params: {total_params}\n')


def try_count_params(
    model: Union[tf.Module, tf_keras.Model],
    trainable_only: bool = False):
  """Count the number of parameters if model is possible.

  Args:
    model: Try to count the number of params in this model.
    trainable_only: Whether to calculate trainable params only. This flag is
      not used when the model has `count_params` attribute.

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
  else:
    total_params = 0
    variables = model.trainable_variables if trainable_only else model.variables
    for var in variables:
      shape = tf.shape(var)
      total_params += tf.math.reduce_prod(shape).numpy()
  return total_params


def try_count_flops(model: Union[tf.Module, tf_keras.Model],
                    inputs_kwargs: Optional[Dict[str, Any]] = None,
                    output_path: Optional[str] = None):
  """Counts and returns model FLOPs.

  Args:
    model: A model instance.
    inputs_kwargs: An optional dictionary of argument pairs specifying inputs'
      shape specifications to getting corresponding concrete function.
    output_path: A file path to write the profiling results to.

  Returns:
    The model's FLOPs.
  """
  if hasattr(model, 'inputs'):
    try:
      # Get input shape and set batch size to 1.
      if model.inputs:
        inputs = [
            tf.TensorSpec([1] + input.shape[1:], input.dtype)
            for input in model.inputs
        ]
        concrete_func = tf.function(model).get_concrete_function(inputs)
      # If model.inputs is invalid, try to use the input to get concrete
      # function for model.call (subclass model).
      else:
        concrete_func = tf.function(model.call).get_concrete_function(
            **inputs_kwargs)
      frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)

      # Calculate FLOPs.
      run_meta = tf.compat.v1.RunMetadata()
      opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
      if output_path is not None:
        opts['output'] = f'file:outfile={output_path}'
      else:
        opts['output'] = 'none'
      flops = tf.compat.v1.profiler.profile(
          graph=frozen_func.graph, run_meta=run_meta, options=opts)
      return flops.total_float_ops
    except Exception as e:  # pylint: disable=broad-except
      logging.info(
          'Failed to count model FLOPs with error %s, because the build() '
          'methods in keras layers were not called. This is probably because '
          'the model was not feed any input, e.g., the max train step already '
          'reached before this run.', e)
      return None
  return None


@ops.RegisterStatistics('Einsum', 'flops')
def _einsum_flops(graph, node):
  """Calculates the compute resources needed for Einsum."""
  assert len(node.input) == 2
  x_shape = tf.compat.v1.graph_util.tensor_shape_from_node_def_name(
      graph, node.input[0])
  y_shape = tf.compat.v1.graph_util.tensor_shape_from_node_def_name(
      graph, node.input[1])
  x_shape.assert_is_fully_defined()
  y_shape.assert_is_fully_defined()
  x_shape = x_shape.as_list()
  y_shape = y_shape.as_list()
  equation = str(node.attr['equation'])
  equation = (
      equation.replace('s:', '')
      .replace('"', '')
      .replace(' ', '')
      .replace('\n', '')
  )
  x_str = equation.split(',')[0]
  y_r_str = equation.split(',')[1]
  y_str = y_r_str.split('->')[0]
  r_str = y_r_str.split('->')[1]
  shape_dic = {}
  contracted = set()
  for indice in x_str + y_str:
    if indice in x_str:
      indice_dim = x_shape[x_str.find(indice)]
    elif indice in y_str:
      indice_dim = y_shape[y_str.find(indice)]
    else:
      raise ValueError('indice {} not found in inputs'.format(indice))
    shape_dic[indice] = indice_dim
    if indice not in r_str:
      contracted.add(indice)
  madds = np.prod([shape_dic[indice] for indice in r_str]) * (
      np.prod([shape_dic[indice] for indice in contracted]))
  flops = 2 * madds
  return ops.OpStats('flops', flops)
