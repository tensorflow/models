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
"""Training utils."""

import json
import os
import pprint
from typing import Any

from absl import logging
import orbit
import tensorflow as tf

from official.core import base_task
from official.core import base_trainer
from official.core import exp_factory
from official.modeling import hyperparams
from official.modeling.hyperparams import config_definitions


def create_trainer(
    params: config_definitions.ExperimentConfig,
    task: base_task.Task,
    model_dir: str,
    train: bool,
    evaluate: bool,
    checkpoint_exporter: Any = None):
  """Create trainer."""
  del model_dir
  logging.info('Running default trainer.')
  trainer = base_trainer.Trainer(
      params, task, train=train, evaluate=evaluate,
      checkpoint_exporter=checkpoint_exporter)
  return trainer


def parse_configuration(flags_obj):
  """Parses ExperimentConfig from flags."""

  # 1. Get the default config from the registered experiment.
  params = exp_factory.get_exp_config(flags_obj.experiment)

  # 2. Get the first level of override from `--config_file`.
  #    `--config_file` is typically used as a template that specifies the common
  #    override for a particular experiment.
  for config_file in flags_obj.config_file or []:
    params = hyperparams.override_params_dict(
        params, config_file, is_strict=True)

  # 3. Override the TPU address.
  params.override({
      'runtime': {
          'tpu': flags_obj.tpu,
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
  params.lock()

  pp = pprint.PrettyPrinter()
  logging.info('Final experiment parameters: %s', pp.pformat(params.as_dict()))

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
  logging.info('get global_step %d from checkpoint %s',
               global_step_restored, ckpt_file_path)
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
  logging.info('Evaluation results at pretrain step %d: %s',
               global_step, serializable_dict)
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
