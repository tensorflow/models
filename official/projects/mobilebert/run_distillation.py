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

# pylint: disable=line-too-long
"""Creating the task and start trainer."""

import pprint

from absl import app
from absl import flags
from absl import logging
import gin
from official.common import distribute_utils
from official.common import flags as tfm_flags
from official.core import config_definitions as cfg
from official.core import train_utils
from official.modeling import hyperparams
from official.modeling import optimization
from official.modeling import performance
from official.modeling.fast_training.progressive import train_lib
from official.modeling.fast_training.progressive import trainer as prog_trainer_lib
from official.nlp.data import pretrain_dataloader
from official.projects.mobilebert import distillation


FLAGS = flags.FLAGS

optimization_config = optimization.OptimizationConfig(
    optimizer=optimization.OptimizerConfig(
        type='lamb',
        lamb=optimization.LAMBConfig(
            weight_decay_rate=0.01,
            exclude_from_weight_decay=['LayerNorm', 'bias', 'norm'],
            clipnorm=1.0)),
    learning_rate=optimization.LrConfig(
        type='polynomial',
        polynomial=optimization.PolynomialLrConfig(
            initial_learning_rate=1.5e-3,
            decay_steps=10000,
            end_learning_rate=1.5e-3)),
    warmup=optimization.WarmupConfig(
        type='linear',
        linear=optimization.LinearWarmupConfig(warmup_learning_rate=0)))


# copy from progressive/utils.py due to the private visibility issue.
def config_override(params, flags_obj):
  """Override ExperimentConfig according to flags."""
  # Change runtime.tpu to the real tpu.
  params.override({
      'runtime': {
          'tpu': flags_obj.tpu,
      }
  })

  # Get the first level of override from `--config_file`.
  #   `--config_file` is typically used as a template that specifies the common
  #   override for a particular experiment.
  for config_file in flags_obj.config_file or []:
    params = hyperparams.override_params_dict(
        params, config_file, is_strict=True)

  # Get the second level of override from `--params_override`.
  #   `--params_override` is typically used as a further override over the
  #   template. For example, one may define a particular template for training
  #   ResNet50 on ImageNet in a config file and pass it via `--config_file`,
  #   then define different learning rates and pass it via `--params_override`.
  if flags_obj.params_override:
    params = hyperparams.override_params_dict(
        params, flags_obj.params_override, is_strict=True)

  params.validate()
  params.lock()

  pp = pprint.PrettyPrinter()
  logging.info('Final experiment parameters: %s', pp.pformat(params.as_dict()))

  model_dir = flags_obj.model_dir
  if 'train' in flags_obj.mode:
    # Pure eval modes do not output yaml files. Otherwise continuous eval job
    # may race against the train job for writing the same file.
    train_utils.serialize_config(params, model_dir)

  return params


def get_exp_config():
  """Get ExperimentConfig."""
  params = cfg.ExperimentConfig(
      task=distillation.BertDistillationTaskConfig(
          train_data=pretrain_dataloader.BertPretrainDataConfig(),
          validation_data=pretrain_dataloader.BertPretrainDataConfig(
              is_training=False)),
      trainer=prog_trainer_lib.ProgressiveTrainerConfig(
          progressive=distillation.BertDistillationProgressiveConfig(),
          optimizer_config=optimization_config,
          train_steps=740000,
          checkpoint_interval=20000))

  return config_override(params, FLAGS)


def main(_):
  logging.info('Parsing config files...')
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
  params = get_exp_config()

  # Sets mixed_precision policy. Using 'mixed_float16' or 'mixed_bfloat16'
  # can have significant impact on model speeds by utilizing float16 in case of
  # GPUs, and bfloat16 in the case of TPUs. loss_scale takes effect only when
  # dtype is float16
  if params.runtime.mixed_precision_dtype:
    performance.set_mixed_precision_policy(params.runtime.mixed_precision_dtype)
  distribution_strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=params.runtime.distribution_strategy,
      all_reduce_alg=params.runtime.all_reduce_alg,
      num_gpus=params.runtime.num_gpus,
      tpu_address=params.runtime.tpu)

  with distribution_strategy.scope():
    task = distillation.BertDistillationTask(
        strategy=distribution_strategy,
        progressive=params.trainer.progressive,
        optimizer_config=params.trainer.optimizer_config,
        task_config=params.task)

  train_lib.run_experiment(
      distribution_strategy=distribution_strategy,
      task=task,
      mode=FLAGS.mode,
      params=params,
      model_dir=FLAGS.model_dir)

if __name__ == '__main__':
  tfm_flags.define_flags()
  app.run(main)
