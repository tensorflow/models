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

"""Runs prediction to generate submission files for GLUE tasks."""
import functools
import json
import os
import pprint

from absl import app
from absl import flags
from absl import logging

import gin
import tensorflow as tf, tf_keras

from official.common import distribute_utils
# Imports registered experiment configs.
from official.common import registry_imports  # pylint: disable=unused-import
from official.core import exp_factory
from official.core import task_factory
from official.core import train_lib
from official.core import train_utils
from official.modeling.hyperparams import params_dict
from official.nlp.finetuning import binary_helper
from official.nlp.finetuning.glue import flags as glue_flags


# Device configs.
flags.DEFINE_string('distribution_strategy', 'tpu',
                    'The Distribution Strategy to use for training.')
flags.DEFINE_string(
    'tpu', '',
    'The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')
flags.DEFINE_integer('num_gpus', 1, 'The number of GPUs to use at each worker.')

_MODE = flags.DEFINE_enum(
    'mode', 'train_eval_and_predict',
    ['train_eval_and_predict', 'train_eval', 'predict'],
    'The mode to run the binary. If `train_eval_and_predict` '
    'it will (1) train on the training data and (2) evaluate on '
    'the validation data and (3) finally generate predictions '
    'on the prediction data; if `train_eval`, it will only '
    'run training and evaluation; if `predict`, it will only '
    'run prediction using the model in `model_dir`.')

# TODO(kitsing) The `params_override` flag is currently not being used.
# Only declared to make xm_job_3p.XMTPUJob happy.
_PARAMS_OVERRIDE = flags.DEFINE_string(
    'params_override', '', 'Overridden parameters.'
)

FLAGS = flags.FLAGS

EXPERIMENT_TYPE = 'bert/sentence_prediction'
BEST_CHECKPOINT_EXPORT_SUBDIR = 'best_ckpt'

EVAL_METRIC_MAP = {
    'AX': 'matthews_corrcoef',
    'COLA': 'matthews_corrcoef',
    'MNLI': 'cls_accuracy',
    'MRPC': 'f1',
    'QNLI': 'cls_accuracy',
    'QQP': 'f1',
    'RTE': 'cls_accuracy',
    'SST-2': 'cls_accuracy',
    'STS-B': 'pearson_spearman_corr',
    'WNLI': 'cls_accuracy',
}

AX_CLASS_NAMES = ['contradiction', 'entailment', 'neutral']
COLA_CLASS_NAMES = ['0', '1']
MNLI_CLASS_NAMES = ['contradiction', 'entailment', 'neutral']
MRPC_CLASS_NAMES = ['0', '1']
QNLI_CLASS_NAMES = ['entailment', 'not_entailment']
QQP_CLASS_NAMES = ['0', '1']
RTE_CLASS_NAMES = ['entailment', 'not_entailment']
SST_2_CLASS_NAMES = ['0', '1']
WNLI_CLASS_NAMES = ['0', '1']


def _override_exp_config_by_file(exp_config, exp_config_files):
  """Overrides an `ExperimentConfig` object by files."""
  for exp_config_file in exp_config_files:
    if not tf.io.gfile.exists(exp_config_file):
      raise ValueError('%s does not exist.' % exp_config_file)
    params_dict.override_params_dict(
        exp_config, exp_config_file, is_strict=True)

  return exp_config


def _override_exp_config_by_flags(exp_config, input_meta_data):
  """Overrides an `ExperimentConfig` object by flags."""
  if FLAGS.task_name in ('AX', 'COLA',):
    override_task_cfg_fn = functools.partial(
        binary_helper.override_sentence_prediction_task_config,
        num_classes=input_meta_data['num_labels'],
        metric_type='matthews_corrcoef')
  elif FLAGS.task_name in ('MNLI', 'QNLI', 'RTE', 'SST-2',
                           'WNLI'):
    override_task_cfg_fn = functools.partial(
        binary_helper.override_sentence_prediction_task_config,
        num_classes=input_meta_data['num_labels'])
  elif FLAGS.task_name in ('QQP', 'MRPC'):
    override_task_cfg_fn = functools.partial(
        binary_helper.override_sentence_prediction_task_config,
        metric_type='f1',
        num_classes=input_meta_data['num_labels'])
  elif FLAGS.task_name in ('STS-B',):
    override_task_cfg_fn = functools.partial(
        binary_helper.override_sentence_prediction_task_config,
        num_classes=1,
        metric_type='pearson_spearman_corr',
        label_type='float')
  else:
    raise ValueError('Task %s not supported.' % FLAGS.task_name)

  binary_helper.override_trainer_cfg(
      exp_config.trainer,
      learning_rate=FLAGS.learning_rate,
      num_epoch=FLAGS.num_epoch,
      global_batch_size=FLAGS.global_batch_size,
      warmup_ratio=FLAGS.warmup_ratio,
      training_data_size=input_meta_data['train_data_size'],
      eval_data_size=input_meta_data['eval_data_size'],
      num_eval_per_epoch=FLAGS.num_eval_per_epoch,
      best_checkpoint_export_subdir=BEST_CHECKPOINT_EXPORT_SUBDIR,
      best_checkpoint_eval_metric=EVAL_METRIC_MAP[FLAGS.task_name],
      best_checkpoint_metric_comp='higher')

  override_task_cfg_fn(
      exp_config.task,
      model_config_file=FLAGS.model_config_file,
      init_checkpoint=FLAGS.init_checkpoint,
      hub_module_url=FLAGS.hub_module_url,
      global_batch_size=FLAGS.global_batch_size,
      train_input_path=FLAGS.train_input_path,
      validation_input_path=FLAGS.validation_input_path,
      seq_length=input_meta_data['max_seq_length'])
  return exp_config


def _get_exp_config(input_meta_data, exp_config_files):
  """Gets an `ExperimentConfig` object."""
  exp_config = exp_factory.get_exp_config(EXPERIMENT_TYPE)

  if exp_config_files:
    logging.info(
        'Loading `ExperimentConfig` from file, and flags will be ignored.')
    exp_config = _override_exp_config_by_file(exp_config, exp_config_files)
  else:
    logging.info('Loading `ExperimentConfig` from flags.')
    exp_config = _override_exp_config_by_flags(exp_config, input_meta_data)

  exp_config.validate()
  exp_config.lock()

  pp = pprint.PrettyPrinter()
  logging.info('Final experiment parameters: %s',
               pp.pformat(exp_config.as_dict()))

  return exp_config


def _write_submission_file(task, seq_length):
  """Writes submission files that can be uploaded to the leaderboard."""
  tf.io.gfile.makedirs(os.path.dirname(FLAGS.test_output_path))
  model = task.build_model()

  ckpt_file = tf.train.latest_checkpoint(
      os.path.join(FLAGS.model_dir, BEST_CHECKPOINT_EXPORT_SUBDIR))
  logging.info('Restoring checkpoints from %s', ckpt_file)
  checkpoint = tf.train.Checkpoint(model=model)
  checkpoint.read(ckpt_file).expect_partial()

  write_fn = binary_helper.write_glue_classification
  write_fn_map = {
      'AX':
          functools.partial(
              write_fn, class_names=AX_CLASS_NAMES),
      'COLA':
          functools.partial(
              write_fn, class_names=COLA_CLASS_NAMES),
      'MNLI':
          functools.partial(
              write_fn, class_names=MNLI_CLASS_NAMES),
      'MRPC':
          functools.partial(
              write_fn, class_names=MRPC_CLASS_NAMES),
      'QNLI':
          functools.partial(
              write_fn, class_names=QNLI_CLASS_NAMES),
      'QQP':
          functools.partial(
              write_fn, class_names=QQP_CLASS_NAMES),
      'RTE':
          functools.partial(
              write_fn, class_names=RTE_CLASS_NAMES),
      'SST-2':
          functools.partial(
              write_fn, class_names=SST_2_CLASS_NAMES),
      'STS-B':
          # No class_names (regression), clip predictions to [0.0, 5.0] per glue
          # benchmark grader.
          functools.partial(
              write_fn, class_names=None, label_type='float',
              min_float_value=0.0, max_float_value=5.0),
      'WNLI':
          functools.partial(
              write_fn, class_names=WNLI_CLASS_NAMES),
  }
  logging.info('Predicting %s', FLAGS.test_input_path)
  write_fn_map[FLAGS.task_name](
      task=task,
      model=model,
      input_file=FLAGS.test_input_path,
      output_file=FLAGS.test_output_path,
      predict_batch_size=(
          task.task_config.train_data.global_batch_size),
      seq_length=seq_length)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  glue_flags.validate_flags(FLAGS, file_exists_fn=tf.io.gfile.exists)

  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
  distribution_strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy,
      num_gpus=FLAGS.num_gpus,
      tpu_address=FLAGS.tpu)

  with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))

  with distribution_strategy.scope():
    task = None
    if 'train_eval' in _MODE.value:
      logging.info('Starting training and eval...')
      logging.info('Model dir: %s', FLAGS.model_dir)

      exp_config = _get_exp_config(
          input_meta_data=input_meta_data,
          exp_config_files=FLAGS.config_file)
      train_utils.serialize_config(exp_config, FLAGS.model_dir)
      task = task_factory.get_task(exp_config.task, logging_dir=FLAGS.model_dir)
      train_lib.run_experiment(
          distribution_strategy=distribution_strategy,
          task=task,
          mode='train_and_eval',
          params=exp_config,
          model_dir=FLAGS.model_dir)

    if 'predict' in _MODE.value:
      logging.info('Starting predict...')
      # When mode is `predict`, `task` will be None.
      if task is None:
        exp_config = _get_exp_config(
            input_meta_data=input_meta_data,
            exp_config_files=[os.path.join(FLAGS.model_dir, 'params.yaml')])
        task = task_factory.get_task(
            exp_config.task, logging_dir=FLAGS.model_dir)
      _write_submission_file(task, input_meta_data['max_seq_length'])


if __name__ == '__main__':
  glue_flags.define_flags()
  flags.mark_flag_as_required('mode')
  flags.mark_flag_as_required('task_name')
  app.run(main)
