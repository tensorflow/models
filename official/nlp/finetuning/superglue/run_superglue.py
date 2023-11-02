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

"""Runs prediction to generate submission files for SuperGLUE tasks."""
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
from official.nlp.finetuning.superglue import flags as superglue_flags

# Device configs.
flags.DEFINE_string('distribution_strategy', 'tpu',
                    'The Distribution Strategy to use for training.')
flags.DEFINE_string(
    'tpu', '',
    'The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')
flags.DEFINE_integer('num_gpus', 1, 'The number of GPUs to use at each worker.')

FLAGS = flags.FLAGS

EXPERIMENT_TYPE = 'bert/sentence_prediction'
BEST_CHECKPOINT_EXPORT_SUBDIR = 'best_ckpt'

EVAL_METRIC_MAP = {
    'AX-b': 'matthews_corrcoef',
    'CB': 'cls_accuracy',
    'COPA': 'cls_accuracy',
    'MULTIRC': 'exact_match',
    'RTE': 'cls_accuracy',
    'WiC': 'cls_accuracy',
    'WSC': 'cls_accuracy',
    'BoolQ': 'cls_accuracy',
    'ReCoRD': 'cls_accuracy',
    'AX-g': 'cls_accuracy',
}

AXG_CLASS_NAMES = ['entailment', 'not_entailment']
RTE_CLASS_NAMES = ['entailment', 'not_entailment']
CB_CLASS_NAMES = ['entailment', 'neutral', 'contradiction']
BOOLQ_CLASS_NAMES = ['True', 'False']


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
  if FLAGS.task_name in 'AX-b':
    override_task_cfg_fn = functools.partial(
        binary_helper.override_sentence_prediction_task_config,
        num_classes=input_meta_data['num_labels'],
        metric_type='matthews_corrcoef')
  elif FLAGS.task_name in ('CB', 'COPA', 'RTE', 'WiC', 'WSC', 'BoolQ', 'ReCoRD',
                           'AX-g'):
    override_task_cfg_fn = functools.partial(
        binary_helper.override_sentence_prediction_task_config,
        num_classes=input_meta_data['num_labels'])
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

  write_fn = binary_helper.write_superglue_classification
  write_fn_map = {
      'RTE': functools.partial(write_fn, class_names=RTE_CLASS_NAMES),
      'AX-g': functools.partial(write_fn, class_names=AXG_CLASS_NAMES),
      'CB': functools.partial(write_fn, class_names=CB_CLASS_NAMES),
      'BoolQ': functools.partial(write_fn, class_names=BOOLQ_CLASS_NAMES)
  }
  logging.info('Predicting %s', FLAGS.test_input_path)
  write_fn_map[FLAGS.task_name](
      task=task,
      model=model,
      input_file=FLAGS.test_input_path,
      output_file=FLAGS.test_output_path,
      predict_batch_size=(task.task_config.train_data.global_batch_size),
      seq_length=seq_length)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  superglue_flags.validate_flags(FLAGS, file_exists_fn=tf.io.gfile.exists)

  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_params)
  distribution_strategy = distribute_utils.get_distribution_strategy(
      distribution_strategy=FLAGS.distribution_strategy,
      num_gpus=FLAGS.num_gpus,
      tpu_address=FLAGS.tpu)

  with tf.io.gfile.GFile(FLAGS.input_meta_data_path, 'rb') as reader:
    input_meta_data = json.loads(reader.read().decode('utf-8'))

  with distribution_strategy.scope():
    task = None
    if 'train_eval' in FLAGS.mode:
      logging.info('Starting training and eval...')
      logging.info('Model dir: %s', FLAGS.model_dir)

      exp_config = _get_exp_config(
          input_meta_data=input_meta_data, exp_config_files=FLAGS.config_file)
      train_utils.serialize_config(exp_config, FLAGS.model_dir)
      task = task_factory.get_task(exp_config.task, logging_dir=FLAGS.model_dir)
      train_lib.run_experiment(
          distribution_strategy=distribution_strategy,
          task=task,
          mode='train_and_eval',
          params=exp_config,
          model_dir=FLAGS.model_dir)

    if 'predict' in FLAGS.mode:
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
  superglue_flags.define_flags()
  flags.mark_flag_as_required('mode')
  flags.mark_flag_as_required('task_name')
  app.run(main)
