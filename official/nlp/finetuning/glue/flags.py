# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Common flags for GLUE finetuning binary."""
from typing import Callable

from absl import flags
from absl import logging


def define_flags():
  """Defines flags."""

  # ===========================================================================
  # Glue binary flags.
  # ===========================================================================
  flags.DEFINE_enum(
      'mode', 'train_eval_and_predict',
      ['train_eval_and_predict', 'train_eval', 'predict'],
      'The mode to run the binary. If `train_eval_and_predict` '
      'it will (1) train on the training data and (2) evaluate on '
      'the validation data and (3) finally generate predictions '
      'on the prediction data; if `train_eval`, it will only '
      'run training and evaluation; if `predict`, it will only '
      'run prediction using the model in `model_dir`.')

  flags.DEFINE_enum('task_name', None, [
      'AX', 'COLA', 'MNLI', 'MRPC', 'QNLI', 'QQP', 'RTE', 'SST-2', 'STS-B',
      'WNLI'
  ], 'The type of GLUE task.')

  flags.DEFINE_string('train_input_path', None,
                      'The file path to the training data.')

  flags.DEFINE_string('validation_input_path', None,
                      'The file path to the evaluation data.')

  flags.DEFINE_string('test_input_path', None,
                      'The file path to the test input data.')

  flags.DEFINE_string('test_output_path', None,
                      'The file path to the test output data.')

  flags.DEFINE_string('model_dir', '', 'The model directory containing '
                      'subdirectories for each task. Only needed for "predict" '
                      'mode. For all other modes, if not provided, a unique '
                      'directory will be created automatically for each run.')

  flags.DEFINE_string(
      'input_meta_data_path', None, 'Path to file that contains '
      'metadata about input file. It is output by the `create_finetuning_data` '
      'binary. Required for all modes except "predict".')

  flags.DEFINE_string('init_checkpoint', '',
                      'Initial checkpoint from a pre-trained BERT model.')

  flags.DEFINE_string(
      'model_config_file', '', 'The config file specifying the architecture '
      'of the pre-trained model. The file can be either a bert_config.json '
      'file or `encoders.EncoderConfig` in yaml file.')

  flags.DEFINE_string(
      'hub_module_url', '', 'TF-Hub path/url to a pretrained model. If '
      'specified, `init_checkpoint` and `model_config_file` flag should not be '
      'used.')

  flags.DEFINE_multi_string('gin_file', None,
                            'List of paths to the gin config files.')

  flags.DEFINE_multi_string('gin_params', None,
                            'Newline separated list of gin parameter bindings.')

  flags.DEFINE_multi_string(
      'config_file', None, 'This is the advanced usage to specify the '
      '`ExperimentConfig` directly. When specified, '
      'we will ignore FLAGS related to `ExperimentConfig` such as '
      '`train_input_path`, `validation_input_path` and following hparams.')

  # ===========================================================================
  # Tuning hparams.
  # ===========================================================================
  flags.DEFINE_integer('global_batch_size', 32,
                       'Global batch size for train/eval/predict.')

  flags.DEFINE_float('learning_rate', 3e-5, 'Initial learning rate.')

  flags.DEFINE_integer('num_epoch', 3, 'Number of training epochs.')

  flags.DEFINE_float('warmup_ratio', 0.1,
                     'Proportion of learning rate warmup steps.')

  flags.DEFINE_integer('num_eval_per_epoch', 2,
                       'Number of evaluations to run per epoch.')


def validate_flags(flags_obj: flags.FlagValues,
                   file_exists_fn: Callable[[str], bool]):
  """Raises ValueError if any flags are misconfigured.

  Args:
    flags_obj: A `flags.FlagValues` object, usually from `flags.FLAG`.
    file_exists_fn: A callable to decide if a file path exists or not.
  """

  def _check_path_exists(flag_path, flag_name):
    if not file_exists_fn(flag_path):
      raise ValueError('Flag `%s` at %s does not exist.' %
                       (flag_name, flag_path))

  def _validate_path(flag_path, flag_name):
    if not flag_path:
      raise ValueError('Flag `%s` must be provided in mode %s.' %
                       (flag_name, flags_obj.mode))
    _check_path_exists(flag_path, flag_name)

  if 'train' in flags_obj.mode:
    _validate_path(flags_obj.train_input_path, 'train_input_path')
    _validate_path(flags_obj.input_meta_data_path, 'input_meta_data_path')

    if flags_obj.gin_file:
      for gin_file in flags_obj.gin_file:
        _check_path_exists(gin_file, 'gin_file')
    if flags_obj.config_file:
      for config_file in flags_obj.config_file:
        _check_path_exists(config_file, 'config_file')

  if 'eval' in flags_obj.mode:
    _validate_path(flags_obj.validation_input_path, 'validation_input_path')

  if flags_obj.mode == 'predict':
    # model_dir is only needed strictly in 'predict' mode.
    _validate_path(flags_obj.model_dir, 'model_dir')

  if 'predict' in flags_obj.mode:
    _validate_path(flags_obj.test_input_path, 'test_input_path')

  if not flags_obj.config_file and flags_obj.mode != 'predict':
    if flags_obj.hub_module_url:
      if flags_obj.init_checkpoint or flags_obj.model_config_file:
        raise ValueError(
            'When `hub_module_url` is specified, `init_checkpoint` and '
            '`model_config_file` should be empty.')
      logging.info(
          'Using the pretrained tf.hub from %s', flags_obj.hub_module_url)
    else:
      if not (flags_obj.init_checkpoint and flags_obj.model_config_file):
        raise ValueError('Both `init_checkpoint` and `model_config_file` '
                         'should be specified if `config_file` is not '
                         'specified.')
      _validate_path(flags_obj.model_config_file, 'model_config_file')
      logging.info(
          'Using the pretrained checkpoint from %s and model_config_file from '
          '%s.', flags_obj.init_checkpoint, flags_obj.model_config_file)
