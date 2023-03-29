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

"""Common flags for importing hyperparameters."""

from absl import flags
from official.utils.flags import core as flags_core

FLAGS = flags.FLAGS


def define_gin_flags():
  """Define common gin configurable flags."""
  flags.DEFINE_multi_string('gin_file', None,
                            'List of paths to the config files.')
  flags.DEFINE_multi_string(
      'gin_param', None, 'Newline separated list of Gin parameter bindings.')


def define_common_hparams_flags():
  """Define the common flags across models."""

  flags.DEFINE_string(
      'model_dir',
      default=None,
      help=('The directory where the model and training/evaluation summaries'
            'are stored.'))

  flags.DEFINE_integer(
      'train_batch_size', default=None, help='Batch size for training.')

  flags.DEFINE_integer(
      'eval_batch_size', default=None, help='Batch size for evaluation.')

  flags.DEFINE_string(
      'precision',
      default=None,
      help=('Precision to use; one of: {bfloat16, float32}'))

  flags.DEFINE_string(
      'config_file',
      default=None,
      help=('A YAML file which specifies overrides. Note that this file can be '
            'used as an override template to override the default parameters '
            'specified in Python. If the same parameter is specified in both '
            '`--config_file` and `--params_override`, the one in '
            '`--params_override` will be used finally.'))

  flags.DEFINE_string(
      'params_override',
      default=None,
      help=('a YAML/JSON string or a YAML file which specifies additional '
            'overrides over the default parameters and those specified in '
            '`--config_file`. Note that this is supposed to be used only to '
            'override the model parameters, but not the parameters like TPU '
            'specific flags. One canonical use case of `--config_file` and '
            '`--params_override` is users first define a template config file '
            'using `--config_file`, then use `--params_override` to adjust the '
            'minimal set of tuning parameters, for example setting up different'
            ' `train_batch_size`. '
            'The final override order of parameters: default_model_params --> '
            'params from config_file --> params in params_override.'
            'See also the help message of `--config_file`.'))
  flags.DEFINE_integer('save_checkpoint_freq', None,
                       'Number of steps to save checkpoint.')


def initialize_common_flags():
  """Define the common flags across models."""
  define_common_hparams_flags()

  flags_core.define_device(tpu=True)
  flags_core.define_base(
      num_gpu=True, model_dir=False, data_dir=False, batch_size=False)
  flags_core.define_distribution(worker_hosts=True, task_index=True)
  flags_core.define_performance(all_reduce_alg=True, num_packs=True)

  # Reset the default value of num_gpus to zero.
  FLAGS.num_gpus = 0

  flags.DEFINE_string(
      'strategy_type', 'mirrored', 'Type of distribute strategy.'
      'One of mirrored, tpu and multiworker.')


def strategy_flags_dict():
  """Returns TPU and/or GPU related flags in a dictionary."""
  return {
      'distribution_strategy': FLAGS.strategy_type,
      # TPUStrategy related flags.
      'tpu': FLAGS.tpu,
      # MultiWorkerMirroredStrategy related flags.
      'all_reduce_alg': FLAGS.all_reduce_alg,
      'worker_hosts': FLAGS.worker_hosts,
      'task_index': FLAGS.task_index,
      # MirroredStrategy and OneDeviceStrategy
      'num_gpus': FLAGS.num_gpus,
      'num_packs': FLAGS.num_packs,
  }


def hparam_flags_dict():
  """Returns model params related flags in a dictionary."""
  return {
      'data_dir': FLAGS.data_dir,
      'model_dir': FLAGS.model_dir,
      'train_batch_size': FLAGS.train_batch_size,
      'eval_batch_size': FLAGS.eval_batch_size,
      'precision': FLAGS.precision,
      'config_file': FLAGS.config_file,
      'params_override': FLAGS.params_override,
  }
