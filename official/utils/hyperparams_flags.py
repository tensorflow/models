# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Common flags for importing hyperparameters."""

from __future__ import absolute_import
from __future__ import division
# from __future__ import google_type_annotations
from __future__ import print_function

from absl import flags

FLAGS = flags.FLAGS


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

  flags.DEFINE_string(
      'strategy_type', 'mirrored', 'Type of distribute strategy.'
      'One of mirrored, tpu and multiworker.')


def initialize_common_flags():
  """Define the common flags across models."""
  key_flags = []
  define_common_hparams_flags()
  flags.DEFINE_string(
      'tpu',
      default=None,
      help='The Cloud TPU to use for training. This should be either the name '
      'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
      'url.')
  # Parameters for MultiWorkerMirroredStrategy
  flags.DEFINE_string(
      'worker_hosts',
      default=None,
      help='Comma-separated list of worker ip:port pairs for running '
      'multi-worker models with distribution strategy.  The user would '
      'start the program on each host with identical value for this flag.')
  flags.DEFINE_integer(
      'task_index', 0,
      'If multi-worker training, the task_index of this worker.')
  flags.DEFINE_integer('save_checkpoint_freq', None,
                       'Number of steps to save checkpoint.')
  return key_flags

