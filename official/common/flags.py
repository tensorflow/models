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

"""The central place to define flags."""

from absl import flags


def define_flags():
  """Defines flags."""
  flags.DEFINE_string(
      'experiment', default=None, help='The experiment type registered.')

  flags.DEFINE_enum(
      'mode',
      default=None,
      enum_values=[
          'train', 'eval', 'train_and_eval', 'continuous_eval',
          'continuous_train_and_eval', 'train_and_validate'
      ],
      help='Mode to run: `train`, `eval`, `train_and_eval`, '
      '`continuous_eval`, `continuous_train_and_eval` and '
      '`train_and_validate` (which is not implemented in '
      'the open source version).')

  flags.DEFINE_string(
      'model_dir',
      default=None,
      help='The directory where the model and training/evaluation summaries'
      'are stored.')

  flags.DEFINE_multi_string(
      'config_file',
      default=None,
      help='YAML/JSON files which specifies overrides. The override order '
      'follows the order of args. Note that each file '
      'can be used as an override template to override the default parameters '
      'specified in Python. If the same parameter is specified in both '
      '`--config_file` and `--params_override`, `config_file` will be used '
      'first, followed by params_override.')

  flags.DEFINE_string(
      'params_override',
      default=None,
      help='a YAML/JSON string or a YAML file which specifies additional '
      'overrides over the default parameters and those specified in '
      '`--config_file`. Note that this is supposed to be used only to override '
      'the model parameters, but not the parameters like TPU specific flags. '
      'One canonical use case of `--config_file` and `--params_override` is '
      'users first define a template config file using `--config_file`, then '
      'use `--params_override` to adjust the minimal set of tuning parameters, '
      'for example setting up different `train_batch_size`. The final override '
      'order of parameters: default_model_params --> params from config_file '
      '--> params in params_override. See also the help message of '
      '`--config_file`.')

  # The libraries rely on gin often make mistakes that include flags inside
  # the library files which causes conflicts.
  try:
    flags.DEFINE_multi_string(
        'gin_file', default=None, help='List of paths to the config files.')
  except flags.DuplicateFlagError:
    pass

  try:
    flags.DEFINE_multi_string(
        'gin_params',
        default=None,
        help='Newline separated list of Gin parameter bindings.')
  except flags.DuplicateFlagError:
    pass

  flags.DEFINE_string(
      'tpu',
      default=None,
      help='The Cloud TPU to use for training. This should be either the name '
      'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
      'url.')

  flags.DEFINE_string(
      'tf_data_service', default=None, help='The tf.data service address')
