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

"""A converter for BERT pretrained checkpoint to QAT BERT checkpoint."""
import tempfile

# Import libraries

from absl import app
from absl import flags
import tensorflow as tf

from official.common import registry_imports  # pylint: disable=unused-import
from official.core import exp_factory
from official.core import task_factory
from official.modeling import hyperparams
from official.projects.qat.nlp import registry_imports as qat_registry_imports  # pylint: disable=unused-import

FLAGS = flags.FLAGS

_EXPERIMENT = flags.DEFINE_string(
    'experiment', default=None,
    help='The experiment type registered for the pretrained model.')

_CONFIG_FILE = flags.DEFINE_multi_string(
    'config_file',
    default=None,
    help='YAML/JSON files which specifies overrides. The override order '
    'follows the order of args. Note that each file '
    'can be used as an override template to override the default parameters '
    'specified in Python. If the same parameter is specified in both '
    '`--config_file` and `--params_override`, `config_file` will be used '
    'first, followed by params_override.')

_PARAMS_OVERRIDE = flags.DEFINE_string(
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

_PRETRAINED_CHECKPOINT = flags.DEFINE_string(
    'pretrained_checkpoint',
    default=None,
    help='The path of pretrained checkpoint for the original bert model.')

_EXPERIEMNT_QAT = flags.DEFINE_string(
    'experiment_qat', default=None,
    help='The experiment type registered for the pretrained model.')

_CONFIG_FILE_QAT = flags.DEFINE_multi_string(
    'config_file_qat',
    default=None,
    help='config_file flag for the qat model.')

_PARAMS_OVERRIDE_QAT = flags.DEFINE_string(
    'params_override_qat',
    default=None,
    help='params_override flag for the qat model.')

_OUTPUT_CHECKPOINT = flags.DEFINE_string(
    'output_checkpoint',
    default=None,
    help='The output checkpoint path for QAT applied BERT model.')


def _build_model(experiment, config_file, params_override):
  """Build the model."""
  params = exp_factory.get_exp_config(experiment)
  for config_file in config_file or []:
    params = hyperparams.override_params_dict(
        params, config_file, is_strict=True)

  if params_override:
    params = hyperparams.override_params_dict(
        params, params_override, is_strict=True)

  task = task_factory.get_task(params.task, logging_dir=tempfile.mkdtemp())
  return task.build_model()


def _set_weights_to_qat(model_from, model_to):
  """Set pretrained weight to QAT applied model."""
  name_to_index = {}
  for index, weight in enumerate(model_to.weights):
    origin_name = weight.name.replace('quant_', '').replace(
        'mobile_bert_embedding_1', 'mobile_bert_embedding')
    name_to_index[origin_name] = index

  model_to_weights = model_to.get_weights()
  for weight, value in zip(model_from.weights, model_from.get_weights()):
    index = name_to_index[weight.name]
    model_to_weights[index] = value
  model_to.set_weights(model_to_weights)


def main(_):
  model = _build_model(
      _EXPERIMENT.value, _CONFIG_FILE.value, _PARAMS_OVERRIDE.value)
  if _PRETRAINED_CHECKPOINT.value is not None:
    ckpt = tf.train.Checkpoint(model=model)
    status = ckpt.restore(_PRETRAINED_CHECKPOINT.value)
    status.expect_partial().assert_existing_objects_matched()

  model_qat = _build_model(
      _EXPERIEMNT_QAT.value, _CONFIG_FILE_QAT.value, _PARAMS_OVERRIDE_QAT.value)

  _set_weights_to_qat(model, model_qat)

  if hasattr(model_qat, 'checkpoint_items'):
    checkpoint_items = model_qat.checkpoint_items
  else:
    checkpoint_items = {}
  ckpt_qat = tf.train.Checkpoint(
      model=model_qat,
      **checkpoint_items)
  ckpt_qat.save(_OUTPUT_CHECKPOINT.value)


if __name__ == '__main__':
  app.run(main)
