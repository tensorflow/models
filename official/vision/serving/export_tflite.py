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

r"""Binary to convert a saved model to tflite model.

It requires a SavedModel exported using export_saved_model.py with batch size 1
and input type `tflite`, and using the same config file used for exporting saved
model. It includes optional post-training quantization. When using integer
quantization, calibration steps need to be provided to calibrate model input.

To convert a SavedModel to a TFLite model:

EXPERIMENT_TYPE = XX
TFLITE_PATH = XX
SAVED_MOODEL_DIR = XX
CONFIG_FILE = XX
export_tflite --experiment=${EXPERIMENT_TYPE} \
              --saved_model_dir=${SAVED_MOODEL_DIR} \
              --tflite_path=${TFLITE_PATH} \
              --config_file=${CONFIG_FILE} \
              --quant_type=fp16 \
              --calibration_steps=500
"""
from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
from official.core import exp_factory
from official.modeling import hyperparams
from official.vision import registry_imports  # pylint: disable=unused-import
from official.vision.serving import export_tflite_lib

FLAGS = flags.FLAGS

_EXPERIMENT = flags.DEFINE_string(
    'experiment',
    None,
    'experiment type, e.g. retinanet_resnetfpn_coco',
    required=True)
_CONFIG_FILE = flags.DEFINE_multi_string(
    'config_file',
    default='',
    help='YAML/JSON files which specifies overrides. The override order '
    'follows the order of args. Note that each file '
    'can be used as an override template to override the default parameters '
    'specified in Python. If the same parameter is specified in both '
    '`--config_file` and `--params_override`, `config_file` will be used '
    'first, followed by params_override.')
_PARAMS_OVERRIDE = flags.DEFINE_string(
    'params_override', '',
    'The JSON/YAML file or string which specifies the parameter to be overriden'
    ' on top of `config_file` template.')
_SAVED_MODEL_DIR = flags.DEFINE_string(
    'saved_model_dir', None, 'The directory to the saved model.', required=True)
_TFLITE_PATH = flags.DEFINE_string(
    'tflite_path', None, 'The path to the output tflite model.', required=True)
_QUANT_TYPE = flags.DEFINE_string(
    'quant_type',
    default=None,
    help='Post training quantization type. Support `int8_fallback`, '
    '`int8_full_fp32_io`, `int8_full`, `fp16`, `qat`, `qat_fp32_io`, '
    '`int8_full_int8_io` and `default`. See '
    'https://www.tensorflow.org/lite/performance/post_training_quantization '
    'for more details.')
_CALIBRATION_STEPS = flags.DEFINE_integer(
    'calibration_steps', 500,
    'The number of calibration steps for integer model.')
_DENYLISTED_OPS = flags.DEFINE_string(
    'denylisted_ops', '', 'The comma-separated string of ops '
    'that are excluded from integer quantization. The name of '
    'ops should be all capital letters, such as CAST or GREATER.'
    'This is useful to exclude certains ops that affects quality or latency. '
    'Valid ops that should not be included are quantization friendly ops, such '
    'as CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, etc.')


def main(_) -> None:
  params = exp_factory.get_exp_config(_EXPERIMENT.value)
  if _CONFIG_FILE.value is not None:
    for config_file in _CONFIG_FILE.value:
      params = hyperparams.override_params_dict(
          params, config_file, is_strict=True)
  if _PARAMS_OVERRIDE.value:
    params = hyperparams.override_params_dict(
        params, _PARAMS_OVERRIDE.value, is_strict=True)

  params.validate()
  params.lock()

  logging.info('Converting SavedModel from %s to TFLite model...',
               _SAVED_MODEL_DIR.value)

  if _DENYLISTED_OPS.value:
    denylisted_ops = list(_DENYLISTED_OPS.value.split(','))
  else:
    denylisted_ops = None
  tflite_model = export_tflite_lib.convert_tflite_model(
      saved_model_dir=_SAVED_MODEL_DIR.value,
      quant_type=_QUANT_TYPE.value,
      params=params,
      calibration_steps=_CALIBRATION_STEPS.value,
      denylisted_ops=denylisted_ops)

  with tf.io.gfile.GFile(_TFLITE_PATH.value, 'wb') as fw:
    fw.write(tflite_model)

  logging.info('TFLite model converted and saved to %s.', _TFLITE_PATH.value)


if __name__ == '__main__':
  app.run(main)
