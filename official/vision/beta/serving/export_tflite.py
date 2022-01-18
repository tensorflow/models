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
from official.common import registry_imports  # pylint: disable=unused-import
from official.core import exp_factory
from official.modeling import hyperparams
from official.vision.beta.serving import export_tflite_lib

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'experiment',
    None,
    'experiment type, e.g. retinanet_resnetfpn_coco',
    required=True)
flags.DEFINE_multi_string(
    'config_file',
    default='',
    help='YAML/JSON files which specifies overrides. The override order '
    'follows the order of args. Note that each file '
    'can be used as an override template to override the default parameters '
    'specified in Python. If the same parameter is specified in both '
    '`--config_file` and `--params_override`, `config_file` will be used '
    'first, followed by params_override.')
flags.DEFINE_string(
    'params_override', '',
    'The JSON/YAML file or string which specifies the parameter to be overriden'
    ' on top of `config_file` template.')
flags.DEFINE_string(
    'saved_model_dir', None, 'The directory to the saved model.', required=True)
flags.DEFINE_string(
    'tflite_path', None, 'The path to the output tflite model.', required=True)
flags.DEFINE_string(
    'quant_type',
    default=None,
    help='Post training quantization type. Support `int8`, `int8_full`, '
    '`fp16`, and `default`. See '
    'https://www.tensorflow.org/lite/performance/post_training_quantization '
    'for more details.')
flags.DEFINE_integer('calibration_steps', 500,
                     'The number of calibration steps for integer model.')


def main(_) -> None:
  params = exp_factory.get_exp_config(FLAGS.experiment)
  if FLAGS.config_file is not None:
    for config_file in FLAGS.config_file:
      params = hyperparams.override_params_dict(
          params, config_file, is_strict=True)
  if FLAGS.params_override:
    params = hyperparams.override_params_dict(
        params, FLAGS.params_override, is_strict=True)

  params.validate()
  params.lock()

  logging.info('Converting SavedModel from %s to TFLite model...',
               FLAGS.saved_model_dir)
  tflite_model = export_tflite_lib.convert_tflite_model(
      saved_model_dir=FLAGS.saved_model_dir,
      quant_type=FLAGS.quant_type,
      params=params,
      calibration_steps=FLAGS.calibration_steps)

  with tf.io.gfile.GFile(FLAGS.tflite_path, 'wb') as fw:
    fw.write(tflite_model)

  logging.info('TFLite model converted and saved to %s.', FLAGS.tflite_path)


if __name__ == '__main__':
  app.run(main)
