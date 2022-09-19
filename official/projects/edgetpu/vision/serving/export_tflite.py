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

# pylint: disable=line-too-long
r"""Export model (float or quantized tflite, and saved model) from a trained checkpoint.

Example:
To export dummy quantized model:
export_tflite --model_name=mobilenet_edgetpu_v2_s --output_dir=/tmp --quantize

Using a training checkpoint:
export_tflite --model_name=mobilenet_edgetpu_v2_s \
--ckpt_path=/path/to/training/checkpoint \
--dataset_dir=/path/to/your/dataset --output_dir=/tmp --quantize

Exporting w/o final squeeze layer:
export_tflite --model_name=mobilenet_edgetpu_v2_xs \
--output_layer=probs \
--dataset_dir=/path/to/your/dataset --output_dir=/tmp --quantize
"""
# pylint: enable=line-too-long
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from official.projects.edgetpu.vision.modeling import common_modules
from official.projects.edgetpu.vision.serving import export_util

flags.DEFINE_string('model_name', None,
                    'Used to build model using experiment config factory.')
flags.DEFINE_string(
    'ckpt_path', None, 'Path to the checkpoint. '
    'If not provided tflite with random parameters is exported.')
flags.DEFINE_enum(
    'ckpt_format', 'tf_checkpoint',
    ['tf_checkpoint', 'keras_checkpoint'],
    'tf_checkpoint is for ckpt files from tf.train.Checkpoint.save() method'
    'keras_checkpoint is for ckpt files from keras.Model.save_weights() method')
flags.DEFINE_bool(
    'export_keras_model', False,
    'Export SavedModel format: if False, export TF SavedModel with'
    'tf.saved_model API; if True, export Keras SavedModel with tf.keras.Model'
    'API.')
flags.DEFINE_string('output_dir', None, 'Directory to output exported files.')
flags.DEFINE_integer(
    'image_size', 224,
    'Size of the input image. Ideally should be the same as the image_size used '
    'in training config.')
flags.DEFINE_bool(
    'fix_batch_size', True, 'Whether to export model with fixed batch size.')
flags.DEFINE_string(
    'output_layer', None,
    'Layer name to take the output from. Can be used to take the output from '
    'an intermediate layer. None means use the original model output.')
flags.DEFINE_string(
    'finalize_method', 'none',
    'Additional layers to be added to customize serving output.\n'
    'Supported are (none|(argmax|resize<?>)[,...]).\n'
    '- none: do not add extra serving layers.\n'
    '- argmax: adds argmax.\n'
    '- squeeze: removes dimensions of size 1 from the shape of a tensor.\n'
    '- resize<?> (for example resize512): adds resize bilinear|nn to <?> size.'
    'For example: --finalize_method=resize128,argmax,resize512,squeeze\n'
    'Will do resize bilinear to 128x128, then argmax then resize nn to 512x512')

# Quantization related parameters
flags.DEFINE_bool(
    'quantize', False,
    'Quantize model before exporting tflite. Note that only the exported '
    'TFLite is quantized not the SavedModel.')
flags.DEFINE_bool('use_experimental_quantizer', True, 'Enables experimental '
                  'quantizer of TFLiteConverter 2.0.')
flags.DEFINE_bool(
    'quantize_less_restrictive', False,
    'Allows non int8 based intermediate types, automatic model output type.')
flags.DEFINE_integer(
    'num_calibration_steps', 100,
    'Number of post-training quantization calibration steps to run.')
flags.DEFINE_string('dataset_name', 'imagenet2012',
                    'Name of the dataset to use for quantization calibration.')
flags.DEFINE_string('dataset_dir', None, 'Dataset location.')
flags.DEFINE_string(
    'dataset_split', 'train',
    'The dataset split (train, validation etc.) to use for calibration.')

FLAGS = flags.FLAGS


def get_export_config_from_flags():
  """Creates ExportConfig from cmd line flags."""
  quantization_config = export_util.QuantizationConfig(
      quantize=FLAGS.quantize,
      quantize_less_restrictive=FLAGS.quantize_less_restrictive,
      use_experimental_quantizer=FLAGS.use_experimental_quantizer,
      num_calibration_steps=FLAGS.num_calibration_steps,
      dataset_name=FLAGS.dataset_name,
      dataset_dir=FLAGS.dataset_dir,
      dataset_split=FLAGS.dataset_split)
  export_config = export_util.ExportConfig(
      model_name=FLAGS.model_name,
      output_layer=FLAGS.output_layer,
      ckpt_path=FLAGS.ckpt_path,
      ckpt_format=FLAGS.ckpt_format,
      output_dir=FLAGS.output_dir,
      image_size=FLAGS.image_size,
      finalize_method=FLAGS.finalize_method.lower().split(','),
      quantization_config=quantization_config)
  return export_config


def run_export():
  """Exports TFLite with PTQ."""
  export_config = get_export_config_from_flags()
  model = export_util.build_experiment_model(
      experiment_type=export_config.model_name)

  if export_config.ckpt_path:
    logging.info('Loading checkpoint from %s', FLAGS.ckpt_path)
    common_modules.load_weights(
        model,
        export_config.ckpt_path,
        checkpoint_format=export_config.ckpt_format)
  else:
    logging.info('No checkpoint provided. Using randomly initialized weights.')

  if export_config.output_layer is not None:
    all_layer_names = {l.name for l in model.layers}
    if export_config.output_layer not in all_layer_names:
      model.summary()
      logging.info(
          'Cannot find the layer %s in the model. See the above summary to '
          'chose an output layer.', export_config.output_layer)
      return
    output_layer = model.get_layer(export_config.output_layer)
    model = tf.keras.Model(model.input, output_layer.output)

  batch_size = 1 if FLAGS.fix_batch_size else None

  model_input = tf.keras.Input(
      shape=(export_config.image_size, export_config.image_size, 3),
      batch_size=batch_size)
  model_output = export_util.finalize_serving(model(model_input), export_config)
  model_for_inference = tf.keras.Model(model_input, model_output)

  # Convert to tflite. Quantize if quantization parameters are specified.
  converter = tf.lite.TFLiteConverter.from_keras_model(model_for_inference)
  export_util.configure_tflite_converter(export_config, converter)
  tflite_buffer = converter.convert()

  # Make sure the base directory exists and write tflite.
  tf.io.gfile.makedirs(os.path.dirname(export_config.output_dir))
  tflite_path = os.path.join(export_config.output_dir,
                             f'{export_config.model_name}.tflite')
  tf.io.gfile.GFile(tflite_path, 'wb').write(tflite_buffer)
  print('TfLite model exported to {}'.format(tflite_path))

  # Export saved model.
  saved_model_path = os.path.join(export_config.output_dir,
                                  export_config.model_name)
  if FLAGS.export_keras_model:
    model_for_inference.save(saved_model_path)
  else:
    tf.saved_model.save(model_for_inference, saved_model_path)
  print('SavedModel exported to {}'.format(saved_model_path))


def main(_):
  run_export()


if __name__ == '__main__':
  flags.mark_flag_as_required('model_name')
  app.run(main)
