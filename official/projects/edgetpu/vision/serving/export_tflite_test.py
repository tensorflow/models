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

"""Tests for export_tflite."""

import itertools
import os

from absl.testing import parameterized
import tensorflow as tf

from official.core import exp_factory
from official.core import task_factory
from official.projects.edgetpu.vision.serving import export_util


def _build_experiment_model(experiment_type):
  """Builds model from experiment type configuration w/o loading checkpoint.

  To reduce test latency and avoid unexpected errors (e.g. checkpoint files not
  exist in the dedicated path), we skip the checkpoint loading for the tests.

  Args:
    experiment_type: model type for the experiment.
  Returns:
    TF/Keras model for the task.
  """
  params = exp_factory.get_exp_config(experiment_type)
  if 'deeplabv3plus_mobilenet_edgetpuv2' in experiment_type:
    params.task.model.backbone.mobilenet_edgetpu.pretrained_checkpoint_path = None
  if 'autoseg_edgetpu' in experiment_type:
    params.task.model.model_params.model_weights_path = None
  params.validate()
  params.lock()
  task = task_factory.get_task(params.task)
  return task.build_model()


def _build_model(config):
  model = _build_experiment_model(config.model_name)
  model_input = tf.keras.Input(
      shape=(config.image_size, config.image_size, 3), batch_size=1)
  model_output = export_util.finalize_serving(model(model_input), config)
  model_for_inference = tf.keras.Model(model_input, model_output)
  return model_for_inference


def _dump_tflite(model, config):
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  export_util.configure_tflite_converter(config, converter)
  tflite_buffer = converter.convert()
  tf.io.gfile.makedirs(os.path.dirname(config.output_dir))
  tflite_path = os.path.join(config.output_dir, f'{config.model_name}.tflite')
  tf.io.gfile.GFile(tflite_path, 'wb').write(tflite_buffer)
  return tflite_path


SEG_MODELS = [
    'autoseg_edgetpu_xs',
]
FINALIZE_METHODS = [
    'resize512,argmax,squeeze', 'resize256,argmax,resize512,squeeze',
    'resize128,argmax,resize512,squeeze'
]


class ExportTfliteTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ('mobilenet_edgetpu_v2_xs', 224),
      ('autoseg_edgetpu_xs', 512),
      ('deeplabv3plus_mobilenet_edgetpuv2_xs_ade20k', 512),
      ('deeplabv3plus_mobilenet_edgetpuv2_xs_ade20k_32', 512),
  )
  def test_model_build_and_export_tflite(self, model_name, image_size):
    tmp_dir = self.create_tempdir().full_path
    config = export_util.ExportConfig(
        model_name=model_name, image_size=image_size, output_dir=tmp_dir)
    config.quantization_config.quantize = False
    model = _build_model(config)
    tflite_path = _dump_tflite(model, config)
    self.assertTrue(tf.io.gfile.exists(tflite_path))

  @parameterized.parameters(
      ('mobilenet_edgetpu_v2_xs', 224),
      ('autoseg_edgetpu_xs', 512),
      ('deeplabv3plus_mobilenet_edgetpuv2_xs_ade20k', 512),
      ('deeplabv3plus_mobilenet_edgetpuv2_xs_ade20k_32', 512),
  )
  def test_model_build_and_export_saved_model(self, model_name, image_size):
    tmp_dir = self.create_tempdir().full_path
    config = export_util.ExportConfig(
        model_name=model_name, image_size=image_size, output_dir=tmp_dir)
    model = _build_model(config)
    saved_model_path = os.path.join(config.output_dir, config.model_name)
    model.save(saved_model_path)
    self.assertTrue(tf.saved_model.contains_saved_model(saved_model_path))

  @parameterized.parameters(itertools.product(SEG_MODELS, FINALIZE_METHODS))
  def test_segmentation_finalize_methods(self, model_name, finalize_method):
    tmp_dir = self.create_tempdir().full_path
    config = export_util.ExportConfig(
        model_name=model_name,
        image_size=512,
        output_dir=tmp_dir,
        finalize_method=finalize_method.split(','))
    config.quantization_config.quantize = False
    model = _build_model(config)
    model_input = tf.random.normal([1, config.image_size, config.image_size, 3])
    self.assertEqual(
        model(model_input).get_shape().as_list(),
        [1, config.image_size, config.image_size])


if __name__ == '__main__':
  tf.test.main()
