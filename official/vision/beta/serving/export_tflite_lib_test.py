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

"""Tests for export_tflite_lib."""
import os

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.distribute import combinations
from official.common import registry_imports  # pylint: disable=unused-import
from official.core import exp_factory
from official.vision.beta.dataloaders import tfexample_utils
from official.vision.beta.serving import detection as detection_serving
from official.vision.beta.serving import export_tflite_lib
from official.vision.beta.serving import image_classification as image_classification_serving
from official.vision.beta.serving import semantic_segmentation as semantic_segmentation_serving


class ExportTfliteLibTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._test_tfrecord_file = os.path.join(self.get_temp_dir(),
                                            'test.tfrecord')
    self._create_test_tfrecord(num_samples=50)

  def _create_test_tfrecord(self, num_samples):
    tfexample_utils.dump_to_tfrecord(self._test_tfrecord_file, [
        tf.train.Example.FromString(
            tfexample_utils.create_classification_example(
                image_height=256, image_width=256)) for _ in range(num_samples)
    ])

  def _export_from_module(self, module, input_type, saved_model_dir):
    signatures = module.get_inference_signatures(
        {input_type: 'serving_default'})
    tf.saved_model.save(module, saved_model_dir, signatures=signatures)

  @combinations.generate(
      combinations.combine(
          experiment=['mobilenet_imagenet'],
          quant_type=[None, 'default', 'fp16', 'int8'],
          input_image_size=[[224, 224]]))
  def test_export_tflite_image_classification(self, experiment, quant_type,
                                              input_image_size):
    params = exp_factory.get_exp_config(experiment)
    params.task.validation_data.input_path = self._test_tfrecord_file
    params.task.train_data.input_path = self._test_tfrecord_file
    temp_dir = self.get_temp_dir()
    module = image_classification_serving.ClassificationModule(
        params=params, batch_size=1, input_image_size=input_image_size)
    self._export_from_module(
        module=module,
        input_type='tflite',
        saved_model_dir=os.path.join(temp_dir, 'saved_model'))

    tflite_model = export_tflite_lib.convert_tflite_model(
        saved_model_dir=os.path.join(temp_dir, 'saved_model'),
        quant_type=quant_type,
        params=params,
        calibration_steps=5)

    self.assertIsInstance(tflite_model, bytes)

  @combinations.generate(
      combinations.combine(
          experiment=['retinanet_mobile_coco'],
          quant_type=[None, 'default', 'fp16'],
          input_image_size=[[256, 256]]))
  def test_export_tflite_detection(self, experiment, quant_type,
                                   input_image_size):
    params = exp_factory.get_exp_config(experiment)
    temp_dir = self.get_temp_dir()
    module = detection_serving.DetectionModule(
        params=params, batch_size=1, input_image_size=input_image_size)
    self._export_from_module(
        module=module,
        input_type='tflite',
        saved_model_dir=os.path.join(temp_dir, 'saved_model'))

    tflite_model = export_tflite_lib.convert_tflite_model(
        saved_model_dir=os.path.join(temp_dir, 'saved_model'),
        quant_type=quant_type,
        params=params,
        calibration_steps=5)

    self.assertIsInstance(tflite_model, bytes)

  @combinations.generate(
      combinations.combine(
          experiment=['seg_deeplabv3_pascal'],
          quant_type=[None, 'default', 'fp16'],
          input_image_size=[[512, 512]]))
  def test_export_tflite_semantic_segmentation(self, experiment, quant_type,
                                               input_image_size):
    params = exp_factory.get_exp_config(experiment)
    temp_dir = self.get_temp_dir()
    module = semantic_segmentation_serving.SegmentationModule(
        params=params, batch_size=1, input_image_size=input_image_size)
    self._export_from_module(
        module=module,
        input_type='tflite',
        saved_model_dir=os.path.join(temp_dir, 'saved_model'))

    tflite_model = export_tflite_lib.convert_tflite_model(
        saved_model_dir=os.path.join(temp_dir, 'saved_model'),
        quant_type=quant_type,
        params=params,
        calibration_steps=5)

    self.assertIsInstance(tflite_model, bytes)

if __name__ == '__main__':
  tf.test.main()
