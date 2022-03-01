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

"""Tests for export_tflite_lib."""
import os

from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.distribute import combinations
from official.core import exp_factory
from official.vision import registry_imports  # pylint: disable=unused-import
from official.vision.dataloaders import tfexample_utils
from official.vision.serving import detection as detection_serving
from official.vision.serving import export_tflite_lib
from official.vision.serving import image_classification as image_classification_serving
from official.vision.serving import semantic_segmentation as semantic_segmentation_serving


class ExportTfliteLibTest(tf.test.TestCase, parameterized.TestCase):

  def _create_test_tfrecord(self, tfrecord_file, example, num_samples):
    examples = [example] * num_samples
    tfexample_utils.dump_to_tfrecord(
        record_file=tfrecord_file, tf_examples=examples)

  def _export_from_module(self, module, input_type, saved_model_dir):
    signatures = module.get_inference_signatures(
        {input_type: 'serving_default'})
    tf.saved_model.save(module, saved_model_dir, signatures=signatures)

  @combinations.generate(
      combinations.combine(
          experiment=['mobilenet_imagenet'],
          quant_type=[None, 'default', 'fp16', 'int8', 'int8_full'],
          input_image_size=[[224, 224]]))
  def test_export_tflite_image_classification(self, experiment, quant_type,
                                              input_image_size):
    test_tfrecord_file = os.path.join(self.get_temp_dir(), 'cls_test.tfrecord')
    example = tf.train.Example.FromString(
        tfexample_utils.create_classification_example(
            image_height=input_image_size[0], image_width=input_image_size[1]))
    self._create_test_tfrecord(
        tfrecord_file=test_tfrecord_file, example=example, num_samples=10)
    params = exp_factory.get_exp_config(experiment)
    params.task.validation_data.input_path = test_tfrecord_file
    params.task.train_data.input_path = test_tfrecord_file
    temp_dir = self.get_temp_dir()
    module = image_classification_serving.ClassificationModule(
        params=params,
        batch_size=1,
        input_image_size=input_image_size,
        input_type='tflite')
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
          input_image_size=[[384, 384]]))
  def test_export_tflite_detection(self, experiment, quant_type,
                                   input_image_size):
    test_tfrecord_file = os.path.join(self.get_temp_dir(), 'det_test.tfrecord')
    example = tfexample_utils.create_detection_test_example(
        image_height=input_image_size[0],
        image_width=input_image_size[1],
        image_channel=3,
        num_instances=10)
    self._create_test_tfrecord(
        tfrecord_file=test_tfrecord_file, example=example, num_samples=10)
    params = exp_factory.get_exp_config(experiment)
    params.task.validation_data.input_path = test_tfrecord_file
    params.task.train_data.input_path = test_tfrecord_file
    temp_dir = self.get_temp_dir()
    module = detection_serving.DetectionModule(
        params=params,
        batch_size=1,
        input_image_size=input_image_size,
        input_type='tflite')
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
          experiment=['mnv2_deeplabv3_pascal'],
          quant_type=[None, 'default', 'fp16', 'int8', 'int8_full'],
          input_image_size=[[512, 512]]))
  def test_export_tflite_semantic_segmentation(self, experiment, quant_type,
                                               input_image_size):
    test_tfrecord_file = os.path.join(self.get_temp_dir(), 'seg_test.tfrecord')
    example = tfexample_utils.create_segmentation_test_example(
        image_height=input_image_size[0],
        image_width=input_image_size[1],
        image_channel=3)
    self._create_test_tfrecord(
        tfrecord_file=test_tfrecord_file, example=example, num_samples=10)
    params = exp_factory.get_exp_config(experiment)
    params.task.validation_data.input_path = test_tfrecord_file
    params.task.train_data.input_path = test_tfrecord_file
    temp_dir = self.get_temp_dir()
    module = semantic_segmentation_serving.SegmentationModule(
        params=params,
        batch_size=1,
        input_image_size=input_image_size,
        input_type='tflite')
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
