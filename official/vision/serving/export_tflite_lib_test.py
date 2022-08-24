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

  def setUp(self):
    super().setUp()
    # Create test data for image classification.
    self.test_tfrecord_file_cls = os.path.join(self.get_temp_dir(),
                                               'cls_test.tfrecord')
    example = tf.train.Example.FromString(
        tfexample_utils.create_classification_example(
            image_height=224, image_width=224))
    self._create_test_tfrecord(
        tfrecord_file=self.test_tfrecord_file_cls,
        example=example,
        num_samples=10)

    # Create test data for object detection.
    self.test_tfrecord_file_det = os.path.join(self.get_temp_dir(),
                                               'det_test.tfrecord')
    example = tfexample_utils.create_detection_test_example(
        image_height=128, image_width=128, image_channel=3, num_instances=10)
    self._create_test_tfrecord(
        tfrecord_file=self.test_tfrecord_file_det,
        example=example,
        num_samples=10)

    # Create test data for semantic segmentation.
    self.test_tfrecord_file_seg = os.path.join(self.get_temp_dir(),
                                               'seg_test.tfrecord')
    example = tfexample_utils.create_segmentation_test_example(
        image_height=512, image_width=512, image_channel=3)
    self._create_test_tfrecord(
        tfrecord_file=self.test_tfrecord_file_seg,
        example=example,
        num_samples=10)

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
          quant_type=[
              None,
              'default',
              'fp16',
              'int8_fallback',
              'int8_full',
              'int8_full_fp32_io',
              'int8_full_int8_io',
          ]))
  def test_export_tflite_image_classification(self, experiment, quant_type):

    params = exp_factory.get_exp_config(experiment)
    params.task.validation_data.input_path = self.test_tfrecord_file_cls
    params.task.train_data.input_path = self.test_tfrecord_file_cls
    params.task.train_data.shuffle_buffer_size = 10
    temp_dir = self.get_temp_dir()
    module = image_classification_serving.ClassificationModule(
        params=params,
        batch_size=1,
        input_image_size=[224, 224],
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
          quant_type=[
              None,
              'default',
              'fp16',
              'int8_fallback',
              'int8_full',
              'int8_full_fp32_io',
              'int8_full_int8_io',
          ]))
  def test_export_tflite_detection(self, experiment, quant_type):

    params = exp_factory.get_exp_config(experiment)
    params.task.validation_data.input_path = self.test_tfrecord_file_det
    params.task.train_data.input_path = self.test_tfrecord_file_det
    params.task.model.num_classes = 2
    params.task.model.backbone.spinenet_mobile.model_id = '49XS'
    params.task.model.input_size = [128, 128, 3]
    params.task.model.detection_generator.nms_version = 'v1'
    params.task.train_data.shuffle_buffer_size = 5
    temp_dir = self.get_temp_dir()
    module = detection_serving.DetectionModule(
        params=params,
        batch_size=1,
        input_image_size=[128, 128],
        input_type='tflite')
    self._export_from_module(
        module=module,
        input_type='tflite',
        saved_model_dir=os.path.join(temp_dir, 'saved_model'))

    tflite_model = export_tflite_lib.convert_tflite_model(
        saved_model_dir=os.path.join(temp_dir, 'saved_model'),
        quant_type=quant_type,
        params=params,
        calibration_steps=1)

    self.assertIsInstance(tflite_model, bytes)

  @combinations.generate(
      combinations.combine(
          experiment=['mnv2_deeplabv3_pascal'],
          quant_type=[
              None,
              'default',
              'fp16',
              'int8_fallback',
              'int8_full',
              'int8_full_fp32_io',
              'int8_full_int8_io',
          ]))
  def test_export_tflite_semantic_segmentation(self, experiment, quant_type):

    params = exp_factory.get_exp_config(experiment)
    params.task.validation_data.input_path = self.test_tfrecord_file_seg
    params.task.train_data.input_path = self.test_tfrecord_file_seg
    params.task.train_data.shuffle_buffer_size = 10
    temp_dir = self.get_temp_dir()
    module = semantic_segmentation_serving.SegmentationModule(
        params=params,
        batch_size=1,
        input_image_size=[512, 512],
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
