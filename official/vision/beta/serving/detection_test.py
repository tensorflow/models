# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test for image classification export lib."""

import io
import os

from absl.testing import parameterized
import numpy as np
from PIL import Image
import tensorflow as tf

from official.common import registry_imports  # pylint: disable=unused-import
from official.core import exp_factory
from official.vision.beta.serving import detection


class DetectionExportTest(tf.test.TestCase, parameterized.TestCase):

  def _get_detection_module(self, experiment_name):
    params = exp_factory.get_exp_config(experiment_name)
    params.task.model.backbone.resnet.model_id = 18
    params.task.model.detection_generator.use_batched_nms = True
    detection_module = detection.DetectionModule(
        params, batch_size=1, input_image_size=[640, 640])
    return detection_module

  def _export_from_module(self, module, input_type, batch_size, save_directory):
    if input_type == 'image_tensor':
      input_signature = tf.TensorSpec(
          shape=[batch_size, 640, 640, 3], dtype=tf.uint8)
      signatures = {
          'serving_default':
              module.inference_from_image_tensors.get_concrete_function(
                  input_signature)
      }
    elif input_type == 'image_bytes':
      input_signature = tf.TensorSpec(shape=[batch_size], dtype=tf.string)
      signatures = {
          'serving_default':
              module.inference_from_image_bytes.get_concrete_function(
                  input_signature)
      }
    elif input_type == 'tf_example':
      input_signature = tf.TensorSpec(shape=[batch_size], dtype=tf.string)
      signatures = {
          'serving_default':
              module.inference_from_tf_example.get_concrete_function(
                  input_signature)
      }
    else:
      raise ValueError('Unrecognized `input_type`')

    tf.saved_model.save(module,
                        save_directory,
                        signatures=signatures)

  def _get_dummy_input(self, input_type, batch_size):
    """Get dummy input for the given input type."""

    if input_type == 'image_tensor':
      return tf.zeros((batch_size, 640, 640, 3), dtype=np.uint8)
    elif input_type == 'image_bytes':
      image = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
      byte_io = io.BytesIO()
      image.save(byte_io, 'PNG')
      return [byte_io.getvalue() for b in range(batch_size)]
    elif input_type == 'tf_example':
      image_tensor = tf.zeros((640, 640, 3), dtype=tf.uint8)
      encoded_jpeg = tf.image.encode_jpeg(tf.constant(image_tensor)).numpy()
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      tf.train.Feature(
                          bytes_list=tf.train.BytesList(value=[encoded_jpeg])),
              })).SerializeToString()
      return [example for b in range(batch_size)]

  @parameterized.parameters(
      ('image_tensor', 'fasterrcnn_resnetfpn_coco'),
      ('image_bytes', 'fasterrcnn_resnetfpn_coco'),
      ('tf_example', 'fasterrcnn_resnetfpn_coco'),
      ('image_tensor', 'maskrcnn_resnetfpn_coco'),
      ('image_bytes', 'maskrcnn_resnetfpn_coco'),
      ('tf_example', 'maskrcnn_resnetfpn_coco'),
      ('image_tensor', 'retinanet_resnetfpn_coco'),
      ('image_bytes', 'retinanet_resnetfpn_coco'),
      ('tf_example', 'retinanet_resnetfpn_coco'),
  )
  def test_export(self, input_type, experiment_name):
    tmp_dir = self.get_temp_dir()
    batch_size = 1

    experiment_name = 'fasterrcnn_resnetfpn_coco'
    module = self._get_detection_module(experiment_name)
    model = module.build_model()

    self._export_from_module(module, input_type, batch_size, tmp_dir)

    self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'saved_model.pb')))
    self.assertTrue(os.path.exists(
        os.path.join(tmp_dir, 'variables', 'variables.index')))
    self.assertTrue(os.path.exists(
        os.path.join(tmp_dir, 'variables', 'variables.data-00000-of-00001')))

    imported = tf.saved_model.load(tmp_dir)
    classification_fn = imported.signatures['serving_default']

    images = self._get_dummy_input(input_type, batch_size)

    processed_images, anchor_boxes, image_shape = module._build_inputs(
        tf.zeros((224, 224, 3), dtype=tf.uint8))
    processed_images = tf.expand_dims(processed_images, 0)
    image_shape = tf.expand_dims(image_shape, 0)
    for l, l_boxes in anchor_boxes.items():
      anchor_boxes[l] = tf.expand_dims(l_boxes, 0)

    expected_outputs = model(
        images=processed_images,
        image_shape=image_shape,
        anchor_boxes=anchor_boxes,
        training=False)
    outputs = classification_fn(tf.constant(images))

    self.assertAllClose(outputs['num_detections'].numpy(),
                        expected_outputs['num_detections'].numpy())

if __name__ == '__main__':
  tf.test.main()
