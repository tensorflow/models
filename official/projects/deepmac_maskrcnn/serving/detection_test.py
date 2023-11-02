# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Test for image detection export lib."""

import io
import os

from absl.testing import parameterized
import numpy as np
from PIL import Image
import tensorflow as tf, tf_keras

from official.core import exp_factory
from official.projects.deepmac_maskrcnn.serving import detection


class DetectionExportTest(tf.test.TestCase, parameterized.TestCase):

  def _get_detection_module(self, experiment_name, image_size=(640, 640)):
    params = exp_factory.get_exp_config(experiment_name)
    params.task.model.backbone.resnet.model_id = 18
    params.task.model.detection_generator.use_batched_nms = True
    detection_module = detection.DetectionModule(
        params, batch_size=1, input_image_size=list(image_size))
    return detection_module

  def _export_from_module(self, module, input_type, save_directory):
    signatures = module.get_inference_signatures(
        {input_type: 'serving_default'})
    tf.saved_model.save(module, save_directory, signatures=signatures)

  def _get_dummy_input(self, input_type, batch_size, image_size):
    """Get dummy input for the given input type."""
    h, w = image_size

    if input_type == 'image_tensor':
      return tf.zeros((batch_size, h, w, 3), dtype=np.uint8)
    elif input_type == 'image_bytes':
      image = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
      byte_io = io.BytesIO()
      image.save(byte_io, 'PNG')
      return [byte_io.getvalue() for b in range(batch_size)]
    elif input_type == 'tf_example':
      image_tensor = tf.zeros((h, w, 3), dtype=tf.uint8)
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
      ('image_tensor', 'deep_mask_head_rcnn_resnetfpn_coco', [640, 640]),
      ('image_bytes', 'deep_mask_head_rcnn_resnetfpn_coco', [640, 384]),
      ('tf_example', 'deep_mask_head_rcnn_resnetfpn_coco', [640, 640]),
  )
  def test_export(self, input_type, experiment_name, image_size):
    self.skipTest('a')
    tmp_dir = self.get_temp_dir()
    module = self._get_detection_module(experiment_name, image_size)

    self._export_from_module(module, input_type, tmp_dir)

    self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'saved_model.pb')))
    self.assertTrue(
        os.path.exists(os.path.join(tmp_dir, 'variables', 'variables.index')))
    self.assertTrue(
        os.path.exists(
            os.path.join(tmp_dir, 'variables',
                         'variables.data-00000-of-00001')))

    imported = tf.saved_model.load(tmp_dir)
    detection_fn = imported.signatures['serving_default']

    images = self._get_dummy_input(
        input_type, batch_size=1, image_size=image_size)

    processed_images, anchor_boxes, image_info = module._build_inputs(
        tf.zeros((224, 224, 3), dtype=tf.uint8))
    image_shape = image_info[1, :]
    image_shape = tf.expand_dims(image_shape, 0)
    processed_images = tf.expand_dims(processed_images, 0)
    for l, l_boxes in anchor_boxes.items():
      anchor_boxes[l] = tf.expand_dims(l_boxes, 0)

    expected_outputs = module.model(
        images=processed_images,
        image_shape=image_shape,
        anchor_boxes=anchor_boxes,
        training=False)
    outputs = detection_fn(tf.constant(images))

    self.assertAllClose(outputs['num_detections'].numpy(),
                        expected_outputs['num_detections'].numpy())

  @parameterized.parameters(
      ('deep_mask_head_rcnn_resnetfpn_coco', [640, 640], 1),
      ('deep_mask_head_rcnn_resnetfpn_coco', [640, 640], 5),
      ('deep_mask_head_rcnn_spinenet_coco', [640, 384], 3),
      ('deep_mask_head_rcnn_spinenet_coco', [640, 384], 9),
  )
  def test_export_image_and_boxes(self, experiment_name, image_size, num_boxes):
    tmp_dir = self.get_temp_dir()
    module = self._get_detection_module(experiment_name)

    self._export_from_module(module, 'image_and_boxes_tensor', tmp_dir)

    self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'saved_model.pb')))
    self.assertTrue(
        os.path.exists(os.path.join(tmp_dir, 'variables', 'variables.index')))
    self.assertTrue(
        os.path.exists(
            os.path.join(tmp_dir, 'variables',
                         'variables.data-00000-of-00001')))

    imported = tf.saved_model.load(tmp_dir)
    detection_fn = imported.signatures['serving_default']

    images = self._get_dummy_input(
        'image_tensor', batch_size=1, image_size=image_size)

    processed_images, anchor_boxes, image_info = module._build_inputs(
        tf.zeros(image_size + [3], dtype=tf.uint8))

    image_shape = image_info[1, :]
    image_shape = image_shape[tf.newaxis]
    processed_images = processed_images[tf.newaxis]
    image_info = image_info[tf.newaxis]

    for l, l_boxes in anchor_boxes.items():
      anchor_boxes[l] = tf.expand_dims(l_boxes, 0)

    boxes = np.zeros((1, num_boxes, 4), dtype=np.float32)
    boxes[:, :, [2, 3]] = 1.0
    boxes = tf.constant(boxes)

    denormalized_boxes = detection.reverse_input_box_transformation(
        boxes, image_info)
    expected_outputs = module.model.call_images_and_boxes(
        images=processed_images, boxes=denormalized_boxes)
    outputs = detection_fn(images=tf.constant(images), boxes=boxes)

    self.assertAllClose(outputs['detection_masks'].numpy(),
                        expected_outputs['detection_masks'].numpy(),
                        rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
  tf.test.main()
