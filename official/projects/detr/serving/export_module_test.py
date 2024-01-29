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

"""Test for DETR export module."""

import io
import os

from absl.testing import parameterized
import numpy as np
from PIL import Image
import tensorflow as tf

from official.core import exp_factory
from official.projects.detr.configs import detr as exp_cfg  # pylint: disable=unused-import
from official.projects.detr.serving import export_module


class ExportModuleTest(tf.test.TestCase, parameterized.TestCase):

  def _get_module(self, input_type):
    params = exp_factory.get_exp_config('detr_coco')
    return export_module.DETRModule(
        params,
        batch_size=1,
        input_image_size=[384, 384],
        input_type=input_type)

  def _export_from_module(self, module, input_type, save_directory):
    signatures = module.get_inference_signatures(
        {input_type: 'serving_default'})
    tf.saved_model.save(module, save_directory, signatures=signatures)

  def _get_dummy_input(self, input_type):
    """Gets dummy input for the given input type."""

    if input_type == 'image_tensor':
      return tf.zeros((1, 384, 384, 3), dtype=np.uint8)
    elif input_type == 'image_bytes':
      image = Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8))
      byte_io = io.BytesIO()
      image.save(byte_io, 'PNG')
      return [byte_io.getvalue()]
    elif input_type == 'tf_example':
      image_tensor = tf.zeros((384, 384, 3), dtype=tf.uint8)
      encoded_jpeg = tf.image.encode_jpeg(tf.constant(image_tensor)).numpy()
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      tf.train.Feature(
                          bytes_list=tf.train.BytesList(value=[encoded_jpeg])),
              })).SerializeToString()
      return [example]

  @parameterized.parameters(
      {'input_type': 'image_tensor'},
      {'input_type': 'image_bytes'},
      {'input_type': 'tf_example'},
  )
  def test_export(self, input_type='image_tensor'):
    tmp_dir = self.get_temp_dir()
    module = self._get_module(input_type)
    self._export_from_module(module, input_type, tmp_dir)

    self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'saved_model.pb')))
    self.assertTrue(
        os.path.exists(os.path.join(tmp_dir, 'variables', 'variables.index')))
    self.assertTrue(
        os.path.exists(
            os.path.join(tmp_dir, 'variables',
                         'variables.data-00000-of-00001')))

    imported = tf.saved_model.load(tmp_dir)
    predict_fn = imported.signatures['serving_default']

    images = self._get_dummy_input(input_type)
    outputs = predict_fn(tf.constant(images))

    self.assertNotEmpty(outputs['detection_boxes'])
    self.assertNotEmpty(outputs['detection_classes'])
    self.assertNotEmpty(outputs['detection_scores'])
    self.assertNotEmpty(outputs['num_detections'])


if __name__ == '__main__':
  tf.test.main()
