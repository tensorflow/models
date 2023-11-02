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

"""Test for semantic segmentation export lib."""

import io
import os

from absl.testing import parameterized
import numpy as np
from PIL import Image
import tensorflow as tf, tf_keras

from official.core import exp_factory
from official.vision import registry_imports  # pylint: disable=unused-import
from official.vision.serving import semantic_segmentation


class SemanticSegmentationExportTest(tf.test.TestCase, parameterized.TestCase):

  def _get_segmentation_module(self,
                               input_type,
                               rescale_output,
                               preserve_aspect_ratio,
                               batch_size=1):
    params = exp_factory.get_exp_config('mnv2_deeplabv3_pascal')
    params.task.export_config.rescale_output = rescale_output
    params.task.train_data.preserve_aspect_ratio = preserve_aspect_ratio
    segmentation_module = semantic_segmentation.SegmentationModule(
        params,
        batch_size=batch_size,
        input_image_size=[112, 112],
        input_type=input_type)
    return segmentation_module

  def _export_from_module(self, module, input_type, save_directory):
    signatures = module.get_inference_signatures(
        {input_type: 'serving_default'})
    tf.saved_model.save(module, save_directory, signatures=signatures)

  def _get_dummy_input(self, input_type, input_image_size):
    """Get dummy input for the given input type."""

    height = input_image_size[0]
    width = input_image_size[1]
    if input_type == 'image_tensor':
      return tf.zeros((1, height, width, 3), dtype=np.uint8)
    elif input_type == 'image_bytes':
      image = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
      byte_io = io.BytesIO()
      image.save(byte_io, 'PNG')
      return [byte_io.getvalue()]
    elif input_type == 'tf_example':
      image_tensor = tf.zeros((height, width, 3), dtype=tf.uint8)
      encoded_jpeg = tf.image.encode_jpeg(tf.constant(image_tensor)).numpy()
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      tf.train.Feature(
                          bytes_list=tf.train.BytesList(value=[encoded_jpeg])),
              })).SerializeToString()
      return [example]
    elif input_type == 'tflite':
      return tf.zeros((1, height, width, 3), dtype=np.float32)

  @parameterized.parameters(
      ('image_tensor', False, [112, 112], False),
      ('image_bytes', False, [112, 112], False),
      ('tf_example', False, [112, 112], True),
      ('tflite', False, [112, 112], False),
      ('image_tensor', True, [112, 56], True),
      ('image_bytes', True, [112, 56], True),
      ('tf_example', True, [56, 112], False),
  )
  def test_export(self, input_type, rescale_output, input_image_size,
                  preserve_aspect_ratio):
    tmp_dir = self.get_temp_dir()
    module = self._get_segmentation_module(
        input_type=input_type,
        rescale_output=rescale_output,
        preserve_aspect_ratio=preserve_aspect_ratio)

    self._export_from_module(module, input_type, tmp_dir)

    self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'saved_model.pb')))
    self.assertTrue(
        os.path.exists(os.path.join(tmp_dir, 'variables', 'variables.index')))
    self.assertTrue(
        os.path.exists(
            os.path.join(tmp_dir, 'variables',
                         'variables.data-00000-of-00001')))

    imported = tf.saved_model.load(tmp_dir)
    segmentation_fn = imported.signatures['serving_default']

    images = self._get_dummy_input(input_type, input_image_size)
    if input_type != 'tflite':
      processed_images, _ = tf.nest.map_structure(
          tf.stop_gradient,
          tf.map_fn(
              module._build_inputs,
              elems=tf.zeros((1, 112, 112, 3), dtype=tf.uint8),
              fn_output_signature=(tf.TensorSpec(
                  shape=[112, 112, 3], dtype=tf.float32),
                                   tf.TensorSpec(
                                       shape=[4, 2], dtype=tf.float32))))
    else:
      processed_images = images

    logits = module.model(processed_images, training=False)['logits']
    if rescale_output:
      expected_output = tf.image.resize(
          logits, input_image_size, method='bilinear')
    else:
      expected_output = tf.image.resize(logits, [112, 112], method='bilinear')
    out = segmentation_fn(tf.constant(images))
    self.assertAllClose(out['logits'].numpy(), expected_output.numpy())

  def test_export_invalid_batch_size(self):
    batch_size = 3
    tmp_dir = self.get_temp_dir()
    module = self._get_segmentation_module(
        input_type='image_tensor',
        rescale_output=True,
        preserve_aspect_ratio=False,
        batch_size=batch_size)
    with self.assertRaisesRegex(ValueError,
                                'Batch size cannot be more than 1.'):
      self._export_from_module(module, 'image_tensor', tmp_dir)


if __name__ == '__main__':
  tf.test.main()
