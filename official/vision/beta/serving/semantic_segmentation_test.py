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
"""Test for semantic segmentation export lib."""

import io
import os

from absl.testing import parameterized
import numpy as np
from PIL import Image
import tensorflow as tf

from official.common import registry_imports  # pylint: disable=unused-import
from official.core import exp_factory
from official.vision.beta.serving import semantic_segmentation


class SemanticSegmentationExportTest(tf.test.TestCase, parameterized.TestCase):

  def _get_segmentation_module(self):
    params = exp_factory.get_exp_config('seg_deeplabv3_pascal')
    params.task.model.backbone.dilated_resnet.model_id = 50
    segmentation_module = semantic_segmentation.SegmentationModule(
        params, batch_size=1, input_image_size=[112, 112])
    return segmentation_module

  def _export_from_module(self, module, input_type, save_directory):
    if input_type == 'image_tensor':
      input_signature = tf.TensorSpec(shape=[None, 112, 112, 3], dtype=tf.uint8)
      signatures = {
          'serving_default':
              module.inference_from_image_tensors.get_concrete_function(
                  input_signature)
      }
    elif input_type == 'image_bytes':
      input_signature = tf.TensorSpec(shape=[None], dtype=tf.string)
      signatures = {
          'serving_default':
              module.inference_from_image_bytes.get_concrete_function(
                  input_signature)
      }
    elif input_type == 'tf_example':
      input_signature = tf.TensorSpec(shape=[None], dtype=tf.string)
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

  def _get_dummy_input(self, input_type):
    """Get dummy input for the given input type."""

    if input_type == 'image_tensor':
      return tf.zeros((1, 112, 112, 3), dtype=np.uint8)
    elif input_type == 'image_bytes':
      image = Image.fromarray(np.zeros((112, 112, 3), dtype=np.uint8))
      byte_io = io.BytesIO()
      image.save(byte_io, 'PNG')
      return [byte_io.getvalue()]
    elif input_type == 'tf_example':
      image_tensor = tf.zeros((112, 112, 3), dtype=tf.uint8)
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

    module = self._get_segmentation_module()
    model = module.build_model()

    self._export_from_module(module, input_type, tmp_dir)

    self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'saved_model.pb')))
    self.assertTrue(os.path.exists(
        os.path.join(tmp_dir, 'variables', 'variables.index')))
    self.assertTrue(os.path.exists(
        os.path.join(tmp_dir, 'variables', 'variables.data-00000-of-00001')))

    imported = tf.saved_model.load(tmp_dir)
    segmentation_fn = imported.signatures['serving_default']

    images = self._get_dummy_input(input_type)
    processed_images = tf.nest.map_structure(
        tf.stop_gradient,
        tf.map_fn(
            module._build_inputs,
            elems=tf.zeros((1, 112, 112, 3), dtype=tf.uint8),
            fn_output_signature=tf.TensorSpec(
                shape=[112, 112, 3], dtype=tf.float32)))
    expected_output = tf.image.resize(
        model(processed_images, training=False), [112, 112], method='bilinear')
    out = segmentation_fn(tf.constant(images))
    self.assertAllClose(out['predicted_masks'].numpy(), expected_output.numpy())

if __name__ == '__main__':
  tf.test.main()
