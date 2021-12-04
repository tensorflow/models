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

# Lint as: python3
"""Test for image classification export lib."""

import io
import os

from absl.testing import parameterized
import numpy as np
from PIL import Image
import tensorflow as tf

from official.common import registry_imports  # pylint: disable=unused-import
from official.core import exp_factory
from official.vision.beta.serving import image_classification


class ImageClassificationExportTest(tf.test.TestCase, parameterized.TestCase):

  def _get_classification_module(self, input_type):
    params = exp_factory.get_exp_config('resnet_imagenet')
    params.task.model.backbone.resnet.model_id = 18
    classification_module = image_classification.ClassificationModule(
        params,
        batch_size=1,
        input_image_size=[224, 224],
        input_type=input_type)
    return classification_module

  def _export_from_module(self, module, input_type, save_directory):
    signatures = module.get_inference_signatures(
        {input_type: 'serving_default'})
    tf.saved_model.save(module,
                        save_directory,
                        signatures=signatures)

  def _get_dummy_input(self, input_type):
    """Get dummy input for the given input type."""

    if input_type == 'image_tensor':
      return tf.zeros((1, 224, 224, 3), dtype=np.uint8)
    elif input_type == 'image_bytes':
      image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
      byte_io = io.BytesIO()
      image.save(byte_io, 'PNG')
      return [byte_io.getvalue()]
    elif input_type == 'tf_example':
      image_tensor = tf.zeros((224, 224, 3), dtype=tf.uint8)
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
      return tf.zeros((1, 224, 224, 3), dtype=np.float32)

  @parameterized.parameters(
      {'input_type': 'image_tensor'},
      {'input_type': 'image_bytes'},
      {'input_type': 'tf_example'},
      {'input_type': 'tflite'},
  )
  def test_export(self, input_type='image_tensor'):
    tmp_dir = self.get_temp_dir()
    module = self._get_classification_module(input_type)
    # Test that the model restores any attrs that are trackable objects
    # (eg: tables, resource variables, keras models/layers, tf.hub modules).
    module.model.test_trackable = tf.keras.layers.InputLayer(input_shape=(4,))

    self._export_from_module(module, input_type, tmp_dir)

    self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'saved_model.pb')))
    self.assertTrue(os.path.exists(
        os.path.join(tmp_dir, 'variables', 'variables.index')))
    self.assertTrue(os.path.exists(
        os.path.join(tmp_dir, 'variables', 'variables.data-00000-of-00001')))

    imported = tf.saved_model.load(tmp_dir)
    classification_fn = imported.signatures['serving_default']

    images = self._get_dummy_input(input_type)
    if input_type != 'tflite':
      processed_images = tf.nest.map_structure(
          tf.stop_gradient,
          tf.map_fn(
              module._build_inputs,
              elems=tf.zeros((1, 224, 224, 3), dtype=tf.uint8),
              fn_output_signature=tf.TensorSpec(
                  shape=[224, 224, 3], dtype=tf.float32)))
    else:
      processed_images = images
    expected_logits = module.model(processed_images, training=False)
    expected_prob = tf.nn.softmax(expected_logits)
    out = classification_fn(tf.constant(images))

    # The imported model should contain any trackable attrs that the original
    # model had.
    self.assertTrue(hasattr(imported.model, 'test_trackable'))
    self.assertAllClose(out['logits'].numpy(), expected_logits.numpy())
    self.assertAllClose(out['probs'].numpy(), expected_prob.numpy())


if __name__ == '__main__':
  tf.test.main()
