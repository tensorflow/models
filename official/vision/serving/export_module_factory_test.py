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

"""Test for vision modules."""

import io
import os

from absl.testing import parameterized
import numpy as np
from PIL import Image
import tensorflow as tf, tf_keras

from official.core import exp_factory
from official.core import export_base
from official.vision import registry_imports  # pylint: disable=unused-import
from official.vision.dataloaders import classification_input
from official.vision.serving import export_module_factory


class ImageClassificationExportTest(tf.test.TestCase, parameterized.TestCase):

  def _get_classification_module(self, input_type, input_image_size):
    params = exp_factory.get_exp_config('resnet_imagenet')
    params.task.model.backbone.resnet.model_id = 18
    module = export_module_factory.create_classification_export_module(
        params, input_type, batch_size=1, input_image_size=input_image_size)
    return module

  def _get_dummy_input(self, input_type):
    """Get dummy input for the given input type."""

    if input_type == 'image_tensor':
      return tf.zeros((1, 32, 32, 3), dtype=np.uint8)
    elif input_type == 'image_bytes':
      image = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
      byte_io = io.BytesIO()
      image.save(byte_io, 'PNG')
      return [byte_io.getvalue()]
    elif input_type == 'tf_example':
      image_tensor = tf.zeros((32, 32, 3), dtype=tf.uint8)
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
    input_image_size = [32, 32]
    tmp_dir = self.get_temp_dir()
    module = self._get_classification_module(input_type, input_image_size)
    # Test that the model restores any attrs that are trackable objects
    # (eg: tables, resource variables, keras models/layers, tf.hub modules).
    module.model.test_trackable = tf_keras.layers.InputLayer(input_shape=(4,))
    ckpt_path = tf.train.Checkpoint(model=module.model).save(
        os.path.join(tmp_dir, 'ckpt'))
    export_dir = export_base.export(
        module, [input_type],
        export_savedmodel_dir=tmp_dir,
        checkpoint_path=ckpt_path,
        timestamped=False)

    self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'saved_model.pb')))
    self.assertTrue(os.path.exists(
        os.path.join(tmp_dir, 'variables', 'variables.index')))
    self.assertTrue(os.path.exists(
        os.path.join(tmp_dir, 'variables', 'variables.data-00000-of-00001')))

    imported = tf.saved_model.load(export_dir)
    classification_fn = imported.signatures['serving_default']

    images = self._get_dummy_input(input_type)

    def preprocess_image_fn(inputs):
      return classification_input.Parser.inference_fn(
          inputs, input_image_size, num_channels=3)

    processed_images = tf.map_fn(
        preprocess_image_fn,
        elems=tf.zeros([1] + input_image_size + [3], dtype=tf.uint8),
        fn_output_signature=tf.TensorSpec(
            shape=input_image_size + [3], dtype=tf.float32))
    expected_logits = module.model(processed_images, training=False)
    expected_prob = tf.nn.softmax(expected_logits)
    out = classification_fn(tf.constant(images))

    # The imported model should contain any trackable attrs that the original
    # model had.
    self.assertTrue(hasattr(imported.model, 'test_trackable'))
    self.assertAllClose(
        out['logits'].numpy(), expected_logits.numpy(), rtol=1e-04, atol=1e-04)
    self.assertAllClose(
        out['probs'].numpy(), expected_prob.numpy(), rtol=1e-04, atol=1e-04)


if __name__ == '__main__':
  tf.test.main()
