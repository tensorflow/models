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

"""Test for semantic_segmentation_3d export lib."""

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

# pylint: disable=unused-import
from official.core import exp_factory
from official.projects.volumetric_models.configs import semantic_segmentation_3d as exp_cfg
from official.projects.volumetric_models.modeling import backbones
from official.projects.volumetric_models.modeling import decoders
from official.projects.volumetric_models.serving import semantic_segmentation_3d


class SemanticSegmentationExportTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._num_channels = 2
    self._input_image_size = [32, 32, 32]
    self._params = exp_factory.get_exp_config('seg_unet3d_test')

    input_shape = self._input_image_size + [self._num_channels]
    self._image_array = np.zeros(shape=input_shape, dtype=np.uint8)

  def _get_segmentation_module(self):
    return semantic_segmentation_3d.SegmentationModule(
        self._params,
        batch_size=1,
        input_image_size=self._input_image_size,
        num_channels=self._num_channels)

  def _export_from_module(self, module, input_type: str, save_directory: str):
    signatures = module.get_inference_signatures(
        {input_type: 'serving_default'})
    tf.saved_model.save(module,
                        save_directory,
                        signatures=signatures)

  def _get_dummy_input(self, input_type):
    """Get dummy input for the given input type."""

    if input_type == 'image_tensor':
      image_tensor = tf.convert_to_tensor(self._image_array, dtype=tf.uint8)
      return tf.expand_dims(image_tensor, axis=0)
    if input_type == 'image_bytes':
      return [self._image_array.tostring()]
    if input_type == 'tf_example':
      encoded_image = self._image_array.tostring()
      example = tf.train.Example(
          features=tf.train.Features(
              feature={
                  'image/encoded':
                      tf.train.Feature(
                          bytes_list=tf.train.BytesList(value=[encoded_image])),
              })).SerializeToString()
      return [example]

  @parameterized.parameters(
      {'input_type': 'image_tensor'},
      {'input_type': 'image_bytes'},
      {'input_type': 'tf_example'},
  )
  def test_export(self, input_type: str = 'image_tensor'):
    tmp_dir = self.get_temp_dir()

    module = self._get_segmentation_module()
    self._export_from_module(module, input_type, tmp_dir)

    # Check if model is successfully exported.
    self.assertTrue(tf.io.gfile.exists(os.path.join(tmp_dir, 'saved_model.pb')))
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(tmp_dir, 'variables', 'variables.index')))
    self.assertTrue(
        tf.io.gfile.exists(
            os.path.join(tmp_dir, 'variables',
                         'variables.data-00000-of-00001')))

    # Get inference signature from loaded SavedModel.
    imported = tf.saved_model.load(tmp_dir)
    segmentation_fn = imported.signatures['serving_default']

    images = self._get_dummy_input(input_type)
    image_tensor = self._get_dummy_input(input_type='image_tensor')

    # Perform inference using loaded SavedModel and model instance and check if
    # outputs equal.
    expected_output = module.model(image_tensor, training=False)
    out = segmentation_fn(tf.constant(images))
    self.assertAllClose(out['logits'].numpy(),
                        expected_output['logits'].numpy())


if __name__ == '__main__':
  tf.test.main()
