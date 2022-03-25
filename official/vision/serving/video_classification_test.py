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


# import io
import os
import random

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.core import exp_factory
from official.vision import registry_imports  # pylint: disable=unused-import
from official.vision.dataloaders import tfexample_utils
from official.vision.serving import video_classification


class VideoClassificationTest(tf.test.TestCase, parameterized.TestCase):

  def _get_classification_module(self):
    params = exp_factory.get_exp_config('video_classification_ucf101')
    params.task.train_data.feature_shape = (8, 64, 64, 3)
    params.task.validation_data.feature_shape = (8, 64, 64, 3)
    params.task.model.backbone.resnet_3d.model_id = 50
    classification_module = video_classification.VideoClassificationModule(
        params, batch_size=1, input_image_size=[8, 64, 64])
    return classification_module

  def _export_from_module(self, module, input_type, save_directory):
    signatures = module.get_inference_signatures(
        {input_type: 'serving_default'})
    tf.saved_model.save(module, save_directory, signatures=signatures)

  def _get_dummy_input(self, input_type, module=None):
    """Get dummy input for the given input type."""

    if input_type == 'image_tensor':
      images = np.random.randint(
          low=0, high=255, size=(1, 8, 64, 64, 3), dtype=np.uint8)
      # images = np.zeros((1, 8, 64, 64, 3), dtype=np.uint8)
      return images, images
    elif input_type == 'tf_example':
      example = tfexample_utils.make_video_test_example(
          image_shape=(64, 64, 3),
          audio_shape=(20, 128),
          label=random.randint(0, 100)).SerializeToString()
      images = tf.nest.map_structure(
          tf.stop_gradient,
          tf.map_fn(
              module._decode_tf_example,
              elems=tf.constant([example]),
              fn_output_signature={
                  video_classification.video_input.IMAGE_KEY: tf.string,
              }))
      images = images[video_classification.video_input.IMAGE_KEY]
      return [example], images
    else:
      raise ValueError(f'{input_type}')

  @parameterized.parameters(
      {'input_type': 'image_tensor'},
      {'input_type': 'tf_example'},
  )
  def test_export(self, input_type):
    tmp_dir = self.get_temp_dir()
    module = self._get_classification_module()

    self._export_from_module(module, input_type, tmp_dir)

    self.assertTrue(os.path.exists(os.path.join(tmp_dir, 'saved_model.pb')))
    self.assertTrue(
        os.path.exists(os.path.join(tmp_dir, 'variables', 'variables.index')))
    self.assertTrue(
        os.path.exists(
            os.path.join(tmp_dir, 'variables',
                         'variables.data-00000-of-00001')))

    imported = tf.saved_model.load(tmp_dir)
    classification_fn = imported.signatures['serving_default']

    images, images_tensor = self._get_dummy_input(input_type, module)
    processed_images = tf.nest.map_structure(
        tf.stop_gradient,
        tf.map_fn(
            module._preprocess_image,
            elems=images_tensor,
            fn_output_signature={
                'image': tf.float32,
            }))
    expected_logits = module.model(processed_images, training=False)
    expected_prob = tf.nn.softmax(expected_logits)
    out = classification_fn(tf.constant(images))

    # The imported model should contain any trackable attrs that the original
    # model had.
    self.assertAllClose(out['logits'].numpy(), expected_logits.numpy())
    self.assertAllClose(out['probs'].numpy(), expected_prob.numpy())


if __name__ == '__main__':
  tf.test.main()
