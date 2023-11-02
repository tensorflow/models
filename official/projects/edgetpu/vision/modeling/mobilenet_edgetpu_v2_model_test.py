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

"""Tests for mobilenet_edgetpu model."""

import os

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.projects.edgetpu.vision.modeling import common_modules
from official.projects.edgetpu.vision.modeling import mobilenet_edgetpu_v2_model


class MobilenetEdgeTPUV2BuildTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(tf.test.TestCase, self).setUp()
    # Ensure no model duplicates
    tf_keras.backend.clear_session()

  def test_create_mobilenet_edgetpu(self):
    model = mobilenet_edgetpu_v2_model.MobilenetEdgeTPUV2()
    self.assertEqual(common_modules.count_params(model), 6069657)

  def test_export_tflite(self):
    model = mobilenet_edgetpu_v2_model.MobilenetEdgeTPUV2()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tmp_dir = self.create_tempdir()
    output_tflite = os.path.join(tmp_dir, 'model_quant.tflite')
    tflite_buffer = converter.convert()
    tf.io.gfile.GFile(output_tflite, 'wb').write(tflite_buffer)
    self.assertTrue(tf.io.gfile.exists(output_tflite))

  def test_model_save_load(self):
    """Serializes and de-serializeds the model."""
    model_builder = mobilenet_edgetpu_v2_model.MobilenetEdgeTPUV2
    model = model_builder.from_name(model_name='mobilenet_edgetpu_v2')
    # Model always has a conv2d layer followed by the input layer, and we
    # compare the weight parameters of this layers for the original model and
    # the save-then-load model.
    first_conv_layer = model.get_layer('stem_conv2d')
    kernel_tensor = first_conv_layer.trainable_weights[0].numpy()
    model.save('/tmp/test_model')
    loaded_model = tf_keras.models.load_model('/tmp/test_model')
    loaded_first_conv_layer = loaded_model.get_layer('stem_conv2d')
    loaded_kernel_tensor = loaded_first_conv_layer.trainable_weights[0].numpy()

    self.assertAllClose(kernel_tensor, loaded_kernel_tensor)

  def test_model_initialization_failure(self):
    """Tests model can only be initialized with predefined model name."""
    model_builder = mobilenet_edgetpu_v2_model.MobilenetEdgeTPUV2
    with self.assertRaises(ValueError):
      _ = model_builder.from_name(model_name='undefined_model_name')


if __name__ == '__main__':
  tf.test.main()
