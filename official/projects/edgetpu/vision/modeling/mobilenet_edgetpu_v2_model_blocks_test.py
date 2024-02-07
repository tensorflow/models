# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for mobilenet_edgetpu_v2_model_blocks."""

import tensorflow as tf, tf_keras

from official.projects.edgetpu.vision.modeling import custom_layers
from official.projects.edgetpu.vision.modeling import mobilenet_edgetpu_v2_model_blocks


class MobilenetEdgetpuV2ModelBlocksTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.model_config = mobilenet_edgetpu_v2_model_blocks.ModelConfig()

  def test_model_creatation(self):
    model_input = tf_keras.layers.Input(shape=(224, 224, 1))
    model_output = mobilenet_edgetpu_v2_model_blocks.mobilenet_edgetpu_v2(
        image_input=model_input,
        config=self.model_config)
    test_model = tf_keras.Model(inputs=model_input, outputs=model_output)
    self.assertIsInstance(test_model, tf_keras.Model)
    self.assertEqual(test_model.input.shape, (None, 224, 224, 1))
    self.assertEqual(test_model.output.shape, (None, 1001))

  def test_model_with_customized_kernel_initializer(self):
    self.model_config.conv_kernel_initializer = 'he_uniform'
    self.model_config.dense_kernel_initializer = 'glorot_normal'
    model_input = tf_keras.layers.Input(shape=(224, 224, 1))
    model_output = mobilenet_edgetpu_v2_model_blocks.mobilenet_edgetpu_v2(
        image_input=model_input,
        config=self.model_config)
    test_model = tf_keras.Model(inputs=model_input, outputs=model_output)

    conv_layer_stack = []
    for layer in test_model.layers:
      if (isinstance(layer, tf_keras.layers.Conv2D) or
          isinstance(layer, tf_keras.layers.DepthwiseConv2D) or
          isinstance(layer, custom_layers.GroupConv2D)):
        conv_layer_stack.append(layer)
    self.assertGreater(len(conv_layer_stack), 2)
    # The last Conv layer is used as a Dense layer.
    for layer in conv_layer_stack[:-1]:
      if isinstance(layer, custom_layers.GroupConv2D):
        self.assertIsInstance(layer.kernel_initializer,
                              tf_keras.initializers.GlorotUniform)
      elif isinstance(layer, tf_keras.layers.Conv2D):
        self.assertIsInstance(layer.kernel_initializer,
                              tf_keras.initializers.HeUniform)
      elif isinstance(layer, tf_keras.layers.DepthwiseConv2D):
        self.assertIsInstance(layer.depthwise_initializer,
                              tf_keras.initializers.HeUniform)

    self.assertIsInstance(conv_layer_stack[-1].kernel_initializer,
                          tf_keras.initializers.GlorotNormal)


if __name__ == '__main__':
  tf.test.main()
