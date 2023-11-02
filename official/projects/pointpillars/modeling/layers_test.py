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

"""Tests for backbones."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.projects.pointpillars.modeling import layers


class ConvBlockTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ([1, 8, 8, 3], 4, 1, False),
      ([1, 8, 8, 3], 4, 2, False),
      ([1, 8, 8, 3], 2, 1, True),
      ([1, 8, 8, 3], 2, 2, True),
  )
  def test_creation(self, input_shape, filters, strides,
                    use_transpose_conv):
    kernel_size = 3
    n, h, w, _ = input_shape
    inputs = tf_keras.Input(shape=input_shape[1:], batch_size=n)
    block = layers.ConvBlock(filters, kernel_size, strides, use_transpose_conv)
    outputs = block(inputs)

    if not use_transpose_conv:
      if strides == 1:
        self.assertAllEqual([n, h, w, filters], outputs.shape.as_list())
      elif strides == 2:
        self.assertAllEqual([n, h/2, w/2, filters], outputs.shape.as_list())
    else:
      if strides == 1:
        self.assertAllEqual([n, h, w, filters], outputs.shape.as_list())
      elif strides == 2:
        self.assertAllEqual([n, h*2, w*2, filters], outputs.shape.as_list())

  def test_serialization(self):
    kwargs = dict(
        filters=3,
        kernel_size=3,
        strides=1,
        use_transpose_conv=False,
        kernel_initializer=None,
        kernel_regularizer=None,
        use_bias=False,
        bias_initializer=None,
        bias_regularizer=None,
        use_sync_bn=True,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        bn_trainable=True,
        activation='relu',
    )
    net = layers.ConvBlock(**kwargs)
    expected_config = kwargs
    self.assertEqual(net.get_config(), expected_config)

    new_net = layers.ConvBlock.from_config(net.get_config())
    self.assertAllEqual(net.get_config(), new_net.get_config())

if __name__ == '__main__':
  tf.test.main()
