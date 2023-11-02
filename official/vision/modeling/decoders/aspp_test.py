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

"""Tests for aspp."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.vision.modeling.backbones import resnet
from official.vision.modeling.decoders import aspp


class ASPPTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (3, [6, 12, 18, 24], 128, 'v1'),
      (3, [6, 12, 18], 128, 'v1'),
      (3, [6, 12], 256, 'v1'),
      (4, [6, 12, 18, 24], 128, 'v2'),
      (4, [6, 12, 18], 128, 'v2'),
      (4, [6, 12], 256, 'v2'),
  )
  def test_network_creation(self, level, dilation_rates, num_filters,
                            spp_layer_version):
    """Test creation of ASPP."""

    input_size = 256
    tf_keras.backend.set_image_data_format('channels_last')

    inputs = tf_keras.Input(shape=(input_size, input_size, 3), batch_size=1)

    backbone = resnet.ResNet(model_id=50)
    network = aspp.ASPP(
        level=level,
        dilation_rates=dilation_rates,
        num_filters=num_filters,
        spp_layer_version=spp_layer_version)

    endpoints = backbone(inputs)
    feats = network(endpoints)

    self.assertIn(str(level), feats)
    self.assertAllEqual(
        [1, input_size // 2**level, input_size // 2**level, num_filters],
        feats[str(level)].shape.as_list())

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        level=3,
        dilation_rates=[6, 12],
        num_filters=256,
        pool_kernel_size=None,
        use_sync_bn=False,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        activation='relu',
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        interpolation='bilinear',
        dropout_rate=0.2,
        use_depthwise_convolution='false',
        spp_layer_version='v1',
        output_tensor=False,
        dtype='float32',
        name='aspp',
        trainable=True)
    network = aspp.ASPP(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = aspp.ASPP.from_config(network.get_config())

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


if __name__ == '__main__':
  tf.test.main()
