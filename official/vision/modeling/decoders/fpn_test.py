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

"""Tests for FPN."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.vision.modeling.backbones import mobilenet
from official.vision.modeling.backbones import resnet
from official.vision.modeling.decoders import fpn


class FPNTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (256, 3, 7, False, False, 'sum'),
      (256, 3, 7, False, True, 'sum'),
      (256, 3, 7, True, False, 'concat'),
      (256, 3, 7, True, True, 'concat'),
  )
  def test_network_creation(self, input_size, min_level, max_level,
                            use_separable_conv, use_keras_layer, fusion_type):
    """Test creation of FPN."""
    tf_keras.backend.set_image_data_format('channels_last')

    inputs = tf_keras.Input(shape=(input_size, input_size, 3), batch_size=1)

    backbone = resnet.ResNet(model_id=50)
    network = fpn.FPN(
        input_specs=backbone.output_specs,
        min_level=min_level,
        max_level=max_level,
        fusion_type=fusion_type,
        use_separable_conv=use_separable_conv,
        use_keras_layer=use_keras_layer)

    endpoints = backbone(inputs)
    feats = network(endpoints)

    for level in range(min_level, max_level + 1):
      self.assertIn(str(level), feats)
      self.assertAllEqual(
          [1, input_size // 2**level, input_size // 2**level, 256],
          feats[str(level)].shape.as_list())

  @parameterized.parameters(
      (256, 3, 7, False, False),
      (256, 3, 7, False, True),
      (256, 3, 7, True, False),
      (256, 3, 7, True, True),
  )
  def test_network_creation_with_mobilenet(self, input_size, min_level,
                                           max_level, use_separable_conv,
                                           use_keras_layer):
    """Test creation of FPN with mobilenet backbone."""
    tf_keras.backend.set_image_data_format('channels_last')

    inputs = tf_keras.Input(shape=(input_size, input_size, 3), batch_size=1)

    backbone = mobilenet.MobileNet(model_id='MobileNetV2')
    network = fpn.FPN(
        input_specs=backbone.output_specs,
        min_level=min_level,
        max_level=max_level,
        use_separable_conv=use_separable_conv,
        use_keras_layer=use_keras_layer)

    endpoints = backbone(inputs)
    feats = network(endpoints)

    for level in range(min_level, max_level + 1):
      self.assertIn(str(level), feats)
      self.assertAllEqual(
          [1, input_size // 2**level, input_size // 2**level, 256],
          feats[str(level)].shape.as_list())

  def test_serialize_deserialize(self):
    # Create a network object that sets all of its config options.
    kwargs = dict(
        input_specs=resnet.ResNet(model_id=50).output_specs,
        min_level=3,
        max_level=7,
        num_filters=256,
        fusion_type='sum',
        use_separable_conv=False,
        use_keras_layer=False,
        use_sync_bn=False,
        activation='relu',
        norm_momentum=0.99,
        norm_epsilon=0.001,
        kernel_initializer='VarianceScaling',
        kernel_regularizer=None,
        bias_regularizer=None,
    )
    network = fpn.FPN(**kwargs)

    expected_config = dict(kwargs)
    self.assertEqual(network.get_config(), expected_config)

    # Create another network object from the first object's config.
    new_network = fpn.FPN.from_config(network.get_config())

    # Validate that the config can be forced to JSON.
    _ = new_network.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(network.get_config(), new_network.get_config())


if __name__ == '__main__':
  tf.test.main()
