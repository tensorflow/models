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
"""Tests for classification network."""

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.vision.beta.modeling import backbones
from official.vision.beta.modeling import classification_model


class ClassificationNetworkTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (128, 50, 'relu'),
      (128, 50, 'relu'),
      (128, 50, 'swish'),
  )
  def test_resnet_network_creation(
      self, input_size, resnet_model_id, activation):
    """Test for creation of a ResNet-50 classifier."""
    inputs = np.random.rand(2, input_size, input_size, 3)

    tf.keras.backend.set_image_data_format('channels_last')

    backbone = backbones.ResNet(
        model_id=resnet_model_id, activation=activation)
    self.assertEqual(backbone.count_params(), 23561152)

    num_classes = 1000
    model = classification_model.ClassificationModel(
        backbone=backbone,
        num_classes=num_classes,
        dropout_rate=0.2,
    )
    self.assertEqual(model.count_params(), 25610152)

    logits = model(inputs)
    self.assertAllEqual([2, num_classes], logits.numpy().shape)

  def test_revnet_network_creation(self):
    """Test for creation of a RevNet-56 classifier."""
    revnet_model_id = 56
    inputs = np.random.rand(2, 224, 224, 3)

    tf.keras.backend.set_image_data_format('channels_last')

    backbone = backbones.RevNet(model_id=revnet_model_id)
    self.assertEqual(backbone.count_params(), 19473792)

    num_classes = 1000
    model = classification_model.ClassificationModel(
        backbone=backbone,
        num_classes=num_classes,
        dropout_rate=0.2,
        add_head_batch_norm=True,
    )
    self.assertEqual(model.count_params(), 22816104)

    logits = model(inputs)
    self.assertAllEqual([2, num_classes], logits.numpy().shape)

  @combinations.generate(
      combinations.combine(
          mobilenet_model_id=[
              'MobileNetV1',
              'MobileNetV2',
              'MobileNetV3Large',
              'MobileNetV3Small',
              'MobileNetV3EdgeTPU',
              'MobileNetMultiAVG',
              'MobileNetMultiMAX',
          ],
          filter_size_scale=[1.0, 0.75],
      ))
  def test_mobilenet_network_creation(self, mobilenet_model_id,
                                      filter_size_scale):
    """Test for creation of a MobileNet classifier."""
    mobilenet_params = {
        ('MobileNetV1', 1.0): 4254889,
        ('MobileNetV1', 0.75): 2602745,
        ('MobileNetV2', 1.0): 3540265,
        ('MobileNetV2', 0.75): 2664345,
        ('MobileNetV3Large', 1.0): 5508713,
        ('MobileNetV3Large', 0.75): 4013897,
        ('MobileNetV3Small', 1.0): 2555993,
        ('MobileNetV3Small', 0.75): 2052577,
        ('MobileNetV3EdgeTPU', 1.0): 4131593,
        ('MobileNetV3EdgeTPU', 0.75): 3019569,
        ('MobileNetMultiAVG', 1.0): 4982857,
        ('MobileNetMultiAVG', 0.75): 3628145,
        ('MobileNetMultiMAX', 1.0): 4453001,
        ('MobileNetMultiMAX', 0.75): 3324257,
    }

    inputs = np.random.rand(2, 224, 224, 3)

    tf.keras.backend.set_image_data_format('channels_last')

    backbone = backbones.MobileNet(
        model_id=mobilenet_model_id, filter_size_scale=filter_size_scale)

    num_classes = 1001
    model = classification_model.ClassificationModel(
        backbone=backbone,
        num_classes=num_classes,
        dropout_rate=0.2,
    )
    self.assertEqual(model.count_params(),
                     mobilenet_params[(mobilenet_model_id, filter_size_scale)])

    logits = model(inputs)
    self.assertAllEqual([2, num_classes], logits.numpy().shape)

  @combinations.generate(
      combinations.combine(
          strategy=[
              strategy_combinations.cloud_tpu_strategy,
              strategy_combinations.one_device_strategy_gpu,
          ],
          use_sync_bn=[False, True],
      ))
  def test_sync_bn_multiple_devices(self, strategy, use_sync_bn):
    """Test for sync bn on TPU and GPU devices."""
    inputs = np.random.rand(64, 128, 128, 3)

    tf.keras.backend.set_image_data_format('channels_last')

    with strategy.scope():
      backbone = backbones.ResNet(model_id=50, use_sync_bn=use_sync_bn)

      model = classification_model.ClassificationModel(
          backbone=backbone,
          num_classes=1000,
          dropout_rate=0.2,
      )
      _ = model(inputs)

  @combinations.generate(
      combinations.combine(
          strategy=[
              strategy_combinations.one_device_strategy_gpu,
          ],
          data_format=['channels_last', 'channels_first'],
          input_dim=[1, 3, 4]))
  def test_data_format_gpu(self, strategy, data_format, input_dim):
    """Test for different data formats on GPU devices."""
    if data_format == 'channels_last':
      inputs = np.random.rand(2, 128, 128, input_dim)
    else:
      inputs = np.random.rand(2, input_dim, 128, 128)
    input_specs = tf.keras.layers.InputSpec(shape=inputs.shape)

    tf.keras.backend.set_image_data_format(data_format)

    with strategy.scope():
      backbone = backbones.ResNet(model_id=50, input_specs=input_specs)

      model = classification_model.ClassificationModel(
          backbone=backbone,
          num_classes=1000,
          input_specs=input_specs,
      )
      _ = model(inputs)

  def test_serialize_deserialize(self):
    """Validate the classification net can be serialized and deserialized."""

    tf.keras.backend.set_image_data_format('channels_last')
    backbone = backbones.ResNet(model_id=50)

    model = classification_model.ClassificationModel(
        backbone=backbone, num_classes=1000)

    config = model.get_config()
    new_model = classification_model.ClassificationModel.from_config(config)

    # Validate that the config can be forced to JSON.
    _ = new_model.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(model.get_config(), new_model.get_config())


if __name__ == '__main__':
  tf.test.main()
