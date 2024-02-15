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

"""Tests for factory functions."""
# Import libraries
from absl.testing import parameterized
import tensorflow as tf, tf_keras

from tensorflow.python.distribute import combinations
from official.vision.configs import backbones as backbones_cfg
from official.vision.configs import backbones_3d as backbones_3d_cfg
from official.vision.configs import common as common_cfg
from official.vision.modeling import backbones
from official.vision.modeling.backbones import factory


class FactoryTest(tf.test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(model_id=[18, 34, 50, 101, 152],))
  def test_resnet_creation(self, model_id):
    """Test creation of ResNet models."""

    network = backbones.ResNet(
        model_id=model_id, se_ratio=0.0, norm_momentum=0.99, norm_epsilon=1e-5)

    backbone_config = backbones_cfg.Backbone(
        type='resnet',
        resnet=backbones_cfg.ResNet(model_id=model_id, se_ratio=0.0))
    norm_activation_config = common_cfg.NormActivation(
        norm_momentum=0.99, norm_epsilon=1e-5, use_sync_bn=False)

    factory_network = factory.build_backbone(
        input_specs=tf_keras.layers.InputSpec(shape=[None, None, None, 3]),
        backbone_config=backbone_config,
        norm_activation_config=norm_activation_config)

    network_config = network.get_config()
    factory_network_config = factory_network.get_config()

    self.assertEqual(network_config, factory_network_config)

  @combinations.generate(
      combinations.combine(
          model_id=['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'],
          se_ratio=[0.0, 0.25],
      ))
  def test_efficientnet_creation(self, model_id, se_ratio):
    """Test creation of EfficientNet models."""

    network = backbones.EfficientNet(
        model_id=model_id,
        se_ratio=se_ratio,
        norm_momentum=0.99,
        norm_epsilon=1e-5)

    backbone_config = backbones_cfg.Backbone(
        type='efficientnet',
        efficientnet=backbones_cfg.EfficientNet(
            model_id=model_id, se_ratio=se_ratio))
    norm_activation_config = common_cfg.NormActivation(
        norm_momentum=0.99, norm_epsilon=1e-5, use_sync_bn=False)

    factory_network = factory.build_backbone(
        input_specs=tf_keras.layers.InputSpec(shape=[None, None, None, 3]),
        backbone_config=backbone_config,
        norm_activation_config=norm_activation_config)

    network_config = network.get_config()
    factory_network_config = factory_network.get_config()

    self.assertEqual(network_config, factory_network_config)

  @combinations.generate(
      combinations.combine(
          model_id=[
              'MobileNetV1',
              'MobileNetV2',
              'MobileNetV3Large',
              'MobileNetV3Small',
              'MobileNetV3EdgeTPU',
              'MobileNetV4ConvSmall',
              'MobileNetV4ConvMedium',
              'MobileNetV4ConvLarge',
              'MobileNetV4HybridMedium',
              'MobileNetV4HybridLarge',
          ],
          filter_size_scale=[1.0, 0.75],
      )
  )
  def test_mobilenet_creation(self, model_id, filter_size_scale):
    """Test creation of Mobilenet models."""

    network = backbones.MobileNet(
        model_id=model_id,
        filter_size_scale=filter_size_scale,
        norm_momentum=0.99,
        norm_epsilon=1e-5)

    backbone_config = backbones_cfg.Backbone(
        type='mobilenet',
        mobilenet=backbones_cfg.MobileNet(
            model_id=model_id, filter_size_scale=filter_size_scale))
    norm_activation_config = common_cfg.NormActivation(
        norm_momentum=0.99, norm_epsilon=1e-5, use_sync_bn=False)

    factory_network = factory.build_backbone(
        input_specs=tf_keras.layers.InputSpec(shape=[None, None, None, 3]),
        backbone_config=backbone_config,
        norm_activation_config=norm_activation_config)

    network_config = network.get_config()
    factory_network_config = factory_network.get_config()

    self.assertEqual(network_config, factory_network_config)

  @combinations.generate(combinations.combine(model_id=['49'],))
  def test_spinenet_creation(self, model_id):
    """Test creation of SpineNet models."""
    input_size = 128
    min_level = 3
    max_level = 7

    input_specs = tf_keras.layers.InputSpec(
        shape=[None, input_size, input_size, 3])
    network = backbones.SpineNet(
        input_specs=input_specs,
        min_level=min_level,
        max_level=max_level,
        norm_momentum=0.99,
        norm_epsilon=1e-5)

    backbone_config = backbones_cfg.Backbone(
        type='spinenet',
        spinenet=backbones_cfg.SpineNet(model_id=model_id))
    norm_activation_config = common_cfg.NormActivation(
        norm_momentum=0.99, norm_epsilon=1e-5, use_sync_bn=False)

    factory_network = factory.build_backbone(
        input_specs=tf_keras.layers.InputSpec(
            shape=[None, input_size, input_size, 3]),
        backbone_config=backbone_config,
        norm_activation_config=norm_activation_config)

    network_config = network.get_config()
    factory_network_config = factory_network.get_config()

    self.assertEqual(network_config, factory_network_config)

  @combinations.generate(
      combinations.combine(model_id=[38, 56, 104],))
  def test_revnet_creation(self, model_id):
    """Test creation of RevNet models."""
    network = backbones.RevNet(
        model_id=model_id, norm_momentum=0.99, norm_epsilon=1e-5)

    backbone_config = backbones_cfg.Backbone(
        type='revnet',
        revnet=backbones_cfg.RevNet(model_id=model_id))
    norm_activation_config = common_cfg.NormActivation(
        norm_momentum=0.99, norm_epsilon=1e-5, use_sync_bn=False)

    factory_network = factory.build_backbone(
        input_specs=tf_keras.layers.InputSpec(shape=[None, None, None, 3]),
        backbone_config=backbone_config,
        norm_activation_config=norm_activation_config)

    network_config = network.get_config()
    factory_network_config = factory_network.get_config()

    self.assertEqual(network_config, factory_network_config)

  @combinations.generate(combinations.combine(model_type=['resnet_3d'],))
  def test_resnet_3d_creation(self, model_type):
    """Test creation of ResNet 3D models."""
    backbone_cfg = backbones_3d_cfg.Backbone3D(type=model_type).get()
    temporal_strides = []
    temporal_kernel_sizes = []
    for block_spec in backbone_cfg.block_specs:
      temporal_strides.append(block_spec.temporal_strides)
      temporal_kernel_sizes.append(block_spec.temporal_kernel_sizes)

    _ = backbones.ResNet3D(
        model_id=backbone_cfg.model_id,
        temporal_strides=temporal_strides,
        temporal_kernel_sizes=temporal_kernel_sizes,
        norm_momentum=0.99,
        norm_epsilon=1e-5)

  @combinations.generate(
      combinations.combine(
          model_id=[
              'MobileDetCPU',
              'MobileDetDSP',
              'MobileDetEdgeTPU',
              'MobileDetGPU'],
          filter_size_scale=[1.0, 0.75],
      ))
  def test_mobiledet_creation(self, model_id, filter_size_scale):
    """Test creation of Mobiledet models."""

    network = backbones.MobileDet(
        model_id=model_id,
        filter_size_scale=filter_size_scale,
        norm_momentum=0.99,
        norm_epsilon=1e-5)

    backbone_config = backbones_cfg.Backbone(
        type='mobiledet',
        mobiledet=backbones_cfg.MobileDet(
            model_id=model_id, filter_size_scale=filter_size_scale))
    norm_activation_config = common_cfg.NormActivation(
        norm_momentum=0.99, norm_epsilon=1e-5, use_sync_bn=False)

    factory_network = factory.build_backbone(
        input_specs=tf_keras.layers.InputSpec(shape=[None, None, None, 3]),
        backbone_config=backbone_config,
        norm_activation_config=norm_activation_config)

    network_config = network.get_config()
    factory_network_config = factory_network.get_config()

    self.assertEqual(network_config, factory_network_config)

if __name__ == '__main__':
  tf.test.main()
