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
"""Tests for factory.py."""

# Import libraries
from absl.testing import parameterized
import tensorflow as tf

from official.vision.beta.configs import backbones
from official.vision.beta.configs import backbones_3d
from official.vision.beta.configs import image_classification as classification_cfg
from official.vision.beta.configs import maskrcnn as maskrcnn_cfg
from official.vision.beta.configs import retinanet as retinanet_cfg
from official.vision.beta.configs import video_classification as video_classification_cfg
from official.vision.beta.modeling import factory
from official.vision.beta.modeling import factory_3d


class ClassificationModelBuilderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ('resnet', (224, 224), 5e-5),
      ('resnet', (224, 224), None),
      ('resnet', (None, None), 5e-5),
      ('resnet', (None, None), None),
  )
  def test_builder(self, backbone_type, input_size, weight_decay):
    num_classes = 2
    input_specs = tf.keras.layers.InputSpec(
        shape=[None, input_size[0], input_size[1], 3])
    model_config = classification_cfg.ImageClassificationModel(
        num_classes=num_classes,
        backbone=backbones.Backbone(type=backbone_type))
    l2_regularizer = (
        tf.keras.regularizers.l2(weight_decay) if weight_decay else None)
    _ = factory.build_classification_model(
        input_specs=input_specs,
        model_config=model_config,
        l2_regularizer=l2_regularizer)


class MaskRCNNBuilderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ('resnet', (640, 640)),
      ('resnet', (None, None)),
  )
  def test_builder(self, backbone_type, input_size):
    num_classes = 2
    input_specs = tf.keras.layers.InputSpec(
        shape=[None, input_size[0], input_size[1], 3])
    model_config = maskrcnn_cfg.MaskRCNN(
        num_classes=num_classes,
        backbone=backbones.Backbone(type=backbone_type))
    l2_regularizer = tf.keras.regularizers.l2(5e-5)
    _ = factory.build_maskrcnn(
        input_specs=input_specs,
        model_config=model_config,
        l2_regularizer=l2_regularizer)


class RetinaNetBuilderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ('resnet', (640, 640)),
      ('resnet', (None, None)),
  )
  def test_builder(self, backbone_type, input_size):
    num_classes = 2
    input_specs = tf.keras.layers.InputSpec(
        shape=[None, input_size[0], input_size[1], 3])
    model_config = retinanet_cfg.RetinaNet(
        num_classes=num_classes,
        backbone=backbones.Backbone(type=backbone_type))
    l2_regularizer = tf.keras.regularizers.l2(5e-5)
    _ = factory.build_retinanet(
        input_specs=input_specs,
        model_config=model_config,
        l2_regularizer=l2_regularizer)


class VideoClassificationModelBuilderTest(parameterized.TestCase,
                                          tf.test.TestCase):

  @parameterized.parameters(
      ('resnet_3d', (8, 224, 224), 5e-5),
      ('resnet_3d', (None, None, None), 5e-5),
  )
  def test_builder(self, backbone_type, input_size, weight_decay):
    input_specs = tf.keras.layers.InputSpec(
        shape=[None, input_size[0], input_size[1], input_size[2], 3])
    model_config = video_classification_cfg.VideoClassificationModel(
        backbone=backbones_3d.Backbone3D(type=backbone_type))
    l2_regularizer = (
        tf.keras.regularizers.l2(weight_decay) if weight_decay else None)
    _ = factory_3d.build_video_classification_model(
        input_specs=input_specs,
        model_config=model_config,
        num_classes=2,
        l2_regularizer=l2_regularizer)


if __name__ == '__main__':
  tf.test.main()
