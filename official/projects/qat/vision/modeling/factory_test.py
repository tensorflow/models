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

"""Tests for factory.py."""

# Import libraries

from absl.testing import parameterized
import tensorflow as tf

from official.projects.qat.vision.configs import common
from official.projects.qat.vision.modeling import factory as qat_factory
from official.vision.configs import backbones
from official.vision.configs import decoders
from official.vision.configs import image_classification as classification_cfg
from official.vision.configs import retinanet as retinanet_cfg
from official.vision.configs import semantic_segmentation as semantic_segmentation_cfg
from official.vision.modeling import factory


class ClassificationModelBuilderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ('resnet', (224, 224), 5e-5),
      ('resnet', (224, 224), None),
      ('resnet', (None, None), 5e-5),
      ('resnet', (None, None), None),
      ('mobilenet', (224, 224), 5e-5),
      ('mobilenet', (224, 224), None),
      ('mobilenet', (None, None), 5e-5),
      ('mobilenet', (None, None), None),
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
    model = factory.build_classification_model(
        input_specs=input_specs,
        model_config=model_config,
        l2_regularizer=l2_regularizer)

    quantization_config = common.Quantization()
    _ = qat_factory.build_qat_classification_model(
        model=model,
        input_specs=input_specs,
        quantization=quantization_config,
        model_config=model_config,
        l2_regularizer=l2_regularizer)


class RetinaNetBuilderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ('spinenet_mobile', (640, 640), False),
  )
  def test_builder(self, backbone_type, input_size, has_attribute_heads):
    num_classes = 2
    input_specs = tf.keras.layers.InputSpec(
        shape=[None, input_size[0], input_size[1], 3])
    if has_attribute_heads:
      attribute_heads_config = [
          retinanet_cfg.AttributeHead(name='att1'),
          retinanet_cfg.AttributeHead(
              name='att2', type='classification', size=2),
      ]
    else:
      attribute_heads_config = None
    model_config = retinanet_cfg.RetinaNet(
        num_classes=num_classes,
        backbone=backbones.Backbone(
            type=backbone_type,
            spinenet_mobile=backbones.SpineNetMobile(
                model_id='49',
                stochastic_depth_drop_rate=0.2,
                min_level=3,
                max_level=7,
                use_keras_upsampling_2d=True)),
        head=retinanet_cfg.RetinaNetHead(
            attribute_heads=attribute_heads_config))
    l2_regularizer = tf.keras.regularizers.l2(5e-5)
    quantization_config = common.Quantization()
    model = factory.build_retinanet(
        input_specs=input_specs,
        model_config=model_config,
        l2_regularizer=l2_regularizer)

    _ = qat_factory.build_qat_retinanet(
        model=model,
        quantization=quantization_config,
        model_config=model_config)
    if has_attribute_heads:
      self.assertEqual(model_config.head.attribute_heads[0].as_dict(),
                       dict(name='att1', type='regression', size=1))
      self.assertEqual(model_config.head.attribute_heads[1].as_dict(),
                       dict(name='att2', type='classification', size=2))


class SegmentationModelBuilderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ('mobilenet', (512, 512), 5e-5),)
  def test_deeplabv3_builder(self, backbone_type, input_size, weight_decay):
    num_classes = 21
    input_specs = tf.keras.layers.InputSpec(
        shape=[None, input_size[0], input_size[1], 3])
    model_config = semantic_segmentation_cfg.SemanticSegmentationModel(
        num_classes=num_classes,
        backbone=backbones.Backbone(
            type=backbone_type,
            mobilenet=backbones.MobileNet(
                model_id='MobileNetV2', output_stride=16)),
        decoder=decoders.Decoder(
            type='aspp',
            aspp=decoders.ASPP(
                level=4,
                num_filters=256,
                dilation_rates=[],
                spp_layer_version='v1',
                output_tensor=True)),
        head=semantic_segmentation_cfg.SegmentationHead(
            level=4,
            low_level=2,
            num_convs=1,
            upsample_factor=2,
            use_depthwise_convolution=True))
    l2_regularizer = (
        tf.keras.regularizers.l2(weight_decay) if weight_decay else None)
    model = factory.build_segmentation_model(
        input_specs=input_specs,
        model_config=model_config,
        l2_regularizer=l2_regularizer)
    quantization_config = common.Quantization()
    _ = qat_factory.build_qat_segmentation_model(
        model=model, quantization=quantization_config, input_specs=input_specs)

  @parameterized.parameters(
      ('mobilenet', (512, 1024), 5e-5),)
  def test_deeplabv3plus_builder(self, backbone_type, input_size, weight_decay):
    num_classes = 19
    input_specs = tf.keras.layers.InputSpec(
        shape=[None, input_size[0], input_size[1], 3])
    model_config = semantic_segmentation_cfg.SemanticSegmentationModel(
        num_classes=num_classes,
        backbone=backbones.Backbone(
            type=backbone_type,
            mobilenet=backbones.MobileNet(
                model_id='MobileNetV2',
                output_stride=16,
                output_intermediate_endpoints=True)),
        decoder=decoders.Decoder(
            type='aspp',
            aspp=decoders.ASPP(
                level=4,
                num_filters=256,
                dilation_rates=[],
                pool_kernel_size=[512, 1024],
                use_depthwise_convolution=False,
                spp_layer_version='v1',
                output_tensor=True)),
        head=semantic_segmentation_cfg.SegmentationHead(
            level=4,
            num_convs=2,
            feature_fusion='deeplabv3plus',
            use_depthwise_convolution=True,
            low_level='2/depthwise',
            low_level_num_filters=48,
            prediction_kernel_size=1,
            upsample_factor=1,
            num_filters=256))
    l2_regularizer = (
        tf.keras.regularizers.l2(weight_decay) if weight_decay else None)
    model = factory.build_segmentation_model(
        input_specs=input_specs,
        model_config=model_config,
        l2_regularizer=l2_regularizer)
    quantization_config = common.Quantization()
    _ = qat_factory.build_qat_segmentation_model(
        model=model, quantization=quantization_config, input_specs=input_specs)

if __name__ == '__main__':
  tf.test.main()
