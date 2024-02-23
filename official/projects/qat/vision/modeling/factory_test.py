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

"""Tests for factory.py."""

# Import libraries

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.projects.qat.vision.configs import common
from official.projects.qat.vision.modeling import factory as qat_factory
from official.projects.qat.vision.modeling.heads import dense_prediction_heads as qat_dense_prediction_heads
from official.vision.configs import backbones
from official.vision.configs import decoders
from official.vision.configs import image_classification as classification_cfg
from official.vision.configs import retinanet as retinanet_cfg
from official.vision.configs import semantic_segmentation as semantic_segmentation_cfg
from official.vision.modeling import factory
from official.vision.modeling.decoders import fpn
from official.vision.modeling.heads import dense_prediction_heads


class ClassificationModelBuilderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ('resnet', 50, (224, 224), 5e-5),
      ('resnet', 50, (224, 224), None),
      ('resnet', 50, (None, None), 5e-5),
      ('resnet', 50, (None, None), None),
      ('mobilenet', 'MobileNetV2', (224, 224), 5e-5),
      ('mobilenet', 'MobileNetV2', (224, 224), None),
      ('mobilenet', 'MobileNetV2', (None, None), 5e-5),
      ('mobilenet', 'MobileNetV2', (None, None), None),
      ('mobilenet', 'MobileNetV4ConvLarge', (224, 224), 5e-5),
  )
  def test_builder(self, backbone_type, model_id, input_size, weight_decay):
    num_classes = 2
    input_specs = tf_keras.layers.InputSpec(
        shape=[None, input_size[0], input_size[1], 3])

    backbone = backbones.Backbone(type=backbone_type)
    if backbone_type == 'resnet':
      backbone.resnet.model_id = model_id
    elif backbone_type == 'mobilenet':
      backbone.mobilenet.model_id = model_id
    else:
      raise ValueError('Unexpected backbone_type', backbone_type)

    model_config = classification_cfg.ImageClassificationModel(
        num_classes=num_classes, backbone=backbone
    )
    l2_regularizer = (
        tf_keras.regularizers.l2(weight_decay) if weight_decay else None)
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
      ('spinenet_mobile', 'identity', (640, 640), False, False),
      ('spinenet_mobile', 'identity', (640, 640), True, False),
      ('mobilenet', 'fpn', (640, 640), True, False),
      ('mobilenet', 'fpn', (640, 640), True, True),
  )
  def test_builder(self,
                   backbone_type,
                   decoder_type,
                   input_size,
                   quantize_detection_head,
                   quantize_detection_decoder):
    num_classes = 2
    input_specs = tf_keras.layers.InputSpec(
        shape=[None, input_size[0], input_size[1], 3])

    if backbone_type == 'spinenet_mobile':
      backbone_config = backbones.Backbone(
          type=backbone_type,
          spinenet_mobile=backbones.SpineNetMobile(
              model_id='49',
              stochastic_depth_drop_rate=0.2,
              min_level=3,
              max_level=7,
              use_keras_upsampling_2d=True))
    elif backbone_type == 'mobilenet':
      backbone_config = backbones.Backbone(
          type=backbone_type,
          mobilenet=backbones.MobileNet(
              model_id='MobileNetV2',
              filter_size_scale=1.0))
    else:
      raise ValueError(
          'backbone_type {} is not supported'.format(backbone_type))

    if decoder_type == 'identity':
      decoder_config = decoders.Decoder(type=decoder_type)
    elif decoder_type == 'fpn':
      decoder_config = decoders.Decoder(
          type=decoder_type,
          fpn=decoders.FPN(
              num_filters=128,
              use_separable_conv=True,
              use_keras_layer=True))
    else:
      raise ValueError(
          'decoder_type {} is not supported'.format(decoder_type))

    model_config = retinanet_cfg.RetinaNet(
        num_classes=num_classes,
        input_size=[input_size[0], input_size[1], 3],
        backbone=backbone_config,
        decoder=decoder_config,
        head=retinanet_cfg.RetinaNetHead(
            attribute_heads=None,
            use_separable_conv=True))

    l2_regularizer = tf_keras.regularizers.l2(5e-5)
    # Build the original float32 retinanet model.
    model = factory.build_retinanet(
        input_specs=input_specs,
        model_config=model_config,
        l2_regularizer=l2_regularizer)

    # Call the model with dummy input to build the head part.
    dummpy_input = tf.zeros([1] + model_config.input_size)
    model(dummpy_input, training=True)

    # Build the QAT model from the original model with quantization config.
    qat_model = qat_factory.build_qat_retinanet(
        model=model,
        quantization=common.Quantization(
            quantize_detection_decoder=quantize_detection_decoder,
            quantize_detection_head=quantize_detection_head),
        model_config=model_config)

    if quantize_detection_head:
      # head become a RetinaNetHeadQuantized when we apply quantization.
      self.assertIsInstance(qat_model.head,
                            qat_dense_prediction_heads.RetinaNetHeadQuantized)
    else:
      # head is a RetinaNetHead if we don't apply quantization on head part.
      self.assertIsInstance(
          qat_model.head, dense_prediction_heads.RetinaNetHead)
      self.assertNotIsInstance(
          qat_model.head, qat_dense_prediction_heads.RetinaNetHeadQuantized)

    if decoder_type == 'FPN':
      if quantize_detection_decoder:
        # FPN decoder become a general keras functional model after applying
        # quantization.
        self.assertNotIsInstance(qat_model.decoder, fpn.FPN)
      else:
        self.assertIsInstance(qat_model.decoder, fpn.FPN)


class SegmentationModelBuilderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ('mobilenet', (512, 512), 5e-5),)
  def test_deeplabv3_builder(self, backbone_type, input_size, weight_decay):
    num_classes = 21
    input_specs = tf_keras.layers.InputSpec(
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
        tf_keras.regularizers.l2(weight_decay) if weight_decay else None)
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
    input_specs = tf_keras.layers.InputSpec(
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
        tf_keras.regularizers.l2(weight_decay) if weight_decay else None)
    model = factory.build_segmentation_model(
        input_specs=input_specs,
        model_config=model_config,
        l2_regularizer=l2_regularizer)
    quantization_config = common.Quantization()
    _ = qat_factory.build_qat_segmentation_model(
        model=model, quantization=quantization_config, input_specs=input_specs)

if __name__ == '__main__':
  tf.test.main()
