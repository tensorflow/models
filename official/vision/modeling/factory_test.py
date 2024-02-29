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
import collections

# Import libraries
from absl.testing import parameterized
import tensorflow as tf, tf_keras

from official.vision.configs import backbones
from official.vision.configs import backbones_3d
from official.vision.configs import decoders
from official.vision.configs import image_classification as classification_cfg
from official.vision.configs import maskrcnn as maskrcnn_cfg
from official.vision.configs import retinanet as retinanet_cfg
from official.vision.configs import video_classification as video_classification_cfg
from official.vision.modeling import factory
from official.vision.modeling import factory_3d


class ClassificationModelBuilderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ('resnet', (224, 224), 5e-5),
      ('resnet', (224, 224), None),
      ('resnet', (None, None), 5e-5),
      ('resnet', (None, None), None),
  )
  def test_builder(self, backbone_type, input_size, weight_decay):
    num_classes = 2
    input_specs = tf_keras.layers.InputSpec(
        shape=[None, input_size[0], input_size[1], 3])
    model_config = classification_cfg.ImageClassificationModel(
        num_classes=num_classes,
        backbone=backbones.Backbone(type=backbone_type))
    l2_regularizer = (
        tf_keras.regularizers.l2(weight_decay) if weight_decay else None)
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
    input_specs = tf_keras.layers.InputSpec(
        shape=[None, input_size[0], input_size[1], 3])
    model_config = maskrcnn_cfg.MaskRCNN(
        num_classes=num_classes,
        backbone=backbones.Backbone(type=backbone_type))
    l2_regularizer = tf_keras.regularizers.l2(5e-5)
    _ = factory.build_maskrcnn(
        input_specs=input_specs,
        model_config=model_config,
        l2_regularizer=l2_regularizer)


class RetinaNetBuilderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      ('resnet', (640, 640), False),
      ('resnet', (None, None), True),
  )
  def test_builder(self, backbone_type, input_size, has_att_heads):
    num_classes = 2
    input_specs = tf_keras.layers.InputSpec(
        shape=[None, input_size[0], input_size[1], 3])
    if has_att_heads:
      attribute_heads_config = [
          retinanet_cfg.AttributeHead(name='att1'),
          retinanet_cfg.AttributeHead(
              name='att2', type='classification', size=2),
      ]
    else:
      attribute_heads_config = None
    model_config = retinanet_cfg.RetinaNet(
        num_classes=num_classes,
        backbone=backbones.Backbone(type=backbone_type),
        head=retinanet_cfg.RetinaNetHead(
            attribute_heads=attribute_heads_config))
    l2_regularizer = tf_keras.regularizers.l2(5e-5)
    _ = factory.build_retinanet(
        input_specs=input_specs,
        model_config=model_config,
        l2_regularizer=l2_regularizer)
    if has_att_heads:
      self.assertEqual(
          model_config.head.attribute_heads[0].as_dict(),
          dict(
              name='att1',
              type='regression',
              size=1,
              prediction_tower_name='',
              num_convs=None,
              num_filters=None,
          ),
      )
      self.assertEqual(
          model_config.head.attribute_heads[1].as_dict(),
          dict(
              name='att2',
              type='classification',
              size=2,
              prediction_tower_name='',
              num_convs=None,
              num_filters=None,
          ),
      )

  def test_build_model_with_custom_anchors_can_run(self):
    image_size = (16, 16)
    input_specs = tf_keras.layers.InputSpec(shape=[None, *image_size, 3])
    model_config = retinanet_cfg.RetinaNet(
        num_classes=5,
        min_level=3,
        max_level=4,
        decoder=decoders.Decoder(type='identity'),
        head=retinanet_cfg.RetinaNetHead(
            num_convs=0, share_level_convs=False,
        )
    )
    anchor_boxes = collections.OrderedDict()
    anchor_boxes['3'] = tf.constant(
        [
            [[3, 4, 5, 6], [3, 4, 5, 6]],
            [[3, 4, 5, 6], [3, 4, 5, 6]],
        ],
        dtype=tf.float32,
    )
    anchor_boxes['4'] = tf.constant(
        [[[3, 4, 5, 6, 3, 4, 5, 6]]], dtype=tf.float32
    )
    model = factory.build_retinanet(
        input_specs=input_specs,
        model_config=model_config,
        anchor_boxes=anchor_boxes,
        num_anchors_per_location={'3': 1, '4': 2},
    )
    test_input = tf.zeros([2, *image_size, 3])
    outputs = model.call(test_input)
    self.assertIn('box_outputs', outputs)
    self.assertIn('3', outputs['box_outputs'])
    self.assertIn('4', outputs['box_outputs'])
    self.assertAllEqual(
        outputs['box_outputs']['3'].numpy().shape, [2, 2, 2, 4 * 1]
    )
    self.assertAllEqual(
        outputs['box_outputs']['4'].numpy().shape, [2, 1, 1, 4 * 2]
    )


class VideoClassificationModelBuilderTest(parameterized.TestCase,
                                          tf.test.TestCase):

  @parameterized.parameters(
      ('resnet_3d', (8, 224, 224), 5e-5),
      ('resnet_3d', (None, None, None), 5e-5),
  )
  def test_builder(self, backbone_type, input_size, weight_decay):
    input_specs = tf_keras.layers.InputSpec(
        shape=[None, input_size[0], input_size[1], input_size[2], 3])
    model_config = video_classification_cfg.VideoClassificationModel(
        backbone=backbones_3d.Backbone3D(type=backbone_type))
    l2_regularizer = (
        tf_keras.regularizers.l2(weight_decay) if weight_decay else None)
    _ = factory_3d.build_video_classification_model(
        input_specs=input_specs,
        model_config=model_config,
        num_classes=2,
        l2_regularizer=l2_regularizer)


if __name__ == '__main__':
  tf.test.main()
