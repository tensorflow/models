# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for RetinaNet models."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf, tf_keras

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.vision.modeling import retinanet_model
from official.vision.modeling.backbones import resnet
from official.vision.modeling.decoders import fpn
from official.vision.modeling.heads import dense_prediction_heads
from official.vision.modeling.layers import detection_generator
from official.vision.ops import anchor


class RetinaNetTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      {
          'use_separable_conv': True,
          'build_anchor_boxes': True,
          'is_training': False,
          'has_att_heads': False,
      },
      {
          'use_separable_conv': False,
          'build_anchor_boxes': True,
          'is_training': False,
          'has_att_heads': False,
      },
      {
          'use_separable_conv': False,
          'build_anchor_boxes': False,
          'is_training': False,
          'has_att_heads': False,
      },
      {
          'use_separable_conv': False,
          'build_anchor_boxes': False,
          'is_training': True,
          'has_att_heads': False,
      },
      {
          'use_separable_conv': False,
          'build_anchor_boxes': True,
          'is_training': True,
          'has_att_heads': True,
      },
      {
          'use_separable_conv': False,
          'build_anchor_boxes': True,
          'is_training': False,
          'has_att_heads': True,
      },
  )
  def test_build_model(self, use_separable_conv, build_anchor_boxes,
                       is_training, has_att_heads):
    num_classes = 3
    min_level = 3
    max_level = 7
    num_scales = 3
    aspect_ratios = [1.0]
    anchor_size = 3
    fpn_num_filters = 256
    head_num_convs = 4
    head_num_filters = 256
    num_anchors_per_location = num_scales * len(aspect_ratios)
    image_size = 384
    images = np.random.rand(2, image_size, image_size, 3)
    image_shape = np.array([[image_size, image_size], [image_size, image_size]])

    if build_anchor_boxes:
      anchor_boxes = anchor.Anchor(
          min_level=min_level,
          max_level=max_level,
          num_scales=num_scales,
          aspect_ratios=aspect_ratios,
          anchor_size=anchor_size,
          image_size=(image_size, image_size)).multilevel_boxes
      for l in anchor_boxes:
        anchor_boxes[l] = tf.tile(
            tf.expand_dims(anchor_boxes[l], axis=0), [2, 1, 1, 1])
    else:
      anchor_boxes = None

    if has_att_heads:
      attribute_heads = [
          dict(
              name='depth', type='regression', size=1, prediction_tower_name='')
      ]
    else:
      attribute_heads = None

    backbone = resnet.ResNet(model_id=50)
    decoder = fpn.FPN(
        input_specs=backbone.output_specs,
        min_level=min_level,
        max_level=max_level,
        num_filters=fpn_num_filters,
        use_separable_conv=use_separable_conv)
    head = dense_prediction_heads.RetinaNetHead(
        min_level=min_level,
        max_level=max_level,
        num_classes=num_classes,
        attribute_heads=attribute_heads,
        num_anchors_per_location=num_anchors_per_location,
        use_separable_conv=use_separable_conv,
        num_convs=head_num_convs,
        num_filters=head_num_filters)
    generator = detection_generator.MultilevelDetectionGenerator(
        max_num_detections=10)
    model = retinanet_model.RetinaNetModel(
        backbone=backbone,
        decoder=decoder,
        head=head,
        detection_generator=generator,
        min_level=min_level,
        max_level=max_level,
        num_scales=num_scales,
        aspect_ratios=aspect_ratios,
        anchor_size=anchor_size)

    _ = model(images, image_shape, anchor_boxes, training=is_training)

  @combinations.generate(
      combinations.combine(
          strategy=[
              strategy_combinations.cloud_tpu_strategy,
              strategy_combinations.one_device_strategy_gpu,
          ],
          image_size=[
              (128, 128),
          ],
          training=[True, False],
          has_att_heads=[True, False],
          output_intermediate_features=[True, False],
          soft_nms_sigma=[None, 0.0, 0.1],
      ))
  def test_forward(self, strategy, image_size, training, has_att_heads,
                   output_intermediate_features, soft_nms_sigma):
    """Test for creation of a R50-FPN RetinaNet."""
    tf_keras.backend.set_image_data_format('channels_last')
    num_classes = 3
    min_level = 3
    max_level = 7
    num_scales = 3
    aspect_ratios = [1.0]
    num_anchors_per_location = num_scales * len(aspect_ratios)

    images = np.random.rand(2, image_size[0], image_size[1], 3)
    image_shape = np.array(
        [[image_size[0], image_size[1]], [image_size[0], image_size[1]]])

    with strategy.scope():
      anchor_gen = anchor.build_anchor_generator(
          min_level=min_level,
          max_level=max_level,
          num_scales=num_scales,
          aspect_ratios=aspect_ratios,
          anchor_size=3)
      anchor_boxes = anchor_gen(image_size)
      for l in anchor_boxes:
        anchor_boxes[l] = tf.tile(
            tf.expand_dims(anchor_boxes[l], axis=0), [2, 1, 1, 1])

      backbone = resnet.ResNet(model_id=50)
      decoder = fpn.FPN(
          input_specs=backbone.output_specs,
          min_level=min_level,
          max_level=max_level)

      if has_att_heads:
        attribute_heads = [
            dict(
                name='depth',
                type='regression',
                size=1,
                prediction_tower_name='')
        ]
      else:
        attribute_heads = None
      head = dense_prediction_heads.RetinaNetHead(
          min_level=min_level,
          max_level=max_level,
          num_classes=num_classes,
          attribute_heads=attribute_heads,
          num_anchors_per_location=num_anchors_per_location)
      generator = detection_generator.MultilevelDetectionGenerator(
          max_num_detections=10,
          nms_version='v1',
          use_cpu_nms=soft_nms_sigma is not None,
          soft_nms_sigma=soft_nms_sigma)
      model = retinanet_model.RetinaNetModel(
          backbone=backbone,
          decoder=decoder,
          head=head,
          detection_generator=generator)

      model_outputs = model(
          images,
          image_shape,
          anchor_boxes,
          output_intermediate_features=output_intermediate_features,
          training=training)

    if training:
      cls_outputs = model_outputs['cls_outputs']
      box_outputs = model_outputs['box_outputs']
      for level in range(min_level, max_level + 1):
        self.assertIn(str(level), cls_outputs)
        self.assertIn(str(level), box_outputs)
        self.assertAllEqual([
            2,
            image_size[0] // 2**level,
            image_size[1] // 2**level,
            num_classes * num_anchors_per_location
        ], cls_outputs[str(level)].numpy().shape)
        self.assertAllEqual([
            2,
            image_size[0] // 2**level,
            image_size[1] // 2**level,
            4 * num_anchors_per_location
        ], box_outputs[str(level)].numpy().shape)
        if has_att_heads:
          att_outputs = model_outputs['attribute_outputs']
          for att in att_outputs.values():
            self.assertAllEqual([
                2, image_size[0] // 2**level, image_size[1] // 2**level,
                1 * num_anchors_per_location
            ], att[str(level)].numpy().shape)
    else:
      self.assertIn('detection_boxes', model_outputs)
      self.assertIn('detection_scores', model_outputs)
      self.assertIn('detection_classes', model_outputs)
      self.assertIn('num_detections', model_outputs)
      self.assertAllEqual(
          [2, 10, 4], model_outputs['detection_boxes'].numpy().shape)
      self.assertAllEqual(
          [2, 10], model_outputs['detection_scores'].numpy().shape)
      self.assertAllEqual(
          [2, 10], model_outputs['detection_classes'].numpy().shape)
      self.assertAllEqual(
          [2,], model_outputs['num_detections'].numpy().shape)
      if has_att_heads:
        self.assertIn('detection_attributes', model_outputs)
        self.assertAllEqual(
            [2, 10, 1],
            model_outputs['detection_attributes']['depth'].numpy().shape)
    if output_intermediate_features:
      for l in range(2, 6):
        self.assertIn('backbone_{}'.format(l), model_outputs)
        self.assertAllEqual([
            2, image_size[0] // 2**l, image_size[1] // 2**l,
            backbone.output_specs[str(l)].as_list()[-1]
        ], model_outputs['backbone_{}'.format(l)].numpy().shape)
      for l in range(min_level, max_level + 1):
        self.assertIn('decoder_{}'.format(l), model_outputs)
        self.assertAllEqual([
            2, image_size[0] // 2**l, image_size[1] // 2**l,
            decoder.output_specs[str(l)].as_list()[-1]
        ], model_outputs['decoder_{}'.format(l)].numpy().shape)

  def test_serialize_deserialize(self):
    """Validate the network can be serialized and deserialized."""
    num_classes = 3
    min_level = 3
    max_level = 7
    num_scales = 3
    aspect_ratios = [1.0]
    num_anchors_per_location = num_scales * len(aspect_ratios)

    backbone = resnet.ResNet(model_id=50)
    decoder = fpn.FPN(
        input_specs=backbone.output_specs,
        min_level=min_level,
        max_level=max_level)
    head = dense_prediction_heads.RetinaNetHead(
        min_level=min_level,
        max_level=max_level,
        num_classes=num_classes,
        num_anchors_per_location=num_anchors_per_location)
    generator = detection_generator.MultilevelDetectionGenerator(
        max_num_detections=10)
    model = retinanet_model.RetinaNetModel(
        backbone=backbone,
        decoder=decoder,
        head=head,
        detection_generator=generator,
        min_level=min_level,
        max_level=max_level,
        num_scales=num_scales,
        aspect_ratios=aspect_ratios,
        anchor_size=3)

    config = model.get_config()
    new_model = retinanet_model.RetinaNetModel.from_config(config)

    # Validate that the config can be forced to JSON.
    _ = new_model.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(model.get_config(), new_model.get_config())


if __name__ == '__main__':
  tf.test.main()
