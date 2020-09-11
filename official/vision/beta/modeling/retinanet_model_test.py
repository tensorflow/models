# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for RetinaNet models."""

# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.vision.beta.modeling import retinanet_model
from official.vision.beta.modeling.backbones import resnet
from official.vision.beta.modeling.decoders import fpn
from official.vision.beta.modeling.heads import dense_prediction_heads
from official.vision.beta.modeling.layers import detection_generator
from official.vision.beta.ops import anchor


class RetinaNetTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (3, 3, 7, 3, [1.0], 50, False, 256, 4, 256, 32244949),
  )
  def test_num_params(self,
                      num_classes,
                      min_level,
                      max_level,
                      num_scales,
                      aspect_ratios,
                      resnet_model_id,
                      use_separable_conv,
                      fpn_num_filters,
                      head_num_convs,
                      head_num_filters,
                      expected_num_params):
    num_anchors_per_location = num_scales * len(aspect_ratios)
    image_size = 384
    images = np.random.rand(2, image_size, image_size, 3)
    image_shape = np.array([[image_size, image_size], [image_size, image_size]])

    anchor_boxes = anchor.Anchor(
        min_level=min_level,
        max_level=max_level,
        num_scales=num_scales,
        aspect_ratios=aspect_ratios,
        anchor_size=3,
        image_size=(image_size, image_size)).multilevel_boxes
    for l in anchor_boxes:
      anchor_boxes[l] = tf.tile(
          tf.expand_dims(anchor_boxes[l], axis=0), [2, 1, 1, 1])

    backbone = resnet.ResNet(model_id=resnet_model_id)
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
        detection_generator=generator)

    _ = model(images, image_shape, anchor_boxes, training=True)
    self.assertEqual(expected_num_params, model.count_params())

  @combinations.generate(
      combinations.combine(
          strategy=[
              strategy_combinations.tpu_strategy,
              strategy_combinations.one_device_strategy_gpu,
          ],
          image_size=[(128, 128),],
          training=[True, False],
      )
  )
  def test_forward(self, strategy, image_size, training):
    """Test for creation of a R50-FPN RetinaNet."""
    tf.keras.backend.set_image_data_format('channels_last')
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
          detection_generator=generator)

      model_outputs = model(
          images,
          image_shape,
          anchor_boxes,
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
        detection_generator=generator)

    config = model.get_config()
    new_model = retinanet_model.RetinaNetModel.from_config(config)

    # Validate that the config can be forced to JSON.
    _ = new_model.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(model.get_config(), new_model.get_config())


if __name__ == '__main__':
  tf.test.main()

