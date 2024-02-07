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

"""Tests for PointPillars models."""

from absl.testing import parameterized
import tensorflow as tf, tf_keras

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.projects.pointpillars.modeling import backbones
from official.projects.pointpillars.modeling import decoders
from official.projects.pointpillars.modeling import featurizers
from official.projects.pointpillars.modeling import heads
from official.projects.pointpillars.modeling import models
from official.projects.pointpillars.utils import utils
from official.vision.modeling.layers import detection_generator


class PointpillarsTest(parameterized.TestCase, tf.test.TestCase):

  @combinations.generate(
      combinations.combine(
          strategy=[
              strategy_combinations.cloud_tpu_strategy,
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_gpu,
              strategy_combinations.mirrored_strategy_with_two_gpus,
          ],
          training=[True, False],
      ))
  def test_all(self, strategy, training):
    tf_keras.backend.set_image_data_format('channels_last')
    num_classes = 2

    h, w, c = 8, 8, 2
    n, p, d = 2, 3, 4
    image_size = [h, w]
    pillars_size = [n, p, d]
    indices_size = [n, 2]
    attribute_heads = [{'name': 'heading', 'type': 'regression', 'size': 1}]

    min_level = 1
    max_level = 2

    anchor_sizes = [(1.1, 1.1)]
    num_anchors_per_location = len(anchor_sizes)

    global_batch_size = 4
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    batch_size = int(global_batch_size / num_replicas)
    pillars = tf_keras.Input(shape=pillars_size, batch_size=batch_size)
    indices = tf_keras.Input(
        shape=indices_size, batch_size=batch_size, dtype=tf.int32)
    image_shape = tf.tile(tf.expand_dims([h, w], axis=0), [batch_size, 1])
    max_num_detections = 4

    # Test model creation.
    with strategy.scope():
      anchor_boxes = utils.generate_anchors(min_level,
                                            max_level,
                                            image_size,
                                            anchor_sizes)
      for l in anchor_boxes:
        anchor_boxes[l] = tf.tile(
            tf.expand_dims(anchor_boxes[l], axis=0), [batch_size, 1, 1, 1])

      featurizer = featurizers.Featurizer(
          image_size=image_size,
          pillars_size=pillars_size,
          train_batch_size=batch_size,
          eval_batch_size=batch_size,
          num_blocks=3,
          num_channels=c
      )
      image = featurizer(pillars, indices, training)
      backbone = backbones.Backbone(
          input_specs=featurizer.output_specs,
          min_level=min_level,
          max_level=max_level,
          num_convs=3
      )
      encoded_feats = backbone(image)
      decoder = decoders.Decoder(
          input_specs=backbone.output_specs)
      decoded_feats = decoder(encoded_feats)
      head = heads.SSDHead(
          num_classes=num_classes,
          num_anchors_per_location=num_anchors_per_location,
          num_params_per_anchor=4,
          attribute_heads=attribute_heads,
          min_level=min_level,
          max_level=max_level
      )
      _ = head(decoded_feats)
      generator = detection_generator.MultilevelDetectionGenerator(
          max_num_detections=max_num_detections,
          nms_version='v1',
          use_cpu_nms=True,
          soft_nms_sigma=0.1)
      model = models.PointPillarsModel(
          featurizer=featurizer,
          backbone=backbone,
          decoder=decoder,
          head=head,
          detection_generator=generator,
          min_level=min_level,
          max_level=max_level,
          image_size=image_size,
          anchor_sizes=anchor_sizes)
      outputs = model(
          pillars,
          indices,
          image_shape,
          anchor_boxes,
          training)

    # Test training and evaluation.
    if training:
      cls_outputs = outputs['cls_outputs']
      box_outputs = outputs['box_outputs']
      for level in range(min_level, max_level+1):
        self.assertIn(str(level), cls_outputs)
        self.assertIn(str(level), box_outputs)
        self.assertAllEqual([
            batch_size,
            h // 2**level,
            w // 2**level,
            num_classes * num_anchors_per_location
        ], cls_outputs[str(level)].shape)
        self.assertAllEqual([
            batch_size,
            h // 2**level,
            w // 2**level,
            4 * num_anchors_per_location
        ], box_outputs[str(level)].shape)
        att_outputs = outputs['attribute_outputs']
        self.assertLen(att_outputs, 1)
        self.assertIn('heading', att_outputs)
        self.assertAllEqual([
            batch_size,
            h // 2**level,
            w // 2**level,
            1 * num_anchors_per_location
        ], att_outputs['heading'][str(level)].shape)
    else:
      self.assertIn('boxes', outputs)
      self.assertIn('scores', outputs)
      self.assertIn('classes', outputs)
      self.assertIn('num_detections', outputs)
      self.assertAllEqual([
          batch_size,
      ], outputs['num_detections'].shape)
      self.assertAllEqual([batch_size, max_num_detections, 4],
                          outputs['boxes'].shape)
      self.assertAllEqual([batch_size, max_num_detections],
                          outputs['scores'].shape)
      self.assertAllEqual([batch_size, max_num_detections],
                          outputs['classes'].shape)
      self.assertIn('attributes', outputs)
      self.assertAllEqual(
          [batch_size, max_num_detections, 1],
          outputs['attributes']['heading'].shape)

    # Test serialization.
    config = model.get_config()
    new_model = models.PointPillarsModel.from_config(config)
    _ = new_model.to_json()
    self.assertAllEqual(model.get_config(), new_model.get_config())


if __name__ == '__main__':
  tf.test.main()
