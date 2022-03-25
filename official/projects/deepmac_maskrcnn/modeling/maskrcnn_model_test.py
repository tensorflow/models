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

"""Tests for maskrcnn_model.py."""

# Import libraries

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.projects.deepmac_maskrcnn.modeling import maskrcnn_model
from official.projects.deepmac_maskrcnn.modeling.heads import instance_heads as deep_instance_heads
from official.vision.modeling.backbones import resnet
from official.vision.modeling.decoders import fpn
from official.vision.modeling.heads import dense_prediction_heads
from official.vision.modeling.heads import instance_heads
from official.vision.modeling.layers import detection_generator
from official.vision.modeling.layers import mask_sampler
from official.vision.modeling.layers import roi_aligner
from official.vision.modeling.layers import roi_generator
from official.vision.modeling.layers import roi_sampler
from official.vision.ops import anchor


def construct_model_and_anchors(image_size, use_gt_boxes_for_masks):
  num_classes = 3
  min_level = 3
  max_level = 4
  num_scales = 3
  aspect_ratios = [1.0]

  anchor_boxes = anchor.Anchor(
      min_level=min_level,
      max_level=max_level,
      num_scales=num_scales,
      aspect_ratios=aspect_ratios,
      anchor_size=3,
      image_size=image_size).multilevel_boxes
  num_anchors_per_location = len(aspect_ratios) * num_scales

  input_specs = tf.keras.layers.InputSpec(shape=[None, None, None, 3])
  backbone = resnet.ResNet(model_id=50, input_specs=input_specs)
  decoder = fpn.FPN(
      min_level=min_level,
      max_level=max_level,
      input_specs=backbone.output_specs)
  rpn_head = dense_prediction_heads.RPNHead(
      min_level=min_level,
      max_level=max_level,
      num_anchors_per_location=num_anchors_per_location)
  detection_head = instance_heads.DetectionHead(
      num_classes=num_classes)
  roi_generator_obj = roi_generator.MultilevelROIGenerator()
  roi_sampler_obj = roi_sampler.ROISampler()
  roi_aligner_obj = roi_aligner.MultilevelROIAligner()
  detection_generator_obj = detection_generator.DetectionGenerator()
  mask_head = deep_instance_heads.DeepMaskHead(
      num_classes=num_classes, upsample_factor=2)
  mask_sampler_obj = mask_sampler.MaskSampler(
      mask_target_size=28, num_sampled_masks=1)
  mask_roi_aligner_obj = roi_aligner.MultilevelROIAligner(crop_size=14)

  model = maskrcnn_model.DeepMaskRCNNModel(
      backbone,
      decoder,
      rpn_head,
      detection_head,
      roi_generator_obj,
      roi_sampler_obj,
      roi_aligner_obj,
      detection_generator_obj,
      mask_head,
      mask_sampler_obj,
      mask_roi_aligner_obj,
      use_gt_boxes_for_masks=use_gt_boxes_for_masks)

  return model, anchor_boxes


class MaskRCNNModelTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (False, False,),
      (False, True,),
      (True, False,),
      (True, True,),
  )
  def test_forward(self, use_gt_boxes_for_masks, training):
    image_size = (256, 256)
    images = np.random.rand(2, image_size[0], image_size[1], 3)
    image_shape = np.array([[224, 100], [100, 224]])
    model, anchor_boxes = construct_model_and_anchors(
        image_size, use_gt_boxes_for_masks)

    gt_boxes = tf.zeros((2, 16, 4), dtype=tf.float32)
    gt_masks = tf.zeros((2, 16, 32, 32))
    gt_classes = tf.zeros((2, 16), dtype=tf.int32)
    results = model(images.astype(np.uint8),
                    image_shape,
                    anchor_boxes,
                    gt_boxes,
                    gt_classes,
                    gt_masks,
                    training=training)

    self.assertIn('rpn_boxes', results)
    self.assertIn('rpn_scores', results)
    if training:
      self.assertIn('class_targets', results)
      self.assertIn('box_targets', results)
      self.assertIn('class_outputs', results)
      self.assertIn('box_outputs', results)
      self.assertIn('mask_outputs', results)
      self.assertEqual(results['mask_targets'].shape,
                       results['mask_outputs'].shape)
    else:
      self.assertIn('detection_boxes', results)
      self.assertIn('detection_scores', results)
      self.assertIn('detection_classes', results)
      self.assertIn('num_detections', results)
      self.assertIn('detection_masks', results)

  @parameterized.parameters(
      [(1, 5), (1, 10), (1, 15), (2, 5), (2, 10), (2, 15)]
  )
  def test_image_and_boxes(self, batch_size, num_boxes):
    image_size = (640, 640)
    images = np.random.rand(1, image_size[0], image_size[1], 3).astype(
        np.float32)
    model, _ = construct_model_and_anchors(
        image_size, use_gt_boxes_for_masks=True)

    boxes = np.zeros((1, num_boxes, 4), dtype=np.float32)
    boxes[:, :, [2, 3]] = 1.0
    boxes = tf.constant(boxes)
    results = model.call_images_and_boxes(images, boxes)
    self.assertIn('detection_masks', results)


if __name__ == '__main__':
  tf.test.main()
