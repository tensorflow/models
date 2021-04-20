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
"""Tests for maskrcnn_model.py."""

import os
# Import libraries
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from official.vision.beta.modeling import maskrcnn_model
from official.vision.beta.modeling.backbones import resnet
from official.vision.beta.modeling.decoders import fpn
from official.vision.beta.modeling.heads import dense_prediction_heads
from official.vision.beta.modeling.heads import instance_heads
from official.vision.beta.modeling.layers import detection_generator
from official.vision.beta.modeling.layers import mask_sampler
from official.vision.beta.modeling.layers import roi_aligner
from official.vision.beta.modeling.layers import roi_generator
from official.vision.beta.modeling.layers import roi_sampler
from official.vision.beta.ops import anchor


class MaskRCNNModelTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (3, 3, 7, 3, [1.0], 50, False, False, 41953246),
  )
  def test_num_params(self,
                      num_classes,
                      min_level,
                      max_level,
                      num_scales,
                      aspect_ratios,
                      resnet_model_id,
                      use_separable_conv,
                      include_mask,
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
        use_separable_conv=use_separable_conv)
    rpn_head = dense_prediction_heads.RPNHead(
        min_level=min_level,
        max_level=max_level,
        num_anchors_per_location=num_anchors_per_location,
        num_convs=1)
    detection_head = instance_heads.DetectionHead(
        num_classes=num_classes)
    roi_generator_obj = roi_generator.MultilevelROIGenerator()
    roi_sampler_obj = roi_sampler.ROISampler()
    roi_aligner_obj = roi_aligner.MultilevelROIAligner()
    detection_generator_obj = detection_generator.DetectionGenerator()
    if include_mask:
      mask_head = instance_heads.MaskHead(
          num_classes=num_classes, upsample_factor=2)
      mask_sampler_obj = mask_sampler.MaskSampler(
          mask_target_size=28, num_sampled_masks=1)
      mask_roi_aligner_obj = roi_aligner.MultilevelROIAligner(crop_size=14)
    else:
      mask_head = None
      mask_sampler_obj = None
      mask_roi_aligner_obj = None
    model = maskrcnn_model.MaskRCNNModel(
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
        mask_roi_aligner_obj)

    gt_boxes = np.array(
        [[[10, 10, 15, 15], [2.5, 2.5, 7.5, 7.5], [-1, -1, -1, -1]],
         [[100, 100, 150, 150], [-1, -1, -1, -1], [-1, -1, -1, -1]]],
        dtype=np.float32)
    gt_classes = np.array([[2, 1, -1], [1, -1, -1]], dtype=np.int32)
    if include_mask:
      gt_masks = np.ones((2, 3, 100, 100))
    else:
      gt_masks = None

    _ = model(images,
              image_shape,
              anchor_boxes,
              gt_boxes,
              gt_classes,
              gt_masks,
              training=True)
    self.assertEqual(expected_num_params, model.count_params())

  @parameterized.parameters(
      (False, False,),
      (False, True,),
      (True, False,),
      (True, True,),
  )
  def test_forward(self, include_mask, training):
    num_classes = 3
    min_level = 3
    max_level = 4
    num_scales = 3
    aspect_ratios = [1.0]
    image_size = (256, 256)
    images = np.random.rand(2, image_size[0], image_size[1], 3)
    image_shape = np.array([[224, 100], [100, 224]])
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
    if include_mask:
      mask_head = instance_heads.MaskHead(
          num_classes=num_classes, upsample_factor=2)
      mask_sampler_obj = mask_sampler.MaskSampler(
          mask_target_size=28, num_sampled_masks=1)
      mask_roi_aligner_obj = roi_aligner.MultilevelROIAligner(crop_size=14)
    else:
      mask_head = None
      mask_sampler_obj = None
      mask_roi_aligner_obj = None
    model = maskrcnn_model.MaskRCNNModel(
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
        mask_roi_aligner_obj)

    gt_boxes = np.array(
        [[[10, 10, 15, 15], [2.5, 2.5, 7.5, 7.5], [-1, -1, -1, -1]],
         [[100, 100, 150, 150], [-1, -1, -1, -1], [-1, -1, -1, -1]]],
        dtype=np.float32)
    gt_classes = np.array([[2, 1, -1], [1, -1, -1]], dtype=np.int32)
    if include_mask:
      gt_masks = np.ones((2, 3, 100, 100))
    else:
      gt_masks = None

    results = model(images,
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
      if include_mask:
        self.assertIn('mask_outputs', results)
    else:
      self.assertIn('detection_boxes', results)
      self.assertIn('detection_scores', results)
      self.assertIn('detection_classes', results)
      self.assertIn('num_detections', results)
      if include_mask:
        self.assertIn('detection_masks', results)

  @parameterized.parameters(
      (False,),
      (True,),
  )
  def test_serialize_deserialize(self, include_mask):
    input_specs = tf.keras.layers.InputSpec(shape=[None, None, None, 3])
    backbone = resnet.ResNet(model_id=50, input_specs=input_specs)
    decoder = fpn.FPN(
        min_level=3,
        max_level=7,
        input_specs=backbone.output_specs)
    rpn_head = dense_prediction_heads.RPNHead(
        min_level=3,
        max_level=7,
        num_anchors_per_location=3)
    detection_head = instance_heads.DetectionHead(
        num_classes=2)
    roi_generator_obj = roi_generator.MultilevelROIGenerator()
    roi_sampler_obj = roi_sampler.ROISampler()
    roi_aligner_obj = roi_aligner.MultilevelROIAligner()
    detection_generator_obj = detection_generator.DetectionGenerator()
    if include_mask:
      mask_head = instance_heads.MaskHead(
          num_classes=2, upsample_factor=2)
      mask_sampler_obj = mask_sampler.MaskSampler(
          mask_target_size=28, num_sampled_masks=1)
      mask_roi_aligner_obj = roi_aligner.MultilevelROIAligner(crop_size=14)
    else:
      mask_head = None
      mask_sampler_obj = None
      mask_roi_aligner_obj = None
    model = maskrcnn_model.MaskRCNNModel(
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
        mask_roi_aligner_obj)

    config = model.get_config()
    new_model = maskrcnn_model.MaskRCNNModel.from_config(config)

    # Validate that the config can be forced to JSON.
    _ = new_model.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(model.get_config(), new_model.get_config())

  @parameterized.parameters(
      (False,),
      (True,),
  )
  def test_checkpoint(self, include_mask):
    input_specs = tf.keras.layers.InputSpec(shape=[None, None, None, 3])
    backbone = resnet.ResNet(model_id=50, input_specs=input_specs)
    decoder = fpn.FPN(
        min_level=3, max_level=7, input_specs=backbone.output_specs)
    rpn_head = dense_prediction_heads.RPNHead(
        min_level=3, max_level=7, num_anchors_per_location=3)
    detection_head = instance_heads.DetectionHead(num_classes=2)
    roi_generator_obj = roi_generator.MultilevelROIGenerator()
    roi_sampler_obj = roi_sampler.ROISampler()
    roi_aligner_obj = roi_aligner.MultilevelROIAligner()
    detection_generator_obj = detection_generator.DetectionGenerator()
    if include_mask:
      mask_head = instance_heads.MaskHead(num_classes=2, upsample_factor=2)
      mask_sampler_obj = mask_sampler.MaskSampler(
          mask_target_size=28, num_sampled_masks=1)
      mask_roi_aligner_obj = roi_aligner.MultilevelROIAligner(crop_size=14)
    else:
      mask_head = None
      mask_sampler_obj = None
      mask_roi_aligner_obj = None
    model = maskrcnn_model.MaskRCNNModel(backbone, decoder, rpn_head,
                                         detection_head, roi_generator_obj,
                                         roi_sampler_obj, roi_aligner_obj,
                                         detection_generator_obj, mask_head,
                                         mask_sampler_obj, mask_roi_aligner_obj)
    expect_checkpoint_items = dict(
        backbone=backbone,
        decoder=decoder,
        rpn_head=rpn_head,
        detection_head=detection_head)
    if include_mask:
      expect_checkpoint_items['mask_head'] = mask_head
    self.assertAllEqual(expect_checkpoint_items, model.checkpoint_items)

    # Test save and load checkpoints.
    ckpt = tf.train.Checkpoint(model=model, **model.checkpoint_items)
    save_dir = self.create_tempdir().full_path
    ckpt.save(os.path.join(save_dir, 'ckpt'))

    partial_ckpt = tf.train.Checkpoint(backbone=backbone)
    partial_ckpt.restore(tf.train.latest_checkpoint(
        save_dir)).expect_partial().assert_existing_objects_matched()

    if include_mask:
      partial_ckpt_mask = tf.train.Checkpoint(
          backbone=backbone, mask_head=mask_head)
      partial_ckpt_mask.restore(tf.train.latest_checkpoint(
          save_dir)).expect_partial().assert_existing_objects_matched()


if __name__ == '__main__':
  tf.test.main()
