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

"""Tests for panoptic_maskrcnn_model.py."""

import os
from absl.testing import parameterized
import tensorflow as tf

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from official.vision.beta.modeling.backbones import resnet
from official.vision.beta.modeling.decoders import aspp
from official.vision.beta.modeling.decoders import fpn
from official.vision.beta.modeling.heads import dense_prediction_heads
from official.vision.beta.modeling.heads import instance_heads
from official.vision.beta.modeling.heads import segmentation_heads
from official.vision.beta.modeling.layers import detection_generator
from official.vision.beta.modeling.layers import mask_sampler
from official.vision.beta.modeling.layers import roi_aligner
from official.vision.beta.modeling.layers import roi_generator
from official.vision.beta.modeling.layers import roi_sampler
from official.vision.beta.ops import anchor
from official.vision.beta.projects.panoptic_maskrcnn.modeling import panoptic_maskrcnn_model
from official.vision.beta.projects.panoptic_maskrcnn.modeling.layers import panoptic_segmentation_generator


class PanopticMaskRCNNModelTest(parameterized.TestCase, tf.test.TestCase):

  @combinations.generate(
      combinations.combine(
          use_separable_conv=[True, False],
          build_anchor_boxes=[True, False],
          shared_backbone=[True, False],
          shared_decoder=[True, False],
          is_training=[True,]))
  def test_build_model(self,
                       use_separable_conv,
                       build_anchor_boxes,
                       shared_backbone,
                       shared_decoder,
                       is_training=True):
    num_classes = 3
    min_level = 2
    max_level = 6
    num_scales = 3
    aspect_ratios = [1.0]
    anchor_size = 3
    resnet_model_id = 50
    segmentation_resnet_model_id = 50
    aspp_dilation_rates = [6, 12, 18]
    aspp_decoder_level = 2
    fpn_decoder_level = 2
    num_anchors_per_location = num_scales * len(aspect_ratios)
    image_size = 128
    images = tf.random.normal([2, image_size, image_size, 3])
    image_info = tf.convert_to_tensor(
        [[[image_size, image_size], [image_size, image_size], [1, 1], [0, 0]],
         [[image_size, image_size], [image_size, image_size], [1, 1], [0, 0]]])
    shared_decoder = shared_decoder and shared_backbone
    if build_anchor_boxes or not is_training:
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
    else:
      anchor_boxes = None

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
    detection_head = instance_heads.DetectionHead(num_classes=num_classes)
    roi_generator_obj = roi_generator.MultilevelROIGenerator()
    roi_sampler_obj = roi_sampler.ROISampler()
    roi_aligner_obj = roi_aligner.MultilevelROIAligner()
    detection_generator_obj = detection_generator.DetectionGenerator()
    panoptic_segmentation_generator_obj = panoptic_segmentation_generator.PanopticSegmentationGenerator(
        output_size=[image_size, image_size],
        max_num_detections=100,
        stuff_classes_offset=90)
    mask_head = instance_heads.MaskHead(
        num_classes=num_classes, upsample_factor=2)
    mask_sampler_obj = mask_sampler.MaskSampler(
        mask_target_size=28, num_sampled_masks=1)
    mask_roi_aligner_obj = roi_aligner.MultilevelROIAligner(crop_size=14)

    if shared_backbone:
      segmentation_backbone = None
    else:
      segmentation_backbone = resnet.ResNet(
          model_id=segmentation_resnet_model_id)
    if not shared_decoder:
      feature_fusion = 'deeplabv3plus'
      level = aspp_decoder_level
      segmentation_decoder = aspp.ASPP(
          level=level, dilation_rates=aspp_dilation_rates)
    else:
      feature_fusion = 'panoptic_fpn_fusion'
      level = fpn_decoder_level
      segmentation_decoder = None
    segmentation_head = segmentation_heads.SegmentationHead(
        num_classes=2,  # stuff and common class for things,
        level=level,
        feature_fusion=feature_fusion,
        decoder_min_level=min_level,
        decoder_max_level=max_level,
        num_convs=2)

    model = panoptic_maskrcnn_model.PanopticMaskRCNNModel(
        backbone,
        decoder,
        rpn_head,
        detection_head,
        roi_generator_obj,
        roi_sampler_obj,
        roi_aligner_obj,
        detection_generator_obj,
        panoptic_segmentation_generator_obj,
        mask_head,
        mask_sampler_obj,
        mask_roi_aligner_obj,
        segmentation_backbone=segmentation_backbone,
        segmentation_decoder=segmentation_decoder,
        segmentation_head=segmentation_head,
        min_level=min_level,
        max_level=max_level,
        num_scales=num_scales,
        aspect_ratios=aspect_ratios,
        anchor_size=anchor_size)

    gt_boxes = tf.convert_to_tensor(
        [[[10, 10, 15, 15], [2.5, 2.5, 7.5, 7.5], [-1, -1, -1, -1]],
         [[100, 100, 150, 150], [-1, -1, -1, -1], [-1, -1, -1, -1]]],
        dtype=tf.float32)
    gt_classes = tf.convert_to_tensor([[2, 1, -1], [1, -1, -1]], dtype=tf.int32)
    gt_masks = tf.ones((2, 3, 100, 100))

    # Results will be checked in test_forward.
    _ = model(
        images,
        image_info,
        anchor_boxes,
        gt_boxes,
        gt_classes,
        gt_masks,
        training=is_training)

  @combinations.generate(
      combinations.combine(
          strategy=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.one_device_strategy_gpu,
          ],
          shared_backbone=[True, False],
          shared_decoder=[True, False],
          training=[True, False],
          generate_panoptic_masks=[True, False]))
  def test_forward(self, strategy, training,
                   shared_backbone, shared_decoder,
                   generate_panoptic_masks):
    num_classes = 3
    min_level = 2
    max_level = 6
    num_scales = 3
    aspect_ratios = [1.0]
    anchor_size = 3
    segmentation_resnet_model_id = 101
    aspp_dilation_rates = [6, 12, 18]
    aspp_decoder_level = 2
    fpn_decoder_level = 2

    class_agnostic_bbox_pred = False
    cascade_class_ensemble = False

    image_size = (256, 256)
    images = tf.random.normal([2, image_size[0], image_size[1], 3])
    image_info = tf.convert_to_tensor(
        [[[224, 100], [224, 100], [1, 1], [0, 0]],
         [[224, 100], [224, 100], [1, 1], [0, 0]]])
    shared_decoder = shared_decoder and shared_backbone
    with strategy.scope():

      anchor_boxes = anchor.Anchor(
          min_level=min_level,
          max_level=max_level,
          num_scales=num_scales,
          aspect_ratios=aspect_ratios,
          anchor_size=anchor_size,
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
          num_classes=num_classes,
          class_agnostic_bbox_pred=class_agnostic_bbox_pred)
      roi_generator_obj = roi_generator.MultilevelROIGenerator()

      roi_sampler_cascade = []
      roi_sampler_obj = roi_sampler.ROISampler()
      roi_sampler_cascade.append(roi_sampler_obj)
      roi_aligner_obj = roi_aligner.MultilevelROIAligner()
      detection_generator_obj = detection_generator.DetectionGenerator()

      if generate_panoptic_masks:
        panoptic_segmentation_generator_obj = panoptic_segmentation_generator.PanopticSegmentationGenerator(
            output_size=list(image_size),
            max_num_detections=100,
            stuff_classes_offset=90)
      else:
        panoptic_segmentation_generator_obj = None

      mask_head = instance_heads.MaskHead(
          num_classes=num_classes, upsample_factor=2)
      mask_sampler_obj = mask_sampler.MaskSampler(
          mask_target_size=28, num_sampled_masks=1)
      mask_roi_aligner_obj = roi_aligner.MultilevelROIAligner(crop_size=14)

      if shared_backbone:
        segmentation_backbone = None
      else:
        segmentation_backbone = resnet.ResNet(
            model_id=segmentation_resnet_model_id)
      if not shared_decoder:
        feature_fusion = 'deeplabv3plus'
        level = aspp_decoder_level
        segmentation_decoder = aspp.ASPP(
            level=level, dilation_rates=aspp_dilation_rates)
      else:
        feature_fusion = 'panoptic_fpn_fusion'
        level = fpn_decoder_level
        segmentation_decoder = None
      segmentation_head = segmentation_heads.SegmentationHead(
          num_classes=2,  # stuff and common class for things,
          level=level,
          feature_fusion=feature_fusion,
          decoder_min_level=min_level,
          decoder_max_level=max_level,
          num_convs=2)

      model = panoptic_maskrcnn_model.PanopticMaskRCNNModel(
          backbone,
          decoder,
          rpn_head,
          detection_head,
          roi_generator_obj,
          roi_sampler_obj,
          roi_aligner_obj,
          detection_generator_obj,
          panoptic_segmentation_generator_obj,
          mask_head,
          mask_sampler_obj,
          mask_roi_aligner_obj,
          segmentation_backbone=segmentation_backbone,
          segmentation_decoder=segmentation_decoder,
          segmentation_head=segmentation_head,
          class_agnostic_bbox_pred=class_agnostic_bbox_pred,
          cascade_class_ensemble=cascade_class_ensemble,
          min_level=min_level,
          max_level=max_level,
          num_scales=num_scales,
          aspect_ratios=aspect_ratios,
          anchor_size=anchor_size)

      gt_boxes = tf.convert_to_tensor(
          [[[10, 10, 15, 15], [2.5, 2.5, 7.5, 7.5], [-1, -1, -1, -1]],
           [[100, 100, 150, 150], [-1, -1, -1, -1], [-1, -1, -1, -1]]],
          dtype=tf.float32)
      gt_classes = tf.convert_to_tensor(
          [[2, 1, -1], [1, -1, -1]], dtype=tf.int32)
      gt_masks = tf.ones((2, 3, 100, 100))

      results = model(
          images,
          image_info,
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
    else:
      self.assertIn('detection_boxes', results)
      self.assertIn('detection_scores', results)
      self.assertIn('detection_classes', results)
      self.assertIn('num_detections', results)
      self.assertIn('detection_masks', results)
      self.assertIn('segmentation_outputs', results)

      self.assertAllEqual(
          [2, image_size[0] // (2**level), image_size[1] // (2**level), 2],
          results['segmentation_outputs'].numpy().shape)

      if generate_panoptic_masks:
        self.assertIn('panoptic_outputs', results)
        self.assertIn('category_mask', results['panoptic_outputs'])
        self.assertIn('instance_mask', results['panoptic_outputs'])
        self.assertAllEqual(
            [2, image_size[0], image_size[1]],
            results['panoptic_outputs']['category_mask'].numpy().shape)
        self.assertAllEqual(
            [2, image_size[0], image_size[1]],
            results['panoptic_outputs']['instance_mask'].numpy().shape)
      else:
        self.assertNotIn('panoptic_outputs', results)

  @combinations.generate(
      combinations.combine(
          shared_backbone=[True, False], shared_decoder=[True, False]))
  def test_serialize_deserialize(self, shared_backbone, shared_decoder):
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
    panoptic_segmentation_generator_obj = panoptic_segmentation_generator.PanopticSegmentationGenerator(
        output_size=[None, None],
        max_num_detections=100,
        stuff_classes_offset=90)
    segmentation_resnet_model_id = 101
    aspp_dilation_rates = [6, 12, 18]
    min_level = 2
    max_level = 6
    aspp_decoder_level = 2
    fpn_decoder_level = 2
    shared_decoder = shared_decoder and shared_backbone
    mask_head = instance_heads.MaskHead(num_classes=2, upsample_factor=2)
    mask_sampler_obj = mask_sampler.MaskSampler(
        mask_target_size=28, num_sampled_masks=1)
    mask_roi_aligner_obj = roi_aligner.MultilevelROIAligner(crop_size=14)

    if shared_backbone:
      segmentation_backbone = None
    else:
      segmentation_backbone = resnet.ResNet(
          model_id=segmentation_resnet_model_id)
    if not shared_decoder:
      feature_fusion = 'deeplabv3plus'
      level = aspp_decoder_level
      segmentation_decoder = aspp.ASPP(
          level=level, dilation_rates=aspp_dilation_rates)
    else:
      feature_fusion = 'panoptic_fpn_fusion'
      level = fpn_decoder_level
      segmentation_decoder = None
    segmentation_head = segmentation_heads.SegmentationHead(
        num_classes=2,  # stuff and common class for things,
        level=level,
        feature_fusion=feature_fusion,
        decoder_min_level=min_level,
        decoder_max_level=max_level,
        num_convs=2)

    model = panoptic_maskrcnn_model.PanopticMaskRCNNModel(
        backbone,
        decoder,
        rpn_head,
        detection_head,
        roi_generator_obj,
        roi_sampler_obj,
        roi_aligner_obj,
        detection_generator_obj,
        panoptic_segmentation_generator_obj,
        mask_head,
        mask_sampler_obj,
        mask_roi_aligner_obj,
        segmentation_backbone=segmentation_backbone,
        segmentation_decoder=segmentation_decoder,
        segmentation_head=segmentation_head,
        min_level=min_level,
        max_level=max_level,
        num_scales=3,
        aspect_ratios=[1.0],
        anchor_size=3)

    config = model.get_config()
    new_model = panoptic_maskrcnn_model.PanopticMaskRCNNModel.from_config(
        config)

    # Validate that the config can be forced to JSON.
    _ = new_model.to_json()

    # If the serialization was successful, the new config should match the old.
    self.assertAllEqual(model.get_config(), new_model.get_config())

  @combinations.generate(
      combinations.combine(
          shared_backbone=[True, False], shared_decoder=[True, False]))
  def test_checkpoint(self, shared_backbone, shared_decoder):
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
    panoptic_segmentation_generator_obj = panoptic_segmentation_generator.PanopticSegmentationGenerator(
        output_size=[None, None],
        max_num_detections=100,
        stuff_classes_offset=90)
    segmentation_resnet_model_id = 101
    aspp_dilation_rates = [6, 12, 18]
    min_level = 2
    max_level = 6
    aspp_decoder_level = 2
    fpn_decoder_level = 2
    shared_decoder = shared_decoder and shared_backbone
    mask_head = instance_heads.MaskHead(num_classes=2, upsample_factor=2)
    mask_sampler_obj = mask_sampler.MaskSampler(
        mask_target_size=28, num_sampled_masks=1)
    mask_roi_aligner_obj = roi_aligner.MultilevelROIAligner(crop_size=14)

    if shared_backbone:
      segmentation_backbone = None
    else:
      segmentation_backbone = resnet.ResNet(
          model_id=segmentation_resnet_model_id)
    if not shared_decoder:
      feature_fusion = 'deeplabv3plus'
      level = aspp_decoder_level
      segmentation_decoder = aspp.ASPP(
          level=level, dilation_rates=aspp_dilation_rates)
    else:
      feature_fusion = 'panoptic_fpn_fusion'
      level = fpn_decoder_level
      segmentation_decoder = None
    segmentation_head = segmentation_heads.SegmentationHead(
        num_classes=2,  # stuff and common class for things,
        level=level,
        feature_fusion=feature_fusion,
        decoder_min_level=min_level,
        decoder_max_level=max_level,
        num_convs=2)

    model = panoptic_maskrcnn_model.PanopticMaskRCNNModel(
        backbone,
        decoder,
        rpn_head,
        detection_head,
        roi_generator_obj,
        roi_sampler_obj,
        roi_aligner_obj,
        detection_generator_obj,
        panoptic_segmentation_generator_obj,
        mask_head,
        mask_sampler_obj,
        mask_roi_aligner_obj,
        segmentation_backbone=segmentation_backbone,
        segmentation_decoder=segmentation_decoder,
        segmentation_head=segmentation_head,
        min_level=max_level,
        max_level=max_level,
        num_scales=3,
        aspect_ratios=[1.0],
        anchor_size=3)
    expect_checkpoint_items = dict(
        backbone=backbone,
        decoder=decoder,
        rpn_head=rpn_head,
        detection_head=[detection_head])
    expect_checkpoint_items['mask_head'] = mask_head
    if not shared_backbone:
      expect_checkpoint_items['segmentation_backbone'] = segmentation_backbone
    if not shared_decoder:
      expect_checkpoint_items['segmentation_decoder'] = segmentation_decoder
    expect_checkpoint_items['segmentation_head'] = segmentation_head
    self.assertAllEqual(expect_checkpoint_items, model.checkpoint_items)

    # Test save and load checkpoints.
    ckpt = tf.train.Checkpoint(model=model, **model.checkpoint_items)
    save_dir = self.create_tempdir().full_path
    ckpt.save(os.path.join(save_dir, 'ckpt'))

    partial_ckpt = tf.train.Checkpoint(backbone=backbone)
    partial_ckpt.read(tf.train.latest_checkpoint(
        save_dir)).expect_partial().assert_existing_objects_matched()

    partial_ckpt_mask = tf.train.Checkpoint(
        backbone=backbone, mask_head=mask_head)
    partial_ckpt_mask.restore(tf.train.latest_checkpoint(
        save_dir)).expect_partial().assert_existing_objects_matched()

    if not shared_backbone:
      partial_ckpt_segmentation = tf.train.Checkpoint(
          segmentation_backbone=segmentation_backbone,
          segmentation_decoder=segmentation_decoder,
          segmentation_head=segmentation_head)
    elif not shared_decoder:
      partial_ckpt_segmentation = tf.train.Checkpoint(
          segmentation_decoder=segmentation_decoder,
          segmentation_head=segmentation_head)
    else:
      partial_ckpt_segmentation = tf.train.Checkpoint(
          segmentation_head=segmentation_head)

    partial_ckpt_segmentation.restore(tf.train.latest_checkpoint(
        save_dir)).expect_partial().assert_existing_objects_matched()


if __name__ == '__main__':
  tf.test.main()
