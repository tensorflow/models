# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Mask R-CNN variant with support for deep mask heads."""

import tensorflow as tf

from official.core import task_factory
from official.projects.deepmac_maskrcnn.configs import deep_mask_head_rcnn as deep_mask_head_rcnn_config
from official.projects.deepmac_maskrcnn.modeling import maskrcnn_model as deep_maskrcnn_model
from official.projects.deepmac_maskrcnn.modeling.heads import instance_heads as deep_instance_heads
from official.vision.modeling import backbones
from official.vision.modeling.decoders import factory as decoder_factory
from official.vision.modeling.heads import dense_prediction_heads
from official.vision.modeling.heads import instance_heads
from official.vision.modeling.layers import detection_generator
from official.vision.modeling.layers import mask_sampler
from official.vision.modeling.layers import roi_aligner
from official.vision.modeling.layers import roi_generator
from official.vision.modeling.layers import roi_sampler
from official.vision.tasks import maskrcnn


# Taken from modeling/factory.py
def build_maskrcnn(input_specs: tf.keras.layers.InputSpec,
                   model_config: deep_mask_head_rcnn_config.DeepMaskHeadRCNN,
                   l2_regularizer: tf.keras.regularizers.Regularizer = None):  # pytype: disable=annotation-type-mismatch  # typed-keras
  """Builds Mask R-CNN model."""
  norm_activation_config = model_config.norm_activation
  backbone = backbones.factory.build_backbone(
      input_specs=input_specs,
      backbone_config=model_config.backbone,
      norm_activation_config=norm_activation_config,
      l2_regularizer=l2_regularizer)

  decoder = decoder_factory.build_decoder(
      input_specs=backbone.output_specs,
      model_config=model_config,
      l2_regularizer=l2_regularizer)

  rpn_head_config = model_config.rpn_head
  roi_generator_config = model_config.roi_generator
  roi_sampler_config = model_config.roi_sampler
  roi_aligner_config = model_config.roi_aligner
  detection_head_config = model_config.detection_head
  generator_config = model_config.detection_generator
  num_anchors_per_location = (
      len(model_config.anchor.aspect_ratios) * model_config.anchor.num_scales)

  rpn_head = dense_prediction_heads.RPNHead(
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      num_anchors_per_location=num_anchors_per_location,
      num_convs=rpn_head_config.num_convs,
      num_filters=rpn_head_config.num_filters,
      use_separable_conv=rpn_head_config.use_separable_conv,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)

  detection_head = instance_heads.DetectionHead(
      num_classes=model_config.num_classes,
      num_convs=detection_head_config.num_convs,
      num_filters=detection_head_config.num_filters,
      use_separable_conv=detection_head_config.use_separable_conv,
      num_fcs=detection_head_config.num_fcs,
      fc_dims=detection_head_config.fc_dims,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)

  roi_generator_obj = roi_generator.MultilevelROIGenerator(
      pre_nms_top_k=roi_generator_config.pre_nms_top_k,
      pre_nms_score_threshold=roi_generator_config.pre_nms_score_threshold,
      pre_nms_min_size_threshold=(
          roi_generator_config.pre_nms_min_size_threshold),
      nms_iou_threshold=roi_generator_config.nms_iou_threshold,
      num_proposals=roi_generator_config.num_proposals,
      test_pre_nms_top_k=roi_generator_config.test_pre_nms_top_k,
      test_pre_nms_score_threshold=(
          roi_generator_config.test_pre_nms_score_threshold),
      test_pre_nms_min_size_threshold=(
          roi_generator_config.test_pre_nms_min_size_threshold),
      test_nms_iou_threshold=roi_generator_config.test_nms_iou_threshold,
      test_num_proposals=roi_generator_config.test_num_proposals,
      use_batched_nms=roi_generator_config.use_batched_nms)

  roi_sampler_obj = roi_sampler.ROISampler(
      mix_gt_boxes=roi_sampler_config.mix_gt_boxes,
      num_sampled_rois=roi_sampler_config.num_sampled_rois,
      foreground_fraction=roi_sampler_config.foreground_fraction,
      foreground_iou_threshold=roi_sampler_config.foreground_iou_threshold,
      background_iou_high_threshold=(
          roi_sampler_config.background_iou_high_threshold),
      background_iou_low_threshold=(
          roi_sampler_config.background_iou_low_threshold))

  roi_aligner_obj = roi_aligner.MultilevelROIAligner(
      crop_size=roi_aligner_config.crop_size,
      sample_offset=roi_aligner_config.sample_offset)

  detection_generator_obj = detection_generator.DetectionGenerator(
      apply_nms=True,
      pre_nms_top_k=generator_config.pre_nms_top_k,
      pre_nms_score_threshold=generator_config.pre_nms_score_threshold,
      nms_iou_threshold=generator_config.nms_iou_threshold,
      max_num_detections=generator_config.max_num_detections,
      nms_version=generator_config.nms_version,
      use_sigmoid_probability=generator_config.use_sigmoid_probability)

  if model_config.include_mask:
    mask_head = deep_instance_heads.DeepMaskHead(
        num_classes=model_config.num_classes,
        upsample_factor=model_config.mask_head.upsample_factor,
        num_convs=model_config.mask_head.num_convs,
        num_filters=model_config.mask_head.num_filters,
        use_separable_conv=model_config.mask_head.use_separable_conv,
        activation=model_config.norm_activation.activation,
        norm_momentum=model_config.norm_activation.norm_momentum,
        norm_epsilon=model_config.norm_activation.norm_epsilon,
        kernel_regularizer=l2_regularizer,
        class_agnostic=model_config.mask_head.class_agnostic,
        convnet_variant=model_config.mask_head.convnet_variant)

    mask_sampler_obj = mask_sampler.MaskSampler(
        mask_target_size=(
            model_config.mask_roi_aligner.crop_size *
            model_config.mask_head.upsample_factor),
        num_sampled_masks=model_config.mask_sampler.num_sampled_masks)

    mask_roi_aligner_obj = roi_aligner.MultilevelROIAligner(
        crop_size=model_config.mask_roi_aligner.crop_size,
        sample_offset=model_config.mask_roi_aligner.sample_offset)
  else:
    mask_head = None
    mask_sampler_obj = None
    mask_roi_aligner_obj = None

  model = deep_maskrcnn_model.DeepMaskRCNNModel(
      backbone=backbone,
      decoder=decoder,
      rpn_head=rpn_head,
      detection_head=detection_head,
      roi_generator=roi_generator_obj,
      roi_sampler=roi_sampler_obj,
      roi_aligner=roi_aligner_obj,
      detection_generator=detection_generator_obj,
      mask_head=mask_head,
      mask_sampler=mask_sampler_obj,
      mask_roi_aligner=mask_roi_aligner_obj,
      class_agnostic_bbox_pred=detection_head_config.class_agnostic_bbox_pred,
      cascade_class_ensemble=detection_head_config.cascade_class_ensemble,
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      num_scales=model_config.anchor.num_scales,
      aspect_ratios=model_config.anchor.aspect_ratios,
      anchor_size=model_config.anchor.anchor_size,
      outer_boxes_scale=model_config.outer_boxes_scale,
      use_gt_boxes_for_masks=model_config.use_gt_boxes_for_masks)
  return model


@task_factory.register_task_cls(deep_mask_head_rcnn_config.DeepMaskHeadRCNNTask)
class DeepMaskHeadRCNNTask(maskrcnn.MaskRCNNTask):
  """Mask R-CNN with support for deep mask heads."""

  def build_model(self):
    """Builds Mask R-CNN model."""

    input_specs = tf.keras.layers.InputSpec(
        shape=[None] + self.task_config.model.input_size)

    l2_weight_decay = self.task_config.losses.l2_weight_decay
    # Divide weight decay by 2.0 to match the implementation of tf.nn.l2_loss.
    # (https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2)
    # (https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss)
    l2_regularizer = (tf.keras.regularizers.l2(
        l2_weight_decay / 2.0) if l2_weight_decay else None)

    model = build_maskrcnn(
        input_specs=input_specs,
        model_config=self.task_config.model,
        l2_regularizer=l2_regularizer)

    if self.task_config.freeze_backbone:
      model.backbone.trainable = False

    # Builds the model through warm-up call.
    dummy_images = tf.keras.Input(self.task_config.model.input_size)
    dummy_image_shape = tf.keras.layers.Input([2])
    _ = model(dummy_images, image_shape=dummy_image_shape, training=False)

    return model
