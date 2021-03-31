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

"""Factory methods to build models."""

# Import libraries

import tensorflow as tf

from official.vision.beta.configs import image_classification as classification_cfg
from official.vision.beta.configs import maskrcnn as maskrcnn_cfg
from official.vision.beta.configs import retinanet as retinanet_cfg
from official.vision.beta.configs import semantic_segmentation as segmentation_cfg
from official.vision.beta.modeling import backbones
from official.vision.beta.modeling import classification_model
from official.vision.beta.modeling import maskrcnn_model
from official.vision.beta.modeling import retinanet_model
from official.vision.beta.modeling import segmentation_model
from official.vision.beta.modeling.decoders import factory as decoder_factory
from official.vision.beta.modeling.heads import dense_prediction_heads
from official.vision.beta.modeling.heads import instance_heads
from official.vision.beta.modeling.heads import segmentation_heads
from official.vision.beta.modeling.layers import detection_generator
from official.vision.beta.modeling.layers import mask_sampler
from official.vision.beta.modeling.layers import roi_aligner
from official.vision.beta.modeling.layers import roi_generator
from official.vision.beta.modeling.layers import roi_sampler


def build_classification_model(
    input_specs: tf.keras.layers.InputSpec,
    model_config: classification_cfg.ImageClassificationModel,
    l2_regularizer: tf.keras.regularizers.Regularizer = None,
    skip_logits_layer: bool = False):
  """Builds the classification model."""
  backbone = backbones.factory.build_backbone(
      input_specs=input_specs,
      model_config=model_config,
      l2_regularizer=l2_regularizer)

  norm_activation_config = model_config.norm_activation
  model = classification_model.ClassificationModel(
      backbone=backbone,
      num_classes=model_config.num_classes,
      input_specs=input_specs,
      dropout_rate=model_config.dropout_rate,
      kernel_regularizer=l2_regularizer,
      add_head_batch_norm=model_config.add_head_batch_norm,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      skip_logits_layer=skip_logits_layer)
  return model


def build_maskrcnn(input_specs: tf.keras.layers.InputSpec,
                   model_config: maskrcnn_cfg.MaskRCNN,
                   l2_regularizer: tf.keras.regularizers.Regularizer = None):
  """Builds Mask R-CNN model."""
  backbone = backbones.factory.build_backbone(
      input_specs=input_specs,
      model_config=model_config,
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
  norm_activation_config = model_config.norm_activation
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
      use_batched_nms=generator_config.use_batched_nms)

  if model_config.include_mask:
    mask_head = instance_heads.MaskHead(
        num_classes=model_config.num_classes,
        upsample_factor=model_config.mask_head.upsample_factor,
        num_convs=model_config.mask_head.num_convs,
        num_filters=model_config.mask_head.num_filters,
        use_separable_conv=model_config.mask_head.use_separable_conv,
        activation=model_config.norm_activation.activation,
        norm_momentum=model_config.norm_activation.norm_momentum,
        norm_epsilon=model_config.norm_activation.norm_epsilon,
        kernel_regularizer=l2_regularizer,
        class_agnostic=model_config.mask_head.class_agnostic)

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

  model = maskrcnn_model.MaskRCNNModel(
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
      mask_roi_aligner=mask_roi_aligner_obj)
  return model


def build_retinanet(input_specs: tf.keras.layers.InputSpec,
                    model_config: retinanet_cfg.RetinaNet,
                    l2_regularizer: tf.keras.regularizers.Regularizer = None):
  """Builds RetinaNet model."""
  backbone = backbones.factory.build_backbone(
      input_specs=input_specs,
      model_config=model_config,
      l2_regularizer=l2_regularizer)

  decoder = decoder_factory.build_decoder(
      input_specs=backbone.output_specs,
      model_config=model_config,
      l2_regularizer=l2_regularizer)

  head_config = model_config.head
  generator_config = model_config.detection_generator
  norm_activation_config = model_config.norm_activation
  num_anchors_per_location = (
      len(model_config.anchor.aspect_ratios) * model_config.anchor.num_scales)

  head = dense_prediction_heads.RetinaNetHead(
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      num_classes=model_config.num_classes,
      num_anchors_per_location=num_anchors_per_location,
      num_convs=head_config.num_convs,
      num_filters=head_config.num_filters,
      attribute_heads=head_config.attribute_heads,
      use_separable_conv=head_config.use_separable_conv,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)

  detection_generator_obj = detection_generator.MultilevelDetectionGenerator(
      apply_nms=True,
      pre_nms_top_k=generator_config.pre_nms_top_k,
      pre_nms_score_threshold=generator_config.pre_nms_score_threshold,
      nms_iou_threshold=generator_config.nms_iou_threshold,
      max_num_detections=generator_config.max_num_detections,
      use_batched_nms=generator_config.use_batched_nms)

  model = retinanet_model.RetinaNetModel(
      backbone, decoder, head, detection_generator_obj)
  return model


def build_segmentation_model(
    input_specs: tf.keras.layers.InputSpec,
    model_config: segmentation_cfg.SemanticSegmentationModel,
    l2_regularizer: tf.keras.regularizers.Regularizer = None):
  """Builds Segmentation model."""
  backbone = backbones.factory.build_backbone(
      input_specs=input_specs,
      model_config=model_config,
      l2_regularizer=l2_regularizer)

  decoder = decoder_factory.build_decoder(
      input_specs=backbone.output_specs,
      model_config=model_config,
      l2_regularizer=l2_regularizer)

  head_config = model_config.head
  norm_activation_config = model_config.norm_activation

  head = segmentation_heads.SegmentationHead(
      num_classes=model_config.num_classes,
      level=head_config.level,
      num_convs=head_config.num_convs,
      num_filters=head_config.num_filters,
      upsample_factor=head_config.upsample_factor,
      feature_fusion=head_config.feature_fusion,
      low_level=head_config.low_level,
      low_level_num_filters=head_config.low_level_num_filters,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)

  model = segmentation_model.SegmentationModel(backbone, decoder, head)
  return model
