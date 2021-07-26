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

"""Factory method to build panoptic segmentation model."""

import tensorflow as tf

from official.vision.beta.modeling import backbones
from official.vision.beta.modeling import factory as models_factory
from official.vision.beta.modeling.decoders import factory as decoder_factory
from official.vision.beta.modeling.heads import segmentation_heads
from official.vision.beta.projects.panoptic_maskrcnn.configs import panoptic_maskrcnn as panoptic_maskrcnn_cfg
from official.vision.beta.projects.panoptic_maskrcnn.modeling import panoptic_maskrcnn_model


def build_panoptic_maskrcnn(
    input_specs: tf.keras.layers.InputSpec,
    model_config: panoptic_maskrcnn_cfg.PanopticMaskRCNN,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:
  """Builds Panoptic Mask R-CNN model.

  This factory function builds the mask rcnn first, builds the non-shared
  semantic segmentation layers, and finally combines the two models to form
  the panoptic segmentation model.

  Args:
    input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
    model_config: Config instance for the panoptic maskrcnn model.
    l2_regularizer: Optional `tf.keras.regularizers.Regularizer`, if specified,
      the model is built with the provided regularization layer.
  Returns:
    tf.keras.Model for the panoptic segmentation model.
  """
  norm_activation_config = model_config.norm_activation
  segmentation_config = model_config.segmentation_model

  # Builds the maskrcnn model.
  maskrcnn_model = models_factory.build_maskrcnn(
      input_specs=input_specs,
      model_config=model_config,
      l2_regularizer=l2_regularizer)

  # Builds the semantic segmentation branch.
  if not model_config.shared_backbone:
    segmentation_backbone = backbones.factory.build_backbone(
        input_specs=input_specs,
        backbone_config=segmentation_config.backbone,
        norm_activation_config=norm_activation_config,
        l2_regularizer=l2_regularizer)
    segmentation_decoder_input_specs = segmentation_backbone.output_specs
  else:
    segmentation_backbone = None
    segmentation_decoder_input_specs = maskrcnn_model.backbone.output_specs

  if not model_config.shared_decoder:
    segmentation_decoder = decoder_factory.build_decoder(
        input_specs=segmentation_decoder_input_specs,
        model_config=segmentation_config,
        l2_regularizer=l2_regularizer)
  else:
    segmentation_decoder = None

  segmentation_head_config = segmentation_config.head
  detection_head_config = model_config.detection_head

  segmentation_head = segmentation_heads.SegmentationHead(
      num_classes=segmentation_config.num_classes,
      level=segmentation_head_config.level,
      num_convs=segmentation_head_config.num_convs,
      prediction_kernel_size=segmentation_head_config.prediction_kernel_size,
      num_filters=segmentation_head_config.num_filters,
      upsample_factor=segmentation_head_config.upsample_factor,
      feature_fusion=segmentation_head_config.feature_fusion,
      low_level=segmentation_head_config.low_level,
      low_level_num_filters=segmentation_head_config.low_level_num_filters,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer)

  # Combines maskrcnn, and segmentation models to build panoptic segmentation
  # model.
  model = panoptic_maskrcnn_model.PanopticMaskRCNNModel(
      backbone=maskrcnn_model.backbone,
      decoder=maskrcnn_model.decoder,
      rpn_head=maskrcnn_model.rpn_head,
      detection_head=maskrcnn_model.detection_head,
      roi_generator=maskrcnn_model.roi_generator,
      roi_sampler=maskrcnn_model.roi_sampler,
      roi_aligner=maskrcnn_model.roi_aligner,
      detection_generator=maskrcnn_model.detection_generator,
      mask_head=maskrcnn_model.mask_head,
      mask_sampler=maskrcnn_model.mask_sampler,
      mask_roi_aligner=maskrcnn_model.mask_roi_aligner,
      segmentation_backbone=segmentation_backbone,
      segmentation_decoder=segmentation_decoder,
      segmentation_head=segmentation_head,
      class_agnostic_bbox_pred=detection_head_config.class_agnostic_bbox_pred,
      cascade_class_ensemble=detection_head_config.cascade_class_ensemble,
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      num_scales=model_config.anchor.num_scales,
      aspect_ratios=model_config.anchor.aspect_ratios,
      anchor_size=model_config.anchor.anchor_size)
  return model
