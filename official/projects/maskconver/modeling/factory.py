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

"""Factory method to build panoptic segmentation model."""
from typing import Optional, Union

import tensorflow as tf, tf_keras

from official.projects.maskconver.configs import maskconver as maskconver_cfg
from official.projects.maskconver.configs import multiscale_maskconver as multiscale_maskconver_cfg
from official.projects.maskconver.modeling import maskconver_model
from official.projects.maskconver.modeling import multiscale_maskconver_model
from official.projects.maskconver.modeling.layers import maskconver_head
from official.projects.maskconver.modeling.layers import multiscale_maskconver_head
from official.projects.maskconver.modeling.layers import panoptic_segmentation_generator
from official.vision.modeling import backbones
from official.vision.modeling import decoders


def build_maskconver_model(
    input_specs: tf_keras.layers.InputSpec,
    model_config: maskconver_cfg.MaskConver,
    l2_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
    backbone: Optional[tf_keras.Model] = None,
    decoder: Optional[Union[tf_keras.Model, tf_keras.layers.Layer]] = None,
    segmentation_inference: bool = False,
) -> tf_keras.Model:
  """Builds Segmentation model."""
  norm_activation_config = model_config.norm_activation
  if not backbone:
    backbone = backbones.factory.build_backbone(
        input_specs=input_specs,
        backbone_config=model_config.backbone,
        norm_activation_config=norm_activation_config,
        l2_regularizer=l2_regularizer)

  if not decoder:
    decoder = decoders.factory.build_decoder(
        input_specs=backbone.output_specs,
        model_config=model_config,
        l2_regularizer=l2_regularizer)

  class_head_config = model_config.class_head
  mask_embedding_head_config = model_config.mask_embedding_head
  per_pixel_embedding_head_config = model_config.per_pixel_embedding_head

  # pylint: disable=line-too-long
  class_head = maskconver_head.MaskConverHead(
      num_classes=model_config.num_classes,
      level=class_head_config.level,
      num_convs=class_head_config.num_convs,
      num_filters=class_head_config.num_filters,
      use_layer_norm=class_head_config.use_layer_norm,
      depthwise_kernel_size=class_head_config.depthwise_kernel_size,
      use_depthwise_convolution=class_head_config.use_depthwise_convolution,
      prediction_kernel_size=class_head_config.prediction_kernel_size,
      upsample_factor=class_head_config.upsample_factor,
      feature_fusion=class_head_config.feature_fusion,
      decoder_min_level=class_head_config.decoder_min_level,
      decoder_max_level=class_head_config.decoder_max_level,
      low_level=class_head_config.low_level,
      low_level_num_filters=class_head_config.low_level_num_filters,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer,
      bias_initializer=tf.constant_initializer(-2.19))

  mask_embedding_head = maskconver_head.MaskConverHead(
      num_classes=model_config.embedding_size,
      level=mask_embedding_head_config.level,
      num_convs=mask_embedding_head_config.num_convs,
      num_filters=mask_embedding_head_config.num_filters,
      use_depthwise_convolution=mask_embedding_head_config.use_depthwise_convolution,
      use_layer_norm=mask_embedding_head_config.use_layer_norm,
      depthwise_kernel_size=mask_embedding_head_config.depthwise_kernel_size,
      prediction_kernel_size=mask_embedding_head_config.prediction_kernel_size,
      upsample_factor=mask_embedding_head_config.upsample_factor,
      feature_fusion=mask_embedding_head_config.feature_fusion,
      decoder_min_level=mask_embedding_head_config.decoder_min_level,
      decoder_max_level=mask_embedding_head_config.decoder_max_level,
      low_level=mask_embedding_head_config.low_level,
      low_level_num_filters=mask_embedding_head_config.low_level_num_filters,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer,
      bias_initializer=tf.constant_initializer(0.0))

  per_pixel_embedding_head = maskconver_head.MaskConverHead(
      num_classes=model_config.embedding_size,
      level=per_pixel_embedding_head_config.level,
      num_convs=per_pixel_embedding_head_config.num_convs,
      num_filters=per_pixel_embedding_head_config.num_filters,
      use_depthwise_convolution=per_pixel_embedding_head_config.use_depthwise_convolution,
      depthwise_kernel_size=per_pixel_embedding_head_config.depthwise_kernel_size,
      use_layer_norm=per_pixel_embedding_head_config.use_layer_norm,
      prediction_kernel_size=per_pixel_embedding_head_config.prediction_kernel_size,
      upsample_factor=per_pixel_embedding_head_config.upsample_factor,
      feature_fusion=per_pixel_embedding_head_config.feature_fusion,
      decoder_min_level=per_pixel_embedding_head_config.decoder_min_level,
      decoder_max_level=per_pixel_embedding_head_config.decoder_max_level,
      low_level=per_pixel_embedding_head_config.low_level,
      low_level_num_filters=per_pixel_embedding_head_config.low_level_num_filters,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer,
      bias_initializer=tf.constant_initializer(0.0))

  proposal_generator = panoptic_segmentation_generator.MaskConverProposalGenerator(
      max_proposals=model_config.num_instances,
      peak_error=1e-6,
      peak_extract_kernel_size=3)

  postprocessing_config = model_config.panoptic_generator
  if segmentation_inference:
    is_thing = [False] * model_config.num_classes
  else:
    is_thing = ([False] + [True] * (model_config.num_thing_classes - 1) +
                [False] *
                (model_config.num_classes - model_config.num_thing_classes))
  panoptic_generator = None
  if postprocessing_config:
    panoptic_generator = panoptic_segmentation_generator.MaskConverPanopticGenerator(
        output_size=model_config.padded_output_size,
        num_classes=model_config.num_classes,
        is_thing=is_thing,
        num_instances=model_config.num_instances,
        object_mask_threshold=postprocessing_config.object_mask_threshold,
        small_area_threshold=postprocessing_config.small_area_threshold,
        overlap_threshold=postprocessing_config.overlap_threshold,
        rescale_predictions=postprocessing_config.rescale_predictions,
        use_hardware_optimization=postprocessing_config.use_hardware_optimization,
    )
  # pylint: enable=line-too-long
  mlp_embedding_head = maskconver_head.MLP(
      hidden_dim=model_config.embedding_size,
      output_dim=model_config.embedding_size,
      num_layers=2,
      activation=norm_activation_config.activation,
      l2_regularizer=l2_regularizer)

  model = maskconver_model.MaskConverModel(
      backbone,
      decoder,
      embedding_head=mask_embedding_head,
      class_head=class_head,
      per_pixel_embeddings_head=per_pixel_embedding_head,
      mlp_embedding_head=mlp_embedding_head,
      proposal_generator=proposal_generator,
      panoptic_generator=panoptic_generator,
      level=model_config.level,
      padded_output_size=model_config.padded_output_size,
      l2_regularizer=l2_regularizer,
      embedding_size=model_config.embedding_size,
      num_classes=model_config.num_classes)
  return model


def build_multiscale_maskconver_model(
    input_specs: tf_keras.layers.InputSpec,
    model_config: multiscale_maskconver_cfg.MultiScaleMaskConver,
    l2_regularizer: Optional[tf_keras.regularizers.Regularizer] = None,
    backbone: Optional[tf_keras.regularizers.Regularizer] = None,
    decoder: Optional[tf_keras.regularizers.Regularizer] = None,
    segmentation_inference: bool = False,
) -> tf_keras.Model:
  """Builds multiscale MaskConver model."""
  norm_activation_config = model_config.norm_activation
  if not backbone:
    backbone = backbones.factory.build_backbone(
        input_specs=input_specs,
        backbone_config=model_config.backbone,
        norm_activation_config=norm_activation_config,
        l2_regularizer=l2_regularizer)

  if not decoder:
    decoder = decoders.factory.build_decoder(
        input_specs=backbone.output_specs,
        model_config=model_config,
        l2_regularizer=l2_regularizer)

    mask_decoder = None
    if model_config.mask_decoder:
      temp_model_config = multiscale_maskconver_cfg.MultiScaleMaskConver()
      temp_model_config.override(model_config)
      temp_model_config.decoder.override(model_config.mask_decoder)
      mask_decoder = decoders.factory.build_decoder(
          input_specs=backbone.output_specs,
          model_config=temp_model_config,
          l2_regularizer=l2_regularizer)

  class_head_config = model_config.class_head
  mask_embedding_head_config = model_config.mask_embedding_head
  per_pixel_embedding_head_config = model_config.per_pixel_embedding_head

  # pylint: disable=line-too-long
  class_head = multiscale_maskconver_head.MultiScaleMaskConverHead(
      num_classes=model_config.num_classes,
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      num_convs=class_head_config.num_convs,
      depthwise_kernel_size=class_head_config.depthwise_kernel_size,
      use_layer_norm=class_head_config.use_layer_norm,
      num_filters=class_head_config.num_filters,
      use_depthwise_convolution=class_head_config.use_depthwise_convolution,
      prediction_kernel_size=class_head_config.prediction_kernel_size,
      upsample_factor=class_head_config.upsample_factor,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer,
      bias_initializer=tf.constant_initializer(-2.19),)

  mask_embedding_head = multiscale_maskconver_head.MultiScaleMaskConverHead(
      num_classes=model_config.embedding_size,
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      num_convs=mask_embedding_head_config.num_convs,
      num_filters=mask_embedding_head_config.num_filters,
      use_depthwise_convolution=mask_embedding_head_config
      .use_depthwise_convolution,
      use_layer_norm=mask_embedding_head_config.use_layer_norm,
      depthwise_kernel_size=mask_embedding_head_config.depthwise_kernel_size,
      prediction_kernel_size=mask_embedding_head_config.prediction_kernel_size,
      upsample_factor=mask_embedding_head_config.upsample_factor,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer,
      bias_initializer=tf.constant_initializer(0.0))

  per_pixel_embedding_head = maskconver_head.MaskConverHead(
      num_classes=model_config.embedding_size,
      level=per_pixel_embedding_head_config.level,
      num_convs=per_pixel_embedding_head_config.num_convs,
      num_filters=per_pixel_embedding_head_config.num_filters,
      use_layer_norm=per_pixel_embedding_head_config.use_layer_norm,
      use_depthwise_convolution=per_pixel_embedding_head_config.use_depthwise_convolution,
      depthwise_kernel_size=per_pixel_embedding_head_config.depthwise_kernel_size,
      prediction_kernel_size=per_pixel_embedding_head_config.prediction_kernel_size,
      upsample_factor=per_pixel_embedding_head_config.upsample_factor,
      feature_fusion=per_pixel_embedding_head_config.feature_fusion,
      decoder_min_level=per_pixel_embedding_head_config.decoder_min_level,
      decoder_max_level=per_pixel_embedding_head_config.decoder_max_level,
      low_level=per_pixel_embedding_head_config.low_level,
      low_level_num_filters=per_pixel_embedding_head_config.low_level_num_filters,
      activation=norm_activation_config.activation,
      use_sync_bn=norm_activation_config.use_sync_bn,
      norm_momentum=norm_activation_config.norm_momentum,
      norm_epsilon=norm_activation_config.norm_epsilon,
      kernel_regularizer=l2_regularizer,
      bias_initializer=tf.constant_initializer(0.0))

  postprocessing_config = model_config.panoptic_generator
  if segmentation_inference:
    is_thing = [False] * model_config.num_classes
  else:
    is_thing = ([False] + [True] * (model_config.num_thing_classes - 1) +
                [False] *
                (model_config.num_classes - model_config.num_thing_classes))
  panoptic_generator = None
  if postprocessing_config:
    panoptic_generator = panoptic_segmentation_generator.MaskConverPanopticGenerator(
        output_size=model_config.padded_output_size,
        num_classes=model_config.num_classes,
        is_thing=is_thing,
        num_instances=model_config.num_instances,
        object_mask_threshold=postprocessing_config.object_mask_threshold,
        small_area_threshold=postprocessing_config.small_area_threshold,
        overlap_threshold=postprocessing_config.overlap_threshold,
        rescale_predictions=postprocessing_config.rescale_predictions,
        use_hardware_optimization=postprocessing_config.use_hardware_optimization,
    )
  # pylint: enable=line-too-long
  mlp_embedding_head = maskconver_head.MLP(
      hidden_dim=1024,
      output_dim=model_config.embedding_size,
      num_layers=2,
      activation=norm_activation_config.activation,
      l2_regularizer=l2_regularizer)

  model = multiscale_maskconver_model.MultiScaleMaskConverModel(
      backbone,
      decoder,
      mask_decoder=mask_decoder,
      embedding_head=mask_embedding_head,
      class_head=class_head,
      per_pixel_embeddings_head=per_pixel_embedding_head,
      mlp_embedding_head=mlp_embedding_head,
      panoptic_generator=panoptic_generator,
      min_level=model_config.min_level,
      max_level=model_config.max_level,
      max_proposals=model_config.num_instances,
      padded_output_size=model_config.padded_output_size,
      l2_regularizer=l2_regularizer,
      embedding_size=model_config.embedding_size,
      num_classes=model_config.num_classes)
  return model
