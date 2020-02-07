# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""A function to build a DetectionModel from configuration."""

import functools

from object_detection.builders import anchor_generator_builder
from object_detection.builders import box_coder_builder
from object_detection.builders import box_predictor_builder
from object_detection.builders import hyperparams_builder
from object_detection.builders import image_resizer_builder
from object_detection.builders import losses_builder
from object_detection.builders import matcher_builder
from object_detection.builders import post_processing_builder
from object_detection.builders import region_similarity_calculator_builder as sim_calc
from object_detection.core import balanced_positive_negative_sampler as sampler
from object_detection.core import post_processing
from object_detection.core import target_assigner
from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.meta_architectures import rfcn_meta_arch
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import faster_rcnn_inception_resnet_v2_feature_extractor as frcnn_inc_res
from object_detection.models import faster_rcnn_inception_resnet_v2_keras_feature_extractor as frcnn_inc_res_keras
from object_detection.models import faster_rcnn_inception_v2_feature_extractor as frcnn_inc_v2
from object_detection.models import faster_rcnn_nas_feature_extractor as frcnn_nas
from object_detection.models import faster_rcnn_pnas_feature_extractor as frcnn_pnas
from object_detection.models import faster_rcnn_resnet_v1_feature_extractor as frcnn_resnet_v1
from object_detection.models import ssd_resnet_v1_fpn_feature_extractor as ssd_resnet_v1_fpn
from object_detection.models import ssd_resnet_v1_fpn_keras_feature_extractor as ssd_resnet_v1_fpn_keras
from object_detection.models import ssd_resnet_v1_ppn_feature_extractor as ssd_resnet_v1_ppn
from object_detection.models.embedded_ssd_mobilenet_v1_feature_extractor import EmbeddedSSDMobileNetV1FeatureExtractor
from object_detection.models.ssd_inception_v2_feature_extractor import SSDInceptionV2FeatureExtractor
from object_detection.models.ssd_inception_v3_feature_extractor import SSDInceptionV3FeatureExtractor
from object_detection.models.ssd_mobilenet_edgetpu_feature_extractor import SSDMobileNetEdgeTPUFeatureExtractor
from object_detection.models.ssd_mobilenet_v1_feature_extractor import SSDMobileNetV1FeatureExtractor
from object_detection.models.ssd_mobilenet_v1_fpn_feature_extractor import SSDMobileNetV1FpnFeatureExtractor
from object_detection.models.ssd_mobilenet_v1_fpn_keras_feature_extractor import SSDMobileNetV1FpnKerasFeatureExtractor
from object_detection.models.ssd_mobilenet_v1_keras_feature_extractor import SSDMobileNetV1KerasFeatureExtractor
from object_detection.models.ssd_mobilenet_v1_ppn_feature_extractor import SSDMobileNetV1PpnFeatureExtractor
from object_detection.models.ssd_mobilenet_v2_feature_extractor import SSDMobileNetV2FeatureExtractor
from object_detection.models.ssd_mobilenet_v2_fpn_feature_extractor import SSDMobileNetV2FpnFeatureExtractor
from object_detection.models.ssd_mobilenet_v2_fpn_keras_feature_extractor import SSDMobileNetV2FpnKerasFeatureExtractor
from object_detection.models.ssd_mobilenet_v2_keras_feature_extractor import SSDMobileNetV2KerasFeatureExtractor
from object_detection.models.ssd_mobilenet_v3_feature_extractor import SSDMobileNetV3LargeFeatureExtractor
from object_detection.models.ssd_mobilenet_v3_feature_extractor import SSDMobileNetV3SmallFeatureExtractor
from object_detection.models.ssd_pnasnet_feature_extractor import SSDPNASNetFeatureExtractor
from object_detection.predictors import rfcn_box_predictor
from object_detection.predictors import rfcn_keras_box_predictor
from object_detection.predictors.heads import mask_head
from object_detection.protos import model_pb2
from object_detection.utils import ops

# A map of names to SSD feature extractors.
SSD_FEATURE_EXTRACTOR_CLASS_MAP = {
    'ssd_inception_v2': SSDInceptionV2FeatureExtractor,
    'ssd_inception_v3': SSDInceptionV3FeatureExtractor,
    'ssd_mobilenet_v1': SSDMobileNetV1FeatureExtractor,
    'ssd_mobilenet_v1_fpn': SSDMobileNetV1FpnFeatureExtractor,
    'ssd_mobilenet_v1_ppn': SSDMobileNetV1PpnFeatureExtractor,
    'ssd_mobilenet_v2': SSDMobileNetV2FeatureExtractor,
    'ssd_mobilenet_v2_fpn': SSDMobileNetV2FpnFeatureExtractor,
    'ssd_mobilenet_v3_large': SSDMobileNetV3LargeFeatureExtractor,
    'ssd_mobilenet_v3_small': SSDMobileNetV3SmallFeatureExtractor,
    'ssd_mobilenet_edgetpu': SSDMobileNetEdgeTPUFeatureExtractor,
    'ssd_resnet50_v1_fpn': ssd_resnet_v1_fpn.SSDResnet50V1FpnFeatureExtractor,
    'ssd_resnet101_v1_fpn': ssd_resnet_v1_fpn.SSDResnet101V1FpnFeatureExtractor,
    'ssd_resnet152_v1_fpn': ssd_resnet_v1_fpn.SSDResnet152V1FpnFeatureExtractor,
    'ssd_resnet50_v1_ppn': ssd_resnet_v1_ppn.SSDResnet50V1PpnFeatureExtractor,
    'ssd_resnet101_v1_ppn':
        ssd_resnet_v1_ppn.SSDResnet101V1PpnFeatureExtractor,
    'ssd_resnet152_v1_ppn':
        ssd_resnet_v1_ppn.SSDResnet152V1PpnFeatureExtractor,
    'embedded_ssd_mobilenet_v1': EmbeddedSSDMobileNetV1FeatureExtractor,
    'ssd_pnasnet': SSDPNASNetFeatureExtractor,
}

SSD_KERAS_FEATURE_EXTRACTOR_CLASS_MAP = {
    'ssd_mobilenet_v1_keras': SSDMobileNetV1KerasFeatureExtractor,
    'ssd_mobilenet_v1_fpn_keras': SSDMobileNetV1FpnKerasFeatureExtractor,
    'ssd_mobilenet_v2_keras': SSDMobileNetV2KerasFeatureExtractor,
    'ssd_mobilenet_v2_fpn_keras': SSDMobileNetV2FpnKerasFeatureExtractor,
    'ssd_resnet50_v1_fpn_keras':
        ssd_resnet_v1_fpn_keras.SSDResNet50V1FpnKerasFeatureExtractor,
    'ssd_resnet101_v1_fpn_keras':
        ssd_resnet_v1_fpn_keras.SSDResNet101V1FpnKerasFeatureExtractor,
    'ssd_resnet152_v1_fpn_keras':
        ssd_resnet_v1_fpn_keras.SSDResNet152V1FpnKerasFeatureExtractor,
}

# A map of names to Faster R-CNN feature extractors.
FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP = {
    'faster_rcnn_nas':
    frcnn_nas.FasterRCNNNASFeatureExtractor,
    'faster_rcnn_pnas':
    frcnn_pnas.FasterRCNNPNASFeatureExtractor,
    'faster_rcnn_inception_resnet_v2':
    frcnn_inc_res.FasterRCNNInceptionResnetV2FeatureExtractor,
    'faster_rcnn_inception_v2':
    frcnn_inc_v2.FasterRCNNInceptionV2FeatureExtractor,
    'faster_rcnn_resnet50':
    frcnn_resnet_v1.FasterRCNNResnet50FeatureExtractor,
    'faster_rcnn_resnet101':
    frcnn_resnet_v1.FasterRCNNResnet101FeatureExtractor,
    'faster_rcnn_resnet152':
    frcnn_resnet_v1.FasterRCNNResnet152FeatureExtractor,
}

FASTER_RCNN_KERAS_FEATURE_EXTRACTOR_CLASS_MAP = {
    'faster_rcnn_inception_resnet_v2_keras':
    frcnn_inc_res_keras.FasterRCNNInceptionResnetV2KerasFeatureExtractor,
}


def _build_ssd_feature_extractor(feature_extractor_config,
                                 is_training,
                                 freeze_batchnorm,
                                 reuse_weights=None):
  """Builds a ssd_meta_arch.SSDFeatureExtractor based on config.

  Args:
    feature_extractor_config: A SSDFeatureExtractor proto config from ssd.proto.
    is_training: True if this feature extractor is being built for training.
    freeze_batchnorm: Whether to freeze batch norm parameters during
      training or not. When training with a small batch size (e.g. 1), it is
      desirable to freeze batch norm update and use pretrained batch norm
      params.
    reuse_weights: if the feature extractor should reuse weights.

  Returns:
    ssd_meta_arch.SSDFeatureExtractor based on config.

  Raises:
    ValueError: On invalid feature extractor type.
  """
  feature_type = feature_extractor_config.type
  is_keras_extractor = feature_type in SSD_KERAS_FEATURE_EXTRACTOR_CLASS_MAP
  depth_multiplier = feature_extractor_config.depth_multiplier
  min_depth = feature_extractor_config.min_depth
  pad_to_multiple = feature_extractor_config.pad_to_multiple
  use_explicit_padding = feature_extractor_config.use_explicit_padding
  use_depthwise = feature_extractor_config.use_depthwise

  if is_keras_extractor:
    conv_hyperparams = hyperparams_builder.KerasLayerHyperparams(
        feature_extractor_config.conv_hyperparams)
  else:
    conv_hyperparams = hyperparams_builder.build(
        feature_extractor_config.conv_hyperparams, is_training)
  override_base_feature_extractor_hyperparams = (
      feature_extractor_config.override_base_feature_extractor_hyperparams)

  if (feature_type not in SSD_FEATURE_EXTRACTOR_CLASS_MAP) and (
      not is_keras_extractor):
    raise ValueError('Unknown ssd feature_extractor: {}'.format(feature_type))

  if is_keras_extractor:
    feature_extractor_class = SSD_KERAS_FEATURE_EXTRACTOR_CLASS_MAP[
        feature_type]
  else:
    feature_extractor_class = SSD_FEATURE_EXTRACTOR_CLASS_MAP[feature_type]
  kwargs = {
      'is_training':
          is_training,
      'depth_multiplier':
          depth_multiplier,
      'min_depth':
          min_depth,
      'pad_to_multiple':
          pad_to_multiple,
      'use_explicit_padding':
          use_explicit_padding,
      'use_depthwise':
          use_depthwise,
      'override_base_feature_extractor_hyperparams':
          override_base_feature_extractor_hyperparams
  }

  if feature_extractor_config.HasField('replace_preprocessor_with_placeholder'):
    kwargs.update({
        'replace_preprocessor_with_placeholder':
            feature_extractor_config.replace_preprocessor_with_placeholder
    })

  if feature_extractor_config.HasField('num_layers'):
    kwargs.update({'num_layers': feature_extractor_config.num_layers})

  if is_keras_extractor:
    kwargs.update({
        'conv_hyperparams': conv_hyperparams,
        'inplace_batchnorm_update': False,
        'freeze_batchnorm': freeze_batchnorm
    })
  else:
    kwargs.update({
        'conv_hyperparams_fn': conv_hyperparams,
        'reuse_weights': reuse_weights,
    })

  if feature_extractor_config.HasField('fpn'):
    kwargs.update({
        'fpn_min_level':
            feature_extractor_config.fpn.min_level,
        'fpn_max_level':
            feature_extractor_config.fpn.max_level,
        'additional_layer_depth':
            feature_extractor_config.fpn.additional_layer_depth,
    })


  return feature_extractor_class(**kwargs)


def _build_ssd_model(ssd_config, is_training, add_summaries):
  """Builds an SSD detection model based on the model config.

  Args:
    ssd_config: A ssd.proto object containing the config for the desired
      SSDMetaArch.
    is_training: True if this model is being built for training purposes.
    add_summaries: Whether to add tf summaries in the model.
  Returns:
    SSDMetaArch based on the config.

  Raises:
    ValueError: If ssd_config.type is not recognized (i.e. not registered in
      model_class_map).
  """
  num_classes = ssd_config.num_classes

  # Feature extractor
  feature_extractor = _build_ssd_feature_extractor(
      feature_extractor_config=ssd_config.feature_extractor,
      freeze_batchnorm=ssd_config.freeze_batchnorm,
      is_training=is_training)

  box_coder = box_coder_builder.build(ssd_config.box_coder)
  matcher = matcher_builder.build(ssd_config.matcher)
  region_similarity_calculator = sim_calc.build(
      ssd_config.similarity_calculator)
  encode_background_as_zeros = ssd_config.encode_background_as_zeros
  negative_class_weight = ssd_config.negative_class_weight
  anchor_generator = anchor_generator_builder.build(
      ssd_config.anchor_generator)
  if feature_extractor.is_keras_model:
    ssd_box_predictor = box_predictor_builder.build_keras(
        hyperparams_fn=hyperparams_builder.KerasLayerHyperparams,
        freeze_batchnorm=ssd_config.freeze_batchnorm,
        inplace_batchnorm_update=False,
        num_predictions_per_location_list=anchor_generator
        .num_anchors_per_location(),
        box_predictor_config=ssd_config.box_predictor,
        is_training=is_training,
        num_classes=num_classes,
        add_background_class=ssd_config.add_background_class)
  else:
    ssd_box_predictor = box_predictor_builder.build(
        hyperparams_builder.build, ssd_config.box_predictor, is_training,
        num_classes, ssd_config.add_background_class)
  image_resizer_fn = image_resizer_builder.build(ssd_config.image_resizer)
  non_max_suppression_fn, score_conversion_fn = post_processing_builder.build(
      ssd_config.post_processing)
  (classification_loss, localization_loss, classification_weight,
   localization_weight, hard_example_miner, random_example_sampler,
   expected_loss_weights_fn) = losses_builder.build(ssd_config.loss)
  normalize_loss_by_num_matches = ssd_config.normalize_loss_by_num_matches
  normalize_loc_loss_by_codesize = ssd_config.normalize_loc_loss_by_codesize

  equalization_loss_config = ops.EqualizationLossConfig(
      weight=ssd_config.loss.equalization_loss.weight,
      exclude_prefixes=ssd_config.loss.equalization_loss.exclude_prefixes)

  target_assigner_instance = target_assigner.TargetAssigner(
      region_similarity_calculator,
      matcher,
      box_coder,
      negative_class_weight=negative_class_weight)

  ssd_meta_arch_fn = ssd_meta_arch.SSDMetaArch
  kwargs = {}

  return ssd_meta_arch_fn(
      is_training=is_training,
      anchor_generator=anchor_generator,
      box_predictor=ssd_box_predictor,
      box_coder=box_coder,
      feature_extractor=feature_extractor,
      encode_background_as_zeros=encode_background_as_zeros,
      image_resizer_fn=image_resizer_fn,
      non_max_suppression_fn=non_max_suppression_fn,
      score_conversion_fn=score_conversion_fn,
      classification_loss=classification_loss,
      localization_loss=localization_loss,
      classification_loss_weight=classification_weight,
      localization_loss_weight=localization_weight,
      normalize_loss_by_num_matches=normalize_loss_by_num_matches,
      hard_example_miner=hard_example_miner,
      target_assigner_instance=target_assigner_instance,
      add_summaries=add_summaries,
      normalize_loc_loss_by_codesize=normalize_loc_loss_by_codesize,
      freeze_batchnorm=ssd_config.freeze_batchnorm,
      inplace_batchnorm_update=ssd_config.inplace_batchnorm_update,
      add_background_class=ssd_config.add_background_class,
      explicit_background_class=ssd_config.explicit_background_class,
      random_example_sampler=random_example_sampler,
      expected_loss_weights_fn=expected_loss_weights_fn,
      use_confidences_as_targets=ssd_config.use_confidences_as_targets,
      implicit_example_weight=ssd_config.implicit_example_weight,
      equalization_loss_config=equalization_loss_config,
      return_raw_detections_during_predict=(
          ssd_config.return_raw_detections_during_predict),
      **kwargs)


def _build_faster_rcnn_feature_extractor(
    feature_extractor_config, is_training, reuse_weights=None,
    inplace_batchnorm_update=False):
  """Builds a faster_rcnn_meta_arch.FasterRCNNFeatureExtractor based on config.

  Args:
    feature_extractor_config: A FasterRcnnFeatureExtractor proto config from
      faster_rcnn.proto.
    is_training: True if this feature extractor is being built for training.
    reuse_weights: if the feature extractor should reuse weights.
    inplace_batchnorm_update: Whether to update batch_norm inplace during
      training. This is required for batch norm to work correctly on TPUs. When
      this is false, user must add a control dependency on
      tf.GraphKeys.UPDATE_OPS for train/loss op in order to update the batch
      norm moving average parameters.

  Returns:
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor based on config.

  Raises:
    ValueError: On invalid feature extractor type.
  """
  if inplace_batchnorm_update:
    raise ValueError('inplace batchnorm updates not supported.')
  feature_type = feature_extractor_config.type
  first_stage_features_stride = (
      feature_extractor_config.first_stage_features_stride)
  batch_norm_trainable = feature_extractor_config.batch_norm_trainable

  if feature_type not in FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP:
    raise ValueError('Unknown Faster R-CNN feature_extractor: {}'.format(
        feature_type))
  feature_extractor_class = FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP[
      feature_type]
  return feature_extractor_class(
      is_training, first_stage_features_stride,
      batch_norm_trainable, reuse_weights=reuse_weights)


def _build_faster_rcnn_keras_feature_extractor(
    feature_extractor_config, is_training,
    inplace_batchnorm_update=False):
  """Builds a faster_rcnn_meta_arch.FasterRCNNKerasFeatureExtractor from config.

  Args:
    feature_extractor_config: A FasterRcnnFeatureExtractor proto config from
      faster_rcnn.proto.
    is_training: True if this feature extractor is being built for training.
    inplace_batchnorm_update: Whether to update batch_norm inplace during
      training. This is required for batch norm to work correctly on TPUs. When
      this is false, user must add a control dependency on
      tf.GraphKeys.UPDATE_OPS for train/loss op in order to update the batch
      norm moving average parameters.

  Returns:
    faster_rcnn_meta_arch.FasterRCNNKerasFeatureExtractor based on config.

  Raises:
    ValueError: On invalid feature extractor type.
  """
  if inplace_batchnorm_update:
    raise ValueError('inplace batchnorm updates not supported.')
  feature_type = feature_extractor_config.type
  first_stage_features_stride = (
      feature_extractor_config.first_stage_features_stride)
  batch_norm_trainable = feature_extractor_config.batch_norm_trainable

  if feature_type not in FASTER_RCNN_KERAS_FEATURE_EXTRACTOR_CLASS_MAP:
    raise ValueError('Unknown Faster R-CNN feature_extractor: {}'.format(
        feature_type))
  feature_extractor_class = FASTER_RCNN_KERAS_FEATURE_EXTRACTOR_CLASS_MAP[
      feature_type]
  return feature_extractor_class(
      is_training, first_stage_features_stride,
      batch_norm_trainable)


def _build_faster_rcnn_model(frcnn_config, is_training, add_summaries):
  """Builds a Faster R-CNN or R-FCN detection model based on the model config.

  Builds R-FCN model if the second_stage_box_predictor in the config is of type
  `rfcn_box_predictor` else builds a Faster R-CNN model.

  Args:
    frcnn_config: A faster_rcnn.proto object containing the config for the
      desired FasterRCNNMetaArch or RFCNMetaArch.
    is_training: True if this model is being built for training purposes.
    add_summaries: Whether to add tf summaries in the model.

  Returns:
    FasterRCNNMetaArch based on the config.

  Raises:
    ValueError: If frcnn_config.type is not recognized (i.e. not registered in
      model_class_map).
  """
  num_classes = frcnn_config.num_classes
  image_resizer_fn = image_resizer_builder.build(frcnn_config.image_resizer)

  is_keras = (frcnn_config.feature_extractor.type in
              FASTER_RCNN_KERAS_FEATURE_EXTRACTOR_CLASS_MAP)

  if is_keras:
    feature_extractor = _build_faster_rcnn_keras_feature_extractor(
        frcnn_config.feature_extractor, is_training,
        inplace_batchnorm_update=frcnn_config.inplace_batchnorm_update)
  else:
    feature_extractor = _build_faster_rcnn_feature_extractor(
        frcnn_config.feature_extractor, is_training,
        inplace_batchnorm_update=frcnn_config.inplace_batchnorm_update)

  number_of_stages = frcnn_config.number_of_stages
  first_stage_anchor_generator = anchor_generator_builder.build(
      frcnn_config.first_stage_anchor_generator)

  first_stage_target_assigner = target_assigner.create_target_assigner(
      'FasterRCNN',
      'proposal',
      use_matmul_gather=frcnn_config.use_matmul_gather_in_matcher)
  first_stage_atrous_rate = frcnn_config.first_stage_atrous_rate
  if is_keras:
    first_stage_box_predictor_arg_scope_fn = (
        hyperparams_builder.KerasLayerHyperparams(
            frcnn_config.first_stage_box_predictor_conv_hyperparams))
  else:
    first_stage_box_predictor_arg_scope_fn = hyperparams_builder.build(
        frcnn_config.first_stage_box_predictor_conv_hyperparams, is_training)
  first_stage_box_predictor_kernel_size = (
      frcnn_config.first_stage_box_predictor_kernel_size)
  first_stage_box_predictor_depth = frcnn_config.first_stage_box_predictor_depth
  first_stage_minibatch_size = frcnn_config.first_stage_minibatch_size
  use_static_shapes = frcnn_config.use_static_shapes and (
      frcnn_config.use_static_shapes_for_eval or is_training)
  first_stage_sampler = sampler.BalancedPositiveNegativeSampler(
      positive_fraction=frcnn_config.first_stage_positive_balance_fraction,
      is_static=(frcnn_config.use_static_balanced_label_sampler and
                 use_static_shapes))
  first_stage_max_proposals = frcnn_config.first_stage_max_proposals
  if (frcnn_config.first_stage_nms_iou_threshold < 0 or
      frcnn_config.first_stage_nms_iou_threshold > 1.0):
    raise ValueError('iou_threshold not in [0, 1.0].')
  if (is_training and frcnn_config.second_stage_batch_size >
      first_stage_max_proposals):
    raise ValueError('second_stage_batch_size should be no greater than '
                     'first_stage_max_proposals.')
  first_stage_non_max_suppression_fn = functools.partial(
      post_processing.batch_multiclass_non_max_suppression,
      score_thresh=frcnn_config.first_stage_nms_score_threshold,
      iou_thresh=frcnn_config.first_stage_nms_iou_threshold,
      max_size_per_class=frcnn_config.first_stage_max_proposals,
      max_total_size=frcnn_config.first_stage_max_proposals,
      use_static_shapes=use_static_shapes,
      use_partitioned_nms=frcnn_config.use_partitioned_nms_in_first_stage,
      use_combined_nms=frcnn_config.use_combined_nms_in_first_stage)
  first_stage_loc_loss_weight = (
      frcnn_config.first_stage_localization_loss_weight)
  first_stage_obj_loss_weight = frcnn_config.first_stage_objectness_loss_weight

  initial_crop_size = frcnn_config.initial_crop_size
  maxpool_kernel_size = frcnn_config.maxpool_kernel_size
  maxpool_stride = frcnn_config.maxpool_stride

  second_stage_target_assigner = target_assigner.create_target_assigner(
      'FasterRCNN',
      'detection',
      use_matmul_gather=frcnn_config.use_matmul_gather_in_matcher)
  if is_keras:
    second_stage_box_predictor = box_predictor_builder.build_keras(
        hyperparams_builder.KerasLayerHyperparams,
        freeze_batchnorm=False,
        inplace_batchnorm_update=False,
        num_predictions_per_location_list=[1],
        box_predictor_config=frcnn_config.second_stage_box_predictor,
        is_training=is_training,
        num_classes=num_classes)
  else:
    second_stage_box_predictor = box_predictor_builder.build(
        hyperparams_builder.build,
        frcnn_config.second_stage_box_predictor,
        is_training=is_training,
        num_classes=num_classes)
  second_stage_batch_size = frcnn_config.second_stage_batch_size
  second_stage_sampler = sampler.BalancedPositiveNegativeSampler(
      positive_fraction=frcnn_config.second_stage_balance_fraction,
      is_static=(frcnn_config.use_static_balanced_label_sampler and
                 use_static_shapes))
  (second_stage_non_max_suppression_fn, second_stage_score_conversion_fn
  ) = post_processing_builder.build(frcnn_config.second_stage_post_processing)
  second_stage_localization_loss_weight = (
      frcnn_config.second_stage_localization_loss_weight)
  second_stage_classification_loss = (
      losses_builder.build_faster_rcnn_classification_loss(
          frcnn_config.second_stage_classification_loss))
  second_stage_classification_loss_weight = (
      frcnn_config.second_stage_classification_loss_weight)
  second_stage_mask_prediction_loss_weight = (
      frcnn_config.second_stage_mask_prediction_loss_weight)

  hard_example_miner = None
  if frcnn_config.HasField('hard_example_miner'):
    hard_example_miner = losses_builder.build_hard_example_miner(
        frcnn_config.hard_example_miner,
        second_stage_classification_loss_weight,
        second_stage_localization_loss_weight)

  crop_and_resize_fn = (
      ops.matmul_crop_and_resize if frcnn_config.use_matmul_crop_and_resize
      else ops.native_crop_and_resize)
  clip_anchors_to_image = (
      frcnn_config.clip_anchors_to_image)

  common_kwargs = {
      'is_training': is_training,
      'num_classes': num_classes,
      'image_resizer_fn': image_resizer_fn,
      'feature_extractor': feature_extractor,
      'number_of_stages': number_of_stages,
      'first_stage_anchor_generator': first_stage_anchor_generator,
      'first_stage_target_assigner': first_stage_target_assigner,
      'first_stage_atrous_rate': first_stage_atrous_rate,
      'first_stage_box_predictor_arg_scope_fn':
      first_stage_box_predictor_arg_scope_fn,
      'first_stage_box_predictor_kernel_size':
      first_stage_box_predictor_kernel_size,
      'first_stage_box_predictor_depth': first_stage_box_predictor_depth,
      'first_stage_minibatch_size': first_stage_minibatch_size,
      'first_stage_sampler': first_stage_sampler,
      'first_stage_non_max_suppression_fn': first_stage_non_max_suppression_fn,
      'first_stage_max_proposals': first_stage_max_proposals,
      'first_stage_localization_loss_weight': first_stage_loc_loss_weight,
      'first_stage_objectness_loss_weight': first_stage_obj_loss_weight,
      'second_stage_target_assigner': second_stage_target_assigner,
      'second_stage_batch_size': second_stage_batch_size,
      'second_stage_sampler': second_stage_sampler,
      'second_stage_non_max_suppression_fn':
      second_stage_non_max_suppression_fn,
      'second_stage_score_conversion_fn': second_stage_score_conversion_fn,
      'second_stage_localization_loss_weight':
      second_stage_localization_loss_weight,
      'second_stage_classification_loss':
      second_stage_classification_loss,
      'second_stage_classification_loss_weight':
      second_stage_classification_loss_weight,
      'hard_example_miner': hard_example_miner,
      'add_summaries': add_summaries,
      'crop_and_resize_fn': crop_and_resize_fn,
      'clip_anchors_to_image': clip_anchors_to_image,
      'use_static_shapes': use_static_shapes,
      'resize_masks': frcnn_config.resize_masks,
      'return_raw_detections_during_predict': (
          frcnn_config.return_raw_detections_during_predict)
  }

  if (isinstance(second_stage_box_predictor,
                 rfcn_box_predictor.RfcnBoxPredictor) or
      isinstance(second_stage_box_predictor,
                 rfcn_keras_box_predictor.RfcnKerasBoxPredictor)):
    return rfcn_meta_arch.RFCNMetaArch(
        second_stage_rfcn_box_predictor=second_stage_box_predictor,
        **common_kwargs)
  else:
    return faster_rcnn_meta_arch.FasterRCNNMetaArch(
        initial_crop_size=initial_crop_size,
        maxpool_kernel_size=maxpool_kernel_size,
        maxpool_stride=maxpool_stride,
        second_stage_mask_rcnn_box_predictor=second_stage_box_predictor,
        second_stage_mask_prediction_loss_weight=(
            second_stage_mask_prediction_loss_weight),
        **common_kwargs)

EXPERIMENTAL_META_ARCH_BUILDER_MAP = {
}


def _build_experimental_model(config, is_training, add_summaries=True):
  return EXPERIMENTAL_META_ARCH_BUILDER_MAP[config.name](
      is_training, add_summaries)

META_ARCHITECURE_BUILDER_MAP = {
    'ssd': _build_ssd_model,
    'faster_rcnn': _build_faster_rcnn_model,
    'experimental_model': _build_experimental_model
}


def build(model_config, is_training, add_summaries=True):
  """Builds a DetectionModel based on the model config.

  Args:
    model_config: A model.proto object containing the config for the desired
      DetectionModel.
    is_training: True if this model is being built for training purposes.
    add_summaries: Whether to add tensorflow summaries in the model graph.
  Returns:
    DetectionModel based on the config.

  Raises:
    ValueError: On invalid meta architecture or model.
  """
  if not isinstance(model_config, model_pb2.DetectionModel):
    raise ValueError('model_config not of type model_pb2.DetectionModel.')

  meta_architecture = model_config.WhichOneof('model')

  if meta_architecture not in META_ARCHITECURE_BUILDER_MAP:
    raise ValueError('Unknown meta architecture: {}'.format(meta_architecture))
  else:
    build_func = META_ARCHITECURE_BUILDER_MAP[meta_architecture]
    return build_func(getattr(model_config, meta_architecture), is_training,
                      add_summaries)
