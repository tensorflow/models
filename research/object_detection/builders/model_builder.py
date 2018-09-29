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
from object_detection.models import faster_rcnn_inception_v2_feature_extractor as frcnn_inc_v2
from object_detection.models import faster_rcnn_nas_feature_extractor as frcnn_nas
from object_detection.models import faster_rcnn_pnas_feature_extractor as frcnn_pnas
from object_detection.models import faster_rcnn_resnet_v1_feature_extractor as frcnn_resnet_v1
from object_detection.models import ssd_resnet_v1_fpn_feature_extractor as ssd_resnet_v1_fpn
from object_detection.models import ssd_resnet_v1_ppn_feature_extractor as ssd_resnet_v1_ppn
from object_detection.models.embedded_ssd_mobilenet_v1_feature_extractor import EmbeddedSSDMobileNetV1FeatureExtractor
from object_detection.models.ssd_inception_v2_feature_extractor import SSDInceptionV2FeatureExtractor
from object_detection.models.ssd_inception_v3_feature_extractor import SSDInceptionV3FeatureExtractor
from object_detection.models.ssd_mobilenet_v1_feature_extractor import SSDMobileNetV1FeatureExtractor
from object_detection.models.ssd_mobilenet_v1_fpn_feature_extractor import SSDMobileNetV1FpnFeatureExtractor
from object_detection.models.ssd_mobilenet_v1_ppn_feature_extractor import SSDMobileNetV1PpnFeatureExtractor
from object_detection.models.ssd_mobilenet_v2_feature_extractor import SSDMobileNetV2FeatureExtractor
from object_detection.models.ssd_mobilenet_v2_fpn_feature_extractor import SSDMobileNetV2FpnFeatureExtractor
from object_detection.predictors import rfcn_box_predictor
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
    'ssd_resnet50_v1_fpn': ssd_resnet_v1_fpn.SSDResnet50V1FpnFeatureExtractor,
    'ssd_resnet101_v1_fpn': ssd_resnet_v1_fpn.SSDResnet101V1FpnFeatureExtractor,
    'ssd_resnet152_v1_fpn': ssd_resnet_v1_fpn.SSDResnet152V1FpnFeatureExtractor,
    'ssd_resnet50_v1_ppn': ssd_resnet_v1_ppn.SSDResnet50V1PpnFeatureExtractor,
    'ssd_resnet101_v1_ppn':
        ssd_resnet_v1_ppn.SSDResnet101V1PpnFeatureExtractor,
    'ssd_resnet152_v1_ppn':
        ssd_resnet_v1_ppn.SSDResnet152V1PpnFeatureExtractor,
    'embedded_ssd_mobilenet_v1': EmbeddedSSDMobileNetV1FeatureExtractor,
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


def build(model_config, is_training, add_summaries=True,
          add_background_class=True):
  """Builds a DetectionModel based on the model config.

  Args:
    model_config: A model.proto object containing the config for the desired
      DetectionModel.
    is_training: True if this model is being built for training purposes.
    add_summaries: Whether to add tensorflow summaries in the model graph.
    add_background_class: Whether to add an implicit background class to one-hot
      encodings of groundtruth labels. Set to false if using groundtruth labels
      with an explicit background class or using multiclass scores instead of
      truth in the case of distillation. Ignored in the case of faster_rcnn.
  Returns:
    DetectionModel based on the config.

  Raises:
    ValueError: On invalid meta architecture or model.
  """
  if not isinstance(model_config, model_pb2.DetectionModel):
    raise ValueError('model_config not of type model_pb2.DetectionModel.')
  meta_architecture = model_config.WhichOneof('model')
  if meta_architecture == 'ssd':
    return _build_ssd_model(model_config.ssd, is_training, add_summaries,
                            add_background_class)
  if meta_architecture == 'faster_rcnn':
    return _build_faster_rcnn_model(model_config.faster_rcnn, is_training,
                                    add_summaries)
  raise ValueError('Unknown meta architecture: {}'.format(meta_architecture))


def _build_ssd_feature_extractor(feature_extractor_config, is_training,
                                 reuse_weights=None):
  """Builds a ssd_meta_arch.SSDFeatureExtractor based on config.

  Args:
    feature_extractor_config: A SSDFeatureExtractor proto config from ssd.proto.
    is_training: True if this feature extractor is being built for training.
    reuse_weights: if the feature extractor should reuse weights.

  Returns:
    ssd_meta_arch.SSDFeatureExtractor based on config.

  Raises:
    ValueError: On invalid feature extractor type.
  """
  feature_type = feature_extractor_config.type
  depth_multiplier = feature_extractor_config.depth_multiplier
  min_depth = feature_extractor_config.min_depth
  pad_to_multiple = feature_extractor_config.pad_to_multiple
  use_explicit_padding = feature_extractor_config.use_explicit_padding
  use_depthwise = feature_extractor_config.use_depthwise
  conv_hyperparams = hyperparams_builder.build(
      feature_extractor_config.conv_hyperparams, is_training)
  override_base_feature_extractor_hyperparams = (
      feature_extractor_config.override_base_feature_extractor_hyperparams)

  if feature_type not in SSD_FEATURE_EXTRACTOR_CLASS_MAP:
    raise ValueError('Unknown ssd feature_extractor: {}'.format(feature_type))

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
      'conv_hyperparams_fn':
          conv_hyperparams,
      'reuse_weights':
          reuse_weights,
      'use_explicit_padding':
          use_explicit_padding,
      'use_depthwise':
          use_depthwise,
      'override_base_feature_extractor_hyperparams':
          override_base_feature_extractor_hyperparams
  }

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


def _build_ssd_model(ssd_config, is_training, add_summaries,
                     add_background_class=True):
  """Builds an SSD detection model based on the model config.

  Args:
    ssd_config: A ssd.proto object containing the config for the desired
      SSDMetaArch.
    is_training: True if this model is being built for training purposes.
    add_summaries: Whether to add tf summaries in the model.
    add_background_class: Whether to add an implicit background class to one-hot
      encodings of groundtruth labels. Set to false if using groundtruth labels
      with an explicit background class or using multiclass scores instead of
      truth in the case of distillation.
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
      is_training=is_training)

  box_coder = box_coder_builder.build(ssd_config.box_coder)
  matcher = matcher_builder.build(ssd_config.matcher)
  region_similarity_calculator = sim_calc.build(
      ssd_config.similarity_calculator)
  encode_background_as_zeros = ssd_config.encode_background_as_zeros
  negative_class_weight = ssd_config.negative_class_weight
  ssd_box_predictor = box_predictor_builder.build(hyperparams_builder.build,
                                                  ssd_config.box_predictor,
                                                  is_training, num_classes)
  anchor_generator = anchor_generator_builder.build(
      ssd_config.anchor_generator)
  image_resizer_fn = image_resizer_builder.build(ssd_config.image_resizer)
  non_max_suppression_fn, score_conversion_fn = post_processing_builder.build(
      ssd_config.post_processing)
  (classification_loss, localization_loss, classification_weight,
   localization_weight, hard_example_miner,
   random_example_sampler) = losses_builder.build(ssd_config.loss)
  normalize_loss_by_num_matches = ssd_config.normalize_loss_by_num_matches
  normalize_loc_loss_by_codesize = ssd_config.normalize_loc_loss_by_codesize
  weight_regression_loss_by_score = (ssd_config.weight_regression_loss_by_score)

  target_assigner_instance = target_assigner.TargetAssigner(
      region_similarity_calculator,
      matcher,
      box_coder,
      negative_class_weight=negative_class_weight,
      weight_regression_loss_by_score=weight_regression_loss_by_score)

  expected_classification_loss_under_sampling = None
  if ssd_config.use_expected_classification_loss_under_sampling:
    expected_classification_loss_under_sampling = functools.partial(
        ops.expected_classification_loss_under_sampling,
        minimum_negative_sampling=ssd_config.minimum_negative_sampling,
        desired_negative_sampling_ratio=ssd_config.
        desired_negative_sampling_ratio)

  ssd_meta_arch_fn = ssd_meta_arch.SSDMetaArch

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
      add_background_class=add_background_class,
      random_example_sampler=random_example_sampler,
      expected_classification_loss_under_sampling=
      expected_classification_loss_under_sampling)


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
      batch_norm_trainable, reuse_weights)


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

  feature_extractor = _build_faster_rcnn_feature_extractor(
      frcnn_config.feature_extractor, is_training,
      frcnn_config.inplace_batchnorm_update)

  number_of_stages = frcnn_config.number_of_stages
  first_stage_anchor_generator = anchor_generator_builder.build(
      frcnn_config.first_stage_anchor_generator)

  first_stage_target_assigner = target_assigner.create_target_assigner(
      'FasterRCNN',
      'proposal',
      use_matmul_gather=frcnn_config.use_matmul_gather_in_matcher)
  first_stage_atrous_rate = frcnn_config.first_stage_atrous_rate
  first_stage_box_predictor_arg_scope_fn = hyperparams_builder.build(
      frcnn_config.first_stage_box_predictor_conv_hyperparams, is_training)
  first_stage_box_predictor_kernel_size = (
      frcnn_config.first_stage_box_predictor_kernel_size)
  first_stage_box_predictor_depth = frcnn_config.first_stage_box_predictor_depth
  first_stage_minibatch_size = frcnn_config.first_stage_minibatch_size
  # TODO(bhattad): When eval is supported using static shapes, add separate
  # use_static_shapes_for_trainig and use_static_shapes_for_evaluation.
  use_static_shapes = frcnn_config.use_static_shapes and is_training
  first_stage_sampler = sampler.BalancedPositiveNegativeSampler(
      positive_fraction=frcnn_config.first_stage_positive_balance_fraction,
      is_static=frcnn_config.use_static_balanced_label_sampler and is_training)
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
      use_static_shapes=use_static_shapes and is_training)
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
  second_stage_box_predictor = box_predictor_builder.build(
      hyperparams_builder.build,
      frcnn_config.second_stage_box_predictor,
      is_training=is_training,
      num_classes=num_classes)
  second_stage_batch_size = frcnn_config.second_stage_batch_size
  second_stage_sampler = sampler.BalancedPositiveNegativeSampler(
      positive_fraction=frcnn_config.second_stage_balance_fraction,
      is_static=frcnn_config.use_static_balanced_label_sampler and is_training)
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
      'resize_masks': frcnn_config.resize_masks
  }

  if isinstance(second_stage_box_predictor,
                rfcn_box_predictor.RfcnBoxPredictor):
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
