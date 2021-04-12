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
import sys

from absl import logging

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
from object_detection.meta_architectures import center_net_meta_arch
from object_detection.meta_architectures import context_rcnn_meta_arch
from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.meta_architectures import rfcn_meta_arch
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.predictors.heads import mask_head
from object_detection.protos import losses_pb2
from object_detection.protos import model_pb2
from object_detection.utils import label_map_util
from object_detection.utils import ops
from object_detection.utils import spatial_transform_ops as spatial_ops
from object_detection.utils import tf_version

## Feature Extractors for TF
## This section conditionally imports different feature extractors based on the
## Tensorflow version.
##
# pylint: disable=g-import-not-at-top
if tf_version.is_tf2():
  from object_detection.models import center_net_hourglass_feature_extractor
  from object_detection.models import center_net_mobilenet_v2_feature_extractor
  from object_detection.models import center_net_mobilenet_v2_fpn_feature_extractor
  from object_detection.models import center_net_resnet_feature_extractor
  from object_detection.models import center_net_resnet_v1_fpn_feature_extractor
  from object_detection.models import faster_rcnn_inception_resnet_v2_keras_feature_extractor as frcnn_inc_res_keras
  from object_detection.models import faster_rcnn_resnet_keras_feature_extractor as frcnn_resnet_keras
  from object_detection.models import ssd_resnet_v1_fpn_keras_feature_extractor as ssd_resnet_v1_fpn_keras
  from object_detection.models import faster_rcnn_resnet_v1_fpn_keras_feature_extractor as frcnn_resnet_fpn_keras
  from object_detection.models.ssd_mobilenet_v1_fpn_keras_feature_extractor import SSDMobileNetV1FpnKerasFeatureExtractor
  from object_detection.models.ssd_mobilenet_v1_keras_feature_extractor import SSDMobileNetV1KerasFeatureExtractor
  from object_detection.models.ssd_mobilenet_v2_fpn_keras_feature_extractor import SSDMobileNetV2FpnKerasFeatureExtractor
  from object_detection.models.ssd_mobilenet_v2_keras_feature_extractor import SSDMobileNetV2KerasFeatureExtractor
  from object_detection.predictors import rfcn_keras_box_predictor
  if sys.version_info[0] >= 3:
    from object_detection.models import ssd_efficientnet_bifpn_feature_extractor as ssd_efficientnet_bifpn

if tf_version.is_tf1():
  from object_detection.models import faster_rcnn_inception_resnet_v2_feature_extractor as frcnn_inc_res
  from object_detection.models import faster_rcnn_inception_v2_feature_extractor as frcnn_inc_v2
  from object_detection.models import faster_rcnn_nas_feature_extractor as frcnn_nas
  from object_detection.models import faster_rcnn_pnas_feature_extractor as frcnn_pnas
  from object_detection.models import faster_rcnn_resnet_v1_feature_extractor as frcnn_resnet_v1
  from object_detection.models import ssd_resnet_v1_fpn_feature_extractor as ssd_resnet_v1_fpn
  from object_detection.models import ssd_resnet_v1_ppn_feature_extractor as ssd_resnet_v1_ppn
  from object_detection.models.embedded_ssd_mobilenet_v1_feature_extractor import EmbeddedSSDMobileNetV1FeatureExtractor
  from object_detection.models.ssd_inception_v2_feature_extractor import SSDInceptionV2FeatureExtractor
  from object_detection.models.ssd_mobilenet_v2_fpn_feature_extractor import SSDMobileNetV2FpnFeatureExtractor
  from object_detection.models.ssd_mobilenet_v2_mnasfpn_feature_extractor import SSDMobileNetV2MnasFPNFeatureExtractor
  from object_detection.models.ssd_inception_v3_feature_extractor import SSDInceptionV3FeatureExtractor
  from object_detection.models.ssd_mobilenet_edgetpu_feature_extractor import SSDMobileNetEdgeTPUFeatureExtractor
  from object_detection.models.ssd_mobilenet_v1_feature_extractor import SSDMobileNetV1FeatureExtractor
  from object_detection.models.ssd_mobilenet_v1_fpn_feature_extractor import SSDMobileNetV1FpnFeatureExtractor
  from object_detection.models.ssd_mobilenet_v1_ppn_feature_extractor import SSDMobileNetV1PpnFeatureExtractor
  from object_detection.models.ssd_mobilenet_v2_feature_extractor import SSDMobileNetV2FeatureExtractor
  from object_detection.models.ssd_mobilenet_v3_feature_extractor import SSDMobileNetV3LargeFeatureExtractor
  from object_detection.models.ssd_mobilenet_v3_feature_extractor import SSDMobileNetV3SmallFeatureExtractor
  from object_detection.models.ssd_mobiledet_feature_extractor import SSDMobileDetCPUFeatureExtractor
  from object_detection.models.ssd_mobiledet_feature_extractor import SSDMobileDetDSPFeatureExtractor
  from object_detection.models.ssd_mobiledet_feature_extractor import SSDMobileDetEdgeTPUFeatureExtractor
  from object_detection.models.ssd_mobiledet_feature_extractor import SSDMobileDetGPUFeatureExtractor
  from object_detection.models.ssd_pnasnet_feature_extractor import SSDPNASNetFeatureExtractor
  from object_detection.predictors import rfcn_box_predictor
# pylint: enable=g-import-not-at-top

if tf_version.is_tf2():
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
      'ssd_efficientnet-b0_bifpn_keras':
          ssd_efficientnet_bifpn.SSDEfficientNetB0BiFPNKerasFeatureExtractor,
      'ssd_efficientnet-b1_bifpn_keras':
          ssd_efficientnet_bifpn.SSDEfficientNetB1BiFPNKerasFeatureExtractor,
      'ssd_efficientnet-b2_bifpn_keras':
          ssd_efficientnet_bifpn.SSDEfficientNetB2BiFPNKerasFeatureExtractor,
      'ssd_efficientnet-b3_bifpn_keras':
          ssd_efficientnet_bifpn.SSDEfficientNetB3BiFPNKerasFeatureExtractor,
      'ssd_efficientnet-b4_bifpn_keras':
          ssd_efficientnet_bifpn.SSDEfficientNetB4BiFPNKerasFeatureExtractor,
      'ssd_efficientnet-b5_bifpn_keras':
          ssd_efficientnet_bifpn.SSDEfficientNetB5BiFPNKerasFeatureExtractor,
      'ssd_efficientnet-b6_bifpn_keras':
          ssd_efficientnet_bifpn.SSDEfficientNetB6BiFPNKerasFeatureExtractor,
      'ssd_efficientnet-b7_bifpn_keras':
          ssd_efficientnet_bifpn.SSDEfficientNetB7BiFPNKerasFeatureExtractor,
  }

  FASTER_RCNN_KERAS_FEATURE_EXTRACTOR_CLASS_MAP = {
      'faster_rcnn_resnet50_keras':
          frcnn_resnet_keras.FasterRCNNResnet50KerasFeatureExtractor,
      'faster_rcnn_resnet101_keras':
          frcnn_resnet_keras.FasterRCNNResnet101KerasFeatureExtractor,
      'faster_rcnn_resnet152_keras':
          frcnn_resnet_keras.FasterRCNNResnet152KerasFeatureExtractor,
      'faster_rcnn_inception_resnet_v2_keras':
      frcnn_inc_res_keras.FasterRCNNInceptionResnetV2KerasFeatureExtractor,
      'faster_rcnn_resnet50_fpn_keras':
          frcnn_resnet_fpn_keras.FasterRCNNResnet50FpnKerasFeatureExtractor,
      'faster_rcnn_resnet101_fpn_keras':
          frcnn_resnet_fpn_keras.FasterRCNNResnet101FpnKerasFeatureExtractor,
      'faster_rcnn_resnet152_fpn_keras':
          frcnn_resnet_fpn_keras.FasterRCNNResnet152FpnKerasFeatureExtractor,
  }

  CENTER_NET_EXTRACTOR_FUNCTION_MAP = {
      'resnet_v2_50':
          center_net_resnet_feature_extractor.resnet_v2_50,
      'resnet_v2_101':
          center_net_resnet_feature_extractor.resnet_v2_101,
      'resnet_v1_18_fpn':
          center_net_resnet_v1_fpn_feature_extractor.resnet_v1_18_fpn,
      'resnet_v1_34_fpn':
          center_net_resnet_v1_fpn_feature_extractor.resnet_v1_34_fpn,
      'resnet_v1_50_fpn':
          center_net_resnet_v1_fpn_feature_extractor.resnet_v1_50_fpn,
      'resnet_v1_101_fpn':
          center_net_resnet_v1_fpn_feature_extractor.resnet_v1_101_fpn,
      'hourglass_10':
          center_net_hourglass_feature_extractor.hourglass_10,
      'hourglass_20':
          center_net_hourglass_feature_extractor.hourglass_20,
      'hourglass_32':
          center_net_hourglass_feature_extractor.hourglass_32,
      'hourglass_52':
          center_net_hourglass_feature_extractor.hourglass_52,
      'hourglass_104':
          center_net_hourglass_feature_extractor.hourglass_104,
      'mobilenet_v2':
          center_net_mobilenet_v2_feature_extractor.mobilenet_v2,
      'mobilenet_v2_fpn':
          center_net_mobilenet_v2_fpn_feature_extractor.mobilenet_v2_fpn,
  }

  FEATURE_EXTRACTOR_MAPS = [
      CENTER_NET_EXTRACTOR_FUNCTION_MAP,
      FASTER_RCNN_KERAS_FEATURE_EXTRACTOR_CLASS_MAP,
      SSD_KERAS_FEATURE_EXTRACTOR_CLASS_MAP
  ]

if tf_version.is_tf1():
  SSD_FEATURE_EXTRACTOR_CLASS_MAP = {
      'ssd_inception_v2':
          SSDInceptionV2FeatureExtractor,
      'ssd_inception_v3':
          SSDInceptionV3FeatureExtractor,
      'ssd_mobilenet_v1':
          SSDMobileNetV1FeatureExtractor,
      'ssd_mobilenet_v1_fpn':
          SSDMobileNetV1FpnFeatureExtractor,
      'ssd_mobilenet_v1_ppn':
          SSDMobileNetV1PpnFeatureExtractor,
      'ssd_mobilenet_v2':
          SSDMobileNetV2FeatureExtractor,
      'ssd_mobilenet_v2_fpn':
          SSDMobileNetV2FpnFeatureExtractor,
      'ssd_mobilenet_v2_mnasfpn':
          SSDMobileNetV2MnasFPNFeatureExtractor,
      'ssd_mobilenet_v3_large':
          SSDMobileNetV3LargeFeatureExtractor,
      'ssd_mobilenet_v3_small':
          SSDMobileNetV3SmallFeatureExtractor,
      'ssd_mobilenet_edgetpu':
          SSDMobileNetEdgeTPUFeatureExtractor,
      'ssd_resnet50_v1_fpn':
          ssd_resnet_v1_fpn.SSDResnet50V1FpnFeatureExtractor,
      'ssd_resnet101_v1_fpn':
          ssd_resnet_v1_fpn.SSDResnet101V1FpnFeatureExtractor,
      'ssd_resnet152_v1_fpn':
          ssd_resnet_v1_fpn.SSDResnet152V1FpnFeatureExtractor,
      'ssd_resnet50_v1_ppn':
          ssd_resnet_v1_ppn.SSDResnet50V1PpnFeatureExtractor,
      'ssd_resnet101_v1_ppn':
          ssd_resnet_v1_ppn.SSDResnet101V1PpnFeatureExtractor,
      'ssd_resnet152_v1_ppn':
          ssd_resnet_v1_ppn.SSDResnet152V1PpnFeatureExtractor,
      'embedded_ssd_mobilenet_v1':
          EmbeddedSSDMobileNetV1FeatureExtractor,
      'ssd_pnasnet':
          SSDPNASNetFeatureExtractor,
      'ssd_mobiledet_cpu':
          SSDMobileDetCPUFeatureExtractor,
      'ssd_mobiledet_dsp':
          SSDMobileDetDSPFeatureExtractor,
      'ssd_mobiledet_edgetpu':
          SSDMobileDetEdgeTPUFeatureExtractor,
      'ssd_mobiledet_gpu':
          SSDMobileDetGPUFeatureExtractor,
  }

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

  CENTER_NET_EXTRACTOR_FUNCTION_MAP = {}

  FEATURE_EXTRACTOR_MAPS = [
      SSD_FEATURE_EXTRACTOR_CLASS_MAP,
      FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP,
      CENTER_NET_EXTRACTOR_FUNCTION_MAP
  ]


def _check_feature_extractor_exists(feature_extractor_type):
  feature_extractors = set().union(*FEATURE_EXTRACTOR_MAPS)
  if feature_extractor_type not in feature_extractors:
    raise ValueError('{} is not supported. See `model_builder.py` for features '
                     'extractors compatible with different versions of '
                     'Tensorflow'.format(feature_extractor_type))


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
  depth_multiplier = feature_extractor_config.depth_multiplier
  min_depth = feature_extractor_config.min_depth
  pad_to_multiple = feature_extractor_config.pad_to_multiple
  use_explicit_padding = feature_extractor_config.use_explicit_padding
  use_depthwise = feature_extractor_config.use_depthwise

  is_keras = tf_version.is_tf2()
  if is_keras:
    conv_hyperparams = hyperparams_builder.KerasLayerHyperparams(
        feature_extractor_config.conv_hyperparams)
  else:
    conv_hyperparams = hyperparams_builder.build(
        feature_extractor_config.conv_hyperparams, is_training)
  override_base_feature_extractor_hyperparams = (
      feature_extractor_config.override_base_feature_extractor_hyperparams)

  if not is_keras and feature_type not in SSD_FEATURE_EXTRACTOR_CLASS_MAP:
    raise ValueError('Unknown ssd feature_extractor: {}'.format(feature_type))

  if is_keras:
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

  if is_keras:
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

  if feature_extractor_config.HasField('bifpn'):
    kwargs.update({
        'bifpn_min_level': feature_extractor_config.bifpn.min_level,
        'bifpn_max_level': feature_extractor_config.bifpn.max_level,
        'bifpn_num_iterations': feature_extractor_config.bifpn.num_iterations,
        'bifpn_num_filters': feature_extractor_config.bifpn.num_filters,
        'bifpn_combine_method': feature_extractor_config.bifpn.combine_method,
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
  _check_feature_extractor_exists(ssd_config.feature_extractor.type)

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
    feature_extractor_config, is_training, reuse_weights=True,
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

  kwargs = {}

  if feature_extractor_config.HasField('conv_hyperparams'):
    kwargs.update({
        'conv_hyperparams':
            hyperparams_builder.KerasLayerHyperparams(
                feature_extractor_config.conv_hyperparams),
        'override_base_feature_extractor_hyperparams':
            feature_extractor_config.override_base_feature_extractor_hyperparams
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

  return feature_extractor_class(
      is_training, first_stage_features_stride,
      batch_norm_trainable, **kwargs)


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
  _check_feature_extractor_exists(frcnn_config.feature_extractor.type)
  is_keras = tf_version.is_tf2()

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
      spatial_ops.multilevel_matmul_crop_and_resize
      if frcnn_config.use_matmul_crop_and_resize
      else spatial_ops.multilevel_native_crop_and_resize)
  clip_anchors_to_image = (
      frcnn_config.clip_anchors_to_image)

  common_kwargs = {
      'is_training':
          is_training,
      'num_classes':
          num_classes,
      'image_resizer_fn':
          image_resizer_fn,
      'feature_extractor':
          feature_extractor,
      'number_of_stages':
          number_of_stages,
      'first_stage_anchor_generator':
          first_stage_anchor_generator,
      'first_stage_target_assigner':
          first_stage_target_assigner,
      'first_stage_atrous_rate':
          first_stage_atrous_rate,
      'first_stage_box_predictor_arg_scope_fn':
          first_stage_box_predictor_arg_scope_fn,
      'first_stage_box_predictor_kernel_size':
          first_stage_box_predictor_kernel_size,
      'first_stage_box_predictor_depth':
          first_stage_box_predictor_depth,
      'first_stage_minibatch_size':
          first_stage_minibatch_size,
      'first_stage_sampler':
          first_stage_sampler,
      'first_stage_non_max_suppression_fn':
          first_stage_non_max_suppression_fn,
      'first_stage_max_proposals':
          first_stage_max_proposals,
      'first_stage_localization_loss_weight':
          first_stage_loc_loss_weight,
      'first_stage_objectness_loss_weight':
          first_stage_obj_loss_weight,
      'second_stage_target_assigner':
          second_stage_target_assigner,
      'second_stage_batch_size':
          second_stage_batch_size,
      'second_stage_sampler':
          second_stage_sampler,
      'second_stage_non_max_suppression_fn':
          second_stage_non_max_suppression_fn,
      'second_stage_score_conversion_fn':
          second_stage_score_conversion_fn,
      'second_stage_localization_loss_weight':
          second_stage_localization_loss_weight,
      'second_stage_classification_loss':
          second_stage_classification_loss,
      'second_stage_classification_loss_weight':
          second_stage_classification_loss_weight,
      'hard_example_miner':
          hard_example_miner,
      'add_summaries':
          add_summaries,
      'crop_and_resize_fn':
          crop_and_resize_fn,
      'clip_anchors_to_image':
          clip_anchors_to_image,
      'use_static_shapes':
          use_static_shapes,
      'resize_masks':
          frcnn_config.resize_masks,
      'return_raw_detections_during_predict':
          frcnn_config.return_raw_detections_during_predict,
      'output_final_box_features':
          frcnn_config.output_final_box_features,
      'output_final_box_rpn_features':
          frcnn_config.output_final_box_rpn_features,
  }

  if ((not is_keras and isinstance(second_stage_box_predictor,
                                   rfcn_box_predictor.RfcnBoxPredictor)) or
      (is_keras and
       isinstance(second_stage_box_predictor,
                  rfcn_keras_box_predictor.RfcnKerasBoxPredictor))):
    return rfcn_meta_arch.RFCNMetaArch(
        second_stage_rfcn_box_predictor=second_stage_box_predictor,
        **common_kwargs)
  elif frcnn_config.HasField('context_config'):
    context_config = frcnn_config.context_config
    common_kwargs.update({
        'attention_bottleneck_dimension':
            context_config.attention_bottleneck_dimension,
        'attention_temperature':
            context_config.attention_temperature,
        'use_self_attention':
            context_config.use_self_attention,
        'use_long_term_attention':
            context_config.use_long_term_attention,
        'self_attention_in_sequence':
            context_config.self_attention_in_sequence,
        'num_attention_heads':
            context_config.num_attention_heads,
        'num_attention_layers':
            context_config.num_attention_layers,
        'attention_position':
            context_config.attention_position
    })
    return context_rcnn_meta_arch.ContextRCNNMetaArch(
        initial_crop_size=initial_crop_size,
        maxpool_kernel_size=maxpool_kernel_size,
        maxpool_stride=maxpool_stride,
        second_stage_mask_rcnn_box_predictor=second_stage_box_predictor,
        second_stage_mask_prediction_loss_weight=(
            second_stage_mask_prediction_loss_weight),
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


# The class ID in the groundtruth/model architecture is usually 0-based while
# the ID in the label map is 1-based. The offset is used to convert between the
# the two.
CLASS_ID_OFFSET = 1
KEYPOINT_STD_DEV_DEFAULT = 1.0


def keypoint_proto_to_params(kp_config, keypoint_map_dict):
  """Converts CenterNet.KeypointEstimation proto to parameter namedtuple."""
  label_map_item = keypoint_map_dict[kp_config.keypoint_class_name]

  classification_loss, localization_loss, _, _, _, _, _ = (
      losses_builder.build(kp_config.loss))

  keypoint_indices = [
      keypoint.id for keypoint in label_map_item.keypoints
  ]
  keypoint_labels = [
      keypoint.label for keypoint in label_map_item.keypoints
  ]
  keypoint_std_dev_dict = {
      label: KEYPOINT_STD_DEV_DEFAULT for label in keypoint_labels
  }
  if kp_config.keypoint_label_to_std:
    for label, value in kp_config.keypoint_label_to_std.items():
      keypoint_std_dev_dict[label] = value
  keypoint_std_dev = [keypoint_std_dev_dict[label] for label in keypoint_labels]
  if kp_config.HasField('heatmap_head_params'):
    heatmap_head_num_filters = list(kp_config.heatmap_head_params.num_filters)
    heatmap_head_kernel_sizes = list(kp_config.heatmap_head_params.kernel_sizes)
  else:
    heatmap_head_num_filters = [256]
    heatmap_head_kernel_sizes = [3]
  if kp_config.HasField('offset_head_params'):
    offset_head_num_filters = list(kp_config.offset_head_params.num_filters)
    offset_head_kernel_sizes = list(kp_config.offset_head_params.kernel_sizes)
  else:
    offset_head_num_filters = [256]
    offset_head_kernel_sizes = [3]
  if kp_config.HasField('regress_head_params'):
    regress_head_num_filters = list(kp_config.regress_head_params.num_filters)
    regress_head_kernel_sizes = list(
        kp_config.regress_head_params.kernel_sizes)
  else:
    regress_head_num_filters = [256]
    regress_head_kernel_sizes = [3]
  return center_net_meta_arch.KeypointEstimationParams(
      task_name=kp_config.task_name,
      class_id=label_map_item.id - CLASS_ID_OFFSET,
      keypoint_indices=keypoint_indices,
      classification_loss=classification_loss,
      localization_loss=localization_loss,
      keypoint_labels=keypoint_labels,
      keypoint_std_dev=keypoint_std_dev,
      task_loss_weight=kp_config.task_loss_weight,
      keypoint_regression_loss_weight=kp_config.keypoint_regression_loss_weight,
      keypoint_heatmap_loss_weight=kp_config.keypoint_heatmap_loss_weight,
      keypoint_offset_loss_weight=kp_config.keypoint_offset_loss_weight,
      heatmap_bias_init=kp_config.heatmap_bias_init,
      keypoint_candidate_score_threshold=(
          kp_config.keypoint_candidate_score_threshold),
      num_candidates_per_keypoint=kp_config.num_candidates_per_keypoint,
      peak_max_pool_kernel_size=kp_config.peak_max_pool_kernel_size,
      unmatched_keypoint_score=kp_config.unmatched_keypoint_score,
      box_scale=kp_config.box_scale,
      candidate_search_scale=kp_config.candidate_search_scale,
      candidate_ranking_mode=kp_config.candidate_ranking_mode,
      offset_peak_radius=kp_config.offset_peak_radius,
      per_keypoint_offset=kp_config.per_keypoint_offset,
      predict_depth=kp_config.predict_depth,
      per_keypoint_depth=kp_config.per_keypoint_depth,
      keypoint_depth_loss_weight=kp_config.keypoint_depth_loss_weight,
      score_distance_offset=kp_config.score_distance_offset,
      clip_out_of_frame_keypoints=kp_config.clip_out_of_frame_keypoints,
      rescore_instances=kp_config.rescore_instances,
      heatmap_head_num_filters=heatmap_head_num_filters,
      heatmap_head_kernel_sizes=heatmap_head_kernel_sizes,
      offset_head_num_filters=offset_head_num_filters,
      offset_head_kernel_sizes=offset_head_kernel_sizes,
      regress_head_num_filters=regress_head_num_filters,
      regress_head_kernel_sizes=regress_head_kernel_sizes)


def object_detection_proto_to_params(od_config):
  """Converts CenterNet.ObjectDetection proto to parameter namedtuple."""
  loss = losses_pb2.Loss()
  # Add dummy classification loss to avoid the loss_builder throwing error.
  # TODO(yuhuic): update the loss builder to take the classification loss
  # directly.
  loss.classification_loss.weighted_sigmoid.CopyFrom(
      losses_pb2.WeightedSigmoidClassificationLoss())
  loss.localization_loss.CopyFrom(od_config.localization_loss)
  _, localization_loss, _, _, _, _, _ = (losses_builder.build(loss))
  return center_net_meta_arch.ObjectDetectionParams(
      localization_loss=localization_loss,
      scale_loss_weight=od_config.scale_loss_weight,
      offset_loss_weight=od_config.offset_loss_weight,
      task_loss_weight=od_config.task_loss_weight)


def object_center_proto_to_params(oc_config):
  """Converts CenterNet.ObjectCenter proto to parameter namedtuple."""
  loss = losses_pb2.Loss()
  # Add dummy localization loss to avoid the loss_builder throwing error.
  # TODO(yuhuic): update the loss builder to take the localization loss
  # directly.
  loss.localization_loss.weighted_l2.CopyFrom(
      losses_pb2.WeightedL2LocalizationLoss())
  loss.classification_loss.CopyFrom(oc_config.classification_loss)
  classification_loss, _, _, _, _, _, _ = (losses_builder.build(loss))
  keypoint_weights_for_center = []
  if oc_config.keypoint_weights_for_center:
    keypoint_weights_for_center = list(oc_config.keypoint_weights_for_center)

  if oc_config.center_head_params:
    center_head_num_filters = list(oc_config.center_head_params.num_filters)
    center_head_kernel_sizes = list(oc_config.center_head_params.kernel_sizes)
  else:
    center_head_num_filters = [256]
    center_head_kernel_sizes = [3]
  return center_net_meta_arch.ObjectCenterParams(
      classification_loss=classification_loss,
      object_center_loss_weight=oc_config.object_center_loss_weight,
      heatmap_bias_init=oc_config.heatmap_bias_init,
      min_box_overlap_iou=oc_config.min_box_overlap_iou,
      max_box_predictions=oc_config.max_box_predictions,
      use_labeled_classes=oc_config.use_labeled_classes,
      keypoint_weights_for_center=keypoint_weights_for_center,
      center_head_num_filters=center_head_num_filters,
      center_head_kernel_sizes=center_head_kernel_sizes)


def mask_proto_to_params(mask_config):
  """Converts CenterNet.MaskEstimation proto to parameter namedtuple."""
  loss = losses_pb2.Loss()
  # Add dummy localization loss to avoid the loss_builder throwing error.
  loss.localization_loss.weighted_l2.CopyFrom(
      losses_pb2.WeightedL2LocalizationLoss())
  loss.classification_loss.CopyFrom(mask_config.classification_loss)
  classification_loss, _, _, _, _, _, _ = (losses_builder.build(loss))
  return center_net_meta_arch.MaskParams(
      classification_loss=classification_loss,
      task_loss_weight=mask_config.task_loss_weight,
      mask_height=mask_config.mask_height,
      mask_width=mask_config.mask_width,
      score_threshold=mask_config.score_threshold,
      heatmap_bias_init=mask_config.heatmap_bias_init)


def densepose_proto_to_params(densepose_config):
  """Converts CenterNet.DensePoseEstimation proto to parameter namedtuple."""
  classification_loss, localization_loss, _, _, _, _, _ = (
      losses_builder.build(densepose_config.loss))
  return center_net_meta_arch.DensePoseParams(
      class_id=densepose_config.class_id,
      classification_loss=classification_loss,
      localization_loss=localization_loss,
      part_loss_weight=densepose_config.part_loss_weight,
      coordinate_loss_weight=densepose_config.coordinate_loss_weight,
      num_parts=densepose_config.num_parts,
      task_loss_weight=densepose_config.task_loss_weight,
      upsample_to_input_res=densepose_config.upsample_to_input_res,
      heatmap_bias_init=densepose_config.heatmap_bias_init)


def tracking_proto_to_params(tracking_config):
  """Converts CenterNet.TrackEstimation proto to parameter namedtuple."""
  loss = losses_pb2.Loss()
  # Add dummy localization loss to avoid the loss_builder throwing error.
  # TODO(yuhuic): update the loss builder to take the localization loss
  # directly.
  loss.localization_loss.weighted_l2.CopyFrom(
      losses_pb2.WeightedL2LocalizationLoss())
  loss.classification_loss.CopyFrom(tracking_config.classification_loss)
  classification_loss, _, _, _, _, _, _ = losses_builder.build(loss)
  return center_net_meta_arch.TrackParams(
      num_track_ids=tracking_config.num_track_ids,
      reid_embed_size=tracking_config.reid_embed_size,
      classification_loss=classification_loss,
      num_fc_layers=tracking_config.num_fc_layers,
      task_loss_weight=tracking_config.task_loss_weight)


def temporal_offset_proto_to_params(temporal_offset_config):
  """Converts CenterNet.TemporalOffsetEstimation proto to param-tuple."""
  loss = losses_pb2.Loss()
  # Add dummy classification loss to avoid the loss_builder throwing error.
  # TODO(yuhuic): update the loss builder to take the classification loss
  # directly.
  loss.classification_loss.weighted_sigmoid.CopyFrom(
      losses_pb2.WeightedSigmoidClassificationLoss())
  loss.localization_loss.CopyFrom(temporal_offset_config.localization_loss)
  _, localization_loss, _, _, _, _, _ = losses_builder.build(loss)
  return center_net_meta_arch.TemporalOffsetParams(
      localization_loss=localization_loss,
      task_loss_weight=temporal_offset_config.task_loss_weight)


def _build_center_net_model(center_net_config, is_training, add_summaries):
  """Build a CenterNet detection model.

  Args:
    center_net_config: A CenterNet proto object with model configuration.
    is_training: True if this model is being built for training purposes.
    add_summaries: Whether to add tf summaries in the model.

  Returns:
    CenterNetMetaArch based on the config.

  """

  image_resizer_fn = image_resizer_builder.build(
      center_net_config.image_resizer)
  _check_feature_extractor_exists(center_net_config.feature_extractor.type)
  feature_extractor = _build_center_net_feature_extractor(
      center_net_config.feature_extractor, is_training)
  object_center_params = object_center_proto_to_params(
      center_net_config.object_center_params)

  object_detection_params = None
  if center_net_config.HasField('object_detection_task'):
    object_detection_params = object_detection_proto_to_params(
        center_net_config.object_detection_task)

  keypoint_params_dict = None
  if center_net_config.keypoint_estimation_task:
    label_map_proto = label_map_util.load_labelmap(
        center_net_config.keypoint_label_map_path)
    keypoint_map_dict = {
        item.name: item for item in label_map_proto.item if item.keypoints
    }
    keypoint_params_dict = {}
    keypoint_class_id_set = set()
    all_keypoint_indices = []
    for task in center_net_config.keypoint_estimation_task:
      kp_params = keypoint_proto_to_params(task, keypoint_map_dict)
      keypoint_params_dict[task.task_name] = kp_params
      all_keypoint_indices.extend(kp_params.keypoint_indices)
      if kp_params.class_id in keypoint_class_id_set:
        raise ValueError(('Multiple keypoint tasks map to the same class id is '
                          'not allowed: %d' % kp_params.class_id))
      else:
        keypoint_class_id_set.add(kp_params.class_id)
    if len(all_keypoint_indices) > len(set(all_keypoint_indices)):
      raise ValueError('Some keypoint indices are used more than once.')

  mask_params = None
  if center_net_config.HasField('mask_estimation_task'):
    mask_params = mask_proto_to_params(center_net_config.mask_estimation_task)

  densepose_params = None
  if center_net_config.HasField('densepose_estimation_task'):
    densepose_params = densepose_proto_to_params(
        center_net_config.densepose_estimation_task)

  track_params = None
  if center_net_config.HasField('track_estimation_task'):
    track_params = tracking_proto_to_params(
        center_net_config.track_estimation_task)

  temporal_offset_params = None
  if center_net_config.HasField('temporal_offset_task'):
    temporal_offset_params = temporal_offset_proto_to_params(
        center_net_config.temporal_offset_task)
  non_max_suppression_fn = None
  if center_net_config.HasField('post_processing'):
    non_max_suppression_fn, _ = post_processing_builder.build(
        center_net_config.post_processing)

  return center_net_meta_arch.CenterNetMetaArch(
      is_training=is_training,
      add_summaries=add_summaries,
      num_classes=center_net_config.num_classes,
      feature_extractor=feature_extractor,
      image_resizer_fn=image_resizer_fn,
      object_center_params=object_center_params,
      object_detection_params=object_detection_params,
      keypoint_params_dict=keypoint_params_dict,
      mask_params=mask_params,
      densepose_params=densepose_params,
      track_params=track_params,
      temporal_offset_params=temporal_offset_params,
      use_depthwise=center_net_config.use_depthwise,
      compute_heatmap_sparse=center_net_config.compute_heatmap_sparse,
      non_max_suppression_fn=non_max_suppression_fn)


def _build_center_net_feature_extractor(feature_extractor_config, is_training):
  """Build a CenterNet feature extractor from the given config."""

  if feature_extractor_config.type not in CENTER_NET_EXTRACTOR_FUNCTION_MAP:
    raise ValueError('\'{}\' is not a known CenterNet feature extractor type'
                     .format(feature_extractor_config.type))
  kwargs = {
      'channel_means': list(feature_extractor_config.channel_means),
      'channel_stds': list(feature_extractor_config.channel_stds),
      'bgr_ordering': feature_extractor_config.bgr_ordering,
      'depth_multiplier': feature_extractor_config.depth_multiplier,
      'use_separable_conv': feature_extractor_config.use_separable_conv,
  }


  return CENTER_NET_EXTRACTOR_FUNCTION_MAP[feature_extractor_config.type](
      **kwargs)


META_ARCH_BUILDER_MAP = {
    'ssd': _build_ssd_model,
    'faster_rcnn': _build_faster_rcnn_model,
    'experimental_model': _build_experimental_model,
    'center_net': _build_center_net_model
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

  if meta_architecture not in META_ARCH_BUILDER_MAP:
    raise ValueError('Unknown meta architecture: {}'.format(meta_architecture))
  else:
    build_func = META_ARCH_BUILDER_MAP[meta_architecture]
    return build_func(getattr(model_config, meta_architecture), is_training,
                      add_summaries)
