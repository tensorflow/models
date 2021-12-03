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

"""Model architecture factory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from official.legacy.detection.modeling.architecture import fpn
from official.legacy.detection.modeling.architecture import heads
from official.legacy.detection.modeling.architecture import identity
from official.legacy.detection.modeling.architecture import nn_ops
from official.legacy.detection.modeling.architecture import resnet
from official.legacy.detection.modeling.architecture import spinenet


def norm_activation_generator(params):
  return nn_ops.norm_activation_builder(
      momentum=params.batch_norm_momentum,
      epsilon=params.batch_norm_epsilon,
      trainable=params.batch_norm_trainable,
      activation=params.activation)


def backbone_generator(params):
  """Generator function for various backbone models."""
  if params.architecture.backbone == 'resnet':
    resnet_params = params.resnet
    backbone_fn = resnet.Resnet(
        resnet_depth=resnet_params.resnet_depth,
        activation=params.norm_activation.activation,
        norm_activation=norm_activation_generator(
            params.norm_activation))
  elif params.architecture.backbone == 'spinenet':
    spinenet_params = params.spinenet
    backbone_fn = spinenet.SpineNetBuilder(model_id=spinenet_params.model_id)
  else:
    raise ValueError('Backbone model `{}` is not supported.'
                     .format(params.architecture.backbone))

  return backbone_fn


def multilevel_features_generator(params):
  """Generator function for various FPN models."""
  if params.architecture.multilevel_features == 'fpn':
    fpn_params = params.fpn
    fpn_fn = fpn.Fpn(
        min_level=params.architecture.min_level,
        max_level=params.architecture.max_level,
        fpn_feat_dims=fpn_params.fpn_feat_dims,
        use_separable_conv=fpn_params.use_separable_conv,
        activation=params.norm_activation.activation,
        use_batch_norm=fpn_params.use_batch_norm,
        norm_activation=norm_activation_generator(
            params.norm_activation))
  elif params.architecture.multilevel_features == 'identity':
    fpn_fn = identity.Identity()
  else:
    raise ValueError('The multi-level feature model `{}` is not supported.'
                     .format(params.architecture.multilevel_features))
  return fpn_fn


def retinanet_head_generator(params):
  """Generator function for RetinaNet head architecture."""
  head_params = params.retinanet_head
  anchors_per_location = params.anchor.num_scales * len(
      params.anchor.aspect_ratios)
  return heads.RetinanetHead(
      params.architecture.min_level,
      params.architecture.max_level,
      params.architecture.num_classes,
      anchors_per_location,
      head_params.num_convs,
      head_params.num_filters,
      head_params.use_separable_conv,
      norm_activation=norm_activation_generator(params.norm_activation))


def rpn_head_generator(params):
  """Generator function for RPN head architecture."""
  head_params = params.rpn_head
  anchors_per_location = params.anchor.num_scales * len(
      params.anchor.aspect_ratios)
  return heads.RpnHead(
      params.architecture.min_level,
      params.architecture.max_level,
      anchors_per_location,
      head_params.num_convs,
      head_params.num_filters,
      head_params.use_separable_conv,
      params.norm_activation.activation,
      head_params.use_batch_norm,
      norm_activation=norm_activation_generator(params.norm_activation))


def oln_rpn_head_generator(params):
  """Generator function for OLN-proposal (OLN-RPN) head architecture."""
  head_params = params.rpn_head
  anchors_per_location = params.anchor.num_scales * len(
      params.anchor.aspect_ratios)
  return heads.OlnRpnHead(
      params.architecture.min_level,
      params.architecture.max_level,
      anchors_per_location,
      head_params.num_convs,
      head_params.num_filters,
      head_params.use_separable_conv,
      params.norm_activation.activation,
      head_params.use_batch_norm,
      norm_activation=norm_activation_generator(params.norm_activation))


def fast_rcnn_head_generator(params):
  """Generator function for Fast R-CNN head architecture."""
  head_params = params.frcnn_head
  return heads.FastrcnnHead(
      params.architecture.num_classes,
      head_params.num_convs,
      head_params.num_filters,
      head_params.use_separable_conv,
      head_params.num_fcs,
      head_params.fc_dims,
      params.norm_activation.activation,
      head_params.use_batch_norm,
      norm_activation=norm_activation_generator(params.norm_activation))


def oln_box_score_head_generator(params):
  """Generator function for Scoring Fast R-CNN head architecture."""
  head_params = params.frcnn_head
  return heads.OlnBoxScoreHead(
      params.architecture.num_classes,
      head_params.num_convs,
      head_params.num_filters,
      head_params.use_separable_conv,
      head_params.num_fcs,
      head_params.fc_dims,
      params.norm_activation.activation,
      head_params.use_batch_norm,
      norm_activation=norm_activation_generator(params.norm_activation))


def mask_rcnn_head_generator(params):
  """Generator function for Mask R-CNN head architecture."""
  head_params = params.mrcnn_head
  return heads.MaskrcnnHead(
      params.architecture.num_classes,
      params.architecture.mask_target_size,
      head_params.num_convs,
      head_params.num_filters,
      head_params.use_separable_conv,
      params.norm_activation.activation,
      head_params.use_batch_norm,
      norm_activation=norm_activation_generator(params.norm_activation))


def oln_mask_score_head_generator(params):
  """Generator function for Scoring Mask R-CNN head architecture."""
  head_params = params.mrcnn_head
  return heads.OlnMaskScoreHead(
      params.architecture.num_classes,
      params.architecture.mask_target_size,
      head_params.num_convs,
      head_params.num_filters,
      head_params.use_separable_conv,
      params.norm_activation.activation,
      head_params.use_batch_norm,
      norm_activation=norm_activation_generator(params.norm_activation))


def shapeprior_head_generator(params):
  """Generator function for shape prior head architecture."""
  head_params = params.shapemask_head
  return heads.ShapemaskPriorHead(
      params.architecture.num_classes,
      head_params.num_downsample_channels,
      head_params.mask_crop_size,
      head_params.use_category_for_mask,
      head_params.shape_prior_path)


def coarsemask_head_generator(params):
  """Generator function for ShapeMask coarse mask head architecture."""
  head_params = params.shapemask_head
  return heads.ShapemaskCoarsemaskHead(
      params.architecture.num_classes,
      head_params.num_downsample_channels,
      head_params.mask_crop_size,
      head_params.use_category_for_mask,
      head_params.num_convs,
      norm_activation=norm_activation_generator(params.norm_activation))


def finemask_head_generator(params):
  """Generator function for Shapemask fine mask head architecture."""
  head_params = params.shapemask_head
  return heads.ShapemaskFinemaskHead(
      params.architecture.num_classes,
      head_params.num_downsample_channels,
      head_params.mask_crop_size,
      head_params.use_category_for_mask,
      head_params.num_convs,
      head_params.upsample_factor)
