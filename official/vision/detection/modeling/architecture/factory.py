# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Model architecture factory."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from official.vision.detection.modeling.architecture import fpn
from official.vision.detection.modeling.architecture import heads
from official.vision.detection.modeling.architecture import identity
from official.vision.detection.modeling.architecture import nn_ops
from official.vision.detection.modeling.architecture import resnet


def batch_norm_relu_generator(params):

  def _batch_norm_op(**kwargs):
    return nn_ops.BatchNormRelu(
        momentum=params.batch_norm_momentum,
        epsilon=params.batch_norm_epsilon,
        trainable=params.batch_norm_trainable,
        **kwargs)

  return _batch_norm_op


def dropblock_generator(params):
  return nn_ops.Dropblock(
      dropblock_keep_prob=params.dropblock_keep_prob,
      dropblock_size=params.dropblock_size)


def backbone_generator(params):
  """Generator function for various backbone models."""
  if params.architecture.backbone == 'resnet':
    resnet_params = params.resnet
    backbone_fn = resnet.Resnet(
        resnet_depth=resnet_params.resnet_depth,
        dropblock=dropblock_generator(resnet_params.dropblock),
        batch_norm_relu=batch_norm_relu_generator(resnet_params.batch_norm))
  else:
    raise ValueError('Backbone model %s is not supported.' %
                     params.architecture.backbone)

  return backbone_fn


def multilevel_features_generator(params):
  """Generator function for various FPN models."""
  if params.architecture.multilevel_features == 'fpn':
    fpn_params = params.fpn
    fpn_fn = fpn.Fpn(
        min_level=fpn_params.min_level,
        max_level=fpn_params.max_level,
        fpn_feat_dims=fpn_params.fpn_feat_dims,
        use_separable_conv=fpn_params.use_separable_conv,
        batch_norm_relu=batch_norm_relu_generator(fpn_params.batch_norm))
  elif params.architecture.multilevel_features == 'identity':
    fpn_fn = identity.Identity()
  else:
    raise ValueError('The multi-level feature model %s is not supported.'
                     % params.architecture.multilevel_features)
  return fpn_fn


def retinanet_head_generator(params):
  """Generator function for RetinaNet head architecture."""
  return heads.RetinanetHead(
      params.min_level,
      params.max_level,
      params.num_classes,
      params.anchors_per_location,
      params.retinanet_head_num_convs,
      params.retinanet_head_num_filters,
      params.use_separable_conv,
      batch_norm_relu=batch_norm_relu_generator(params.batch_norm))


def rpn_head_generator(params):
  """Generator function for RPN head architecture."""
  return heads.RpnHead(params.min_level,
                       params.max_level,
                       params.anchors_per_location,
                       params.num_convs,
                       params.num_filters,
                       params.use_separable_conv,
                       params.use_batch_norm,
                       batch_norm_relu=batch_norm_relu_generator(
                           params.batch_norm))


def fast_rcnn_head_generator(params):
  """Generator function for Fast R-CNN head architecture."""
  return heads.FastrcnnHead(params.num_classes,
                            params.num_convs,
                            params.num_filters,
                            params.use_separable_conv,
                            params.num_fcs,
                            params.fc_dims,
                            params.use_batch_norm,
                            batch_norm_relu=batch_norm_relu_generator(
                                params.batch_norm))


def mask_rcnn_head_generator(params):
  """Generator function for Mask R-CNN head architecture."""
  return heads.MaskrcnnHead(params.num_classes,
                            params.mask_target_size,
                            params.num_convs,
                            params.num_filters,
                            params.use_separable_conv,
                            params.use_batch_norm,
                            batch_norm_relu=batch_norm_relu_generator(
                                params.batch_norm))


def shapeprior_head_generator(params):
  """Generator function for Shapemask head architecture."""
  raise NotImplementedError('Unimplemented')


def coarsemask_head_generator(params):
  """Generator function for Shapemask head architecture."""
  raise NotImplementedError('Unimplemented')


def finemask_head_generator(params):
  """Generator function for Shapemask head architecture."""
  raise NotImplementedError('Unimplemented')
