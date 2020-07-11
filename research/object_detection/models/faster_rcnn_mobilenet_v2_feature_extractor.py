# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""Mobilenet v1 Faster R-CNN implementation."""
import numpy as np

import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.utils import shape_utils
from object_detection.utils import ops
from nets.mobilenet import mobilenet
from nets.mobilenet import mobilenet_v2

slim = contrib_slim

class FasterRCNNMobilenetV2FeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
  """Faster R-CNN Mobilenet V1 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               batch_norm_trainable=False,
               reuse_weights=None,
               weight_decay=0.0,
               depth_multiplier=1.0,
               min_depth=16,
               skip_last_stride=False,
               conv_depth_ratio_in_percentage=100):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      skip_last_stride: Skip the last stride if True.
      conv_depth_ratio_in_percentage: Conv depth ratio in percentage. Only
        applied if skip_last_stride is True.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16.
    """
    if first_stage_features_stride != 8 and first_stage_features_stride != 16:
      raise ValueError('`first_stage_features_stride` must be 8 or 16.')
    self._depth_multiplier = depth_multiplier
    self._min_depth = min_depth
    self._skip_last_stride = skip_last_stride
    self._conv_depth_ratio_in_percentage = conv_depth_ratio_in_percentage
    super(FasterRCNNMobilenetV2FeatureExtractor, self).__init__(
        is_training, first_stage_features_stride, batch_norm_trainable,
        reuse_weights, weight_decay)

  def preprocess(self, resized_inputs):
    """Faster R-CNN Mobilenet V1 preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0

  def _extract_proposal_features(self, preprocessed_inputs, scope):
    """Extracts first stage RPN features.

    Args:
      preprocessed_inputs: A [batch, height, width, channels] float32 tensor
        representing a batch of images.
      scope: A scope name.

    Returns:
      rpn_feature_map: A tensor with shape [batch, height, width, depth]
      activations: A dictionary mapping feature extractor tensor names to
        tensors

    Raises:
      InvalidArgumentError: If the spatial size of `preprocessed_inputs`
        (height or width) is less than 33.
      ValueError: If the created network is missing the required activation.
    """

    preprocessed_inputs = shape_utils.check_min_image_dim(
        33, preprocessed_inputs)

    with tf.variable_scope('MobilenetV2', reuse=self._reuse_weights) as scope:
      with slim.arg_scope(
          mobilenet_v2.training_scope(is_training=None, bn_decay=0.9997)), \
          slim.arg_scope(
              [mobilenet.depth_multiplier], min_depth=self._min_depth):
          _, activations = mobilenet_v2.mobilenet_base(
              preprocessed_inputs,
              final_endpoint='layer_19',
              min_depth=self._min_depth,
              depth_multiplier=self._depth_multiplier,
              scope=scope)

    return activations['layer_19'], activations

  def _extract_box_classifier_features(self, proposal_feature_maps, scope):
    """Extracts second stage box classifier features.

    Args:
      proposal_feature_maps: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.
      scope: A scope name (unused).

    Returns:
      proposal_classifier_features: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, height, width, depth]
        representing box classifier features for each proposal.
    """
    net = proposal_feature_maps

    conv_depth = 1024
    if self._skip_last_stride:
      conv_depth_ratio = float(self._conv_depth_ratio_in_percentage) / 100.0
      conv_depth = int(float(conv_depth) * conv_depth_ratio)

    depth = lambda d: max(int(d * 1.0), 16)
    with tf.variable_scope('MobilenetV2', reuse=self._reuse_weights):
      with slim.arg_scope(
          mobilenet_v2.training_scope(is_training=None, weight_decay=self._weight_decay)):
        with slim.arg_scope(
            [slim.conv2d, slim.separable_conv2d], padding='SAME'):
          net = slim.separable_conv2d(
              net,
              depth(conv_depth), [3, 3],
              depth_multiplier=1,
              stride=2,
              scope='Conv2d_12_pointwise')
          return slim.separable_conv2d(
              net,
              depth(conv_depth), [3, 3],
              depth_multiplier=1,
              stride=1,
              scope='Conv2d_13_pointwise')
