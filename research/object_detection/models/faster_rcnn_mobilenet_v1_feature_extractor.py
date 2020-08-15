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

import tensorflow.compat.v1 as tf
import tf_slim as slim

from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.utils import shape_utils
from nets import mobilenet_v1


def _get_mobilenet_conv_no_last_stride_defs(conv_depth_ratio_in_percentage):
  if conv_depth_ratio_in_percentage not in [25, 50, 75, 100]:
    raise ValueError(
        'Only the following ratio percentages are supported: 25, 50, 75, 100')
  conv_depth_ratio_in_percentage = float(conv_depth_ratio_in_percentage) / 100.0
  channels = np.array([
      32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024
  ], dtype=np.float32)
  channels = (channels * conv_depth_ratio_in_percentage).astype(np.int32)
  return [
      mobilenet_v1.Conv(kernel=[3, 3], stride=2, depth=channels[0]),
      mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=channels[1]),
      mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=2, depth=channels[2]),
      mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=channels[3]),
      mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=2, depth=channels[4]),
      mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=channels[5]),
      mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=2, depth=channels[6]),
      mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=channels[7]),
      mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=channels[8]),
      mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=channels[9]),
      mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=channels[10]),
      mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=channels[11]),
      mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=channels[12]),
      mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=channels[13])
  ]


class FasterRCNNMobilenetV1FeatureExtractor(
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
    super(FasterRCNNMobilenetV1FeatureExtractor, self).__init__(
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

    preprocessed_inputs.get_shape().assert_has_rank(4)
    preprocessed_inputs = shape_utils.check_min_image_dim(
        min_dim=33, image_tensor=preprocessed_inputs)

    with slim.arg_scope(
        mobilenet_v1.mobilenet_v1_arg_scope(
            is_training=self._train_batch_norm,
            weight_decay=self._weight_decay)):
      with tf.variable_scope('MobilenetV1',
                             reuse=self._reuse_weights) as scope:
        params = {}
        if self._skip_last_stride:
          params['conv_defs'] = _get_mobilenet_conv_no_last_stride_defs(
              conv_depth_ratio_in_percentage=self.
              _conv_depth_ratio_in_percentage)
        _, activations = mobilenet_v1.mobilenet_v1_base(
            preprocessed_inputs,
            final_endpoint='Conv2d_11_pointwise',
            min_depth=self._min_depth,
            depth_multiplier=self._depth_multiplier,
            scope=scope,
            **params)
    return activations['Conv2d_11_pointwise'], activations

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
    with tf.variable_scope('MobilenetV1', reuse=self._reuse_weights):
      with slim.arg_scope(
          mobilenet_v1.mobilenet_v1_arg_scope(
              is_training=self._train_batch_norm,
              weight_decay=self._weight_decay)):
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
