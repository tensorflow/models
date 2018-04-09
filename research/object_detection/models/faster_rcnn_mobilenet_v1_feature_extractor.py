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
import tensorflow as tf

from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.utils import shape_utils
from nets import mobilenet_v1

slim = tf.contrib.slim


_MOBILENET_V1_100_CONV_NO_LAST_STRIDE_DEFS = [
    mobilenet_v1.Conv(kernel=[3, 3], stride=2, depth=32),
    mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=1024),
    mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=1024)
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
               skip_last_stride=False):
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

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16.
    """
    if first_stage_features_stride != 8 and first_stage_features_stride != 16:
      raise ValueError('`first_stage_features_stride` must be 8 or 16.')
    self._depth_multiplier = depth_multiplier
    self._min_depth = min_depth
    self._skip_last_stride = skip_last_stride
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
          params['conv_defs'] = _MOBILENET_V1_100_CONV_NO_LAST_STRIDE_DEFS
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
              depth(1024), [3, 3],
              depth_multiplier=1,
              stride=2,
              scope='Conv2d_12_pointwise')
          return slim.separable_conv2d(
              net,
              depth(1024), [3, 3],
              depth_multiplier=1,
              stride=1,
              scope='Conv2d_13_pointwise')
