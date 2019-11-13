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

"""Inception V2 Faster R-CNN implementation.

See "Rethinking the Inception Architecture for Computer Vision"
https://arxiv.org/abs/1512.00567
"""
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

from object_detection.meta_architectures import faster_rcnn_meta_arch
from nets import inception_v2

slim = contrib_slim


def _batch_norm_arg_scope(list_ops,
                          use_batch_norm=True,
                          batch_norm_decay=0.9997,
                          batch_norm_epsilon=0.001,
                          batch_norm_scale=False,
                          train_batch_norm=False):
  """Slim arg scope for InceptionV2 batch norm."""
  if use_batch_norm:
    batch_norm_params = {
        'is_training': train_batch_norm,
        'scale': batch_norm_scale,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon
    }
    normalizer_fn = slim.batch_norm
  else:
    normalizer_fn = None
    batch_norm_params = None

  return slim.arg_scope(list_ops,
                        normalizer_fn=normalizer_fn,
                        normalizer_params=batch_norm_params)


class FasterRCNNInceptionV2FeatureExtractor(
    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
  """Faster R-CNN Inception V2 feature extractor implementation."""

  def __init__(self,
               is_training,
               first_stage_features_stride,
               batch_norm_trainable=False,
               reuse_weights=None,
               weight_decay=0.0,
               depth_multiplier=1.0,
               min_depth=16):
    """Constructor.

    Args:
      is_training: See base class.
      first_stage_features_stride: See base class.
      batch_norm_trainable: See base class.
      reuse_weights: See base class.
      weight_decay: See base class.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.

    Raises:
      ValueError: If `first_stage_features_stride` is not 8 or 16.
    """
    if first_stage_features_stride != 8 and first_stage_features_stride != 16:
      raise ValueError('`first_stage_features_stride` must be 8 or 16.')
    self._depth_multiplier = depth_multiplier
    self._min_depth = min_depth
    super(FasterRCNNInceptionV2FeatureExtractor, self).__init__(
        is_training, first_stage_features_stride, batch_norm_trainable,
        reuse_weights, weight_decay)

  def preprocess(self, resized_inputs):
    """Faster R-CNN Inception V2 preprocessing.

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
    shape_assert = tf.Assert(
        tf.logical_and(tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
                       tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
        ['image size must at least be 33 in both height and width.'])

    with tf.control_dependencies([shape_assert]):
      with tf.variable_scope('InceptionV2',
                             reuse=self._reuse_weights) as scope:
        with _batch_norm_arg_scope([slim.conv2d, slim.separable_conv2d],
                                   batch_norm_scale=True,
                                   train_batch_norm=self._train_batch_norm):
          _, activations = inception_v2.inception_v2_base(
              preprocessed_inputs,
              final_endpoint='Mixed_4e',
              min_depth=self._min_depth,
              depth_multiplier=self._depth_multiplier,
              scope=scope)

    return activations['Mixed_4e'], activations

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

    depth = lambda d: max(int(d * self._depth_multiplier), self._min_depth)
    trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

    data_format = 'NHWC'
    concat_dim = 3 if data_format == 'NHWC' else 1

    with tf.variable_scope('InceptionV2', reuse=self._reuse_weights):
      with slim.arg_scope(
          [slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
          stride=1,
          padding='SAME',
          data_format=data_format):
        with _batch_norm_arg_scope([slim.conv2d, slim.separable_conv2d],
                                   batch_norm_scale=True,
                                   train_batch_norm=self._train_batch_norm):

          with tf.variable_scope('Mixed_5a'):
            with tf.variable_scope('Branch_0'):
              branch_0 = slim.conv2d(
                  net, depth(128), [1, 1],
                  weights_initializer=trunc_normal(0.09),
                  scope='Conv2d_0a_1x1')
              branch_0 = slim.conv2d(branch_0, depth(192), [3, 3], stride=2,
                                     scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_1'):
              branch_1 = slim.conv2d(
                  net, depth(192), [1, 1],
                  weights_initializer=trunc_normal(0.09),
                  scope='Conv2d_0a_1x1')
              branch_1 = slim.conv2d(branch_1, depth(256), [3, 3],
                                     scope='Conv2d_0b_3x3')
              branch_1 = slim.conv2d(branch_1, depth(256), [3, 3], stride=2,
                                     scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_2'):
              branch_2 = slim.max_pool2d(net, [3, 3], stride=2,
                                         scope='MaxPool_1a_3x3')
            net = tf.concat([branch_0, branch_1, branch_2], concat_dim)

          with tf.variable_scope('Mixed_5b'):
            with tf.variable_scope('Branch_0'):
              branch_0 = slim.conv2d(net, depth(352), [1, 1],
                                     scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
              branch_1 = slim.conv2d(
                  net, depth(192), [1, 1],
                  weights_initializer=trunc_normal(0.09),
                  scope='Conv2d_0a_1x1')
              branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],
                                     scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
              branch_2 = slim.conv2d(
                  net, depth(160), [1, 1],
                  weights_initializer=trunc_normal(0.09),
                  scope='Conv2d_0a_1x1')
              branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                     scope='Conv2d_0b_3x3')
              branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                     scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
              branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
              branch_3 = slim.conv2d(
                  branch_3, depth(128), [1, 1],
                  weights_initializer=trunc_normal(0.1),
                  scope='Conv2d_0b_1x1')
            net = tf.concat([branch_0, branch_1, branch_2, branch_3],
                            concat_dim)

          with tf.variable_scope('Mixed_5c'):
            with tf.variable_scope('Branch_0'):
              branch_0 = slim.conv2d(net, depth(352), [1, 1],
                                     scope='Conv2d_0a_1x1')
            with tf.variable_scope('Branch_1'):
              branch_1 = slim.conv2d(
                  net, depth(192), [1, 1],
                  weights_initializer=trunc_normal(0.09),
                  scope='Conv2d_0a_1x1')
              branch_1 = slim.conv2d(branch_1, depth(320), [3, 3],
                                     scope='Conv2d_0b_3x3')
            with tf.variable_scope('Branch_2'):
              branch_2 = slim.conv2d(
                  net, depth(192), [1, 1],
                  weights_initializer=trunc_normal(0.09),
                  scope='Conv2d_0a_1x1')
              branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                     scope='Conv2d_0b_3x3')
              branch_2 = slim.conv2d(branch_2, depth(224), [3, 3],
                                     scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
              branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
              branch_3 = slim.conv2d(
                  branch_3, depth(128), [1, 1],
                  weights_initializer=trunc_normal(0.1),
                  scope='Conv2d_0b_1x1')
            proposal_classifier_features = tf.concat(
                [branch_0, branch_1, branch_2, branch_3], concat_dim)

    return proposal_classifier_features
