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
"""SSD Feature Pyramid Network (FPN) feature extractors based on Resnet v1.

See https://arxiv.org/abs/1708.02002 for details.
"""

import tensorflow as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import ops
from object_detection.utils import shape_utils
from nets import resnet_v1

slim = tf.contrib.slim


class _SSDResnetV1FpnFeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """SSD FPN feature extractor based on Resnet v1 architecture."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               resnet_base_fn,
               resnet_scope_name,
               fpn_scope_name,
               batch_norm_trainable=True,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False):
    """SSD FPN feature extractor based on Resnet v1 architecture.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
        UNUSED currently.
      min_depth: minimum feature extractor depth. UNUSED Currently.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: tf slim arg_scope for conv2d and separable_conv2d ops.
      resnet_base_fn: base resnet network to use.
      resnet_scope_name: scope name under which to construct resnet
      fpn_scope_name: scope name under which to construct the feature pyramid
        network.
      batch_norm_trainable: Whether to update batch norm parameters during
        training or not. When training with a small batch size
        (e.g. 1), it is desirable to disable batch norm update and use
        pretrained batch norm params.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.

    Raises:
      ValueError: On supplying invalid arguments for unused arguments.
    """
    super(_SSDResnetV1FpnFeatureExtractor, self).__init__(
        is_training, depth_multiplier, min_depth, pad_to_multiple,
        conv_hyperparams, batch_norm_trainable, reuse_weights,
        use_explicit_padding)
    if self._depth_multiplier != 1.0:
      raise ValueError('Only depth 1.0 is supported, found: {}'.
                       format(self._depth_multiplier))
    if self._use_explicit_padding is True:
      raise ValueError('Explicit padding is not a valid option.')
    self._resnet_base_fn = resnet_base_fn
    self._resnet_scope_name = resnet_scope_name
    self._fpn_scope_name = fpn_scope_name

  def preprocess(self, resized_inputs):
    """SSD preprocessing.

    VGG style channel mean subtraction as described here:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-mdnge.

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    channel_means = [123.68, 116.779, 103.939]
    return resized_inputs - [[channel_means]]

  def _filter_features(self, image_features):
    # TODO(rathodv): Change resnet endpoint to strip scope prefixes instead
    # of munging the scope here.
    filtered_image_features = dict({})
    for key, feature in image_features.items():
      feature_name = key.split('/')[-1]
      if feature_name in ['block2', 'block3', 'block4']:
        filtered_image_features[feature_name] = feature
    return filtered_image_features

  def extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]

    Raises:
      ValueError: depth multiplier is not supported.
    """
    if self._depth_multiplier != 1.0:
      raise ValueError('Depth multiplier not supported.')

    preprocessed_inputs = shape_utils.check_min_image_dim(
        129, preprocessed_inputs)

    with tf.variable_scope(
        self._resnet_scope_name, reuse=self._reuse_weights) as scope:
      with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        _, image_features = self._resnet_base_fn(
            inputs=ops.pad_to_multiple(preprocessed_inputs,
                                       self._pad_to_multiple),
            num_classes=None,
            is_training=self._is_training and self._batch_norm_trainable,
            global_pool=False,
            output_stride=None,
            store_non_strided_activations=True,
            scope=scope)
      image_features = self._filter_features(image_features)
      last_feature_map = image_features['block4']
    with tf.variable_scope(self._fpn_scope_name, reuse=self._reuse_weights):
      with slim.arg_scope(self._conv_hyperparams):
        for i in range(5, 7):
          last_feature_map = slim.conv2d(
              last_feature_map,
              num_outputs=256,
              kernel_size=[3, 3],
              stride=2,
              padding='SAME',
              scope='block{}'.format(i))
          image_features['bottomup_{}'.format(i)] = last_feature_map
        feature_maps = feature_map_generators.fpn_top_down_feature_maps(
            [
                image_features[key] for key in
                ['block2', 'block3', 'block4', 'bottomup_5', 'bottomup_6']
            ],
            depth=256,
            scope='top_down_features')
    return feature_maps.values()


class SSDResnet50V1FpnFeatureExtractor(_SSDResnetV1FpnFeatureExtractor):

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               batch_norm_trainable=True,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False):
    """Resnet50 v1 FPN Feature Extractor for SSD Models.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: tf slim arg_scope for conv2d and separable_conv2d ops.
      batch_norm_trainable: Whether to update batch norm parameters during
        training or not. When training with a small batch size
        (e.g. 1), it is desirable to disable batch norm update and use
        pretrained batch norm params.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
    """
    super(SSDResnet50V1FpnFeatureExtractor, self).__init__(
        is_training, depth_multiplier, min_depth, pad_to_multiple,
        conv_hyperparams, resnet_v1.resnet_v1_50, 'resnet_v1_50', 'fpn',
        batch_norm_trainable, reuse_weights, use_explicit_padding)


class SSDResnet101V1FpnFeatureExtractor(_SSDResnetV1FpnFeatureExtractor):

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               batch_norm_trainable=True,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False):
    """Resnet101 v1 FPN Feature Extractor for SSD Models.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: tf slim arg_scope for conv2d and separable_conv2d ops.
      batch_norm_trainable: Whether to update batch norm parameters during
        training or not. When training with a small batch size
        (e.g. 1), it is desirable to disable batch norm update and use
        pretrained batch norm params.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
    """
    super(SSDResnet101V1FpnFeatureExtractor, self).__init__(
        is_training, depth_multiplier, min_depth, pad_to_multiple,
        conv_hyperparams, resnet_v1.resnet_v1_101, 'resnet_v1_101', 'fpn',
        batch_norm_trainable, reuse_weights, use_explicit_padding)


class SSDResnet152V1FpnFeatureExtractor(_SSDResnetV1FpnFeatureExtractor):

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               batch_norm_trainable=True,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False):
    """Resnet152 v1 FPN Feature Extractor for SSD Models.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: tf slim arg_scope for conv2d and separable_conv2d ops.
      batch_norm_trainable: Whether to update batch norm parameters during
        training or not. When training with a small batch size
        (e.g. 1), it is desirable to disable batch norm update and use
        pretrained batch norm params.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
    """
    super(SSDResnet152V1FpnFeatureExtractor, self).__init__(
        is_training, depth_multiplier, min_depth, pad_to_multiple,
        conv_hyperparams, resnet_v1.resnet_v1_152, 'resnet_v1_152', 'fpn',
        batch_norm_trainable, reuse_weights, use_explicit_padding)
