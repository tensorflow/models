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
"""SSD feature extractors based on Resnet v1 and PPN architectures."""

import tensorflow as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import context_manager
from object_detection.utils import ops
from object_detection.utils import shape_utils
from nets import resnet_v1

slim = tf.contrib.slim


class _SSDResnetPpnFeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """SSD feature extractor based on resnet architecture and PPN."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               resnet_base_fn,
               resnet_scope_name,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               base_feature_map_depth=1024,
               num_layers=6,
               override_base_feature_extractor_hyperparams=False,
               use_bounded_activations=False):
    """Resnet based PPN Feature Extractor for SSD Models.

    See go/pooling-pyramid for more details about PPN.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      resnet_base_fn: base resnet network to use.
      resnet_scope_name: scope name to construct resnet
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False.
      use_depthwise: Whether to use depthwise convolutions. Default is False.
      base_feature_map_depth: Depth of the base feature before the max pooling.
      num_layers: Number of layers used to make predictions. They are pooled
        from the base feature.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
      use_bounded_activations: Whether or not to use bounded activations for
        resnet v1 bottleneck residual unit. Bounded activations better lend
        themselves to quantized inference.
    """
    super(_SSDResnetPpnFeatureExtractor, self).__init__(
        is_training, depth_multiplier, min_depth, pad_to_multiple,
        conv_hyperparams_fn, reuse_weights, use_explicit_padding, use_depthwise,
        override_base_feature_extractor_hyperparams)
    self._resnet_base_fn = resnet_base_fn
    self._resnet_scope_name = resnet_scope_name
    self._base_feature_map_depth = base_feature_map_depth
    self._num_layers = num_layers
    self._use_bounded_activations = use_bounded_activations

  def _filter_features(self, image_features):
    # TODO(rathodv): Change resnet endpoint to strip scope prefixes instead
    # of munging the scope here.
    filtered_image_features = dict({})
    for key, feature in image_features.items():
      feature_name = key.split('/')[-1]
      if feature_name in ['block2', 'block3', 'block4']:
        filtered_image_features[feature_name] = feature
    return filtered_image_features

  def preprocess(self, resized_inputs):
    """SSD preprocessing.

    VGG style channel mean subtraction as described here:
    https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-mdnge.
    Note that if the number of channels is not equal to 3, the mean subtraction
    will be skipped and the original resized_inputs will be returned.

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    if resized_inputs.shape.as_list()[3] == 3:
      channel_means = [123.68, 116.779, 103.939]
      return resized_inputs - [[channel_means]]
    else:
      return resized_inputs

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
        with (slim.arg_scope(self._conv_hyperparams_fn())
              if self._override_base_feature_extractor_hyperparams else
              context_manager.IdentityContextManager()):
          with slim.arg_scope(
              [resnet_v1.bottleneck],
              use_bounded_activations=self._use_bounded_activations):
            _, activations = self._resnet_base_fn(
                inputs=ops.pad_to_multiple(preprocessed_inputs,
                                           self._pad_to_multiple),
                num_classes=None,
                is_training=None,
                global_pool=False,
                output_stride=None,
                store_non_strided_activations=True,
                scope=scope)

      with slim.arg_scope(self._conv_hyperparams_fn()):
        feature_maps = feature_map_generators.pooling_pyramid_feature_maps(
            base_feature_map_depth=self._base_feature_map_depth,
            num_layers=self._num_layers,
            image_features={
                'image_features': self._filter_features(activations)['block3']
            })
    return feature_maps.values()


class SSDResnet50V1PpnFeatureExtractor(_SSDResnetPpnFeatureExtractor):
  """PPN Resnet50 v1 Feature Extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False):
    """Resnet50 v1 Feature Extractor for SSD Models.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False.
      use_depthwise: Whether to use depthwise convolutions. Default is False.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(SSDResnet50V1PpnFeatureExtractor, self).__init__(
        is_training, depth_multiplier, min_depth, pad_to_multiple,
        conv_hyperparams_fn, resnet_v1.resnet_v1_50, 'resnet_v1_50',
        reuse_weights, use_explicit_padding, use_depthwise,
        override_base_feature_extractor_hyperparams=(
            override_base_feature_extractor_hyperparams))


class SSDResnet101V1PpnFeatureExtractor(_SSDResnetPpnFeatureExtractor):
  """PPN Resnet101 v1 Feature Extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False):
    """Resnet101 v1 Feature Extractor for SSD Models.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False.
      use_depthwise: Whether to use depthwise convolutions. Default is False.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(SSDResnet101V1PpnFeatureExtractor, self).__init__(
        is_training, depth_multiplier, min_depth, pad_to_multiple,
        conv_hyperparams_fn, resnet_v1.resnet_v1_101, 'resnet_v1_101',
        reuse_weights, use_explicit_padding, use_depthwise,
        override_base_feature_extractor_hyperparams=(
            override_base_feature_extractor_hyperparams))


class SSDResnet152V1PpnFeatureExtractor(_SSDResnetPpnFeatureExtractor):
  """PPN Resnet152 v1 Feature Extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               override_base_feature_extractor_hyperparams=False):
    """Resnet152 v1 Feature Extractor for SSD Models.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False.
      use_depthwise: Whether to use depthwise convolutions. Default is False.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(SSDResnet152V1PpnFeatureExtractor, self).__init__(
        is_training, depth_multiplier, min_depth, pad_to_multiple,
        conv_hyperparams_fn, resnet_v1.resnet_v1_152, 'resnet_v1_152',
        reuse_weights, use_explicit_padding, use_depthwise,
        override_base_feature_extractor_hyperparams=(
            override_base_feature_extractor_hyperparams))
