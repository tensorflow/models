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
from object_detection.utils import context_manager
from object_detection.utils import ops
from object_detection.utils import shape_utils
from nets import resnet_v1

slim = tf.contrib.slim


class SSDResnetV1FpnFeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """SSD FPN feature extractor based on Resnet v1 architecture."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               resnet_base_fn,
               resnet_scope_name,
               fpn_scope_name,
               fpn_min_level=3,
               fpn_max_level=7,
               additional_layer_depth=256,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               use_native_resize_op=False,
               override_base_feature_extractor_hyperparams=False):
    """SSD FPN feature extractor based on Resnet v1 architecture.

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
      resnet_scope_name: scope name under which to construct resnet
      fpn_scope_name: scope name under which to construct the feature pyramid
        network.
      fpn_min_level: the highest resolution feature map to use in FPN. The valid
        values are {2, 3, 4, 5} which map to Resnet blocks {1, 2, 3, 4}
        respectively.
      fpn_max_level: the smallest resolution feature map to construct or use in
        FPN. FPN constructions uses features maps starting from fpn_min_level
        upto the fpn_max_level. In the case that there are not enough feature
        maps in the backbone network, additional feature maps are created by
        applying stride 2 convolutions until we get the desired number of fpn
        levels.
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
      use_native_resize_op: Whether to use tf.image.nearest_neighbor_resize
        to do upsampling in FPN. Default is false.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.

    Raises:
      ValueError: On supplying invalid arguments for unused arguments.
    """
    super(SSDResnetV1FpnFeatureExtractor, self).__init__(
        is_training=is_training,
        depth_multiplier=depth_multiplier,
        min_depth=min_depth,
        pad_to_multiple=pad_to_multiple,
        conv_hyperparams_fn=conv_hyperparams_fn,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)
    if self._use_explicit_padding is True:
      raise ValueError('Explicit padding is not a valid option.')
    self._resnet_base_fn = resnet_base_fn
    self._resnet_scope_name = resnet_scope_name
    self._fpn_scope_name = fpn_scope_name
    self._fpn_min_level = fpn_min_level
    self._fpn_max_level = fpn_max_level
    self._additional_layer_depth = additional_layer_depth
    self._use_native_resize_op = use_native_resize_op

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

  def _filter_features(self, image_features):
    # TODO(rathodv): Change resnet endpoint to strip scope prefixes instead
    # of munging the scope here.
    filtered_image_features = dict({})
    for key, feature in image_features.items():
      feature_name = key.split('/')[-1]
      if feature_name in ['block1', 'block2', 'block3', 'block4']:
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
    """
    preprocessed_inputs = shape_utils.check_min_image_dim(
        129, preprocessed_inputs)

    with tf.variable_scope(
        self._resnet_scope_name, reuse=self._reuse_weights) as scope:
      with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        with (slim.arg_scope(self._conv_hyperparams_fn())
              if self._override_base_feature_extractor_hyperparams else
              context_manager.IdentityContextManager()):
          _, image_features = self._resnet_base_fn(
              inputs=ops.pad_to_multiple(preprocessed_inputs,
                                         self._pad_to_multiple),
              num_classes=None,
              is_training=None,
              global_pool=False,
              output_stride=None,
              store_non_strided_activations=True,
              min_base_depth=self._min_depth,
              depth_multiplier=self._depth_multiplier,
              scope=scope)
          image_features = self._filter_features(image_features)
      depth_fn = lambda d: max(int(d * self._depth_multiplier), self._min_depth)
      with slim.arg_scope(self._conv_hyperparams_fn()):
        with tf.variable_scope(self._fpn_scope_name,
                               reuse=self._reuse_weights):
          base_fpn_max_level = min(self._fpn_max_level, 5)
          feature_block_list = []
          for level in range(self._fpn_min_level, base_fpn_max_level + 1):
            feature_block_list.append('block{}'.format(level - 1))
          fpn_features = feature_map_generators.fpn_top_down_feature_maps(
              [(key, image_features[key]) for key in feature_block_list],
              depth=depth_fn(self._additional_layer_depth),
              use_native_resize_op=self._use_native_resize_op)
          feature_maps = []
          for level in range(self._fpn_min_level, base_fpn_max_level + 1):
            feature_maps.append(
                fpn_features['top_down_block{}'.format(level - 1)])
          last_feature_map = fpn_features['top_down_block{}'.format(
              base_fpn_max_level - 1)]
          # Construct coarse features
          for i in range(base_fpn_max_level, self._fpn_max_level):
            last_feature_map = slim.conv2d(
                last_feature_map,
                num_outputs=depth_fn(self._additional_layer_depth),
                kernel_size=[3, 3],
                stride=2,
                padding='SAME',
                scope='bottom_up_block{}'.format(i))
            feature_maps.append(last_feature_map)
    return feature_maps


class SSDResnet50V1FpnFeatureExtractor(SSDResnetV1FpnFeatureExtractor):
  """SSD Resnet50 V1 FPN feature extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               fpn_min_level=3,
               fpn_max_level=7,
               additional_layer_depth=256,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               use_native_resize_op=False,
               override_base_feature_extractor_hyperparams=False):
    """SSD Resnet50 V1 FPN feature extractor based on Resnet v1 architecture.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      fpn_min_level: the minimum level in feature pyramid networks.
      fpn_max_level: the maximum level in feature pyramid networks.
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
      use_native_resize_op: Whether to use tf.image.nearest_neighbor_resize
        to do upsampling in FPN. Default is false.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(SSDResnet50V1FpnFeatureExtractor, self).__init__(
        is_training,
        depth_multiplier,
        min_depth,
        pad_to_multiple,
        conv_hyperparams_fn,
        resnet_v1.resnet_v1_50,
        'resnet_v1_50',
        'fpn',
        fpn_min_level,
        fpn_max_level,
        additional_layer_depth,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        use_native_resize_op=use_native_resize_op,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)


class SSDResnet101V1FpnFeatureExtractor(SSDResnetV1FpnFeatureExtractor):
  """SSD Resnet101 V1 FPN feature extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               fpn_min_level=3,
               fpn_max_level=7,
               additional_layer_depth=256,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               use_native_resize_op=False,
               override_base_feature_extractor_hyperparams=False):
    """SSD Resnet101 V1 FPN feature extractor based on Resnet v1 architecture.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      fpn_min_level: the minimum level in feature pyramid networks.
      fpn_max_level: the maximum level in feature pyramid networks.
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
      use_native_resize_op: Whether to use tf.image.nearest_neighbor_resize
        to do upsampling in FPN. Default is false.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(SSDResnet101V1FpnFeatureExtractor, self).__init__(
        is_training,
        depth_multiplier,
        min_depth,
        pad_to_multiple,
        conv_hyperparams_fn,
        resnet_v1.resnet_v1_101,
        'resnet_v1_101',
        'fpn',
        fpn_min_level,
        fpn_max_level,
        additional_layer_depth,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        use_native_resize_op=use_native_resize_op,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)


class SSDResnet152V1FpnFeatureExtractor(SSDResnetV1FpnFeatureExtractor):
  """SSD Resnet152 V1 FPN feature extractor."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams_fn,
               fpn_min_level=3,
               fpn_max_level=7,
               additional_layer_depth=256,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False,
               use_native_resize_op=False,
               override_base_feature_extractor_hyperparams=False):
    """SSD Resnet152 V1 FPN feature extractor based on Resnet v1 architecture.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the
        base feature extractor.
      fpn_min_level: the minimum level in feature pyramid networks.
      fpn_max_level: the maximum level in feature pyramid networks.
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False. UNUSED currently.
      use_depthwise: Whether to use depthwise convolutions. UNUSED currently.
      use_native_resize_op: Whether to use tf.image.nearest_neighbor_resize
        to do upsampling in FPN. Default is false.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(SSDResnet152V1FpnFeatureExtractor, self).__init__(
        is_training,
        depth_multiplier,
        min_depth,
        pad_to_multiple,
        conv_hyperparams_fn,
        resnet_v1.resnet_v1_152,
        'resnet_v1_152',
        'fpn',
        fpn_min_level,
        fpn_max_level,
        additional_layer_depth,
        reuse_weights=reuse_weights,
        use_explicit_padding=use_explicit_padding,
        use_depthwise=use_depthwise,
        use_native_resize_op=use_native_resize_op,
        override_base_feature_extractor_hyperparams=
        override_base_feature_extractor_hyperparams)
