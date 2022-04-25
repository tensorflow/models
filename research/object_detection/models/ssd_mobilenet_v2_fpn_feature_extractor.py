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

"""SSD MobilenetV2 FPN Feature Extractor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
from six.moves import range
import tensorflow.compat.v1 as tf
import tf_slim as slim

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import context_manager
from object_detection.utils import ops
from object_detection.utils import shape_utils
from nets.mobilenet import mobilenet
from nets.mobilenet import mobilenet_v2


# A modified config of mobilenet v2 that makes it more detection friendly.
def _create_modified_mobilenet_config():
  conv_defs = copy.deepcopy(mobilenet_v2.V2_DEF)
  conv_defs['spec'][-1] = mobilenet.op(
      slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=256)
  return conv_defs


class SSDMobileNetV2FpnFeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """SSD Feature Extractor using MobilenetV2 FPN features."""

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
    """SSD FPN feature extractor based on Mobilenet v2 architecture.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d
        and separable_conv2d ops in the layers that are added on top of the base
        feature extractor.
      fpn_min_level: the highest resolution feature map to use in FPN. The valid
        values are {2, 3, 4, 5} which map to MobileNet v2 layers
        {layer_4, layer_7, layer_14, layer_19}, respectively.
      fpn_max_level: the smallest resolution feature map to construct or use in
        FPN. FPN constructions uses features maps starting from fpn_min_level
        upto the fpn_max_level. In the case that there are not enough feature
        maps in the backbone network, additional feature maps are created by
        applying stride 2 convolutions until we get the desired number of fpn
        levels.
      additional_layer_depth: additional feature map layer channel depth.
      reuse_weights: whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False.
      use_depthwise: Whether to use depthwise convolutions. Default is False.
      use_native_resize_op: Whether to use tf.image.nearest_neighbor_resize
        to do upsampling in FPN. Default is false.
      override_base_feature_extractor_hyperparams: Whether to override
        hyperparameters of the base feature extractor with the one from
        `conv_hyperparams_fn`.
    """
    super(SSDMobileNetV2FpnFeatureExtractor, self).__init__(
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
    self._fpn_min_level = fpn_min_level
    self._fpn_max_level = fpn_max_level
    self._additional_layer_depth = additional_layer_depth
    self._conv_defs = None
    if self._use_depthwise:
      self._conv_defs = _create_modified_mobilenet_config()
    self._use_native_resize_op = use_native_resize_op

  def preprocess(self, resized_inputs):
    """SSD preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0

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
        33, preprocessed_inputs)

    with tf.variable_scope('MobilenetV2', reuse=self._reuse_weights) as scope:
      with slim.arg_scope(
          mobilenet_v2.training_scope(is_training=None, bn_decay=0.9997)), \
          slim.arg_scope(
              [mobilenet.depth_multiplier], min_depth=self._min_depth):
        with (slim.arg_scope(self._conv_hyperparams_fn())
              if self._override_base_feature_extractor_hyperparams else
              context_manager.IdentityContextManager()):
          _, image_features = mobilenet_v2.mobilenet_base(
              ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
              final_endpoint='layer_19',
              depth_multiplier=self._depth_multiplier,
              conv_defs=self._conv_defs,
              use_explicit_padding=self._use_explicit_padding,
              scope=scope)
      depth_fn = lambda d: max(int(d * self._depth_multiplier), self._min_depth)
      with slim.arg_scope(self._conv_hyperparams_fn()):
        with tf.variable_scope('fpn', reuse=self._reuse_weights):
          feature_blocks = [
              'layer_4', 'layer_7', 'layer_14', 'layer_19'
          ]
          base_fpn_max_level = min(self._fpn_max_level, 5)
          feature_block_list = []
          for level in range(self._fpn_min_level, base_fpn_max_level + 1):
            feature_block_list.append(feature_blocks[level - 2])
          fpn_features = feature_map_generators.fpn_top_down_feature_maps(
              [(key, image_features[key]) for key in feature_block_list],
              depth=depth_fn(self._additional_layer_depth),
              use_depthwise=self._use_depthwise,
              use_explicit_padding=self._use_explicit_padding,
              use_native_resize_op=self._use_native_resize_op)
          feature_maps = []
          for level in range(self._fpn_min_level, base_fpn_max_level + 1):
            feature_maps.append(fpn_features['top_down_{}'.format(
                feature_blocks[level - 2])])
          last_feature_map = fpn_features['top_down_{}'.format(
              feature_blocks[base_fpn_max_level - 2])]
          # Construct coarse features
          padding = 'VALID' if self._use_explicit_padding else 'SAME'
          kernel_size = 3
          for i in range(base_fpn_max_level + 1, self._fpn_max_level + 1):
            if self._use_depthwise:
              conv_op = functools.partial(
                  slim.separable_conv2d, depth_multiplier=1)
            else:
              conv_op = slim.conv2d
            if self._use_explicit_padding:
              last_feature_map = ops.fixed_padding(
                  last_feature_map, kernel_size)
            last_feature_map = conv_op(
                last_feature_map,
                num_outputs=depth_fn(self._additional_layer_depth),
                kernel_size=[kernel_size, kernel_size],
                stride=2,
                padding=padding,
                scope='bottom_up_Conv2d_{}'.format(i - base_fpn_max_level + 19))
            feature_maps.append(last_feature_map)
    return feature_maps
