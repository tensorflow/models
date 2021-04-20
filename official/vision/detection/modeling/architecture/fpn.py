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

"""Feature Pyramid Networks.

Feature Pyramid Networks were proposed in:
[1] Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan,
    , and Serge Belongie
    Feature Pyramid Networks for Object Detection. CVPR 2017.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

from official.vision.detection.modeling.architecture import nn_ops
from official.vision.detection.ops import spatial_transform_ops


class Fpn(object):
  """Feature pyramid networks."""

  def __init__(self,
               min_level=3,
               max_level=7,
               fpn_feat_dims=256,
               use_separable_conv=False,
               activation='relu',
               use_batch_norm=True,
               norm_activation=nn_ops.norm_activation_builder(
                   activation='relu')):
    """FPN initialization function.

    Args:
      min_level: `int` minimum level in FPN output feature maps.
      max_level: `int` maximum level in FPN output feature maps.
      fpn_feat_dims: `int` number of filters in FPN layers.
      use_separable_conv: `bool`, if True use separable convolution for
        convolution in FPN layers.
      use_batch_norm: 'bool', indicating whether batchnorm layers are added.
      norm_activation: an operation that includes a normalization layer
        followed by an optional activation layer.
    """
    self._min_level = min_level
    self._max_level = max_level
    self._fpn_feat_dims = fpn_feat_dims
    if use_separable_conv:
      self._conv2d_op = functools.partial(
          tf.keras.layers.SeparableConv2D, depth_multiplier=1)
    else:
      self._conv2d_op = tf.keras.layers.Conv2D
    if activation == 'relu':
      self._activation_op = tf.nn.relu
    elif activation == 'swish':
      self._activation_op = tf.nn.swish
    else:
      raise ValueError('Unsupported activation `{}`.'.format(activation))
    self._use_batch_norm = use_batch_norm
    self._norm_activation = norm_activation

    self._norm_activations = {}
    self._lateral_conv2d_op = {}
    self._post_hoc_conv2d_op = {}
    self._coarse_conv2d_op = {}
    for level in range(self._min_level, self._max_level + 1):
      if self._use_batch_norm:
        self._norm_activations[level] = norm_activation(
            use_activation=False, name='p%d-bn' % level)
      self._lateral_conv2d_op[level] = self._conv2d_op(
          filters=self._fpn_feat_dims,
          kernel_size=(1, 1),
          padding='same',
          name='l%d' % level)
      self._post_hoc_conv2d_op[level] = self._conv2d_op(
          filters=self._fpn_feat_dims,
          strides=(1, 1),
          kernel_size=(3, 3),
          padding='same',
          name='post_hoc_d%d' % level)
      self._coarse_conv2d_op[level] = self._conv2d_op(
          filters=self._fpn_feat_dims,
          strides=(2, 2),
          kernel_size=(3, 3),
          padding='same',
          name='p%d' % level)

  def __call__(self, multilevel_features, is_training=None):
    """Returns the FPN features for a given multilevel features.

    Args:
      multilevel_features: a `dict` containing `int` keys for continuous feature
        levels, e.g., [2, 3, 4, 5]. The values are corresponding features with
        shape [batch_size, height_l, width_l, num_filters].
      is_training: `bool` if True, the model is in training mode.

    Returns:
      a `dict` containing `int` keys for continuous feature levels
      [min_level, min_level + 1, ..., max_level]. The values are corresponding
      FPN features with shape [batch_size, height_l, width_l, fpn_feat_dims].
    """
    input_levels = list(multilevel_features.keys())
    if min(input_levels) > self._min_level:
      raise ValueError(
          'The minimum backbone level %d should be '%(min(input_levels)) +
          'less or equal to FPN minimum level %d.:'%(self._min_level))
    backbone_max_level = min(max(input_levels), self._max_level)
    with tf.name_scope('fpn'):
      # Adds lateral connections.
      feats_lateral = {}
      for level in range(self._min_level, backbone_max_level + 1):
        feats_lateral[level] = self._lateral_conv2d_op[level](
            multilevel_features[level])

      # Adds top-down path.
      feats = {backbone_max_level: feats_lateral[backbone_max_level]}
      for level in range(backbone_max_level - 1, self._min_level - 1, -1):
        feats[level] = spatial_transform_ops.nearest_upsampling(
            feats[level + 1], 2) + feats_lateral[level]

      # Adds post-hoc 3x3 convolution kernel.
      for level in range(self._min_level, backbone_max_level + 1):
        feats[level] = self._post_hoc_conv2d_op[level](feats[level])

      # Adds coarser FPN levels introduced for RetinaNet.
      for level in range(backbone_max_level + 1, self._max_level + 1):
        feats_in = feats[level - 1]
        if level > backbone_max_level + 1:
          feats_in = self._activation_op(feats_in)
        feats[level] = self._coarse_conv2d_op[level](feats_in)
      if self._use_batch_norm:
        # Adds batch_norm layer.
        for level in range(self._min_level, self._max_level + 1):
          feats[level] = self._norm_activations[level](
              feats[level], is_training=is_training)
    return feats
