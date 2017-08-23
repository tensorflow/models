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

"""Functions to generate a list of feature maps based on image features.

Provides several feature map generators that can be used to build object
detection feature extractors.

Object detection feature extractors usually are built by stacking two components
- A base feature extractor such as Inception V3 and a feature map generator.
Feature map generators build on the base feature extractors and produce a list
of final feature maps.
"""
import collections
import tensorflow as tf
from object_detection.utils import ops
slim = tf.contrib.slim


def get_depth_fn(depth_multiplier, min_depth):
  """Builds a callable to compute depth (output channels) of conv filters.

  Args:
    depth_multiplier: a multiplier for the nominal depth.
    min_depth: a lower bound on the depth of filters.

  Returns:
    A callable that takes in a nominal depth and returns the depth to use.
  """
  def multiply_depth(depth):
    new_depth = int(depth * depth_multiplier)
    return max(new_depth, min_depth)
  return multiply_depth


def multi_resolution_feature_maps(feature_map_layout, depth_multiplier,
                                  min_depth, insert_1x1_conv, image_features):
  """Generates multi resolution feature maps from input image features.

  Generates multi-scale feature maps for detection as in the SSD papers by
  Liu et al: https://arxiv.org/pdf/1512.02325v2.pdf, See Sec 2.1.

  More specifically, it performs the following two tasks:
  1) If a layer name is provided in the configuration, returns that layer as a
     feature map.
  2) If a layer name is left as an empty string, constructs a new feature map
     based on the spatial shape and depth configuration. Note that the current
     implementation only supports generating new layers using convolution of
     stride 2 resulting in a spatial resolution reduction by a factor of 2.

  An example of the configuration for Inception V3:
  {
    'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', ''],
    'layer_depth': [-1, -1, -1, 512, 256, 128],
    'anchor_strides': [16, 32, 64, -1, -1, -1]
  }

  Args:
    feature_map_layout: Dictionary of specifications for the feature map
      layouts in the following format (Inception V2/V3 respectively):
      {
        'from_layer': ['Mixed_3c', 'Mixed_4c', 'Mixed_5c', '', '', ''],
        'layer_depth': [-1, -1, -1, 512, 256, 128],
        'anchor_strides': [16, 32, 64, -1, -1, -1]
      }
      or
      {
        'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', '', ''],
        'layer_depth': [-1, -1, -1, 512, 256, 128],
        'anchor_strides': [16, 32, 64, -1, -1, -1]
      }
      If 'from_layer' is specified, the specified feature map is directly used
      as a box predictor layer, and the layer_depth is directly infered from the
      feature map (instead of using the provided 'layer_depth' parameter). In
      this case, our convention is to set 'layer_depth' to -1 for clarity.
      Otherwise, if 'from_layer' is an empty string, then the box predictor
      layer will be built from the previous layer using convolution operations.
      Note that the current implementation only supports generating new layers
      using convolutions of stride 2 (resulting in a spatial resolution
      reduction by a factor of 2), and will be extended to a more flexible
      design. Finally, the optional 'anchor_strides' can be used to specify the
      anchor stride at each layer where 'from_layer' is specified. Our
      convention is to set 'anchor_strides' to -1 whenever at the positions that
      'from_layer' is an empty string, and anchor strides at these layers will
      be inferred from the previous layer's anchor strides and the current
      layer's stride length. In the case where 'anchor_strides' is not
      specified, the anchor strides will default to the image width and height
      divided by the number of anchors.
    depth_multiplier: Depth multiplier for convolutional layers.
    min_depth: Minimum depth for convolutional layers.
    insert_1x1_conv: A boolean indicating whether an additional 1x1 convolution
      should be inserted before shrinking the feature map.
    image_features: A dictionary of handles to activation tensors from the
      base feature extractor.

  Returns:
    feature_maps: an OrderedDict mapping keys (feature map names) to
      tensors where each tensor has shape [batch, height_i, width_i, depth_i].

  Raises:
    ValueError: if the number entries in 'from_layer' and
      'layer_depth' do not match.
    ValueError: if the generated layer does not have the same resolution
      as specified.
  """
  depth_fn = get_depth_fn(depth_multiplier, min_depth)

  feature_map_keys = []
  feature_maps = []
  base_from_layer = ''
  feature_map_strides = None
  use_depthwise = False
  if 'anchor_strides' in feature_map_layout:
    feature_map_strides = (feature_map_layout['anchor_strides'])
  if 'use_depthwise' in feature_map_layout:
    use_depthwise = feature_map_layout['use_depthwise']
  for index, (from_layer, layer_depth) in enumerate(
      zip(feature_map_layout['from_layer'], feature_map_layout['layer_depth'])):
    if from_layer:
      feature_map = image_features[from_layer]
      base_from_layer = from_layer
      feature_map_keys.append(from_layer)
    else:
      pre_layer = feature_maps[-1]
      intermediate_layer = pre_layer
      if insert_1x1_conv:
        layer_name = '{}_1_Conv2d_{}_1x1_{}'.format(
            base_from_layer, index, depth_fn(layer_depth / 2))
        intermediate_layer = slim.conv2d(
            pre_layer,
            depth_fn(layer_depth / 2), [1, 1],
            padding='SAME',
            stride=1,
            scope=layer_name)
      stride = 2
      layer_name = '{}_2_Conv2d_{}_3x3_s2_{}'.format(
          base_from_layer, index, depth_fn(layer_depth))
      if use_depthwise:
        feature_map = slim.separable_conv2d(
            ops.pad_to_multiple(intermediate_layer, stride),
            None, [3, 3],
            depth_multiplier=1,
            padding='SAME',
            stride=stride,
            scope=layer_name + '_depthwise')
        feature_map = slim.conv2d(
            feature_map,
            depth_fn(layer_depth), [1, 1],
            padding='SAME',
            stride=1,
            scope=layer_name)
      else:
        feature_map = slim.conv2d(
            ops.pad_to_multiple(intermediate_layer, stride),
            depth_fn(layer_depth), [3, 3],
            padding='SAME',
            stride=stride,
            scope=layer_name)

      if (index > 0 and feature_map_strides and
          feature_map_strides[index - 1] > 0):
        feature_map_strides[index] = (
            stride * feature_map_strides[index - 1])
      feature_map_keys.append(layer_name)
    feature_maps.append(feature_map)
  return collections.OrderedDict(
      [(x, y) for (x, y) in zip(feature_map_keys, feature_maps)])
