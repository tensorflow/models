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
     By default convolution kernel size is set to 3, and it can be customized
     by caller.

  An example of the configuration for Inception V3:
  {
    'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', ''],
    'layer_depth': [-1, -1, -1, 512, 256, 128]
  }

  Args:
    feature_map_layout: Dictionary of specifications for the feature map
      layouts in the following format (Inception V2/V3 respectively):
      {
        'from_layer': ['Mixed_3c', 'Mixed_4c', 'Mixed_5c', '', '', ''],
        'layer_depth': [-1, -1, -1, 512, 256, 128]
      }
      or
      {
        'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', '', ''],
        'layer_depth': [-1, -1, -1, 512, 256, 128]
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
      design. Convolution kernel size is set to 3 by default, and can be
      customized by 'conv_kernel_size' parameter (similarily, 'conv_kernel_size'
      should be set to -1 if 'from_layer' is specified). The created convolution
      operation will be a normal 2D convolution by default, and a depthwise
      convolution followed by 1x1 convolution if 'use_depthwise' is set to True.
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
  use_explicit_padding = False
  if 'use_explicit_padding' in feature_map_layout:
    use_explicit_padding = feature_map_layout['use_explicit_padding']
  use_depthwise = False
  if 'use_depthwise' in feature_map_layout:
    use_depthwise = feature_map_layout['use_depthwise']
  for index, from_layer in enumerate(feature_map_layout['from_layer']):
    layer_depth = feature_map_layout['layer_depth'][index]
    conv_kernel_size = 3
    if 'conv_kernel_size' in feature_map_layout:
      conv_kernel_size = feature_map_layout['conv_kernel_size'][index]
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
      layer_name = '{}_2_Conv2d_{}_{}x{}_s2_{}'.format(
          base_from_layer, index, conv_kernel_size, conv_kernel_size,
          depth_fn(layer_depth))
      stride = 2
      padding = 'SAME'
      if use_explicit_padding:
        padding = 'VALID'
        intermediate_layer = ops.fixed_padding(
            intermediate_layer, conv_kernel_size)
      if use_depthwise:
        feature_map = slim.separable_conv2d(
            intermediate_layer,
            None, [conv_kernel_size, conv_kernel_size],
            depth_multiplier=1,
            padding=padding,
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
            intermediate_layer,
            depth_fn(layer_depth), [conv_kernel_size, conv_kernel_size],
            padding=padding,
            stride=stride,
            scope=layer_name)
      feature_map_keys.append(layer_name)
    feature_maps.append(feature_map)
  return collections.OrderedDict(
      [(x, y) for (x, y) in zip(feature_map_keys, feature_maps)])


def fpn_top_down_feature_maps(image_features, depth, scope=None):
  """Generates `top-down` feature maps for Feature Pyramid Networks.

  See https://arxiv.org/abs/1612.03144 for details.

  Args:
    image_features: list of tuples of (tensor_name, image_feature_tensor).
      Spatial resolutions of succesive tensors must reduce exactly by a factor
      of 2.
    depth: depth of output feature maps.
    scope: A scope name to wrap this op under.

  Returns:
    feature_maps: an OrderedDict mapping keys (feature map names) to
      tensors where each tensor has shape [batch, height_i, width_i, depth_i].
  """
  with tf.name_scope(scope, 'top_down'):
    num_levels = len(image_features)
    output_feature_maps_list = []
    output_feature_map_keys = []
    with slim.arg_scope(
        [slim.conv2d], padding='SAME', stride=1):
      top_down = slim.conv2d(
          image_features[-1][1],
          depth, [1, 1], activation_fn=None, normalizer_fn=None,
          scope='projection_%d' % num_levels)
      output_feature_maps_list.append(top_down)
      output_feature_map_keys.append(
          'top_down_%s' % image_features[-1][0])

      for level in reversed(range(num_levels - 1)):
        top_down = ops.nearest_neighbor_upsampling(top_down, 2)
        residual = slim.conv2d(
            image_features[level][1], depth, [1, 1],
            activation_fn=None, normalizer_fn=None,
            scope='projection_%d' % (level + 1))
        top_down += residual
        output_feature_maps_list.append(slim.conv2d(
            top_down,
            depth, [3, 3],
            scope='smoothing_%d' % (level + 1)))
        output_feature_map_keys.append('top_down_%s' % image_features[level][0])
      return collections.OrderedDict(
          reversed(zip(output_feature_map_keys, output_feature_maps_list)))


def pooling_pyramid_feature_maps(base_feature_map_depth, num_layers,
                                 image_features):
  """Generates pooling pyramid feature maps.

  The pooling pyramid feature maps is motivated by
  multi_resolution_feature_maps. The main difference are that it is simpler and
  reduces the number of free parameters.

  More specifically:
   - Instead of using convolutions to shrink the feature map, it uses max
     pooling, therefore totally gets rid of the parameters in convolution.
   - By pooling feature from larger map up to a single cell, it generates
     features in the same feature space.
   - Instead of independently making box predictions from individual maps, it
     shares the same classifier across different feature maps, therefore reduces
     the "mis-calibration" across different scales.

  See go/ppn-detection for more details.

  Args:
    base_feature_map_depth: Depth of the base feature before the max pooling.
    num_layers: Number of layers used to make predictions. They are pooled
      from the base feature.
    image_features: A dictionary of handles to activation tensors from the
      feature extractor.

  Returns:
    feature_maps: an OrderedDict mapping keys (feature map names) to
      tensors where each tensor has shape [batch, height_i, width_i, depth_i].
  Raises:
    ValueError: image_features does not contain exactly one entry
  """
  if len(image_features) != 1:
    raise ValueError('image_features should be a dictionary of length 1.')
  image_features = image_features[image_features.keys()[0]]

  feature_map_keys = []
  feature_maps = []
  feature_map_key = 'Base_Conv2d_1x1_%d' % base_feature_map_depth
  if base_feature_map_depth > 0:
    image_features = slim.conv2d(
        image_features,
        base_feature_map_depth,
        [1, 1],  # kernel size
        padding='SAME', stride=1, scope=feature_map_key)
    # Add a 1x1 max-pooling node (a no op node) immediately after the conv2d for
    # TPU v1 compatibility.  Without the following dummy op, TPU runtime
    # compiler will combine the convolution with one max-pooling below into a
    # single cycle, so getting the conv2d feature becomes impossible.
    image_features = slim.max_pool2d(
        image_features, [1, 1], padding='SAME', stride=1, scope=feature_map_key)
  feature_map_keys.append(feature_map_key)
  feature_maps.append(image_features)
  feature_map = image_features
  with slim.arg_scope([slim.max_pool2d], padding='SAME', stride=2):
    for i in range(num_layers - 1):
      feature_map_key = 'MaxPool2d_%d_2x2' % i
      feature_map = slim.max_pool2d(
          feature_map, [2, 2], padding='SAME', scope=feature_map_key)
      feature_map_keys.append(feature_map_key)
      feature_maps.append(feature_map)
  return collections.OrderedDict(
      [(x, y) for (x, y) in zip(feature_map_keys, feature_maps)])

