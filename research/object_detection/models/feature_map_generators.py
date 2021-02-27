# Lint as: python2, python3
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import functools
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf
import tf_slim as slim
from object_detection.utils import ops
from object_detection.utils import shape_utils

# Activation bound used for TPU v1. Activations will be clipped to
# [-ACTIVATION_BOUND, ACTIVATION_BOUND] when training with
# use_bounded_activations enabled.
ACTIVATION_BOUND = 6.0


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


def create_conv_block(
    use_depthwise, kernel_size, padding, stride, layer_name, conv_hyperparams,
    is_training, freeze_batchnorm, depth):
  """Create Keras layers for depthwise & non-depthwise convolutions.

  Args:
    use_depthwise: Whether to use depthwise separable conv instead of regular
      conv.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of the
      filters. Can be an int if both values are the same.
    padding: One of 'VALID' or 'SAME'.
    stride: A list of length 2: [stride_height, stride_width], specifying the
      convolution stride. Can be an int if both strides are the same.
    layer_name: String. The name of the layer.
    conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
      containing hyperparameters for convolution ops.
    is_training: Indicates whether the feature generator is in training mode.
    freeze_batchnorm: Bool. Whether to freeze batch norm parameters during
      training or not. When training with a small batch size (e.g. 1), it is
      desirable to freeze batch norm update and use pretrained batch norm
      params.
    depth: Depth of output feature maps.

  Returns:
    A list of conv layers.
  """
  layers = []
  if use_depthwise:
    kwargs = conv_hyperparams.params()
    # Both the regularizer and initializer apply to the depthwise layer,
    # so we remap the kernel_* to depthwise_* here.
    kwargs['depthwise_regularizer'] = kwargs['kernel_regularizer']
    kwargs['depthwise_initializer'] = kwargs['kernel_initializer']
    layers.append(
        tf.keras.layers.SeparableConv2D(
            depth, [kernel_size, kernel_size],
            depth_multiplier=1,
            padding=padding,
            strides=stride,
            name=layer_name + '_depthwise_conv',
            **kwargs))
  else:
    layers.append(tf.keras.layers.Conv2D(
        depth,
        [kernel_size, kernel_size],
        padding=padding,
        strides=stride,
        name=layer_name + '_conv',
        **conv_hyperparams.params()))
  layers.append(
      conv_hyperparams.build_batch_norm(
          training=(is_training and not freeze_batchnorm),
          name=layer_name + '_batchnorm'))
  layers.append(
      conv_hyperparams.build_activation_layer(
          name=layer_name))
  return layers


class KerasMultiResolutionFeatureMaps(tf.keras.Model):
  """Generates multi resolution feature maps from input image features.

  A Keras model that generates multi-scale feature maps for detection as in the
  SSD papers by Liu et al: https://arxiv.org/pdf/1512.02325v2.pdf, See Sec 2.1.

  More specifically, when called on inputs it performs the following two tasks:
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

  When this feature generator object is called on input image_features:
    Args:
      image_features: A dictionary of handles to activation tensors from the
        base feature extractor.

    Returns:
      feature_maps: an OrderedDict mapping keys (feature map names) to
        tensors where each tensor has shape [batch, height_i, width_i, depth_i].
  """

  def __init__(self,
               feature_map_layout,
               depth_multiplier,
               min_depth,
               insert_1x1_conv,
               is_training,
               conv_hyperparams,
               freeze_batchnorm,
               name=None):
    """Constructor.

    Args:
      feature_map_layout: Dictionary of specifications for the feature map
        layouts in the following format (Inception V2/V3 respectively):
        {
          'from_layer': ['Mixed_3c', 'Mixed_4c', 'Mixed_5c', '', '', ''],
          'layer_depth': [-1, -1, -1, 512, 256, 128]
        }
        or
        {
          'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', ''],
          'layer_depth': [-1, -1, -1, 512, 256, 128]
        }
        If 'from_layer' is specified, the specified feature map is directly used
        as a box predictor layer, and the layer_depth is directly infered from
        the feature map (instead of using the provided 'layer_depth' parameter).
        In this case, our convention is to set 'layer_depth' to -1 for clarity.
        Otherwise, if 'from_layer' is an empty string, then the box predictor
        layer will be built from the previous layer using convolution
        operations. Note that the current implementation only supports
        generating new layers using convolutions of stride 2 (resulting in a
        spatial resolution reduction by a factor of 2), and will be extended to
        a more flexible design. Convolution kernel size is set to 3 by default,
        and can be customized by 'conv_kernel_size' parameter (similarily,
        'conv_kernel_size' should be set to -1 if 'from_layer' is specified).
        The created convolution operation will be a normal 2D convolution by
        default, and a depthwise convolution followed by 1x1 convolution if
        'use_depthwise' is set to True.
      depth_multiplier: Depth multiplier for convolutional layers.
      min_depth: Minimum depth for convolutional layers.
      insert_1x1_conv: A boolean indicating whether an additional 1x1
        convolution should be inserted before shrinking the feature map.
      is_training: Indicates whether the feature generator is in training mode.
      conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
        containing hyperparameters for convolution ops.
      freeze_batchnorm: Bool. Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      name: A string name scope to assign to the model. If 'None', Keras
        will auto-generate one from the class name.
    """
    super(KerasMultiResolutionFeatureMaps, self).__init__(name=name)

    self.feature_map_layout = feature_map_layout
    self.convolutions = []

    depth_fn = get_depth_fn(depth_multiplier, min_depth)

    base_from_layer = ''
    use_explicit_padding = False
    if 'use_explicit_padding' in feature_map_layout:
      use_explicit_padding = feature_map_layout['use_explicit_padding']
    use_depthwise = False
    if 'use_depthwise' in feature_map_layout:
      use_depthwise = feature_map_layout['use_depthwise']
    for index, from_layer in enumerate(feature_map_layout['from_layer']):
      net = []
      layer_depth = feature_map_layout['layer_depth'][index]
      conv_kernel_size = 3
      if 'conv_kernel_size' in feature_map_layout:
        conv_kernel_size = feature_map_layout['conv_kernel_size'][index]
      if from_layer:
        base_from_layer = from_layer
      else:
        if insert_1x1_conv:
          layer_name = '{}_1_Conv2d_{}_1x1_{}'.format(
              base_from_layer, index, depth_fn(layer_depth // 2))
          net.append(tf.keras.layers.Conv2D(depth_fn(layer_depth // 2),
                                            [1, 1],
                                            padding='SAME',
                                            strides=1,
                                            name=layer_name + '_conv',
                                            **conv_hyperparams.params()))
          net.append(
              conv_hyperparams.build_batch_norm(
                  training=(is_training and not freeze_batchnorm),
                  name=layer_name + '_batchnorm'))
          net.append(
              conv_hyperparams.build_activation_layer(
                  name=layer_name))

        layer_name = '{}_2_Conv2d_{}_{}x{}_s2_{}'.format(
            base_from_layer, index, conv_kernel_size, conv_kernel_size,
            depth_fn(layer_depth))
        stride = 2
        padding = 'SAME'
        if use_explicit_padding:
          padding = 'VALID'
          # We define this function here while capturing the value of
          # conv_kernel_size, to avoid holding a reference to the loop variable
          # conv_kernel_size inside of a lambda function
          def fixed_padding(features, kernel_size=conv_kernel_size):
            return ops.fixed_padding(features, kernel_size)
          net.append(tf.keras.layers.Lambda(fixed_padding))
        # TODO(rathodv): Add some utilities to simplify the creation of
        # Depthwise & non-depthwise convolutions w/ normalization & activations
        if use_depthwise:
          net.append(tf.keras.layers.DepthwiseConv2D(
              [conv_kernel_size, conv_kernel_size],
              depth_multiplier=1,
              padding=padding,
              strides=stride,
              name=layer_name + '_depthwise_conv',
              **conv_hyperparams.params()))
          net.append(
              conv_hyperparams.build_batch_norm(
                  training=(is_training and not freeze_batchnorm),
                  name=layer_name + '_depthwise_batchnorm'))
          net.append(
              conv_hyperparams.build_activation_layer(
                  name=layer_name + '_depthwise'))

          net.append(tf.keras.layers.Conv2D(depth_fn(layer_depth), [1, 1],
                                            padding='SAME',
                                            strides=1,
                                            name=layer_name + '_conv',
                                            **conv_hyperparams.params()))
          net.append(
              conv_hyperparams.build_batch_norm(
                  training=(is_training and not freeze_batchnorm),
                  name=layer_name + '_batchnorm'))
          net.append(
              conv_hyperparams.build_activation_layer(
                  name=layer_name))

        else:
          net.append(tf.keras.layers.Conv2D(
              depth_fn(layer_depth),
              [conv_kernel_size, conv_kernel_size],
              padding=padding,
              strides=stride,
              name=layer_name + '_conv',
              **conv_hyperparams.params()))
          net.append(
              conv_hyperparams.build_batch_norm(
                  training=(is_training and not freeze_batchnorm),
                  name=layer_name + '_batchnorm'))
          net.append(
              conv_hyperparams.build_activation_layer(
                  name=layer_name))

      # Until certain bugs are fixed in checkpointable lists,
      # this net must be appended only once it's been filled with layers
      self.convolutions.append(net)

  def call(self, image_features, training=None):
    """Generate the multi-resolution feature maps.

    Executed when calling the `.__call__` method on input.

    Args:
      image_features: A dictionary of handles to activation tensors from the
        base feature extractor.
      training: A boolean, True when in training mode. If not specified,
        defaults to (in order of priority): the training mode of the outer
        `Layer.call`; the default mode set by
        `tf.keras.backend.set_learning_phase`; or the default value for
        `training` in the call signature.

    Returns:
      feature_maps: an OrderedDict mapping keys (feature map names) to
        tensors where each tensor has shape [batch, height_i, width_i, depth_i].
    """
    feature_maps = []
    feature_map_keys = []

    for index, from_layer in enumerate(self.feature_map_layout['from_layer']):
      if from_layer:
        feature_map = image_features[from_layer]
        feature_map_keys.append(from_layer)
      else:
        feature_map = feature_maps[-1]
        for layer in self.convolutions[index]:
          feature_map = layer(feature_map, training=training)
        layer_name = self.convolutions[index][-1].name
        feature_map_keys.append(layer_name)
      feature_maps.append(feature_map)
    return collections.OrderedDict(
        [(x, y) for (x, y) in zip(feature_map_keys, feature_maps)])


def multi_resolution_feature_maps(feature_map_layout, depth_multiplier,
                                  min_depth, insert_1x1_conv, image_features,
                                  pool_residual=False):
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
        'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', ''],
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
    pool_residual: Whether to add an average pooling layer followed by a
      residual connection between subsequent feature maps when the channel
      depth match. For example, with option 'layer_depth': [-1, 512, 256, 256],
      a pooling and residual layer is added between the third and forth feature
      map. This option is better used with Weight Shared Convolution Box
      Predictor when all feature maps have the same channel depth to encourage
      more consistent features across multi-scale feature maps.

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
      pre_layer_depth = pre_layer.get_shape().as_list()[3]
      intermediate_layer = pre_layer
      if insert_1x1_conv:
        layer_name = '{}_1_Conv2d_{}_1x1_{}'.format(
            base_from_layer, index, depth_fn(layer_depth // 2))
        intermediate_layer = slim.conv2d(
            pre_layer,
            depth_fn(layer_depth // 2), [1, 1],
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
        if pool_residual and pre_layer_depth == depth_fn(layer_depth):
          feature_map += slim.avg_pool2d(
              pre_layer, [3, 3],
              padding='SAME',
              stride=2,
              scope=layer_name + '_pool')
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


class KerasFpnTopDownFeatureMaps(tf.keras.Model):
  """Generates Keras based `top-down` feature maps for Feature Pyramid Networks.

  See https://arxiv.org/abs/1612.03144 for details.
  """

  def __init__(self,
               num_levels,
               depth,
               is_training,
               conv_hyperparams,
               freeze_batchnorm,
               use_depthwise=False,
               use_explicit_padding=False,
               use_bounded_activations=False,
               use_native_resize_op=False,
               scope=None,
               name=None):
    """Constructor.

    Args:
      num_levels: the number of image features.
      depth: depth of output feature maps.
      is_training: Indicates whether the feature generator is in training mode.
      conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
        containing hyperparameters for convolution ops.
      freeze_batchnorm: Bool. Whether to freeze batch norm parameters during
        training or not. When training with a small batch size (e.g. 1), it is
        desirable to freeze batch norm update and use pretrained batch norm
        params.
      use_depthwise: whether to use depthwise separable conv instead of regular
        conv.
      use_explicit_padding: whether to use explicit padding.
      use_bounded_activations: Whether or not to clip activations to range
        [-ACTIVATION_BOUND, ACTIVATION_BOUND]. Bounded activations better lend
        themselves to quantized inference.
      use_native_resize_op: If True, uses tf.image.resize_nearest_neighbor op
        for the upsampling process instead of reshape and broadcasting
        implementation.
      scope: A scope name to wrap this op under.
      name: A string name scope to assign to the model. If 'None', Keras
        will auto-generate one from the class name.
    """
    super(KerasFpnTopDownFeatureMaps, self).__init__(name=name)

    self.scope = scope if scope else 'top_down'
    self.top_layers = []
    self.residual_blocks = []
    self.top_down_blocks = []
    self.reshape_blocks = []
    self.conv_layers = []

    padding = 'VALID' if use_explicit_padding else 'SAME'
    stride = 1
    kernel_size = 3
    def clip_by_value(features):
      return tf.clip_by_value(features, -ACTIVATION_BOUND, ACTIVATION_BOUND)

    # top layers
    self.top_layers.append(tf.keras.layers.Conv2D(
        depth, [1, 1], strides=stride, padding=padding,
        name='projection_%d' % num_levels,
        **conv_hyperparams.params(use_bias=True)))
    if use_bounded_activations:
      self.top_layers.append(tf.keras.layers.Lambda(
          clip_by_value, name='clip_by_value'))

    for level in reversed(list(range(num_levels - 1))):
      # to generate residual from image features
      residual_net = []
      # to preprocess top_down (the image feature map from last layer)
      top_down_net = []
      # to reshape top_down according to residual if necessary
      reshaped_residual = []
      # to apply convolution layers to feature map
      conv_net = []

      # residual block
      residual_net.append(tf.keras.layers.Conv2D(
          depth, [1, 1], padding=padding, strides=1,
          name='projection_%d' % (level + 1),
          **conv_hyperparams.params(use_bias=True)))
      if use_bounded_activations:
        residual_net.append(tf.keras.layers.Lambda(
            clip_by_value, name='clip_by_value'))

      # top-down block
      # TODO (b/128922690): clean-up of ops.nearest_neighbor_upsampling
      if use_native_resize_op:
        def resize_nearest_neighbor(image):
          image_shape = shape_utils.combined_static_and_dynamic_shape(image)
          return tf.image.resize_nearest_neighbor(
              image, [image_shape[1] * 2, image_shape[2] * 2])
        top_down_net.append(tf.keras.layers.Lambda(
            resize_nearest_neighbor, name='nearest_neighbor_upsampling'))
      else:
        def nearest_neighbor_upsampling(image):
          return ops.nearest_neighbor_upsampling(image, scale=2)
        top_down_net.append(tf.keras.layers.Lambda(
            nearest_neighbor_upsampling, name='nearest_neighbor_upsampling'))

      # reshape block
      if use_explicit_padding:
        def reshape(inputs):
          residual_shape = tf.shape(inputs[0])
          return inputs[1][:, :residual_shape[1], :residual_shape[2], :]
        reshaped_residual.append(
            tf.keras.layers.Lambda(reshape, name='reshape'))

      # down layers
      if use_bounded_activations:
        conv_net.append(tf.keras.layers.Lambda(
            clip_by_value, name='clip_by_value'))

      if use_explicit_padding:
        def fixed_padding(features, kernel_size=kernel_size):
          return ops.fixed_padding(features, kernel_size)
        conv_net.append(tf.keras.layers.Lambda(
            fixed_padding, name='fixed_padding'))

      layer_name = 'smoothing_%d' % (level + 1)
      conv_block = create_conv_block(
          use_depthwise, kernel_size, padding, stride, layer_name,
          conv_hyperparams, is_training, freeze_batchnorm, depth)
      conv_net.extend(conv_block)

      self.residual_blocks.append(residual_net)
      self.top_down_blocks.append(top_down_net)
      self.reshape_blocks.append(reshaped_residual)
      self.conv_layers.append(conv_net)

  def call(self, image_features, training=None):
    """Generate the multi-resolution feature maps.

    Executed when calling the `.__call__` method on input.

    Args:
      image_features: list of tuples of (tensor_name, image_feature_tensor).
        Spatial resolutions of succesive tensors must reduce exactly by a factor
        of 2.
      training: A boolean, True when in training mode. If not specified,
        defaults to (in order of priority): the training mode of the outer
        `Layer.call`; the default mode set by
        `tf.keras.backend.set_learning_phase`; or the default value for
        `training` in the call signature.

    Returns:
      feature_maps: an OrderedDict mapping keys (feature map names) to
        tensors where each tensor has shape [batch, height_i, width_i, depth_i].
    """
    output_feature_maps_list = []
    output_feature_map_keys = []

    with tf.name_scope(self.scope):
      top_down = image_features[-1][1]
      for layer in self.top_layers:
        top_down = layer(top_down, training=training)
      output_feature_maps_list.append(top_down)
      output_feature_map_keys.append('top_down_%s' % image_features[-1][0])

      num_levels = len(image_features)
      for index, level in enumerate(reversed(list(range(num_levels - 1)))):
        residual = image_features[level][1]
        top_down = output_feature_maps_list[-1]
        for layer in self.residual_blocks[index]:
          residual = layer(residual, training=training)
        for layer in self.top_down_blocks[index]:
          top_down = layer(top_down, training=training)
        for layer in self.reshape_blocks[index]:
          top_down = layer([residual, top_down], training=training)
        top_down += residual
        for layer in self.conv_layers[index]:
          top_down = layer(top_down, training=training)
        output_feature_maps_list.append(top_down)
        output_feature_map_keys.append('top_down_%s' % image_features[level][0])
    return collections.OrderedDict(reversed(
        list(zip(output_feature_map_keys, output_feature_maps_list))))


def fpn_top_down_feature_maps(image_features,
                              depth,
                              use_depthwise=False,
                              use_explicit_padding=False,
                              use_bounded_activations=False,
                              scope=None,
                              use_native_resize_op=False):
  """Generates `top-down` feature maps for Feature Pyramid Networks.

  See https://arxiv.org/abs/1612.03144 for details.

  Args:
    image_features: list of tuples of (tensor_name, image_feature_tensor).
      Spatial resolutions of succesive tensors must reduce exactly by a factor
      of 2.
    depth: depth of output feature maps.
    use_depthwise: whether to use depthwise separable conv instead of regular
      conv.
    use_explicit_padding: whether to use explicit padding.
    use_bounded_activations: Whether or not to clip activations to range
      [-ACTIVATION_BOUND, ACTIVATION_BOUND]. Bounded activations better lend
      themselves to quantized inference.
    scope: A scope name to wrap this op under.
    use_native_resize_op: If True, uses tf.image.resize_nearest_neighbor op for
      the upsampling process instead of reshape and broadcasting implementation.

  Returns:
    feature_maps: an OrderedDict mapping keys (feature map names) to
      tensors where each tensor has shape [batch, height_i, width_i, depth_i].
  """
  with tf.name_scope(scope, 'top_down'):
    num_levels = len(image_features)
    output_feature_maps_list = []
    output_feature_map_keys = []
    padding = 'VALID' if use_explicit_padding else 'SAME'
    kernel_size = 3
    with slim.arg_scope(
        [slim.conv2d, slim.separable_conv2d], padding=padding, stride=1):
      top_down = slim.conv2d(
          image_features[-1][1],
          depth, [1, 1], activation_fn=None, normalizer_fn=None,
          scope='projection_%d' % num_levels)
      if use_bounded_activations:
        top_down = tf.clip_by_value(top_down, -ACTIVATION_BOUND,
                                    ACTIVATION_BOUND)
      output_feature_maps_list.append(top_down)
      output_feature_map_keys.append(
          'top_down_%s' % image_features[-1][0])

      for level in reversed(list(range(num_levels - 1))):
        if use_native_resize_op:
          with tf.name_scope('nearest_neighbor_upsampling'):
            top_down_shape = shape_utils.combined_static_and_dynamic_shape(
                top_down)
            top_down = tf.image.resize_nearest_neighbor(
                top_down, [top_down_shape[1] * 2, top_down_shape[2] * 2])
        else:
          top_down = ops.nearest_neighbor_upsampling(top_down, scale=2)
        residual = slim.conv2d(
            image_features[level][1], depth, [1, 1],
            activation_fn=None, normalizer_fn=None,
            scope='projection_%d' % (level + 1))
        if use_bounded_activations:
          residual = tf.clip_by_value(residual, -ACTIVATION_BOUND,
                                      ACTIVATION_BOUND)
        if use_explicit_padding:
          # slice top_down to the same shape as residual
          residual_shape = tf.shape(residual)
          top_down = top_down[:, :residual_shape[1], :residual_shape[2], :]
        top_down += residual
        if use_bounded_activations:
          top_down = tf.clip_by_value(top_down, -ACTIVATION_BOUND,
                                      ACTIVATION_BOUND)
        if use_depthwise:
          conv_op = functools.partial(slim.separable_conv2d, depth_multiplier=1)
        else:
          conv_op = slim.conv2d
        pre_output = top_down
        if use_explicit_padding:
          pre_output = ops.fixed_padding(pre_output, kernel_size)
        output_feature_maps_list.append(conv_op(
            pre_output,
            depth, [kernel_size, kernel_size],
            scope='smoothing_%d' % (level + 1)))
        output_feature_map_keys.append('top_down_%s' % image_features[level][0])
      return collections.OrderedDict(reversed(
          list(zip(output_feature_map_keys, output_feature_maps_list))))


def pooling_pyramid_feature_maps(base_feature_map_depth, num_layers,
                                 image_features, replace_pool_with_conv=False):
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
    replace_pool_with_conv: Whether or not to replace pooling operations with
      convolutions in the PPN. Default is False.

  Returns:
    feature_maps: an OrderedDict mapping keys (feature map names) to
      tensors where each tensor has shape [batch, height_i, width_i, depth_i].
  Raises:
    ValueError: image_features does not contain exactly one entry
  """
  if len(image_features) != 1:
    raise ValueError('image_features should be a dictionary of length 1.')
  image_features = image_features[list(image_features.keys())[0]]

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
  if replace_pool_with_conv:
    with slim.arg_scope([slim.conv2d], padding='SAME', stride=2):
      for i in range(num_layers - 1):
        feature_map_key = 'Conv2d_{}_3x3_s2_{}'.format(i,
                                                       base_feature_map_depth)
        feature_map = slim.conv2d(
            feature_map, base_feature_map_depth, [3, 3], scope=feature_map_key)
        feature_map_keys.append(feature_map_key)
        feature_maps.append(feature_map)
  else:
    with slim.arg_scope([slim.max_pool2d], padding='SAME', stride=2):
      for i in range(num_layers - 1):
        feature_map_key = 'MaxPool2d_%d_2x2' % i
        feature_map = slim.max_pool2d(
            feature_map, [2, 2], padding='SAME', scope=feature_map_key)
        feature_map_keys.append(feature_map_key)
        feature_maps.append(feature_map)
  return collections.OrderedDict(
      [(x, y) for (x, y) in zip(feature_map_keys, feature_maps)])
