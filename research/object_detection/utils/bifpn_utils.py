# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Functions to manipulate feature map pyramids, such as for FPNs and BiFPNs.

Includes utility functions to facilitate feature pyramid map manipulations,
such as combining multiple feature maps, upsampling or downsampling feature
maps, and applying blocks of convolution, batchnorm, and activation layers.
"""
from six.moves import range
import tensorflow as tf
from object_detection.utils import ops
from object_detection.utils import shape_utils


def create_conv_block(name, num_filters, kernel_size, strides, padding,
                      use_separable, apply_batchnorm, apply_activation,
                      conv_hyperparams, is_training, freeze_batchnorm):
  """Create Keras layers for regular or separable convolutions.

  Args:
    name: String. The name of the layer.
    num_filters: Number of filters (channels) for the output feature maps.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of the
      filters, or a single int if both values are the same.
    strides: A list of length 2: [stride_height, stride_width], specifying the
      convolution stride, or a single int if both strides are the same.
    padding: One of 'VALID' or 'SAME'.
    use_separable: Bool. Whether to use depthwise separable convolution instead
      of regular convolution.
    apply_batchnorm: Bool. Whether to apply a batch normalization layer after
      convolution, constructed according to the conv_hyperparams.
    apply_activation: Bool. Whether to apply an activation layer after
      convolution, constructed according to the conv_hyperparams.
    conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
      containing hyperparameters for convolution ops.
    is_training: Bool. Whether the feature generator is in training mode.
    freeze_batchnorm: Bool. Whether to freeze batch norm parameters during
      training or not. When training with a small batch size (e.g. 1), it is
      desirable to freeze batch norm update and use pretrained batch norm
      params.

  Returns:
    A list of keras layers, including (regular or seperable) convolution, and
    optionally batch normalization and activation layers.
  """
  layers = []
  if use_separable:
    kwargs = conv_hyperparams.params()
    # Both the regularizer and initializer apply to the depthwise layer,
    # so we remap the kernel_* to depthwise_* here.
    kwargs['depthwise_regularizer'] = kwargs['kernel_regularizer']
    kwargs['depthwise_initializer'] = kwargs['kernel_initializer']
    # TODO(aom): Verify that the pointwise regularizer/initializer should be set
    # here, since this is not the case in feature_map_generators.py
    kwargs['pointwise_regularizer'] = kwargs['kernel_regularizer']
    kwargs['pointwise_initializer'] = kwargs['kernel_initializer']
    layers.append(
        tf.keras.layers.SeparableConv2D(
            filters=num_filters,
            kernel_size=kernel_size,
            depth_multiplier=1,
            padding=padding,
            strides=strides,
            name=name + '_separable_conv',
            **kwargs))
  else:
    layers.append(
        tf.keras.layers.Conv2D(
            filters=num_filters,
            kernel_size=kernel_size,
            padding=padding,
            strides=strides,
            name=name + '_conv',
            **conv_hyperparams.params()))

  if apply_batchnorm:
    layers.append(
        conv_hyperparams.build_batch_norm(
            training=(is_training and not freeze_batchnorm),
            name=name + '_batchnorm'))

  if apply_activation:
    layers.append(
        conv_hyperparams.build_activation_layer(name=name + '_activation'))

  return layers


def create_downsample_feature_map_ops(scale, downsample_method,
                                      conv_hyperparams, is_training,
                                      freeze_batchnorm, name):
  """Creates Keras layers for downsampling feature maps.

  Args:
    scale: Int. The scale factor by which to downsample input feature maps. For
      example, in the case of a typical feature map pyramid, the scale factor
      between level_i and level_i+1 is 2.
    downsample_method: String. The method used for downsampling. Currently
      supported methods include 'max_pooling', 'avg_pooling', and
      'depthwise_conv'.
    conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
      containing hyperparameters for convolution ops.
    is_training: Bool. Whether the feature generator is in training mode.
    freeze_batchnorm: Bool. Whether to freeze batch norm parameters during
      training or not. When training with a small batch size (e.g. 1), it is
      desirable to freeze batch norm update and use pretrained batch norm
      params.
    name: String. The name used to prefix the constructed layers.

  Returns:
    A list of Keras layers which will downsample input feature maps by the
    desired scale factor.
  """
  layers = []
  padding = 'SAME'
  stride = int(scale)
  kernel_size = stride + 1
  if downsample_method == 'max_pooling':
    layers.append(
        tf.keras.layers.MaxPooling2D(
            pool_size=kernel_size,
            strides=stride,
            padding=padding,
            name=name + '_downsample_max_x{}'.format(stride)))
  elif downsample_method == 'avg_pooling':
    layers.append(
        tf.keras.layers.AveragePooling2D(
            pool_size=kernel_size,
            strides=stride,
            padding=padding,
            name=name + '_downsample_avg_x{}'.format(stride)))
  elif downsample_method == 'depthwise_conv':
    layers.append(
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=stride,
            padding=padding,
            name=name + '_downsample_depthwise_x{}'.format(stride)))
    layers.append(
        conv_hyperparams.build_batch_norm(
            training=(is_training and not freeze_batchnorm),
            name=name + '_downsample_batchnorm'))
    layers.append(
        conv_hyperparams.build_activation_layer(name=name +
                                                '_downsample_activation'))
  else:
    raise ValueError('Unknown downsample method: {}'.format(downsample_method))

  return layers


def create_upsample_feature_map_ops(scale, use_native_resize_op, name):
  """Creates Keras layers for upsampling feature maps.

  Args:
    scale: Int. The scale factor by which to upsample input feature maps. For
      example, in the case of a typical feature map pyramid, the scale factor
      between level_i and level_i-1 is 2.
    use_native_resize_op: If True, uses tf.image.resize_nearest_neighbor op for
      the upsampling process instead of reshape and broadcasting implementation.
    name: String. The name used to prefix the constructed layers.

  Returns:
    A list of Keras layers which will upsample input feature maps by the
    desired scale factor.
  """
  layers = []
  if use_native_resize_op:

    def resize_nearest_neighbor(image):
      image_shape = shape_utils.combined_static_and_dynamic_shape(image)
      return tf.image.resize_nearest_neighbor(
          image, [image_shape[1] * scale, image_shape[2] * scale])

    layers.append(
        tf.keras.layers.Lambda(
            resize_nearest_neighbor,
            name=name + 'nearest_neighbor_upsampling_x{}'.format(scale)))
  else:

    def nearest_neighbor_upsampling(image):
      return ops.nearest_neighbor_upsampling(image, scale=scale)

    layers.append(
        tf.keras.layers.Lambda(
            nearest_neighbor_upsampling,
            name=name + 'nearest_neighbor_upsampling_x{}'.format(scale)))

  return layers


def create_resample_feature_map_ops(input_scale_factor, output_scale_factor,
                                    downsample_method, use_native_resize_op,
                                    conv_hyperparams, is_training,
                                    freeze_batchnorm, name):
  """Creates Keras layers for downsampling or upsampling feature maps.

  Args:
    input_scale_factor: Int. Scale factor of the input feature map. For example,
      for a feature pyramid where each successive level halves its spatial
      resolution, the scale factor of a level is 2^level. The input and output
      scale factors are used to compute the scale for upsampling or downsamling,
      so they should be evenly divisible.
    output_scale_factor: Int. Scale factor of the output feature map. See
      input_scale_factor for additional details.
    downsample_method: String. The method used for downsampling. See
      create_downsample_feature_map_ops for details on supported methods.
    use_native_resize_op: If True, uses tf.image.resize_nearest_neighbor op for
      the upsampling process instead of reshape and broadcasting implementation.
      See create_upsample_feature_map_ops for details.
    conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object
      containing hyperparameters for convolution ops.
    is_training: Bool. Whether the feature generator is in training mode.
    freeze_batchnorm: Bool. Whether to freeze batch norm parameters during
      training or not. When training with a small batch size (e.g. 1), it is
      desirable to freeze batch norm update and use pretrained batch norm
      params.
    name: String. The name used to prefix the constructed layers.

  Returns:
    A list of Keras layers which will downsample or upsample input feature maps
    to match the desired output feature map scale.
  """
  if input_scale_factor < output_scale_factor:
    if output_scale_factor % input_scale_factor != 0:
      raise ValueError('Invalid scale factor: input scale 1/{} not divisible by'
                       'output scale 1/{}'.format(input_scale_factor,
                                                  output_scale_factor))
    scale = output_scale_factor // input_scale_factor
    return create_downsample_feature_map_ops(scale, downsample_method,
                                             conv_hyperparams, is_training,
                                             freeze_batchnorm, name)
  elif input_scale_factor > output_scale_factor:
    if input_scale_factor % output_scale_factor != 0:
      raise ValueError('Invalid scale factor: input scale 1/{} not a divisor of'
                       'output scale 1/{}'.format(input_scale_factor,
                                                  output_scale_factor))
    scale = input_scale_factor // output_scale_factor
    return create_upsample_feature_map_ops(scale, use_native_resize_op, name)
  else:
    return []


# TODO(aom): Add tests for this module in a followup CL.
class BiFPNCombineLayer(tf.keras.layers.Layer):
  """Combines multiple input feature maps into a single output feature map.

  A Keras layer which combines multiple input feature maps into a single output
  feature map, according to the desired combination method. Options for
  combining feature maps include simple summation, or several types of weighted
  sums using learned weights for each input feature map. These include
  'weighted_sum', 'attention', and 'fast_attention'. For more details, see the
  EfficientDet paper by Tan et al, see arxiv.org/abs/1911.09070.

  Specifically, this layer takes a list of tensors as input, all of the same
  shape, and returns a single tensor, also of the same shape.
  """

  def __init__(self, combine_method, **kwargs):
    """Constructor.

    Args:
      combine_method: String. The method used to combine the input feature maps
        into a single output feature map. One of 'sum', 'weighted_sum',
        'attention', or 'fast_attention'.
      **kwargs: Additional Keras layer arguments.
    """
    super(BiFPNCombineLayer, self).__init__(**kwargs)
    self.combine_method = combine_method

  def _combine_weighted_sum(self, inputs):
    return tf.squeeze(
        tf.linalg.matmul(tf.stack(inputs, axis=-1), self.per_input_weights),
        axis=[-1])

  def _combine_attention(self, inputs):
    normalized_weights = tf.nn.softmax(self.per_input_weights)
    return tf.squeeze(
        tf.linalg.matmul(tf.stack(inputs, axis=-1), normalized_weights),
        axis=[-1])

  def _combine_fast_attention(self, inputs):
    weights_non_neg = tf.nn.relu(self.per_input_weights)
    normalizer = tf.reduce_sum(weights_non_neg) + 0.0001
    normalized_weights = weights_non_neg / normalizer
    return tf.squeeze(
        tf.linalg.matmul(tf.stack(inputs, axis=-1), normalized_weights),
        axis=[-1])

  def build(self, input_shape):
    if not isinstance(input_shape, list):
      raise ValueError('A BiFPN combine layer should be called '
                       'on a list of inputs.')
    if len(input_shape) < 2:
      raise ValueError('A BiFPN combine layer should be called '
                       'on a list of at least 2 inputs. '
                       'Got ' + str(len(input_shape)) + ' inputs.')
    if self.combine_method == 'sum':
      self._combine_op = tf.keras.layers.Add()
    elif self.combine_method == 'weighted_sum':
      self._combine_op = self._combine_weighted_sum
    elif self.combine_method == 'attention':
      self._combine_op = self._combine_attention
    elif self.combine_method == 'fast_attention':
      self._combine_op = self._combine_fast_attention
    else:
      raise ValueError('Unknown combine type: {}'.format(self.combine_method))
    if self.combine_method in {'weighted_sum', 'attention', 'fast_attention'}:
      self.per_input_weights = self.add_weight(
          name='bifpn_combine_weights',
          shape=(len(input_shape), 1),
          initializer='ones',
          trainable=True)
    super(BiFPNCombineLayer, self).build(input_shape)

  def call(self, inputs):
    """Combines multiple input feature maps into a single output feature map.

    Executed when calling the `.__call__` method on input.

    Args:
      inputs: A list of tensors where all tensors have the same shape, [batch,
        height_i, width_i, depth_i].

    Returns:
      A single tensor, with the same shape as the input tensors,
        [batch, height_i, width_i, depth_i].
    """
    return self._combine_op(inputs)

  def compute_output_shape(self, input_shape):
    output_shape = input_shape[0]
    for i in range(1, len(input_shape)):
      if input_shape[i] != output_shape:
        raise ValueError(
            'Inputs could not be combined. Shapes should match, '
            'but input_shape[0] is {} while input_shape[{}] is {}'.format(
                output_shape, i, input_shape[i]))
