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
"""Defines the CycleGAN generator and discriminator networks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow.compat.v1 as tf
import tf_slim as slim
from tensorflow.python.framework import tensor_util


def cyclegan_arg_scope(instance_norm_center=True,
                       instance_norm_scale=True,
                       instance_norm_epsilon=0.001,
                       weights_init_stddev=0.02,
                       weight_decay=0.0):
  """Returns a default argument scope for all generators and discriminators.

  Args:
    instance_norm_center: Whether instance normalization applies centering.
    instance_norm_scale: Whether instance normalization applies scaling.
    instance_norm_epsilon: Small float added to the variance in the instance
      normalization to avoid dividing by zero.
    weights_init_stddev: Standard deviation of the random values to initialize
      the convolution kernels with.
    weight_decay: Magnitude of weight decay applied to all convolution kernel
      variables of the generator.

  Returns:
    An arg-scope.
  """
  instance_norm_params = {
      'center': instance_norm_center,
      'scale': instance_norm_scale,
      'epsilon': instance_norm_epsilon,
  }

  weights_regularizer = None
  if weight_decay and weight_decay > 0.0:
    weights_regularizer = slim.l2_regularizer(weight_decay)

  with slim.arg_scope(
      [slim.conv2d],
      normalizer_fn=slim.instance_norm,
      normalizer_params=instance_norm_params,
      weights_initializer=tf.random_normal_initializer(
          0, weights_init_stddev),
      weights_regularizer=weights_regularizer) as sc:
    return sc


def cyclegan_upsample(net, num_outputs, stride, method='conv2d_transpose',
                      pad_mode='REFLECT', align_corners=False):
  """Upsamples the given inputs.

  Args:
    net: A Tensor of size [batch_size, height, width, filters].
    num_outputs: The number of output filters.
    stride: A list of 2 scalars or a 1x2 Tensor indicating the scale,
      relative to the inputs, of the output dimensions. For example, if kernel
      size is [2, 3], then the output height and width will be twice and three
      times the input size.
    method: The upsampling method: 'nn_upsample_conv', 'bilinear_upsample_conv',
      or 'conv2d_transpose'.
    pad_mode: mode for tf.pad, one of "CONSTANT", "REFLECT", or "SYMMETRIC".
    align_corners: option for method, 'bilinear_upsample_conv'. If true, the
      centers of the 4 corner pixels of the input and output tensors are
      aligned, preserving the values at the corner pixels.

  Returns:
    A Tensor which was upsampled using the specified method.

  Raises:
    ValueError: if `method` is not recognized.
  """
  with tf.variable_scope('upconv'):
    net_shape = tf.shape(input=net)
    height = net_shape[1]
    width = net_shape[2]

    # Reflection pad by 1 in spatial dimensions (axes 1, 2 = h, w) to make a 3x3
    # 'valid' convolution produce an output with the same dimension as the
    # input.
    spatial_pad_1 = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])

    if method == 'nn_upsample_conv':
      net = tf.image.resize(
          net, [stride[0] * height, stride[1] * width],
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      net = tf.pad(tensor=net, paddings=spatial_pad_1, mode=pad_mode)
      net = slim.conv2d(net, num_outputs, kernel_size=[3, 3], padding='valid')
    elif method == 'bilinear_upsample_conv':
      net = tf.image.resize_bilinear(
          net, [stride[0] * height, stride[1] * width],
          align_corners=align_corners)
      net = tf.pad(tensor=net, paddings=spatial_pad_1, mode=pad_mode)
      net = slim.conv2d(net, num_outputs, kernel_size=[3, 3], padding='valid')
    elif method == 'conv2d_transpose':
      # This corrects 1 pixel offset for images with even width and height.
      # conv2d is left aligned and conv2d_transpose is right aligned for even
      # sized images (while doing 'SAME' padding).
      # Note: This doesn't reflect actual model in paper.
      net = slim.conv2d_transpose(
          net, num_outputs, kernel_size=[3, 3], stride=stride, padding='valid')
      net = net[:, 1:, 1:, :]
    else:
      raise ValueError('Unknown method: [%s]' % method)

    return net


def _dynamic_or_static_shape(tensor):
  shape = tf.shape(input=tensor)
  static_shape = tensor_util.constant_value(shape)
  return static_shape if static_shape is not None else shape


def cyclegan_generator_resnet(images,
                              arg_scope_fn=cyclegan_arg_scope,
                              num_resnet_blocks=6,
                              num_filters=64,
                              upsample_fn=cyclegan_upsample,
                              kernel_size=3,
                              tanh_linear_slope=0.0,
                              is_training=False):
  """Defines the cyclegan resnet network architecture.

  As closely as possible following
  https://github.com/junyanz/CycleGAN/blob/master/models/architectures.lua#L232

  FYI: This network requires input height and width to be divisible by 4 in
  order to generate an output with shape equal to input shape. Assertions will
  catch this if input dimensions are known at graph construction time, but
  there's no protection if unknown at graph construction time (you'll see an
  error).

  Args:
    images: Input image tensor of shape [batch_size, h, w, 3].
    arg_scope_fn: Function to create the global arg_scope for the network.
    num_resnet_blocks: Number of ResNet blocks in the middle of the generator.
    num_filters: Number of filters of the first hidden layer.
    upsample_fn: Upsampling function for the decoder part of the generator.
    kernel_size: Size w or list/tuple [h, w] of the filter kernels for all inner
      layers.
    tanh_linear_slope: Slope of the linear function to add to the tanh over the
      logits.
    is_training: Whether the network is created in training mode or inference
      only mode. Not actually needed, just for compliance with other generator
      network functions.

  Returns:
    A `Tensor` representing the model output and a dictionary of model end
      points.

  Raises:
    ValueError: If the input height or width is known at graph construction time
      and not a multiple of 4.
  """
  # Neither dropout nor batch norm -> dont need is_training
  del is_training

  end_points = {}

  input_size = images.shape.as_list()
  height, width = input_size[1], input_size[2]
  if height and height % 4 != 0:
    raise ValueError('The input height must be a multiple of 4.')
  if width and width % 4 != 0:
    raise ValueError('The input width must be a multiple of 4.')
  num_outputs = input_size[3]

  if not isinstance(kernel_size, (list, tuple)):
    kernel_size = [kernel_size, kernel_size]

  kernel_height = kernel_size[0]
  kernel_width = kernel_size[1]
  pad_top = (kernel_height - 1) // 2
  pad_bottom = kernel_height // 2
  pad_left = (kernel_width - 1) // 2
  pad_right = kernel_width // 2
  paddings = np.array(
      [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
      dtype=np.int32)
  spatial_pad_3 = np.array([[0, 0], [3, 3], [3, 3], [0, 0]])

  with slim.arg_scope(arg_scope_fn()):

    ###########
    # Encoder #
    ###########
    with tf.variable_scope('input'):
      # 7x7 input stage
      net = tf.pad(tensor=images, paddings=spatial_pad_3, mode='REFLECT')
      net = slim.conv2d(net, num_filters, kernel_size=[7, 7], padding='VALID')
      end_points['encoder_0'] = net

    with tf.variable_scope('encoder'):
      with slim.arg_scope([slim.conv2d],
                          kernel_size=kernel_size,
                          stride=2,
                          activation_fn=tf.nn.relu,
                          padding='VALID'):

        net = tf.pad(tensor=net, paddings=paddings, mode='REFLECT')
        net = slim.conv2d(net, num_filters * 2)
        end_points['encoder_1'] = net
        net = tf.pad(tensor=net, paddings=paddings, mode='REFLECT')
        net = slim.conv2d(net, num_filters * 4)
        end_points['encoder_2'] = net

    ###################
    # Residual Blocks #
    ###################
    with tf.variable_scope('residual_blocks'):
      with slim.arg_scope([slim.conv2d],
                          kernel_size=kernel_size,
                          stride=1,
                          activation_fn=tf.nn.relu,
                          padding='VALID'):
        for block_id in xrange(num_resnet_blocks):
          with tf.variable_scope('block_{}'.format(block_id)):
            res_net = tf.pad(tensor=net, paddings=paddings, mode='REFLECT')
            res_net = slim.conv2d(res_net, num_filters * 4)
            res_net = tf.pad(tensor=res_net, paddings=paddings, mode='REFLECT')
            res_net = slim.conv2d(res_net, num_filters * 4, activation_fn=None)
            net += res_net

            end_points['resnet_block_%d' % block_id] = net

    ###########
    # Decoder #
    ###########
    with tf.variable_scope('decoder'):

      with slim.arg_scope([slim.conv2d],
                          kernel_size=kernel_size,
                          stride=1,
                          activation_fn=tf.nn.relu):

        with tf.variable_scope('decoder1'):
          net = upsample_fn(net, num_outputs=num_filters * 2, stride=[2, 2])
        end_points['decoder1'] = net

        with tf.variable_scope('decoder2'):
          net = upsample_fn(net, num_outputs=num_filters, stride=[2, 2])
        end_points['decoder2'] = net

    with tf.variable_scope('output'):
      net = tf.pad(tensor=net, paddings=spatial_pad_3, mode='REFLECT')
      logits = slim.conv2d(
          net,
          num_outputs, [7, 7],
          activation_fn=None,
          normalizer_fn=None,
          padding='valid')
      logits = tf.reshape(logits, _dynamic_or_static_shape(images))

      end_points['logits'] = logits
      end_points['predictions'] = tf.tanh(logits) + logits * tanh_linear_slope

  return end_points['predictions'], end_points
