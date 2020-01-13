# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition for Gated Separable 3D network (S3D-G).

The network architecture is proposed by:
  Saining Xie, Chen Sun, Jonathan Huang, Zhuowen Tu and Kevin Murphy,
  Rethinking Spatiotemporal Feature Learning For Video Understanding.
  https://arxiv.org/abs/1712.04851.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers

from nets import i3d_utils

# pylint: disable=g-long-lambda
trunc_normal = lambda stddev: tf.compat.v1.truncated_normal_initializer(
    0.0, stddev)
conv3d_spatiotemporal = i3d_utils.conv3d_spatiotemporal
inception_block_v1_3d = i3d_utils.inception_block_v1_3d

# Orignaly, arg_scope = slim.arg_scope and layers = slim, now switch to more
# update-to-date tf.contrib.* API.
arg_scope = contrib_framework.arg_scope
layers = contrib_layers


def s3dg_arg_scope(weight_decay=1e-7,
                   batch_norm_decay=0.999,
                   batch_norm_epsilon=0.001):
  """Defines default arg_scope for S3D-G.

  Args:
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.

  Returns:
    sc: An arg_scope to use for the models.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # Turns off fused batch norm.
      'fused': False,
      # collection containing the moving mean and moving variance.
      'variables_collections': {
          'beta': None,
          'gamma': None,
          'moving_mean': ['moving_vars'],
          'moving_variance': ['moving_vars'],
      }
  }

  with arg_scope(
      [layers.conv3d, conv3d_spatiotemporal],
      weights_regularizer=layers.l2_regularizer(weight_decay),
      activation_fn=tf.nn.relu,
      normalizer_fn=layers.batch_norm,
      normalizer_params=batch_norm_params):
    with arg_scope([conv3d_spatiotemporal], separable=True) as sc:
      return sc


def self_gating(input_tensor, scope, data_format='NDHWC'):
  """Feature gating as used in S3D-G.

  Transforms the input features by aggregating features from all
  spatial and temporal locations, and applying gating conditioned
  on the aggregated features. More details can be found at:
  https://arxiv.org/abs/1712.04851

  Args:
    input_tensor: A 5-D float tensor of size [batch_size, num_frames,
      height, width, channels].
    scope: scope for `variable_scope`.
    data_format: An optional string from: "NDHWC", "NCDHW". Defaults to "NDHWC".
      The data format of the input and output data. With the default format
      "NDHWC", the data is stored in the order of: [batch, in_depth, in_height,
      in_width, in_channels]. Alternatively, the format could be "NCDHW", the
      data storage order is:
      [batch, in_channels, in_depth, in_height, in_width].

  Returns:
    A tensor with the same shape as input_tensor.
  """

  index_c = data_format.index('C')
  index_d = data_format.index('D')
  index_h = data_format.index('H')
  index_w = data_format.index('W')
  input_shape = input_tensor.get_shape().as_list()
  t = input_shape[index_d]
  w = input_shape[index_w]
  h = input_shape[index_h]
  num_channels = input_shape[index_c]

  spatiotemporal_average = layers.avg_pool3d(
      input_tensor, [t, w, h],
      stride=1,
      data_format=data_format,
      scope=scope + '/self_gating/avg_pool3d')

  weights = layers.conv3d(
      spatiotemporal_average,
      num_channels, [1, 1, 1],
      activation_fn=None,
      normalizer_fn=None,
      biases_initializer=None,
      data_format=data_format,
      weights_initializer=trunc_normal(0.01),
      scope=scope + '/self_gating/transformer_W')

  tile_multiples = [1, t, w, h]
  tile_multiples.insert(index_c, 1)
  weights = tf.tile(weights, tile_multiples)
  weights = tf.nn.sigmoid(weights)

  return tf.multiply(weights, input_tensor)


def s3dg_base(inputs,
              first_temporal_kernel_size=3,
              temporal_conv_startat='Conv2d_2c_3x3',
              gating_startat='Conv2d_2c_3x3',
              final_endpoint='Mixed_5c',
              min_depth=16,
              depth_multiplier=1.0,
              data_format='NDHWC',
              scope='InceptionV1'):
  """Defines the I3D/S3DG base architecture.

  Note that we use the names as defined in Inception V1 to facilitate checkpoint
  conversion from an image-trained Inception V1 checkpoint to I3D checkpoint.

  Args:
    inputs: A 5-D float tensor of size [batch_size, num_frames, height, width,
      channels].
    first_temporal_kernel_size: Specifies the temporal kernel size for the first
      conv3d filter. A larger value slows down the model but provides little
      accuracy improvement. The default is 7 in the original I3D and S3D-G but 3
      gives better performance. Must be set to one of 1, 3, 5 or 7.
    temporal_conv_startat: Specifies the first conv block to use 3D or separable
      3D convs rather than 2D convs (implemented as [1, k, k] 3D conv). This is
      used to construct the inverted pyramid models. 'Conv2d_2c_3x3' is the
      first valid block to use separable 3D convs. If provided block name is
      not present, all valid blocks will use separable 3D convs. Note that
      'Conv2d_1a_7x7' cannot be made into a separable 3D conv, but can be made
      into a 2D or 3D conv using the `first_temporal_kernel_size` option.
    gating_startat: Specifies the first conv block to use self gating.
      'Conv2d_2c_3x3' is the first valid block to use self gating. If provided
      block name is not present, all valid blocks will use separable 3D convs.
    final_endpoint: Specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
      'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
      'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    data_format: An optional string from: "NDHWC", "NCDHW". Defaults to "NDHWC".
      The data format of the input and output data. With the default format
      "NDHWC", the data is stored in the order of: [batch, in_depth, in_height,
      in_width, in_channels]. Alternatively, the format could be "NCDHW", the
      data storage order is:
      [batch, in_channels, in_depth, in_height, in_width].
    scope: Optional variable_scope.

  Returns:
    A dictionary from components of the network to the corresponding activation.

  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values, or
      if depth_multiplier <= 0.
  """

  assert data_format in ['NDHWC', 'NCDHW']
  end_points = {}
  t = 1
  # For inverted pyramid models, we start with gating switched off.
  use_gating = False
  self_gating_fn = None
  def gating_fn(inputs, scope):
    return self_gating(inputs, scope, data_format=data_format)

  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  with tf.compat.v1.variable_scope(scope, 'InceptionV1', [inputs]):
    with arg_scope([layers.conv3d], weights_initializer=trunc_normal(0.01)):
      with arg_scope(
          [layers.conv3d, layers.max_pool3d, conv3d_spatiotemporal],
          stride=1,
          data_format=data_format,
          padding='SAME'):
        # batch_size x 32 x 112 x 112 x 64
        end_point = 'Conv2d_1a_7x7'
        if first_temporal_kernel_size not in [1, 3, 5, 7]:
          raise ValueError(
              'first_temporal_kernel_size can only be 1, 3, 5 or 7.')
        # Separable conv is slow when used at first conv layer.
        net = conv3d_spatiotemporal(
            inputs,
            depth(64), [first_temporal_kernel_size, 7, 7],
            stride=2,
            separable=False,
            scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points
        # batch_size x 32 x 56 x 56 x 64
        end_point = 'MaxPool_2a_3x3'
        net = layers.max_pool3d(
            net, [1, 3, 3], stride=[1, 2, 2], scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points
        # batch_size x 32 x 56 x 56 x 64
        end_point = 'Conv2d_2b_1x1'
        net = layers.conv3d(net, depth(64), [1, 1, 1], scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points
        # batch_size x 32 x 56 x 56 x 192
        end_point = 'Conv2d_2c_3x3'
        if temporal_conv_startat == end_point:
          t = 3
        if gating_startat == end_point:
          use_gating = True
          self_gating_fn = gating_fn
        net = conv3d_spatiotemporal(net, depth(192), [t, 3, 3], scope=end_point)
        if use_gating:
          net = self_gating(net, scope=end_point, data_format=data_format)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points
        # batch_size x 32 x 28 x 28 x 192
        end_point = 'MaxPool_3a_3x3'
        net = layers.max_pool3d(
            net, [1, 3, 3], stride=[1, 2, 2], scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        # batch_size x 32 x 28 x 28 x 256
        end_point = 'Mixed_3b'
        if temporal_conv_startat == end_point:
          t = 3
        if gating_startat == end_point:
          use_gating = True
          self_gating_fn = gating_fn
        net = inception_block_v1_3d(
            net,
            num_outputs_0_0a=depth(64),
            num_outputs_1_0a=depth(96),
            num_outputs_1_0b=depth(128),
            num_outputs_2_0a=depth(16),
            num_outputs_2_0b=depth(32),
            num_outputs_3_0b=depth(32),
            temporal_kernel_size=t,
            self_gating_fn=self_gating_fn,
            data_format=data_format,
            scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        end_point = 'Mixed_3c'
        if temporal_conv_startat == end_point:
          t = 3
        if gating_startat == end_point:
          use_gating = True
          self_gating_fn = gating_fn
        net = inception_block_v1_3d(
            net,
            num_outputs_0_0a=depth(128),
            num_outputs_1_0a=depth(128),
            num_outputs_1_0b=depth(192),
            num_outputs_2_0a=depth(32),
            num_outputs_2_0b=depth(96),
            num_outputs_3_0b=depth(64),
            temporal_kernel_size=t,
            self_gating_fn=self_gating_fn,
            data_format=data_format,
            scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        end_point = 'MaxPool_4a_3x3'
        net = layers.max_pool3d(
            net, [3, 3, 3], stride=[2, 2, 2], scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        # batch_size x 16 x 14 x 14 x 512
        end_point = 'Mixed_4b'
        if temporal_conv_startat == end_point:
          t = 3
        if gating_startat == end_point:
          use_gating = True
          self_gating_fn = gating_fn
        net = inception_block_v1_3d(
            net,
            num_outputs_0_0a=depth(192),
            num_outputs_1_0a=depth(96),
            num_outputs_1_0b=depth(208),
            num_outputs_2_0a=depth(16),
            num_outputs_2_0b=depth(48),
            num_outputs_3_0b=depth(64),
            temporal_kernel_size=t,
            self_gating_fn=self_gating_fn,
            data_format=data_format,
            scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        # batch_size x 16 x 14 x 14 x 512
        end_point = 'Mixed_4c'
        if temporal_conv_startat == end_point:
          t = 3
        if gating_startat == end_point:
          use_gating = True
          self_gating_fn = gating_fn
        net = inception_block_v1_3d(
            net,
            num_outputs_0_0a=depth(160),
            num_outputs_1_0a=depth(112),
            num_outputs_1_0b=depth(224),
            num_outputs_2_0a=depth(24),
            num_outputs_2_0b=depth(64),
            num_outputs_3_0b=depth(64),
            temporal_kernel_size=t,
            self_gating_fn=self_gating_fn,
            data_format=data_format,
            scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        # batch_size x 16 x 14 x 14 x 512
        end_point = 'Mixed_4d'
        if temporal_conv_startat == end_point:
          t = 3
        if gating_startat == end_point:
          use_gating = True
          self_gating_fn = gating_fn
        net = inception_block_v1_3d(
            net,
            num_outputs_0_0a=depth(128),
            num_outputs_1_0a=depth(128),
            num_outputs_1_0b=depth(256),
            num_outputs_2_0a=depth(24),
            num_outputs_2_0b=depth(64),
            num_outputs_3_0b=depth(64),
            temporal_kernel_size=t,
            self_gating_fn=self_gating_fn,
            data_format=data_format,
            scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        # batch_size x 16 x 14 x 14 x 528
        end_point = 'Mixed_4e'
        if temporal_conv_startat == end_point:
          t = 3
        if gating_startat == end_point:
          use_gating = True
          self_gating_fn = gating_fn
        net = inception_block_v1_3d(
            net,
            num_outputs_0_0a=depth(112),
            num_outputs_1_0a=depth(144),
            num_outputs_1_0b=depth(288),
            num_outputs_2_0a=depth(32),
            num_outputs_2_0b=depth(64),
            num_outputs_3_0b=depth(64),
            temporal_kernel_size=t,
            self_gating_fn=self_gating_fn,
            data_format=data_format,
            scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        # batch_size x 16 x 14 x 14 x 832
        end_point = 'Mixed_4f'
        if temporal_conv_startat == end_point:
          t = 3
        if gating_startat == end_point:
          use_gating = True
          self_gating_fn = gating_fn
        net = inception_block_v1_3d(
            net,
            num_outputs_0_0a=depth(256),
            num_outputs_1_0a=depth(160),
            num_outputs_1_0b=depth(320),
            num_outputs_2_0a=depth(32),
            num_outputs_2_0b=depth(128),
            num_outputs_3_0b=depth(128),
            temporal_kernel_size=t,
            self_gating_fn=self_gating_fn,
            data_format=data_format,
            scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        end_point = 'MaxPool_5a_2x2'
        net = layers.max_pool3d(
            net, [2, 2, 2], stride=[2, 2, 2], scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        # batch_size x 8 x 7 x 7 x 832
        end_point = 'Mixed_5b'
        if temporal_conv_startat == end_point:
          t = 3
        if gating_startat == end_point:
          use_gating = True
          self_gating_fn = gating_fn
        net = inception_block_v1_3d(
            net,
            num_outputs_0_0a=depth(256),
            num_outputs_1_0a=depth(160),
            num_outputs_1_0b=depth(320),
            num_outputs_2_0a=depth(32),
            num_outputs_2_0b=depth(128),
            num_outputs_3_0b=depth(128),
            temporal_kernel_size=t,
            self_gating_fn=self_gating_fn,
            data_format=data_format,
            scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points

        # batch_size x 8 x 7 x 7 x 1024
        end_point = 'Mixed_5c'
        if temporal_conv_startat == end_point:
          t = 3
        if gating_startat == end_point:
          use_gating = True
          self_gating_fn = gating_fn
        net = inception_block_v1_3d(
            net,
            num_outputs_0_0a=depth(384),
            num_outputs_1_0a=depth(192),
            num_outputs_1_0b=depth(384),
            num_outputs_2_0a=depth(48),
            num_outputs_2_0b=depth(128),
            num_outputs_3_0b=depth(128),
            temporal_kernel_size=t,
            self_gating_fn=self_gating_fn,
            data_format=data_format,
            scope=end_point)
        end_points[end_point] = net
        if final_endpoint == end_point:
          return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def s3dg(inputs,
         num_classes=1000,
         first_temporal_kernel_size=3,
         temporal_conv_startat='Conv2d_2c_3x3',
         gating_startat='Conv2d_2c_3x3',
         final_endpoint='Mixed_5c',
         min_depth=16,
         depth_multiplier=1.0,
         dropout_keep_prob=0.8,
         is_training=True,
         prediction_fn=layers.softmax,
         spatial_squeeze=True,
         reuse=None,
         data_format='NDHWC',
         scope='InceptionV1'):
  """Defines the S3D-G architecture.

  The default image size used to train this network is 224x224.

  Args:
    inputs: A 5-D float tensor of size [batch_size, num_frames, height, width,
      channels].
    num_classes: number of predicted classes.
    first_temporal_kernel_size: Specifies the temporal kernel size for the first
      conv3d filter. A larger value slows down the model but provides little
      accuracy improvement. Must be set to one of 1, 3, 5 or 7.
    temporal_conv_startat: Specifies the first conv block to use separable 3D
      convs rather than 2D convs (implemented as [1, k, k] 3D conv). This is
      used to construct the inverted pyramid models. 'Conv2d_2c_3x3' is the
      first valid block to use separable 3D convs. If provided block name is
      not present, all valid blocks will use separable 3D convs.
    gating_startat: Specifies the first conv block to use self gating.
      'Conv2d_2c_3x3' is the first valid block to use self gating. If provided
      block name is not present, all valid blocks will use separable 3D convs.
    final_endpoint: Specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c',
      'MaxPool_4a_3x3', 'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e',
      'Mixed_4f', 'MaxPool_5a_2x2', 'Mixed_5b', 'Mixed_5c']
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    dropout_keep_prob: the percentage of activation values that are retained.
    is_training: whether is training or not.
    prediction_fn: a function to get predictions out of logits.
    spatial_squeeze: if True, logits is of shape is [B, C], if false logits is
        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    data_format: An optional string from: "NDHWC", "NCDHW". Defaults to "NDHWC".
      The data format of the input and output data. With the default format
      "NDHWC", the data is stored in the order of: [batch, in_depth, in_height,
      in_width, in_channels]. Alternatively, the format could be "NCDHW", the
      data storage order is:
      [batch, in_channels, in_depth, in_height, in_width].
    scope: Optional variable_scope.

  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, num_classes]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
  assert data_format in ['NDHWC', 'NCDHW']
  # Final pooling and prediction
  with tf.compat.v1.variable_scope(
      scope, 'InceptionV1', [inputs, num_classes], reuse=reuse) as scope:
    with arg_scope(
        [layers.batch_norm, layers.dropout], is_training=is_training):
      net, end_points = s3dg_base(
          inputs,
          first_temporal_kernel_size=first_temporal_kernel_size,
          temporal_conv_startat=temporal_conv_startat,
          gating_startat=gating_startat,
          final_endpoint=final_endpoint,
          min_depth=min_depth,
          depth_multiplier=depth_multiplier,
          data_format=data_format,
          scope=scope)
      with tf.compat.v1.variable_scope('Logits'):
        if data_format.startswith('NC'):
          net = tf.transpose(a=net, perm=[0, 2, 3, 4, 1])
        kernel_size = i3d_utils.reduced_kernel_size_3d(net, [2, 7, 7])
        net = layers.avg_pool3d(
            net,
            kernel_size,
            stride=1,
            data_format='NDHWC',
            scope='AvgPool_0a_7x7')
        net = layers.dropout(net, dropout_keep_prob, scope='Dropout_0b')
        logits = layers.conv3d(
            net,
            num_classes, [1, 1, 1],
            activation_fn=None,
            normalizer_fn=None,
            data_format='NDHWC',
            scope='Conv2d_0c_1x1')
        # Temporal average pooling.
        logits = tf.reduce_mean(input_tensor=logits, axis=1)
        if spatial_squeeze:
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points


s3dg.default_image_size = 224
