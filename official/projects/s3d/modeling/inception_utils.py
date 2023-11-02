# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Contains modules related to Inception networks."""
from typing import Callable, Dict, Optional, Sequence, Set, Text, Tuple, Type, Union

import tensorflow as tf, tf_keras

from official.modeling import tf_utils
from official.projects.s3d.modeling import net_utils
from official.vision.modeling.layers import nn_blocks_3d

INCEPTION_V1_CONV_ENDPOINTS = [
    'Conv2d_1a_7x7', 'Conv2d_2c_3x3', 'Mixed_3b', 'Mixed_3c', 'Mixed_4b',
    'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_4f', 'Mixed_5b', 'Mixed_5c'
]

# Mapping from endpoint to branch filters. The endpoint shapes below are
# specific for input 64x224x224.
INCEPTION_V1_ARCH_SKELETON = [
    ('Mixed_3b', [[64], [96, 128], [16, 32], [32]]),  # 32x28x28x256
    ('Mixed_3c', [[128], [128, 192], [32, 96], [64]]),  # 32x28x28x480
    ('MaxPool_4a_3x3', [[3, 3, 3], [2, 2, 2]]),  # 16x14x14x480
    ('Mixed_4b', [[192], [96, 208], [16, 48], [64]]),  # 16x14x14x512
    ('Mixed_4c', [[160], [112, 224], [24, 64], [64]]),  # 16x14x14x512
    ('Mixed_4d', [[128], [128, 256], [24, 64], [64]]),  # 16x14x14x512
    ('Mixed_4e', [[112], [144, 288], [32, 64], [64]]),  # 16x14x14x528
    ('Mixed_4f', [[256], [160, 320], [32, 128], [128]]),  # 16x14x14x832
    ('MaxPool_5a_2x2', [[2, 2, 2], [2, 2, 2]]),  # 8x7x7x832
    ('Mixed_5b', [[256], [160, 320], [32, 128], [128]]),  # 8x7x7x832
    ('Mixed_5c', [[384], [192, 384], [48, 128], [128]]),  # 8x7x7x1024
]

INCEPTION_V1_LOCAL_SKELETON = [
    ('MaxPool_5a_2x2_local', [[2, 2, 2], [2, 2, 2]]),  # 8x7x7x832
    ('Mixed_5b_local', [[256], [160, 320], [32, 128], [128]]),  # 8x7x7x832
    ('Mixed_5c_local', [[384], [192, 384], [48, 128], [128]]),  # 8x7x7x1024
]

initializers = tf_keras.initializers
regularizers = tf_keras.regularizers


def inception_v1_stem_cells(
    inputs: tf.Tensor,
    depth_multiplier: float,
    final_endpoint: Text,
    temporal_conv_endpoints: Optional[Set[Text]] = None,
    self_gating_endpoints: Optional[Set[Text]] = None,
    temporal_conv_type: Text = '3d',
    first_temporal_kernel_size: int = 7,
    use_sync_bn: bool = False,
    norm_momentum: float = 0.999,
    norm_epsilon: float = 0.001,
    temporal_conv_initializer: Union[
        Text, initializers.Initializer] = initializers.TruncatedNormal(
            mean=0.0, stddev=0.01),
    kernel_initializer: Union[Text,
                              initializers.Initializer] = 'truncated_normal',
    kernel_regularizer: Union[Text, regularizers.Regularizer] = 'l2',
    parameterized_conv_layer: Type[
        net_utils.ParameterizedConvLayer] = net_utils.ParameterizedConvLayer,
    layer_naming_fn: Callable[[Text], Text] = lambda end_point: None,
) -> Tuple[tf.Tensor, Dict[Text, tf.Tensor]]:
  """Stem cells used in the original I3D/S3D model.

  Args:
    inputs: A 5-D float tensor of size [batch_size, num_frames, height, width,
      channels].
    depth_multiplier: A float to reduce/increase number of channels.
    final_endpoint: Specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',
      'Conv2d_2c_3x3', 'MaxPool_3a_3x3'].
    temporal_conv_endpoints: Specifies the endpoints where to perform temporal
      convolution.
    self_gating_endpoints: Specifies the endpoints where to perform self gating.
    temporal_conv_type: '3d' for I3D model and '2+1d' for S3D model.
    first_temporal_kernel_size: temporal kernel size of the first convolution
      layer.
    use_sync_bn: If True, use synchronized batch normalization.
    norm_momentum: A `float` of normalization momentum for the moving average.
    norm_epsilon: A `float` added to variance to avoid dividing by zero.
    temporal_conv_initializer: Weight initializer for temporal convolution
      inside the cell. It only applies to 2+1d and 1+2d cases.
    kernel_initializer: Weight initializer for convolutional layers other than
      temporal convolution.
    kernel_regularizer: Weight regularizer for all convolutional layers.
    parameterized_conv_layer: class for parameterized conv layer.
    layer_naming_fn: function to customize conv / pooling layer names given
      endpoint name of the block. This is mainly used to creat model that is
      compatible with TF1 checkpoints.

  Returns:
    A dictionary from components of the network to the corresponding activation.
  """

  if temporal_conv_endpoints is None:
    temporal_conv_endpoints = set()
  if self_gating_endpoints is None:
    self_gating_endpoints = set()
  if use_sync_bn:
    batch_norm = tf_keras.layers.experimental.SyncBatchNormalization
  else:
    batch_norm = tf_keras.layers.BatchNormalization
  if tf_keras.backend.image_data_format() == 'channels_last':
    bn_axis = -1
  else:
    bn_axis = 1

  end_points = {}
  # batch_size x 32 x 112 x 112 x 64
  end_point = 'Conv2d_1a_7x7'
  net = tf_keras.layers.Conv3D(
      filters=net_utils.apply_depth_multiplier(64, depth_multiplier),
      kernel_size=[first_temporal_kernel_size, 7, 7],
      strides=[2, 2, 2],
      padding='same',
      use_bias=False,
      kernel_initializer=tf_utils.clone_initializer(kernel_initializer),
      kernel_regularizer=kernel_regularizer,
      name=layer_naming_fn(end_point))(
          inputs)
  net = batch_norm(
      axis=bn_axis,
      momentum=norm_momentum,
      epsilon=norm_epsilon,
      scale=False,
      gamma_initializer='ones',
      name=layer_naming_fn(end_point + '/BatchNorm'))(
          net)
  net = tf.nn.relu(net)
  end_points[end_point] = net
  if final_endpoint == end_point:
    return net, end_points
  # batch_size x 32 x 56 x 56 x 64
  end_point = 'MaxPool_2a_3x3'
  net = tf_keras.layers.MaxPool3D(
      pool_size=[1, 3, 3],
      strides=[1, 2, 2],
      padding='same',
      name=layer_naming_fn(end_point))(
          net)
  end_points[end_point] = net
  if final_endpoint == end_point:
    return net, end_points
  # batch_size x 32 x 56 x 56 x 64
  end_point = 'Conv2d_2b_1x1'
  net = tf_keras.layers.Conv3D(
      filters=net_utils.apply_depth_multiplier(64, depth_multiplier),
      strides=[1, 1, 1],
      kernel_size=[1, 1, 1],
      padding='same',
      use_bias=False,
      kernel_initializer=tf_utils.clone_initializer(kernel_initializer),
      kernel_regularizer=kernel_regularizer,
      name=layer_naming_fn(end_point))(
          net)
  net = batch_norm(
      axis=bn_axis,
      momentum=norm_momentum,
      epsilon=norm_epsilon,
      scale=False,
      gamma_initializer='ones',
      name=layer_naming_fn(end_point + '/BatchNorm'))(
          net)
  net = tf.nn.relu(net)
  end_points[end_point] = net
  if final_endpoint == end_point:
    return net, end_points
  # batch_size x 32 x 56 x 56 x 192
  end_point = 'Conv2d_2c_3x3'
  if end_point not in temporal_conv_endpoints:
    temporal_conv_type = '2d'
  net = parameterized_conv_layer(
      conv_type=temporal_conv_type,
      kernel_size=3,
      filters=net_utils.apply_depth_multiplier(192, depth_multiplier),
      strides=[1, 1, 1],
      rates=[1, 1, 1],
      use_sync_bn=use_sync_bn,
      norm_momentum=norm_momentum,
      norm_epsilon=norm_epsilon,
      temporal_conv_initializer=temporal_conv_initializer,
      kernel_initializer=tf_utils.clone_initializer(kernel_initializer),
      kernel_regularizer=kernel_regularizer,
      name=layer_naming_fn(end_point))(
          net)
  if end_point in self_gating_endpoints:
    net = nn_blocks_3d.SelfGating(
        filters=net_utils.apply_depth_multiplier(192, depth_multiplier),
        name=layer_naming_fn(end_point + '/self_gating'))(
            net)
  end_points[end_point] = net
  if final_endpoint == end_point:
    return net, end_points
  # batch_size x 32 x 28 x 28 x 192
  end_point = 'MaxPool_3a_3x3'
  net = tf_keras.layers.MaxPool3D(
      pool_size=[1, 3, 3],
      strides=[1, 2, 2],
      padding='same',
      name=layer_naming_fn(end_point))(
          net)
  end_points[end_point] = net
  return net, end_points


def _construct_branch_3_layers(
    channels: int,
    swap_pool_and_1x1x1: bool,
    pool_type: Text,
    batch_norm_layer: tf_keras.layers.Layer,
    kernel_initializer: Union[Text, initializers.Initializer],
    kernel_regularizer: Union[Text, regularizers.Regularizer],
):
  """Helper function for Branch 3 inside Inception module."""
  kernel_size = [1, 3, 3] if pool_type == '2d' else [3] * 3

  conv = tf_keras.layers.Conv3D(
      filters=channels,
      kernel_size=[1, 1, 1],
      padding='same',
      use_bias=False,
      kernel_initializer=kernel_initializer,
      kernel_regularizer=kernel_regularizer)
  activation = tf_keras.layers.Activation('relu')
  pool = tf_keras.layers.MaxPool3D(
      pool_size=kernel_size, strides=[1, 1, 1], padding='same')
  if swap_pool_and_1x1x1:
    branch_3_layers = [conv, batch_norm_layer, activation, pool]
  else:
    branch_3_layers = [pool, conv, batch_norm_layer, activation]
  return branch_3_layers


class InceptionV1CellLayer(tf_keras.layers.Layer):
  """A single Tensorflow 2 cell used in the original I3D/S3D model."""

  def __init__(
      self,
      branch_filters: Sequence[Sequence[int]],
      conv_type: Text = '3d',
      temporal_dilation_rate: int = 1,
      swap_pool_and_1x1x1: bool = False,
      use_self_gating_on_branch: bool = False,
      use_self_gating_on_cell: bool = False,
      use_sync_bn: bool = False,
      norm_momentum: float = 0.999,
      norm_epsilon: float = 0.001,
      temporal_conv_initializer: Union[
          Text, initializers.Initializer] = initializers.TruncatedNormal(
              mean=0.0, stddev=0.01),
      kernel_initializer: Union[Text,
                                initializers.Initializer] = 'truncated_normal',
      kernel_regularizer: Union[Text, regularizers.Regularizer] = 'l2',
      parameterized_conv_layer: Type[
          net_utils.ParameterizedConvLayer] = net_utils.ParameterizedConvLayer,
      **kwargs):
    """A cell structure inspired by Inception V1.

    Args:
      branch_filters: Specifies the number of filters in four branches
        (Branch_0, Branch_1, Branch_2, Branch_3). Single number for Branch_0 and
        Branch_3. For Branch_1 and Branch_2, each need to specify two numbers,
        one for 1x1x1 and one for 3x3x3.
      conv_type: The type of parameterized convolution. Currently, we support
        '2d', '3d', '2+1d', '1+2d'.
      temporal_dilation_rate: The dilation rate for temporal convolution.
      swap_pool_and_1x1x1: A boolean flag indicates that whether to swap the
        order of convolution and max pooling in Branch_3.
      use_self_gating_on_branch: Whether or not to apply self gating on each
        branch of the inception cell.
      use_self_gating_on_cell: Whether or not to apply self gating on each cell
        after the concatenation of all branches.
      use_sync_bn: If True, use synchronized batch normalization.
      norm_momentum: A `float` of normalization momentum for the moving average.
      norm_epsilon: A `float` added to variance to avoid dividing by zero.
      temporal_conv_initializer: Weight initializer for temporal convolution
        inside the cell. It only applies to 2+1d and 1+2d cases.
      kernel_initializer: Weight initializer for convolutional layers other than
        temporal convolution.
      kernel_regularizer: Weight regularizer for all convolutional layers.
      parameterized_conv_layer: class for parameterized conv layer.
      **kwargs: keyword arguments to be passed.

    Returns:
      out_tensor: A 5-D float tensor of size [batch_size, num_frames, height,
        width, channels].
    """
    super(InceptionV1CellLayer, self).__init__(**kwargs)

    self._branch_filters = branch_filters
    self._conv_type = conv_type
    self._temporal_dilation_rate = temporal_dilation_rate
    self._swap_pool_and_1x1x1 = swap_pool_and_1x1x1
    self._use_self_gating_on_branch = use_self_gating_on_branch
    self._use_self_gating_on_cell = use_self_gating_on_cell
    self._use_sync_bn = use_sync_bn
    self._norm_momentum = norm_momentum
    self._norm_epsilon = norm_epsilon
    self._temporal_conv_initializer = temporal_conv_initializer
    self._kernel_initializer = kernel_initializer
    self._kernel_regularizer = kernel_regularizer
    self._parameterized_conv_layer = parameterized_conv_layer
    if use_sync_bn:
      self._norm = tf_keras.layers.experimental.SyncBatchNormalization
    else:
      self._norm = tf_keras.layers.BatchNormalization

    if tf_keras.backend.image_data_format() == 'channels_last':
      self._channel_axis = -1
    else:
      self._channel_axis = 1

  def _build_branch_params(self):
    branch_0_params = [
        # Conv3D
        dict(
            filters=self._branch_filters[0][0],
            kernel_size=[1, 1, 1],
            padding='same',
            use_bias=False,
            kernel_initializer=tf_utils.clone_initializer(
                self._kernel_initializer),
            kernel_regularizer=self._kernel_regularizer),
        # norm
        dict(
            axis=self._channel_axis,
            momentum=self._norm_momentum,
            epsilon=self._norm_epsilon,
            scale=False,
            gamma_initializer='ones'),
        # relu
        dict(),
    ]
    branch_1_params = [
        # Conv3D
        dict(
            filters=self._branch_filters[1][0],
            kernel_size=[1, 1, 1],
            padding='same',
            use_bias=False,
            kernel_initializer=tf_utils.clone_initializer(
                self._kernel_initializer),
            kernel_regularizer=self._kernel_regularizer),
        # norm
        dict(
            axis=self._channel_axis,
            momentum=self._norm_momentum,
            epsilon=self._norm_epsilon,
            scale=False,
            gamma_initializer='ones'),
        # relu
        dict(),
        # ParameterizedConvLayer
        dict(
            conv_type=self._conv_type,
            kernel_size=3,
            filters=self._branch_filters[1][1],
            strides=[1, 1, 1],
            rates=[self._temporal_dilation_rate, 1, 1],
            use_sync_bn=self._use_sync_bn,
            norm_momentum=self._norm_momentum,
            norm_epsilon=self._norm_epsilon,
            temporal_conv_initializer=self._temporal_conv_initializer,
            kernel_initializer=tf_utils.clone_initializer(
                self._kernel_initializer),
            kernel_regularizer=self._kernel_regularizer),
    ]
    branch_2_params = [
        # Conv3D
        dict(
            filters=self._branch_filters[2][0],
            kernel_size=[1, 1, 1],
            padding='same',
            use_bias=False,
            kernel_initializer=tf_utils.clone_initializer(
                self._kernel_initializer),
            kernel_regularizer=self._kernel_regularizer),
        # norm
        dict(
            axis=self._channel_axis,
            momentum=self._norm_momentum,
            epsilon=self._norm_epsilon,
            scale=False,
            gamma_initializer='ones'),
        # relu
        dict(),
        # ParameterizedConvLayer
        dict(
            conv_type=self._conv_type,
            kernel_size=3,
            filters=self._branch_filters[2][1],
            strides=[1, 1, 1],
            rates=[self._temporal_dilation_rate, 1, 1],
            use_sync_bn=self._use_sync_bn,
            norm_momentum=self._norm_momentum,
            norm_epsilon=self._norm_epsilon,
            temporal_conv_initializer=self._temporal_conv_initializer,
            kernel_initializer=tf_utils.clone_initializer(
                self._kernel_initializer),
            kernel_regularizer=self._kernel_regularizer)
    ]
    branch_3_params = [
        # Conv3D
        dict(
            filters=self._branch_filters[3][0],
            kernel_size=[1, 1, 1],
            padding='same',
            use_bias=False,
            kernel_initializer=tf_utils.clone_initializer(
                self._kernel_initializer),
            kernel_regularizer=self._kernel_regularizer),
        # norm
        dict(
            axis=self._channel_axis,
            momentum=self._norm_momentum,
            epsilon=self._norm_epsilon,
            scale=False,
            gamma_initializer='ones'),
        # relu
        dict(),
        # pool
        dict(
            pool_size=([1, 3, 3] if self._conv_type == '2d' else [3] * 3),
            strides=[1, 1, 1],
            padding='same')
    ]

    if self._use_self_gating_on_branch:
      branch_0_params.append(dict(filters=self._branch_filters[0][0]))
      branch_1_params.append(dict(filters=self._branch_filters[1][1]))
      branch_2_params.append(dict(filters=self._branch_filters[2][1]))
      branch_3_params.append(dict(filters=self._branch_filters[3][0]))

    out_gating_params = []
    if self._use_self_gating_on_cell:
      out_channels = (
          self._branch_filters[0][0] + self._branch_filters[1][1] +
          self._branch_filters[2][1] + self._branch_filters[3][0])
      out_gating_params.append(dict(filters=out_channels))

    return [
        branch_0_params, branch_1_params, branch_2_params, branch_3_params,
        out_gating_params
    ]

  def build(self, input_shape):
    branch_params = self._build_branch_params()

    self._branch_0_layers = [
        tf_keras.layers.Conv3D(**branch_params[0][0]),
        self._norm(**branch_params[0][1]),
        tf_keras.layers.Activation('relu', **branch_params[0][2]),
    ]

    self._branch_1_layers = [
        tf_keras.layers.Conv3D(**branch_params[1][0]),
        self._norm(**branch_params[1][1]),
        tf_keras.layers.Activation('relu', **branch_params[1][2]),
        self._parameterized_conv_layer(**branch_params[1][3]),
    ]

    self._branch_2_layers = [
        tf_keras.layers.Conv3D(**branch_params[2][0]),
        self._norm(**branch_params[2][1]),
        tf_keras.layers.Activation('relu', **branch_params[2][2]),
        self._parameterized_conv_layer(**branch_params[2][3])
    ]

    if self._swap_pool_and_1x1x1:
      self._branch_3_layers = [
          tf_keras.layers.Conv3D(**branch_params[3][0]),
          self._norm(**branch_params[3][1]),
          tf_keras.layers.Activation('relu', **branch_params[3][2]),
          tf_keras.layers.MaxPool3D(**branch_params[3][3]),
      ]
    else:
      self._branch_3_layers = [
          tf_keras.layers.MaxPool3D(**branch_params[3][3]),
          tf_keras.layers.Conv3D(**branch_params[3][0]),
          self._norm(**branch_params[3][1]),
          tf_keras.layers.Activation('relu', **branch_params[3][2]),
      ]

    if self._use_self_gating_on_branch:
      self._branch_0_layers.append(
          nn_blocks_3d.SelfGating(**branch_params[0][-1]))
      self._branch_1_layers.append(
          nn_blocks_3d.SelfGating(**branch_params[1][-1]))
      self._branch_2_layers.append(
          nn_blocks_3d.SelfGating(**branch_params[2][-1]))
      self._branch_3_layers.append(
          nn_blocks_3d.SelfGating(**branch_params[3][-1]))

    if self._use_self_gating_on_cell:
      self.cell_self_gating = nn_blocks_3d.SelfGating(**branch_params[4][0])

    super(InceptionV1CellLayer, self).build(input_shape)

  def call(self, inputs):
    x = inputs
    for layer in self._branch_0_layers:
      x = layer(x)
    branch_0 = x

    x = inputs
    for layer in self._branch_1_layers:
      x = layer(x)
    branch_1 = x

    x = inputs
    for layer in self._branch_2_layers:
      x = layer(x)
    branch_2 = x

    x = inputs
    for layer in self._branch_3_layers:
      x = layer(x)
    branch_3 = x
    out_tensor = tf.concat([branch_0, branch_1, branch_2, branch_3],
                           axis=self._channel_axis)
    if self._use_self_gating_on_cell:
      out_tensor = self.cell_self_gating(out_tensor)
    return out_tensor
