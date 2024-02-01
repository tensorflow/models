# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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

"""Contains common building blocks for 3D networks."""
import tensorflow as tf, tf_keras

from official.vision.modeling.layers import nn_blocks_3d
from official.vision.modeling.layers import nn_layers

SelfGating = nn_blocks_3d.SelfGating


class BottleneckBlock3D(nn_blocks_3d.BottleneckBlock3D):
  """Creates a 3D bottleneck block."""

  def build(self, input_shape):
    self._shortcut_maxpool = tf_keras.layers.MaxPool3D(
        pool_size=[1, 1, 1],
        strides=[
            self._temporal_strides, self._spatial_strides, self._spatial_strides
        ])

    self._shortcut_conv = tf_keras.layers.Conv3D(
        filters=4 * self._filters,
        kernel_size=1,
        strides=[
            self._temporal_strides, self._spatial_strides, self._spatial_strides
        ],
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        name='shortcut_conv')
    self._norm0 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        name='shortcut_conv/batch_norm')

    self._temporal_conv = tf_keras.layers.Conv3D(
        filters=self._filters,
        kernel_size=[self._temporal_kernel_size, 1, 1],
        strides=[self._temporal_strides, 1, 1],
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        name='temporal_conv')
    self._norm1 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        name='temporal_conv/batch_norm')

    self._spatial_conv = tf_keras.layers.Conv3D(
        filters=self._filters,
        kernel_size=[1, 3, 3],
        strides=[1, self._spatial_strides, self._spatial_strides],
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        name='spatial_conv')
    self._norm2 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        name='spatial_conv/batch_norm')

    self._expand_conv = tf_keras.layers.Conv3D(
        filters=4 * self._filters,
        kernel_size=[1, 1, 1],
        strides=[1, 1, 1],
        padding='same',
        use_bias=False,
        kernel_initializer=self._kernel_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        name='expand_conv')
    self._norm3 = self._norm(
        axis=self._bn_axis,
        momentum=self._norm_momentum,
        epsilon=self._norm_epsilon,
        name='expand_conv/batch_norm/')

    if self._se_ratio and self._se_ratio > 0 and self._se_ratio <= 1:
      self._squeeze_excitation = nn_layers.SqueezeExcitation(
          in_filters=self._filters * 4,
          out_filters=self._filters * 4,
          se_ratio=self._se_ratio,
          use_3d_input=True,
          kernel_initializer=self._kernel_initializer,
          kernel_regularizer=self._kernel_regularizer,
          bias_regularizer=self._bias_regularizer,
          name='se_layer')
    else:
      self._squeeze_excitation = None

    if self._stochastic_depth_drop_rate:
      self._stochastic_depth = nn_layers.StochasticDepth(
          self._stochastic_depth_drop_rate)
    else:
      self._stochastic_depth = None

    if self._use_self_gating:
      self._self_gating = SelfGating(filters=4 * self._filters,
                                     name='self_gating')
    else:
      self._self_gating = None
