# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Neural network operations commonly shared by the architectures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl import logging
import tensorflow as tf


class NormActivation(tf.keras.layers.Layer):
  """Combined Normalization and Activation layers."""

  def __init__(self,
               momentum=0.997,
               epsilon=1e-4,
               trainable=True,
               init_zero=False,
               use_activation=True,
               activation='relu',
               fused=True,
               name=None):
    """A class to construct layers for a batch normalization followed by a ReLU.

    Args:
      momentum: momentum for the moving average.
      epsilon: small float added to variance to avoid dividing by zero.
      trainable: `bool`, if True also add variables to the graph collection
        GraphKeys.TRAINABLE_VARIABLES. If False, freeze batch normalization
        layer.
      init_zero: `bool` if True, initializes scale parameter of batch
          normalization with 0. If False, initialize it with 1.
      fused: `bool` fused option in batch normalziation.
      use_actiation: `bool`, whether to add the optional activation layer after
        the batch normalization layer.
      activation: 'string', the type of the activation layer. Currently support
        `relu` and `swish`.
      name: `str` name for the operation.
    """
    super(NormActivation, self).__init__(trainable=trainable)
    if init_zero:
      gamma_initializer = tf.keras.initializers.Zeros()
    else:
      gamma_initializer = tf.keras.initializers.Ones()
    self._normalization_op = tf.keras.layers.BatchNormalization(
        momentum=momentum,
        epsilon=epsilon,
        center=True,
        scale=True,
        trainable=trainable,
        fused=fused,
        gamma_initializer=gamma_initializer,
        name=name)
    self._use_activation = use_activation
    if activation == 'relu':
      self._activation_op = tf.nn.relu
    elif activation == 'swish':
      self._activation_op = tf.nn.swish
    else:
      raise ValueError('Unsupported activation `{}`.'.format(activation))

  def __call__(self, inputs, is_training=None):
    """Builds the normalization layer followed by an optional activation layer.

    Args:
      inputs: `Tensor` of shape `[batch, channels, ...]`.
      is_training: `boolean`, if True if model is in training mode.

    Returns:
      A normalized `Tensor` with the same `data_format`.
    """
    # We will need to keep training=None by default, so that it can be inherit
    # from keras.Model.training
    if is_training and self.trainable:
      is_training = True
    inputs = self._normalization_op(inputs, training=is_training)

    if self._use_activation:
      inputs = self._activation_op(inputs)
    return inputs


def norm_activation_builder(momentum=0.997,
                            epsilon=1e-4,
                            trainable=True,
                            activation='relu',
                            **kwargs):
  return functools.partial(
      NormActivation,
      momentum=momentum,
      epsilon=epsilon,
      trainable=trainable,
      activation='relu',
      **kwargs)
