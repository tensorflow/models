# Copyright 2021 The TensorFlow Authors All Rights Reserved.
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
"""Pooling layers definitions."""

import tensorflow as tf


class MAC(tf.keras.layers.Layer):
  """Global max pooling (MAC) layer.

  Maximum Activations of Convolutions (MAC) is simply constructed by
  max-pooling over all dimensions per feature map. See
  https://arxiv.org/abs/1511.05879 for a reference.
  """

  def call(self, x, axis=None):
    """Invokes the MAC pooling instance.

    Args:
      x: [B, H, W, D] A float32 Tensor.
      axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.

    Returns:
      output: [B, D] A float32 Tensor.
    """
    if axis is None:
      axis = [1, 2]
    return mac(x, axis=axis)


class SPoC(tf.keras.layers.Layer):
  """Average pooling (SPoC) layer.

  Sum-pooled convolutional features (SPoC) is based on the sum pooling of the
  deep features. See https://arxiv.org/pdf/1510.07493.pdf for a reference.
  """

  def call(self, x, axis=None):
    """Invokes the SPoC instance.

    Args:
      x: [B, H, W, D] A float32 Tensor.
      axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.

    Returns:
      output: [B, D] A float32 Tensor.
    """
    if axis is None:
      axis = [1, 2]
    return spoc(x, axis)


class GeM(tf.keras.layers.Layer):
  """Generalized mean pooling (GeM) layer.

  Generalized Mean Pooling (GeM) computes the generalized mean of each
  channel in a tensor. See https://arxiv.org/abs/1711.02512 for a reference.
  """

  def __init__(self, power=3.):
    """Initialization of the generalized mean pooling (GeM) layer.

    Args:
      power:  Float power > 0 is an inverse exponent parameter, used during the
        generalized mean pooling computation. Setting this exponent as power > 1
        increases the contrast of the pooled feature map and focuses on the
        salient features of the image. GeM is a generalization of the average
        pooling commonly used in classification networks (power = 1) and of
        spatial max-pooling layer (power = inf).
    """
    super(GeM, self).__init__()
    self.power = power
    self.eps = 1e-6

  def call(self, x, axis=None):
    """Invokes the GeM instance.

    Args:
      x: [B, H, W, D] A float32 Tensor.
      axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.

    Returns:
      output: [B, D] A float32 Tensor.
    """
    if axis is None:
      axis = [1, 2]
    return gem(x, power=self.power, eps=self.eps, axis=axis)


class GeMPooling2D(tf.keras.layers.Layer):
  """Generalized mean pooling (GeM) pooling operation for spatial data."""

  def __init__(self,
               power=20.,
               pool_size=(2, 2),
               strides=None,
               padding='valid',
               data_format='channels_last'):
    """Initialization of GeMPooling2D.

    Args:
      power: Float, power > 0. is an inverse exponent parameter (GeM power).
      pool_size: Integer or tuple of 2 integers, factors by which to downscale
        (vertical, horizontal)
      strides: Integer, tuple of 2 integers, or None. Strides values. If None,
        it will default to `pool_size`.
      padding: One of `valid` or `same`. `valid` means no padding. `same`
        results in padding evenly to the left/right or up/down of the input such
        that output has the same height/width dimension as the input.
      data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch, height, width,
        channels)` while `channels_first` corresponds to inputs with shape
        `(batch, channels, height, width)`.
    """
    super(GeMPooling2D, self).__init__()
    self.power = power
    self.eps = 1e-6
    self.pool_size = pool_size
    self.strides = strides
    self.padding = padding.upper()
    data_format_conv = {
        'channels_last': 'NHWC',
        'channels_first': 'NCHW',
    }
    self.data_format = data_format_conv[data_format]

  def call(self, x):
    tmp = tf.pow(x, self.power)
    tmp = tf.nn.avg_pool(tmp, self.pool_size, self.strides, self.padding,
                         self.data_format)
    out = tf.pow(tmp, 1. / self.power)
    return out


def mac(x, axis=None):
  """Performs global max pooling (MAC).

  Args:
    x: [B, H, W, D] A float32 Tensor.
    axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.

  Returns:
    output: [B, D] A float32 Tensor.
  """
  if axis is None:
    axis = [1, 2]
  return tf.reduce_max(x, axis=axis, keepdims=False)


def spoc(x, axis=None):
  """Performs average pooling (SPoC).

  Args:
    x: [B, H, W, D] A float32 Tensor.
    axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.

  Returns:
    output: [B, D] A float32 Tensor.
  """
  if axis is None:
    axis = [1, 2]
  return tf.reduce_mean(x, axis=axis, keepdims=False)


def gem(x, axis=None, power=3., eps=1e-6):
  """Performs generalized mean pooling (GeM).

  Args:
    x: [B, H, W, D] A float32 Tensor.
    axis: Dimensions to reduce. By default, dimensions [1, 2] are reduced.
    power: Float, power > 0 is an inverse exponent parameter (GeM power).
    eps: Float, parameter for numerical stability.

  Returns:
    output: [B, D] A float32 Tensor.
  """
  if axis is None:
    axis = [1, 2]
  tmp = tf.pow(tf.maximum(x, eps), power)
  out = tf.pow(tf.reduce_mean(tmp, axis=axis, keepdims=False), 1. / power)
  return out
