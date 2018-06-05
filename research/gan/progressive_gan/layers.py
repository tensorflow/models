# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Layers for a progressive GAN model.

This module contains basic building blocks to build a progressive GAN model.

See https://arxiv.org/abs/1710.10196 for details about the model.

See https://github.com/tkarras/progressive_growing_of_gans for the original
theano implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf


def pixel_norm(images, epsilon=1.0e-8):
  """Pixel normalization.

  For each pixel a[i,j,k] of image in HWC format, normalize its value to
  b[i,j,k] = a[i,j,k] / SQRT(SUM_k(a[i,j,k]^2) / C + eps).

  Args:
    images: A 4D `Tensor` of NHWC format.
    epsilon: A small positive number to avoid division by zero.

  Returns:
    A 4D `Tensor` with pixel-wise normalized channels.
  """
  return images * tf.rsqrt(
      tf.reduce_mean(tf.square(images), axis=3, keepdims=True) + epsilon)


def _get_validated_scale(scale):
  """Returns the scale guaranteed to be a positive integer."""
  scale = int(scale)
  if scale <= 0:
    raise ValueError('`scale` must be a positive integer.')
  return scale


def downscale(images, scale):
  """Box downscaling of images.

  Args:
    images: A 4D `Tensor` in NHWC format.
    scale: A positive integer scale.

  Returns:
    A 4D `Tensor` of `images` down scaled by a factor `scale`.

  Raises:
    ValueError: If `scale` is not a positive integer.
  """
  scale = _get_validated_scale(scale)
  if scale == 1:
    return images
  return tf.nn.avg_pool(
      images,
      ksize=[1, scale, scale, 1],
      strides=[1, scale, scale, 1],
      padding='VALID')


def upscale(images, scale):
  """Box upscaling (also called nearest neighbors) of images.

  Args:
    images: A 4D `Tensor` in NHWC format.
    scale: A positive integer scale.

  Returns:
    A 4D `Tensor` of `images` up scaled by a factor `scale`.

  Raises:
    ValueError: If `scale` is not a positive integer.
  """
  scale = _get_validated_scale(scale)
  if scale == 1:
    return images
  return tf.batch_to_space(
      tf.tile(images, [scale**2, 1, 1, 1]),
      crops=[[0, 0], [0, 0]],
      block_size=scale)


def minibatch_mean_stddev(x):
  """Computes the standard deviation average.

  This is used by the discriminator as a form of batch discrimination.

  Args:
    x: A `Tensor` for which to compute the standard deviation average. The first
        dimension must be batch size.

  Returns:
    A scalar `Tensor` which is the mean variance of variable x.
  """
  mean, var = tf.nn.moments(x, axes=[0])
  del mean
  return tf.reduce_mean(tf.sqrt(var))


def scalar_concat(tensor, scalar):
  """Concatenates a scalar to the last dimension of a tensor.

  Args:
    tensor: A `Tensor`.
    scalar: a scalar `Tensor` to concatenate to tensor `tensor`.

  Returns:
    A `Tensor`. If `tensor` has shape [...,N], the result R has shape
    [...,N+1] and R[...,N] = scalar.

  Raises:
    ValueError: If `tensor` is a scalar `Tensor`.
  """
  ndims = tensor.shape.ndims
  if ndims < 1:
    raise ValueError('`tensor` must have number of dimensions >= 1.')
  shape = tf.shape(tensor)
  return tf.concat(
      [tensor, tf.ones([shape[i] for i in range(ndims - 1)] + [1]) * scalar],
      axis=ndims - 1)


def he_initializer_scale(shape, slope=1.0):
  """The scale of He neural network initializer.

  Args:
    shape: A list of ints representing the dimensions of a tensor.
    slope: A float representing the slope of the ReLu following the layer.

  Returns:
    A float of he initializer scale.
  """
  fan_in = np.prod(shape[:-1])
  return np.sqrt(2. / ((1. + slope**2) * fan_in))


def _custom_layer_impl(apply_kernel, kernel_shape, bias_shape, activation,
                       he_initializer_slope, use_weight_scaling):
  """Helper function to implement custom_xxx layer.

  Args:
    apply_kernel: A function that transforms kernel to output.
    kernel_shape: An integer tuple or list of the kernel shape.
    bias_shape: An integer tuple or list of the bias shape.
    activation: An activation function to be applied. None means no
        activation.
    he_initializer_slope: A float slope for the He initializer.
    use_weight_scaling: Whether to apply weight scaling.

  Returns:
    A `Tensor` computed as apply_kernel(kernel) + bias where kernel is a
    `Tensor` variable with shape `kernel_shape`, bias is a `Tensor` variable
    with shape `bias_shape`.
  """
  kernel_scale = he_initializer_scale(kernel_shape, he_initializer_slope)
  init_scale, post_scale = kernel_scale, 1.0
  if use_weight_scaling:
    init_scale, post_scale = post_scale, init_scale

  kernel_initializer = tf.random_normal_initializer(stddev=init_scale)

  bias = tf.get_variable(
      'bias', shape=bias_shape, initializer=tf.zeros_initializer())

  output = post_scale * apply_kernel(kernel_shape, kernel_initializer) + bias

  if activation is not None:
    output = activation(output)
  return output


def custom_conv2d(x,
                  filters,
                  kernel_size,
                  strides=(1, 1),
                  padding='SAME',
                  activation=None,
                  he_initializer_slope=1.0,
                  use_weight_scaling=True,
                  scope='custom_conv2d',
                  reuse=None):
  """Custom conv2d layer.

  In comparison with tf.layers.conv2d this implementation use the He initializer
  to initialize convolutional kernel and the weight scaling trick (if
  `use_weight_scaling` is True) to equalize learning rates. See
  https://arxiv.org/abs/1710.10196 for more details.

  Args:
    x: A `Tensor` of NHWC format.
    filters: An int of output channels.
    kernel_size: An integer or a int tuple of [kernel_height, kernel_width].
    strides: A list of strides.
    padding: One of "VALID" or "SAME".
    activation: An activation function to be applied. None means no
        activation. Defaults to None.
    he_initializer_slope: A float slope for the He initializer. Defaults to 1.0.
    use_weight_scaling: Whether to apply weight scaling. Defaults to True.
    scope: A string or variable scope.
    reuse: Whether to reuse the weights. Defaults to None.

  Returns:
    A `Tensor` of NHWC format where the last dimension has size `filters`.
  """
  if not isinstance(kernel_size, (list, tuple)):
    kernel_size = [kernel_size] * 2
  kernel_size = list(kernel_size)

  def _apply_kernel(kernel_shape, kernel_initializer):
    return tf.layers.conv2d(
        x,
        filters=filters,
        kernel_size=kernel_shape[0:2],
        strides=strides,
        padding=padding,
        use_bias=False,
        kernel_initializer=kernel_initializer)

  with tf.variable_scope(scope, reuse=reuse):
    return _custom_layer_impl(
        _apply_kernel,
        kernel_shape=kernel_size + [x.shape.as_list()[3], filters],
        bias_shape=(filters,),
        activation=activation,
        he_initializer_slope=he_initializer_slope,
        use_weight_scaling=use_weight_scaling)


def custom_dense(x,
                 units,
                 activation=None,
                 he_initializer_slope=1.0,
                 use_weight_scaling=True,
                 scope='custom_dense',
                 reuse=None):
  """Custom dense layer.

  In comparison with tf.layers.dense This implementation use the He
  initializer to initialize weights and the weight scaling trick
  (if `use_weight_scaling` is True) to equalize learning rates. See
  https://arxiv.org/abs/1710.10196 for more details.

  Args:
    x: A `Tensor`.
    units: An int of the last dimension size of output.
    activation: An activation function to be applied. None means no
        activation. Defaults to None.
    he_initializer_slope: A float slope for the He initializer. Defaults to 1.0.
    use_weight_scaling: Whether to apply weight scaling. Defaults to True.
    scope: A string or variable scope.
    reuse: Whether to reuse the weights. Defaults to None.

  Returns:
    A `Tensor` where the last dimension has size `units`.
  """
  x = tf.contrib.layers.flatten(x)

  def _apply_kernel(kernel_shape, kernel_initializer):
    return tf.layers.dense(
        x,
        kernel_shape[1],
        use_bias=False,
        kernel_initializer=kernel_initializer)

  with tf.variable_scope(scope, reuse=reuse):
    return _custom_layer_impl(
        _apply_kernel,
        kernel_shape=(x.shape.as_list()[-1], units),
        bias_shape=(units,),
        activation=activation,
        he_initializer_slope=he_initializer_slope,
        use_weight_scaling=use_weight_scaling)
