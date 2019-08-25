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
"""Layers for a StarGAN model.

This module contains basic layers to build a StarGAN model.

See https://arxiv.org/abs/1711.09020 for details about the model.

See https://github.com/yunjey/StarGAN for the original pytorvh implementation.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf

import ops


def generator_down_sample(input_net, final_num_outputs=256):
  """Down-sampling module in Generator.

  Down sampling pathway of the Generator Architecture:

  PyTorch Version:
  https://github.com/yunjey/StarGAN/blob/fbdb6a6ce2a4a92e1dc034faec765e0dbe4b8164/model.py#L32

  Notes:
    We require dimension 1 and dimension 2 of the input_net to be fully defined
    for the correct down sampling.

  Args:
    input_net: Tensor of shape (batch_size, h, w, c + num_class).
    final_num_outputs: (int) Number of hidden unit for the final layer.

  Returns:
    Tensor of shape (batch_size, h / 4, w / 4, 256).

  Raises:
    ValueError: If final_num_outputs are not divisible by 4,
      or input_net does not have a rank of 4,
      or dimension 1 and dimension 2 of input_net are not defined at graph
      construction time,
      or dimension 1 and dimension 2 of input_net are not divisible by 4.
  """

  if final_num_outputs % 4 != 0:
    raise ValueError('Final number outputs need to be divisible by 4.')

  # Check the rank of input_net.
  input_net.shape.assert_has_rank(4)

  # Check dimension 1 and dimension 2 are defined and divisible by 4.
  if input_net.shape[1]:
    if input_net.shape[1] % 4 != 0:
      raise ValueError(
          'Dimension 1 of the input should be divisible by 4, but is {} '
          'instead.'.
          format(input_net.shape[1]))
  else:
    raise ValueError('Dimension 1 of the input should be explicitly defined.')

  # Check dimension 1 and dimension 2 are defined and divisible by 4.
  if input_net.shape[2]:
    if input_net.shape[2] % 4 != 0:
      raise ValueError(
          'Dimension 2 of the input should be divisible by 4, but is {} '
          'instead.'.
          format(input_net.shape[2]))
  else:
    raise ValueError('Dimension 2 of the input should be explicitly defined.')

  with tf.variable_scope('generator_down_sample'):

    with tf.contrib.framework.arg_scope(
        [tf.contrib.layers.conv2d],
        padding='VALID',
        biases_initializer=None,
        normalizer_fn=tf.contrib.layers.instance_norm,
        activation_fn=tf.nn.relu):

      down_sample = ops.pad(input_net, 3)
      down_sample = tf.contrib.layers.conv2d(
          inputs=down_sample,
          num_outputs=final_num_outputs / 4,
          kernel_size=7,
          stride=1,
          scope='conv_0')

      down_sample = ops.pad(down_sample, 1)
      down_sample = tf.contrib.layers.conv2d(
          inputs=down_sample,
          num_outputs=final_num_outputs / 2,
          kernel_size=4,
          stride=2,
          scope='conv_1')

      down_sample = ops.pad(down_sample, 1)
      output_net = tf.contrib.layers.conv2d(
          inputs=down_sample,
          num_outputs=final_num_outputs,
          kernel_size=4,
          stride=2,
          scope='conv_2')

    return output_net


def _residual_block(input_net,
                    num_outputs,
                    kernel_size,
                    stride=1,
                    padding_size=0,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=None,
                    name='residual_block'):
  """Residual Block.

  Input Tensor X - > Conv1 -> IN -> ReLU -> Conv2 -> IN + X

  PyTorch Version:
  https://github.com/yunjey/StarGAN/blob/fbdb6a6ce2a4a92e1dc034faec765e0dbe4b8164/model.py#L7

  Args:
    input_net: Tensor as input.
    num_outputs: (int) number of output channels for Convolution.
    kernel_size: (int) size of the square kernel for Convolution.
    stride: (int) stride for Convolution. Default to 1.
    padding_size: (int) padding size for Convolution. Default to 0.
    activation_fn: Activation function.
    normalizer_fn: Normalization function.
    name: Name scope

  Returns:
    Residual Tensor with the same shape as the input tensor.
  """

  with tf.variable_scope(name):

    with tf.contrib.framework.arg_scope(
        [tf.contrib.layers.conv2d],
        num_outputs=num_outputs,
        kernel_size=kernel_size,
        stride=stride,
        padding='VALID',
        normalizer_fn=normalizer_fn,
        activation_fn=None):

      res_block = ops.pad(input_net, padding_size)
      res_block = tf.contrib.layers.conv2d(inputs=res_block, scope='conv_0')

      res_block = activation_fn(res_block, name='activation_0')

      res_block = ops.pad(res_block, padding_size)
      res_block = tf.contrib.layers.conv2d(inputs=res_block, scope='conv_1')

      output_net = res_block + input_net

  return output_net


def generator_bottleneck(input_net, residual_block_num=6, num_outputs=256):
  """Bottleneck module in Generator.

  Residual bottleneck pathway in Generator.

  PyTorch Version:
  https://github.com/yunjey/StarGAN/blob/fbdb6a6ce2a4a92e1dc034faec765e0dbe4b8164/model.py#L40

  Args:
    input_net: Tensor of shape (batch_size, h / 4, w / 4, 256).
    residual_block_num: (int) Number of residual_block_num. Default to 6 per the
      original implementation.
    num_outputs: (int) Number of hidden unit in the residual bottleneck. Default
      to 256 per the original implementation.

  Returns:
    Tensor of shape (batch_size, h / 4, w / 4, 256).

  Raises:
    ValueError: If the rank of the input tensor is not 4,
      or the last channel of the input_tensor is not explicitly defined,
      or the last channel of the input_tensor is not the same as num_outputs.
  """

  # Check the rank of input_net.
  input_net.shape.assert_has_rank(4)

  # Check dimension 4 of the input_net.
  if input_net.shape[-1]:
    if input_net.shape[-1] != num_outputs:
      raise ValueError(
          'The last dimension of the input_net should be the same as '
          'num_outputs: but {} vs. {} instead.'.format(input_net.shape[-1],
                                                       num_outputs))
  else:
    raise ValueError(
        'The last dimension of the input_net should be explicitly defined.')

  with tf.variable_scope('generator_bottleneck'):

    bottleneck = input_net

    for i in range(residual_block_num):

      bottleneck = _residual_block(
          input_net=bottleneck,
          num_outputs=num_outputs,
          kernel_size=3,
          stride=1,
          padding_size=1,
          activation_fn=tf.nn.relu,
          normalizer_fn=tf.contrib.layers.instance_norm,
          name='residual_block_{}'.format(i))

  return bottleneck


def generator_up_sample(input_net, num_outputs):
  """Up-sampling module in Generator.

  Up sampling path for image generation in the Generator.

  PyTorch Version:
  https://github.com/yunjey/StarGAN/blob/fbdb6a6ce2a4a92e1dc034faec765e0dbe4b8164/model.py#L44

  Args:
    input_net: Tensor of shape (batch_size, h / 4, w / 4, 256).
    num_outputs: (int) Number of channel for the output tensor.

  Returns:
    Tensor of shape (batch_size, h, w, num_outputs).
  """

  with tf.variable_scope('generator_up_sample'):

    with tf.contrib.framework.arg_scope(
        [tf.contrib.layers.conv2d_transpose],
        kernel_size=4,
        stride=2,
        padding='VALID',
        normalizer_fn=tf.contrib.layers.instance_norm,
        activation_fn=tf.nn.relu):

      up_sample = tf.contrib.layers.conv2d_transpose(
          inputs=input_net, num_outputs=128, scope='deconv_0')
      up_sample = up_sample[:, 1:-1, 1:-1, :]

      up_sample = tf.contrib.layers.conv2d_transpose(
          inputs=up_sample, num_outputs=64, scope='deconv_1')
      up_sample = up_sample[:, 1:-1, 1:-1, :]

    output_net = ops.pad(up_sample, 3)
    output_net = tf.contrib.layers.conv2d(
        inputs=output_net,
        num_outputs=num_outputs,
        kernel_size=7,
        stride=1,
        padding='VALID',
        activation_fn=tf.nn.tanh,
        normalizer_fn=None,
        biases_initializer=None,
        scope='conv_0')

  return output_net


def discriminator_input_hidden(input_net, hidden_layer=6, init_num_outputs=64):
  """Input Layer + Hidden Layer in the Discriminator.

  Feature extraction pathway in the Discriminator.

  PyTorch Version:
  https://github.com/yunjey/StarGAN/blob/fbdb6a6ce2a4a92e1dc034faec765e0dbe4b8164/model.py#L68

  Args:
    input_net: Tensor of shape (batch_size, h, w, 3) as batch of images.
    hidden_layer: (int) Number of hidden layers. Default to 6 per the original
      implementation.
    init_num_outputs: (int) Number of hidden unit in the first hidden layer. The
      number of hidden unit double after each layer. Default to 64 per the
      original implementation.

  Returns:
    Tensor of shape (batch_size, h / 64, w / 64, 2048) as features.
  """

  num_outputs = init_num_outputs

  with tf.variable_scope('discriminator_input_hidden'):

    hidden = input_net

    for i in range(hidden_layer):

      hidden = ops.pad(hidden, 1)
      hidden = tf.contrib.layers.conv2d(
          inputs=hidden,
          num_outputs=num_outputs,
          kernel_size=4,
          stride=2,
          padding='VALID',
          activation_fn=None,
          normalizer_fn=None,
          scope='conv_{}'.format(i))
      hidden = tf.nn.leaky_relu(hidden, alpha=0.01)

      num_outputs = 2 * num_outputs

  return hidden


def discriminator_output_source(input_net):
  """Output Layer for Source in the Discriminator.

  Determine if the image is real/fake based on the feature extracted. We follow
  the original paper design where the output is not a simple (batch_size) shape
  Tensor but rather a (batch_size, 2, 2, 2048) shape Tensor. We will get the
  correct shape later when we piece things together.

  PyTorch Version:
  https://github.com/yunjey/StarGAN/blob/fbdb6a6ce2a4a92e1dc034faec765e0dbe4b8164/model.py#L79

  Args:
    input_net: Tensor of shape (batch_size, h / 64, w / 64, 2048) as features.

  Returns:
    Tensor of shape (batch_size, h / 64, w / 64, 1) as the score.
  """

  with tf.variable_scope('discriminator_output_source'):

    output_src = ops.pad(input_net, 1)
    output_src = tf.contrib.layers.conv2d(
        inputs=output_src,
        num_outputs=1,
        kernel_size=3,
        stride=1,
        padding='VALID',
        activation_fn=None,
        normalizer_fn=None,
        biases_initializer=None,
        scope='conv')

  return output_src


def discriminator_output_class(input_net, class_num):
  """Output Layer for Domain Classification in the Discriminator.

  The original paper use convolution layer where the kernel size is the height
  and width of the Tensor. We use an equivalent operation here where we first
  flatten the Tensor to shape (batch_size, K) and a fully connected layer.

  PyTorch Version:
  https://github.com/yunjey/StarGAN/blob/fbdb6a6ce2a4a92e1dc034faec765e0dbe4b8164/model.py#L80https

  Args:
    input_net: Tensor of shape (batch_size, h / 64, w / 64, 2028).
    class_num: Number of output classes to be predicted.

  Returns:
    Tensor of shape (batch_size, class_num).
  """

  with tf.variable_scope('discriminator_output_class'):

    output_cls = tf.contrib.layers.flatten(input_net, scope='flatten')
    output_cls = tf.contrib.layers.fully_connected(
        inputs=output_cls,
        num_outputs=class_num,
        activation_fn=None,
        normalizer_fn=None,
        biases_initializer=None,
        scope='fully_connected')

  return output_cls
