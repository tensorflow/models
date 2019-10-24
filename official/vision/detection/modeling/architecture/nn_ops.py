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

from absl import logging
import tensorflow.compat.v2 as tf
from tensorflow.python.keras import backend


class BatchNormRelu(tf.keras.layers.Layer):
  """Combined Batch Normalization and ReLU layers."""

  def __init__(self,
               momentum=0.997,
               epsilon=1e-4,
               trainable=True,
               relu=True,
               init_zero=False,
               fused=True,
               name=None):
    """A class to construct layers for a batch normalization followed by a ReLU.

    Args:
      momentum: momentum for the moving average.
      epsilon: small float added to variance to avoid dividing by zero.
      trainable: `boolean`, if True also add variables to the graph collection
        GraphKeys.TRAINABLE_VARIABLES. If False, freeze batch normalization
        layer.
      relu: `bool` if False, omits the ReLU operation.
      init_zero: `bool` if True, initializes scale parameter of batch
          normalization with 0. If False, initialize it with 1.
      fused: `bool` fused option in batch normalziation.
      name: `str` name for the operation.
    """
    self._use_relu = relu
    self._trainable = trainable
    if init_zero:
      gamma_initializer = tf.keras.initializers.Zeros()
    else:
      gamma_initializer = tf.keras.initializers.Ones()
    self._batch_norm_op = tf.keras.layers.BatchNormalization(
        momentum=momentum,
        epsilon=epsilon,
        center=True,
        scale=True,
        trainable=trainable,
        fused=fused,
        gamma_initializer=gamma_initializer,
        name=name)

  def __call__(self, inputs, is_training=None):
    """Builds layers for a batch normalization followed by a ReLU.

    Args:
      inputs: `Tensor` of shape `[batch, channels, ...]`.
      is_training: `boolean`, if True if model is in training mode.

    Returns:
      A normalized `Tensor` with the same `data_format`.
    """
    # We will need to keep training=None by default, so that it can be inherit
    # from keras.Model.training
    if is_training and self._trainable:
      is_training = True
    inputs = self._batch_norm_op(inputs, training=is_training)

    if self._use_relu:
      inputs = tf.nn.relu(inputs)
    return inputs


class Dropblock(object):
  """DropBlock: a regularization method for convolutional neural networks.

    DropBlock is a form of structured dropout, where units in a contiguous
    region of a feature map are dropped together. DropBlock works better than
    dropout on convolutional layers due to the fact that activation units in
    convolutional layers are spatially correlated.
    See https://arxiv.org/pdf/1810.12890.pdf for details.
  """

  def __init__(self,
               dropblock_keep_prob=None,
               dropblock_size=None,
               data_format='channels_last'):
    self._dropblock_keep_prob = dropblock_keep_prob
    self._dropblock_size = dropblock_size
    self._data_format = data_format

  def __call__(self, net, is_training=False):
    """Builds Dropblock layer.

    Args:
      net: `Tensor` input tensor.
      is_training: `bool` if True, the model is in training mode.

    Returns:
      A version of input tensor with DropBlock applied.
    """
    if not is_training or self._dropblock_keep_prob is None:
      return net

    logging.info('Applying DropBlock: dropblock_size {}, net.shape {}'.format(
        self._dropblock_size, net.shape))

    if self._data_format == 'channels_last':
      _, height, width, _ = net.get_shape().as_list()
    else:
      _, _, height, width = net.get_shape().as_list()

    total_size = width * height
    dropblock_size = min(self._dropblock_size, min(width, height))
    # Seed_drop_rate is the gamma parameter of DropBlcok.
    seed_drop_rate = (
        1.0 - self._dropblock_keep_prob) * total_size / dropblock_size**2 / (
            (width - self._dropblock_size + 1) *
            (height - self._dropblock_size + 1))

    # Forces the block to be inside the feature map.
    w_i, h_i = tf.meshgrid(tf.range(width), tf.range(height))
    valid_block = tf.logical_and(
        tf.logical_and(w_i >= int(dropblock_size // 2),
                       w_i < width - (dropblock_size - 1) // 2),
        tf.logical_and(h_i >= int(dropblock_size // 2),
                       h_i < width - (dropblock_size - 1) // 2))

    if self._data_format == 'channels_last':
      valid_block = tf.reshape(valid_block, [1, height, width, 1])
    else:
      valid_block = tf.reshape(valid_block, [1, 1, height, width])

    randnoise = tf.random.uniform(net.shape, dtype=tf.float32)
    valid_block = tf.cast(valid_block, dtype=tf.float32)
    seed_keep_rate = tf.cast(1 - seed_drop_rate, dtype=tf.float32)
    block_pattern = (1 - valid_block + seed_keep_rate + randnoise) >= 1
    block_pattern = tf.cast(block_pattern, dtype=tf.float32)

    if self._data_format == 'channels_last':
      ksize = [1, self._dropblock_size, self._dropblock_size, 1]
    else:
      ksize = [1, 1, self._dropblock_size, self._dropblock_size]
    block_pattern = -tf.nn.max_pool2d(
        -block_pattern,
        ksize=ksize,
        strides=[1, 1, 1, 1],
        padding='SAME',
        data_format='NHWC' if self._data_format == 'channels_last' else 'NCHW')

    percent_ones = tf.cast(
        tf.reduce_sum(input_tensor=block_pattern), tf.float32) / tf.cast(
            tf.size(input=block_pattern), tf.float32)

    net = net / tf.cast(percent_ones, net.dtype) * tf.cast(
        block_pattern, net.dtype)
    return net
