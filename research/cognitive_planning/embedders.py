# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Interface for different embedders for modalities."""

import abc
import numpy as np
import tensorflow as tf
import preprocessing
from tensorflow.contrib.slim.nets import resnet_v2

slim = tf.contrib.slim


class Embedder(object):
  """Represents the embedder for different modalities.

  Modalities can be semantic segmentation, depth channel, object detection and
  so on, which require specific embedder for them.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def build(self, observation):
    """Builds the model to embed the observation modality.

    Args:
      observation: tensor that contains the raw observation from modality.
    Returns:
      Embedding tensor for the given observation tensor.
    """
    raise NotImplementedError(
        'Needs to be implemented as part of Embedder Interface')


class DetectionBoxEmbedder(Embedder):
  """Represents the model that encodes the detection boxes from images."""

  def __init__(self, rnn_state_size, scope=None):
    self._rnn_state_size = rnn_state_size
    self._scope = scope

  def build(self, observations):
    """Builds the model to embed object detection observations.

    Args:
      observations: a tuple of (dets, det_num).
        dets is a tensor of BxTxLxE that has the detection boxes in all the
          images of the batch. B is the batch size, T is the maximum length of
          episode, L is the maximum number of detections per image in the batch
          and E is the size of each detection embedding.
        det_num is a tensor of BxT that contains the number of detected boxes
          each image of each sequence in the batch.
    Returns:
      For each image in the batch, returns the accumulative embedding of all the
      detection boxes in that image.
    """
    with tf.variable_scope(self._scope, default_name=''):
      shape = observations[0].shape
      dets = tf.reshape(observations[0], [-1, shape[-2], shape[-1]])
      det_num = tf.reshape(observations[1], [-1])
      lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self._rnn_state_size)
      batch_size = tf.shape(dets)[0]
      lstm_outputs, _ = tf.nn.dynamic_rnn(
          cell=lstm_cell,
          inputs=dets,
          sequence_length=det_num,
          initial_state=lstm_cell.zero_state(batch_size, dtype=tf.float32),
          dtype=tf.float32)
      # Gathering the last state of each sequence in the batch.
      batch_range = tf.range(batch_size)
      indices = tf.stack([batch_range, det_num - 1], axis=1)
      last_lstm_outputs = tf.gather_nd(lstm_outputs, indices)
      last_lstm_outputs = tf.reshape(last_lstm_outputs,
                                     [-1, shape[1], self._rnn_state_size])
    return last_lstm_outputs


class ResNet(Embedder):
  """Residual net embedder for image data."""

  def __init__(self, params, *args, **kwargs):
    super(ResNet, self).__init__(*args, **kwargs)
    self._params = params
    self._extra_train_ops = []

  def build(self, images):
    shape = images.get_shape().as_list()
    if len(shape) == 5:
      images = tf.reshape(images,
                          [shape[0] * shape[1], shape[2], shape[3], shape[4]])
    embedding = self._build_model(images)
    if len(shape) == 5:
      embedding = tf.reshape(embedding, [shape[0], shape[1], -1])

    return embedding

  @property
  def extra_train_ops(self):
    return self._extra_train_ops

  def _build_model(self, images):
    """Builds the model."""

    # Convert images to floats and normalize them.
    images = tf.to_float(images)
    bs = images.get_shape().as_list()[0]
    images = [
        tf.image.per_image_standardization(tf.squeeze(i))
        for i in tf.split(images, bs)
    ]
    images = tf.concat([tf.expand_dims(i, axis=0) for i in images], axis=0)

    with tf.variable_scope('init'):
      x = self._conv('init_conv', images, 3, 3, 16, self._stride_arr(1))

    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    if self._params.use_bottleneck:
      res_func = self._bottleneck_residual
      filters = [16, 64, 128, 256]
    else:
      res_func = self._residual
      filters = [16, 16, 32, 128]

    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])
    for i in xrange(1, self._params.num_residual_units):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)

    with tf.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])
    for i in xrange(1, self._params.num_residual_units):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)

    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
    for i in xrange(1, self._params.num_residual_units):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)

    with tf.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, self._params.relu_leakiness)

    with tf.variable_scope('pool_logit'):
      x = self._global_avg_pooling(x)

    return x

  def _stride_arr(self, stride):
    return [1, stride, stride, 1]

  def _batch_norm(self, name, x):
    """batch norm implementation."""
    with tf.variable_scope(name):
      params_shape = [x.shape[-1]]

      beta = tf.get_variable(
          'beta',
          params_shape,
          tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma',
          params_shape,
          tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self._params.is_train:
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean',
            params_shape,
            tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance',
            params_shape,
            tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(
            tf.assign_moving_average(moving_mean, mean, 0.9))
        self._extra_train_ops.append(
            tf.assign_moving_average(moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean',
            params_shape,
            tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance',
            params_shape,
            tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
        tf.summary.histogram(mean.op.name, mean)
        tf.summary.histogram(variance.op.name, variance)
      # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.shape)
      return y

  def _residual(self,
                x,
                in_filter,
                out_filter,
                stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""

    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self._params.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self._params.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self._params.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0], [(out_filter - in_filter) // 2,
                                              (out_filter - in_filter) // 2]])
      x += orig_x

    return x

  def _bottleneck_residual(self,
                           x,
                           in_filter,
                           out_filter,
                           stride,
                           activate_before_residual=False):
    """A residual convolutional layer with a bottleneck.

    The layer is a composite of three convolutional layers with a ReLU non-
    linearity and batch normalization after each linear convolution. The depth
    if the second and third layer is out_filter / 4 (hence it is a bottleneck).

    Args:
      x: a float 4 rank Tensor representing the input to the layer.
      in_filter: a python integer representing depth of the input.
      out_filter: a python integer representing depth of the output.
      stride: a python integer denoting the stride of the layer applied before
        the first convolution.
      activate_before_residual: a python boolean. If True, then a ReLU is
        applied as a first operation on the input x before everything else.
    Returns:
      A 4 rank Tensor with batch_size = batch size of input, width and height =
      width / stride and height / stride of the input and depth = out_filter.
    """
    if activate_before_residual:
      with tf.variable_scope('common_bn_relu'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self._params.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_bn_relu'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self._params.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 1, in_filter, out_filter / 4, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self._params.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter / 4, out_filter / 4,
                     [1, 1, 1, 1])

    with tf.variable_scope('sub3'):
      x = self._batch_norm('bn3', x)
      x = self._relu(x, self._params.relu_leakiness)
      x = self._conv('conv3', x, 1, out_filter / 4, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
      x += orig_x

    return x

  def _decay(self):
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        costs.append(tf.nn.l2_loss(var))

    return tf.mul(self._params.weight_decay_rate, tf.add_n(costs))

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32,
          initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    x = tf.reshape(x, [self._params.batch_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable(
        'biases', [out_dim], initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pooling(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])


class MLPEmbedder(Embedder):
  """Embedder of vectorial data.

  The net is a multi-layer perceptron, with ReLU nonlinearities in all layers
  except the last one.
  """

  def __init__(self, layers, *args, **kwargs):
    """Constructs MLPEmbedder.

    Args:
      layers: a list of python integers representing layer sizes.
      *args: arguments for super constructor.
      **kwargs: keyed arguments for super constructor.
    """
    super(MLPEmbedder, self).__init__(*args, **kwargs)
    self._layers = layers

  def build(self, features):
    shape = features.get_shape().as_list()
    if len(shape) == 3:
      features = tf.reshape(features, [shape[0] * shape[1], shape[2]])
    x = features
    for i, dim in enumerate(self._layers):
      with tf.variable_scope('layer_%i' % i):
        x = self._fully_connected(x, dim)
        if i < len(self._layers) - 1:
          x = self._relu(x)

    if len(shape) == 3:
      x = tf.reshape(x, shape[:-1] + [self._layers[-1]])
    return x

  def _fully_connected(self, x, out_dim):
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.variance_scaling_initializer(distribution='uniform'))
    b = tf.get_variable(
        'biases', [out_dim], initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _relu(self, x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')


class SmallNetworkEmbedder(Embedder):
  """Embedder for image like observations.

  The network is comprised of multiple conv layers and a fully connected layer
  at the end. The number of conv layers and the parameters are configured from
  params.
  """

  def __init__(self, params, *args, **kwargs):
    """Constructs the small network.

    Args:
      params: params should be tf.hparams type. params need to have a list of
        conv_sizes, conv_strides, conv_channels. The length of these lists
        should be equal to each other and to the number of conv layers in the
        network. Plus, it also needs to have boolean variable named to_one_hot
        which indicates whether the input should be converted to one hot or not.
        The size of the fully connected layer is specified by
        params.embedding_size.

      *args: The rest of the parameters.
      **kwargs: the reset of the parameters.

    Raises:
      ValueError: If the length of params.conv_strides, params.conv_sizes, and
        params.conv_channels are not equal.

    """

    super(SmallNetworkEmbedder, self).__init__(*args, **kwargs)
    self._params = params
    if len(self._params.conv_sizes) != len(self._params.conv_strides):
      raise ValueError(
          'Conv sizes and strides should have the same length: {} != {}'.format(
              len(self._params.conv_sizes), len(self._params.conv_strides)))

    if len(self._params.conv_sizes) != len(self._params.conv_channels):
      raise ValueError(
          'Conv sizes and channels should have the same length: {} != {}'.
          format(len(self._params.conv_sizes), len(self._params.conv_channels)))

  def build(self, images):
    """Builds the embedder with the given speicifcation.

    Args:
      images: a tensor that contains the input images which has the shape of
        NxTxHxWxC where N is the batch size, T is the maximum length of the
        sequence, H and W are the height and width of the images and C is the
        number of channels.

    Returns:
      A tensor that is the embedding of the images.
    """

    shape = images.get_shape().as_list()
    images = tf.reshape(images,
                        [shape[0] * shape[1], shape[2], shape[3], shape[4]])

    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(self._params.weight_decay_rate),
        biases_initializer=tf.zeros_initializer()):
      with slim.arg_scope([slim.conv2d], padding='SAME'):
        # convert the image to one hot if needed.
        if self._params.to_one_hot:
          net = tf.one_hot(
              tf.squeeze(tf.to_int32(images), axis=[-1]),
              self._params.one_hot_length)
        else:
          net = images

        p = self._params
        # Adding conv layers with the specified configurations.
        for conv_id, kernel_stride_channel in enumerate(
            zip(p.conv_sizes, p.conv_strides, p.conv_channels)):
          kernel_size, stride, channels = kernel_stride_channel
          net = slim.conv2d(
              net,
              channels, [kernel_size, kernel_size],
              stride,
              scope='conv_{}'.format(conv_id + 1))

        net = slim.flatten(net)
        net = slim.fully_connected(net, self._params.embedding_size, scope='fc')

        output = tf.reshape(net, [shape[0], shape[1], -1])
        return output


class ResNet50Embedder(Embedder):
  """Uses ResNet50 to embed input images."""

  def build(self, images):
    """Builds a ResNet50 embedder for the input images.

    It assumes that the range of the pixel values in the images tensor is
      [0,255] and should be castable to tf.uint8.

    Args:
      images: a tensor that contains the input images which has the shape of
          NxTxHxWx3 where N is the batch size, T is the maximum length of the
          sequence, H and W are the height and width of the images and C is the
          number of channels.
    Returns:
      The embedding of the input image with the shape of NxTxL where L is the
        embedding size of the output.

    Raises:
      ValueError: if the shape of the input does not agree with the expected
      shape explained in the Args section.
    """
    shape = images.get_shape().as_list()
    if len(shape) != 5:
      raise ValueError(
          'The tensor shape should have 5 elements, {} is provided'.format(
              len(shape)))
    if shape[4] != 3:
      raise ValueError('Three channels are expected for the input image')

    images = tf.cast(images, tf.uint8)
    images = tf.reshape(images,
                        [shape[0] * shape[1], shape[2], shape[3], shape[4]])
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):

      def preprocess_fn(x):
        x = tf.expand_dims(x, 0)
        x = tf.image.resize_bilinear(x, [299, 299],
                                       align_corners=False)
        return(tf.squeeze(x, [0]))

      images = tf.map_fn(preprocess_fn, images, dtype=tf.float32)

      net, _ = resnet_v2.resnet_v2_50(
          images, is_training=False, global_pool=True)
      output = tf.reshape(net, [shape[0], shape[1], -1])
      return output


class IdentityEmbedder(Embedder):
  """This embedder just returns the input as the output.

  Used for modalitites that the embedding of the modality is the same as the
  modality itself. For example, it can be used for one_hot goal.
  """

  def build(self, images):
    return images
