# Copyright 2016 The TensorFlow Authors All Rights Reserved.
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

"""Cross Convolutional Model.

https://arxiv.org/pdf/1607.02586v1.pdf
"""
import math
import sys

import tensorflow as tf

slim = tf.contrib.slim


class CrossConvModel(object):

  def __init__(self, image_diff_list, params):
    """Constructor.

    Args:
      image_diff_list: A list of (image, diff) tuples, with shape
          [batch_size, image_size, image_size, 3] and image_sizes as
          [32, 64, 128, 256].
      params: Dict of parameters.
    """
    self.images = [i for (i, _) in image_diff_list]
    # Move the diff to the positive realm.
    self.diffs = [(d + params['scale']) / 2 for (i, d) in image_diff_list]
    self.params = params

  def Build(self):
    with tf.device('/gpu:0'):
      with slim.arg_scope([slim.conv2d],
                          activation_fn=tf.nn.relu,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params={'is_training':
                                             self.params['is_training']}):
        self._BuildMotionKernel()
        encoded_images = self._BuildImageEncoder()
        cross_conved_images = self._CrossConv(encoded_images)
        self._BuildImageDecoder(cross_conved_images)
        self._BuildLoss()

      image = self.images[1]
      diff = self.diffs[1]

      self.global_step = tf.Variable(0, name='global_step', trainable=False)

      if self.params['is_training']:
        self._BuildTrainOp()

      diff = diff * 2.0 - self.params['scale']
      diff_output = self.diff_output * 2.0 - self.params['scale']
      concat_image = tf.concat(
          1, [image, image + diff_output, image + diff, diff_output])
      tf.summary.image('origin_predict_expect_predictdiff', concat_image)
      self.summary_op = tf.summary.merge_all()
      return self.loss

  def _BuildTrainOp(self):
    lrn_rate = tf.maximum(
        0.01,  # min_lr_rate.
        tf.train.exponential_decay(
            self.params['learning_rate'], self.global_step, 10000, 0.5))
    tf.summary.scalar('learning rate', lrn_rate)
    optimizer = tf.train.GradientDescentOptimizer(lrn_rate)
    self.train_op = slim.learning.create_train_op(
        self.loss, optimizer, global_step=self.global_step)

  def _BuildLoss(self):
    # 1. reconstr_loss seems doesn't do better than l2 loss.
    # 2. Only works when using reduce_mean. reduce_sum doesn't work.
    # 3. It seems kl loss doesn't play an important role.
    self.loss = 0
    with tf.variable_scope('loss'):
      if self.params['l2_loss']:
        l2_loss = tf.reduce_mean(tf.square(self.diff_output - self.diffs[1]))
        tf.summary.scalar('l2_loss', l2_loss)
        self.loss += l2_loss
      if self.params['reconstr_loss']:
        reconstr_loss = (-tf.reduce_mean(
            self.diffs[1] * (1e-10 + self.diff_output) +
            (1-self.diffs[1]) * tf.log(1e-10 + 1 - self.diff_output)))
        reconstr_loss = tf.check_numerics(reconstr_loss, 'reconstr_loss')
        tf.summary.scalar('reconstr_loss', reconstr_loss)
        self.loss += reconstr_loss
      if self.params['kl_loss']:
        kl_loss = (0.5 * tf.reduce_mean(
            tf.square(self.z_mean) + tf.square(self.z_stddev) -
            2 * self.z_stddev_log - 1))
        tf.summary.scalar('kl_loss', kl_loss)
        self.loss += kl_loss

      tf.summary.scalar('loss', self.loss)

  def _BuildMotionKernel(self):
    image = self.images[-2]
    diff = self.diffs[-2]
    shape = image.get_shape().as_list()
    assert shape[1] == shape[2] and shape[1] == 128
    batch_size = shape[0]

    net = tf.concat(3, [image, diff])
    with tf.variable_scope('motion_encoder'):
      with slim.arg_scope([slim.conv2d], padding='VALID'):
        net = slim.conv2d(net, 96, [5, 5], stride=1)
        net = slim.max_pool2d(net, [2, 2])
        net = slim.conv2d(net, 96, [5, 5], stride=1)
        net = slim.max_pool2d(net, [2, 2])
        net = slim.conv2d(net, 128, [5, 5], stride=1)
        net = slim.conv2d(net, 128, [5, 5], stride=1)
        net = slim.max_pool2d(net, [2, 2])
        net = slim.conv2d(net, 256, [4, 4], stride=1)
        net = slim.conv2d(net, 256, [3, 3], stride=1)

        z = tf.reshape(net, shape=[batch_size, -1])
        self.z_mean, self.z_stddev_log = tf.split(
            split_dim=1, num_split=2, value=z)
        self.z_stddev = tf.exp(self.z_stddev_log)

        epsilon = tf.random_normal(
            self.z_mean.get_shape().as_list(), 0, 1, dtype=tf.float32)
        kernel = self.z_mean + tf.multiply(self.z_stddev, epsilon)

        width = int(math.sqrt(kernel.get_shape().as_list()[1] // 128))
        kernel = tf.reshape(kernel, [batch_size, width, width, 128])
    with tf.variable_scope('kernel_decoder'):
      with slim.arg_scope([slim.conv2d], padding='SAME'):
        kernel = slim.conv2d(kernel, 128, [5, 5], stride=1)
        self.kernel = slim.conv2d(kernel, 128, [5, 5], stride=1)

    sys.stderr.write('kernel shape: %s\n' % kernel.get_shape())

  def _BuildImageEncoder(self):
    feature_maps = []
    for (i, image) in enumerate(self.images):
      with tf.variable_scope('image_encoder_%d' % i):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME'):
          net = slim.conv2d(image, 64, [5, 5], stride=1)
          net = slim.conv2d(net, 64, [5, 5], stride=1)
          net = slim.max_pool2d(net, [5, 5])
          net = slim.conv2d(net, 64, [5, 5], stride=1)
          net = slim.conv2d(net, 32, [5, 5], stride=1)
          net = slim.max_pool2d(net, [2, 2])
      sys.stderr.write('image_conv shape: %s\n' % net.get_shape())
      feature_maps.append(net)
    return feature_maps

  def _CrossConvHelper(self, encoded_image, kernel):
    """Cross Convolution.

      The encoded image and kernel are of the same shape. Namely
      [batch_size, image_size, image_size, channels]. They are split
      into [image_size, image_size] image squares [kernel_size, kernel_size]
      kernel squares. kernel squares are used to convolute image squares.
    """
    images = tf.expand_dims(encoded_image, 0)
    kernels = tf.expand_dims(kernel, 3)
    return tf.nn.depthwise_conv2d(images, kernels, [1, 1, 1, 1], 'SAME')

  def _CrossConv(self, encoded_images):
    """Apply the motion kernel on the encoded_images."""
    cross_conved_images = []
    kernels = tf.split(split_dim=3, num_split=4, value=self.kernel)
    for (i, encoded_image) in enumerate(encoded_images):
      with tf.variable_scope('cross_conv_%d' % i):
        kernel = kernels[i]

        encoded_image = tf.unstack(encoded_image, axis=0)
        kernel = tf.unstack(kernel, axis=0)
        assert len(encoded_image) == len(kernel)
        assert len(encoded_image) == self.params['batch_size']
        conved_image = []
        for j in xrange(len(encoded_image)):
          conved_image.append(self._CrossConvHelper(
              encoded_image[j], kernel[j]))
        cross_conved_images.append(tf.concat(0, conved_image))
        sys.stderr.write('cross_conved shape: %s\n' %
                         cross_conved_images[-1].get_shape())
    return cross_conved_images

  def _Deconv(self, net, out_filters, kernel_size, stride):
    shape = net.get_shape().as_list()
    in_filters = shape[3]
    kernel_shape = [kernel_size, kernel_size, out_filters, in_filters]

    weights = tf.get_variable(
        name='weights',
        shape=kernel_shape,
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.01))


    out_height = shape[1] * stride
    out_width = shape[2] * stride
    batch_size = shape[0]

    output_shape = [batch_size, out_height, out_width, out_filters]
    net = tf.nn.conv2d_transpose(net, weights, output_shape,
                                 [1, stride, stride, 1], padding='SAME')
    slim.batch_norm(net)
    return net

  def _BuildImageDecoder(self, cross_conved_images):
    """Decode the cross_conved feature maps into the predicted images."""
    nets = []
    for i, cross_conved_image in enumerate(cross_conved_images):
      with tf.variable_scope('image_decoder_%d' % i):
        stride = 64 / cross_conved_image.get_shape().as_list()[1]
        # TODO(xpan): Alternative solution for upsampling?
        nets.append(self._Deconv(
            cross_conved_image, 64, kernel_size=3, stride=stride))

    net = tf.concat(3, nets)
    net = slim.conv2d(net, 128, [9, 9], padding='SAME', stride=1)
    net = slim.conv2d(net, 128, [1, 1], padding='SAME', stride=1)
    net = slim.conv2d(net, 3, [1, 1], padding='SAME', stride=1)
    self.diff_output = net
    sys.stderr.write('diff_output shape: %s\n' % self.diff_output.get_shape())
