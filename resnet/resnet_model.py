# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""ResNet model.

Related papers:
https://arxiv.org/pdf/1603.05027v2.pdf
https://arxiv.org/pdf/1512.03385v1.pdf
https://arxiv.org/pdf/1605.07146v1.pdf
"""
from collections import namedtuple

import random
import numpy as np
import tensorflow as tf

from tensorflow.python.training import moving_averages


HParams = namedtuple('HParams',
                     'batch_size, num_classes, num_gpus, initial_learning_rate, '
                     'lr_decay_steps, lr_decay_factor, optimizer, num_layers, '
                     'prob_depth, use_bottleneck, weight_decay_rate, relu_leakiness')

TOWER_NAME = 'tower'

class ResNet(object):
  """ResNet model."""
  def __init__(self, hps, images, labels, train):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self.hps = hps
    self._images = images
    self.labels = labels
    self.train = train

    self._extra_train_ops = []

  def inference(self):
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    if self.train:
      self._build_train_op()
    else:
      self.logits, self.predictions = self._build_model(_images)
    self.summaries = tf.merge_all_summaries()

  def _build_train_op(self):
    """Build training specific ops for the graph."""
    self.learning_rate = tf.train.exponential_decay(self.hps.initial_learning_rate,
                                                    self.global_step,
                                                    self.hps.lr_decay_steps,
                                                    self.hps.lr_decay_factor,
                                                    staircase=True)
    tf.scalar_summary('learning rate', self.learning_rate)

    if self.hps.optimizer == 'SGD':
      optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
    elif self.hps.optimizer == 'Momentum':
      optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)

    assert self.hps.batch_size % self.hps.num_gpus == 0, ('Batch size must be divisible by number of GPUs')
    self.split_batch_size = int(self.hps.batch_size / self.hps.num_gpus)

    # Split the batch of images and labels for towers.
    images_splits = tf.split(0, self.hps.num_gpus, self._images)
    labels_splits = tf.split(0, self.hps.num_gpus, self.labels)

    # Calculate the gradients for each model tower.
    tower_grads = []
    for i in xrange(self.hps.num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
          # Forward Propagation
          logits, predictions = self._build_model(images_splits[i])

          # Calculate loss
          self._build_loss(logits, labels_splits[i])

          # Reuse variables for the next tower.
          tf.get_variable_scope().reuse_variables()

          # Retain the summaries from the final tower.
          summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

          # Calculate the gradients for the batch of data on this gpu.
          _grads = optimizer.compute_gradients(self.loss)

          # Merge the gradients
          tower_grads.append(_grads)

    # Moving Average
    moving_avg = tf.train.ExponentialMovingAverage(0.99, num_updates=self.global_step, 
                                                   name='moving_avg')
    self._extra_train_ops.append(moving_avg.apply([self.loss]))
    tf.scalar_summary('losses', moving_avg.average(self.loss))


    grads = self._average_gradients(tower_grads)

    #for grad, var in grads:
    #  if grad is not None:
    #    summaries.append(
    #        tf.histogram_summary(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_op = optimizer.apply_gradients(grads, global_step=self.global_step, 
                                         name='train_step')

    # Add histograms for trainable variables.
    #for var in tf.trainable_variables():
    #  summaries.append(tf.histogram_summary(var.op.name, var))

    train_ops = [apply_op] + self._extra_train_ops
    self.train_op = tf.group(*train_ops)

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self, input_images):
    """Build the core model within the graph."""
    with tf.variable_scope('init'):
      x = self._conv('init_conv', input_images, 3, 3, 16, self._stride_arr(1))
    
    self.L = 1.5 / (self.hps.num_layers - 2)
    strides = [1, 2, 2]
    activate_before_residual = [True, True, True]
    if self.hps.use_bottleneck:
      res_func = self._bottleneck_residual
      filters = [16, 64, 128, 256]
    else:
      res_func = self._residual
      filters = [16, 16, 32, 64]
      # Uncomment the following codes to use w28-10 wide residual network.
      # It is more memory efficient than very deep residual network and has
      # comparably good performance.
      # https://arxiv.org/pdf/1605.07146v1.pdf
      # filters = [16, 160, 320, 640]
      # Update hps.num_residual_units to 9
    
    assert (self.hps.num_layers - 2) % 9 == 0
    num_residual_units = (self.hps.num_layers - 2) / 9 - 1

    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]), 1,
                   activate_before_residual[0])
    for i in xrange(1, num_residual_units):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), i+1, True)

    with tf.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]), num_residual_units+1, 
                   activate_before_residual[1])
    for i in xrange(1, num_residual_units):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), num_residual_units+i+1, True)

    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]), 2*num_residual_units+1,
                   activate_before_residual[2])
    for i in xrange(1, num_residual_units):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), 2*num_residual_units+i+1, True)

    with tf.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._global_avg_pool(x)

    with tf.variable_scope('logit'):
      logits = self._fully_connected(x, self.hps.num_classes)
      predictions = tf.nn.softmax(logits)

    return logits, predictions

  def _build_loss(self, logits, input_labels):
    sparse_labels = tf.reshape(input_labels, [self.split_batch_size, 1])
    indices = tf.reshape(tf.range(self.split_batch_size), [self.split_batch_size, 1])
    concated = tf.concat(1, [indices, sparse_labels])
    one_hot_labels = tf.sparse_to_dense(concated, [self.split_batch_size, self.hps.num_classes], 1.0, 0.0)
    xent = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_labels)
    self.loss = tf.reduce_mean(xent, name='xent')
    self.loss += self._decay()

  def _average_gradients(self, tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      # Note that each grad_and_vars looks like the following:
      #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
      grads = []
      for g, _ in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
        expanded_g = tf.expand_dims(g, 0)

        # Append on a 'tower' dimension which we will average over below.
        grads.append(expanded_g)

      # Average over the 'tower' dimension.
      grad = tf.concat(0, grads)
      grad = tf.reduce_mean(grad, 0)

      # Keep in mind that the Variables are redundant because they are shared
      # across towers. So .. we will just return the first tower's pointer to
      # the Variable.
      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    return average_grads

  # TODO(xpan): Consider batch_norm in contrib/layers/python/layers/layers.py
  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.variable_scope(name):
      params_shape = [x.get_shape()[-1]]

      beta = tf.get_variable(
          'beta', params_shape, tf.float32,
          initializer=tf.constant_initializer(0.0, tf.float32))
      gamma = tf.get_variable(
          'gamma', params_shape, tf.float32,
          initializer=tf.constant_initializer(1.0, tf.float32))

      if self.train:
        mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')

        moving_mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)

        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_mean, mean, 0.9))
        self._extra_train_ops.append(moving_averages.assign_moving_average(
            moving_variance, variance, 0.9))
      else:
        mean = tf.get_variable(
            'moving_mean', params_shape, tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False)
        variance = tf.get_variable(
            'moving_variance', params_shape, tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False)
        tf.histogram_summary(mean.op.name, mean)
        tf.histogram_summary(variance.op.name, variance)
      # elipson used to be 1e-5. Maybe 0.001 solves NaN problem in deeper net.
      y = tf.nn.batch_normalization(
          x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y

  def _residual(self, x, in_filter, out_filter, stride, layer,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])

      # Stochastic Depth
      def fn1(): return x + orig_x
      def fn2(): return orig_x
      prob = 1-self.L*layer*(1-self.hps.prob_depth)
      output = tf.cond(tf.logical_or(tf.greater_equal(prob - random.uniform(0, 1), 0), not self.train), fn1, fn2)

    tf.logging.info('image after unit %s', output.get_shape())
    return output

  def _bottleneck_residual(self, x, in_filter, out_filter, stride, layer,
                           activate_before_residual=False):
    """Bottleneck resisual unit with 3 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('common_bn_relu'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        orig_x = x
    else:
      with tf.variable_scope('residual_bn_relu'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 1, in_filter, out_filter/4, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv2', x, 3, out_filter/4, out_filter/4, [1, 1, 1, 1])

    with tf.variable_scope('sub3'):
      x = self._batch_norm('bn3', x)
      x = self._relu(x, self.hps.relu_leakiness)
      x = self._conv('conv3', x, 1, out_filter/4, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)

      # Stochastic Depth
      def fn1(): return x + orig_x
      def fn2(): return orig_x
      prob = 1-self.L*layer*(1-self.hps.prob_depth)
      output = tf.cond(tf.logical_or(tf.greater_equal(prob - random.uniform(0, 1), 0), not self.train), fn1, fn2)

    tf.logging.info('image after unit %s', output.get_shape())
    return output

  def _decay(self):
    """L2 weight decay loss."""
    losses = []
    for var in tf.trainable_variables():
      if var.op.name.find(r'DW') > 0:
        losses.append(tf.nn.l2_loss(var))
        # tf.histogram_summary(var.op.name, var)

    return tf.mul(self.hps.weight_decay_rate, tf.add_n(losses))

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.select(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    if self.train:
      input_size = self.split_batch_size
    else:
      input_size = self.hps.batch_size

    x = tf.reshape(x, [input_size, -1])
    w = tf.get_variable(
        'DW', [x.get_shape()[1], out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])
