# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains convenience wrappers for typical Neural Network TensorFlow layers.

   Additionally it maintains a collection with update_ops that need to be
   updated after the ops have been computed, for exmaple to update moving means
   and moving variances of batch_norm.

   Ops that have different behavior during training or eval have an is_training
   parameter. Additionally Ops that contain variables.variable have a trainable
   parameter, which control if the ops variables are trainable or not.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf

from tensorflow.python.training import moving_averages

from inception.slim import losses
from inception.slim import scopes
from inception.slim import variables

# Used to keep the update ops done by batch_norm.
UPDATE_OPS_COLLECTION = '_update_ops_'


@scopes.add_arg_scope
def batch_norm(inputs,
               decay=0.999,
               scale=False,
               epsilon=0.001,
               moving_vars='moving_vars',
               activation=None,
               is_training=True,
               trainable=True,
               restore=True,
               scope=None):
  """Adds a Batch Normalization layer.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels]
            or [batch_size, channels].
    decay: decay for the moving average.
    scale: If True, multiply by gamma. If False, gamma is
      not used. When the next layer is linear (also e.g. ReLU), this can be
      disabled since the scaling can be done by the next layer.
    epsilon: small float added to variance to avoid dividing by zero.
    moving_vars: collection to store the moving_mean and moving_variance.
    activation: activation function.
    is_training: whether or not the model is in training mode.
    trainable: whether or not the variables should be trainable or not.
    restore: whether or not the variables should be marked for restore.
    scope: Optional scope for variable_op_scope.

  Returns:
    a tensor representing the output of the operation.

  """
  inputs_shape = inputs.get_shape()
  with tf.variable_op_scope([inputs], scope, 'BatchNorm'):
    axis = range(len(inputs_shape) - 1)
    params_shape = inputs_shape[-1:]
    with scopes.arg_scope([variables.variable], restore=restore):
      # Allocate parameters for the beta and gamma of the normalization.
      beta = variables.variable('beta',
                                params_shape,
                                initializer=tf.zeros_initializer,
                                trainable=trainable)
      if scale:
        gamma = variables.variable('gamma',
                                   params_shape,
                                   initializer=tf.ones,
                                   trainable=trainable)
      else:
        gamma = None
      # Create moving_mean and moving_variance add them to moving_vars and
      # GraphKeys.MOVING_AVERAGE_VARIABLES collections.
      with scopes.arg_scope([variables.variable], trainable=False,
                            collections=[
                                moving_vars,
                                tf.GraphKeys.MOVING_AVERAGE_VARIABLES]):
        moving_mean = variables.variable('moving_mean',
                                         params_shape,
                                         initializer=tf.zeros_initializer)
        moving_variance = variables.variable('moving_variance',
                                             params_shape,
                                             initializer=tf.ones)
    if is_training:
      # Calculate the moments based on the individual batch.
      mean, variance = tf.nn.moments(inputs, axis)

      update_moving_mean = moving_averages.assign_moving_average(
          moving_mean, mean, decay)
      tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
      update_moving_variance = moving_averages.assign_moving_average(
          moving_variance, variance, decay)
      tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
    else:
      # Just use the moving_mean and moving_variance.
      mean = moving_mean
      variance = moving_variance
    # Normalize the activations.
    outputs = tf.nn.batch_normalization(
        inputs, mean, variance, beta, gamma, epsilon)
    outputs.set_shape(inputs.get_shape())
    if activation:
      outputs = activation(outputs)
    return outputs


@scopes.add_arg_scope
def conv2d(inputs,
           num_filters_out,
           kernel_size,
           stride=1,
           padding='SAME',
           activation=tf.nn.relu,
           stddev=0.01,
           bias=0.0,
           weight_decay=0,
           batch_norm_params=None,
           is_training=True,
           trainable=True,
           restore=True,
           scope=None):
  """Adds a 2D convolution followed by an optional batch_norm layer.

  conv2d creates a variable called 'weights', representing the convolutional
  kernel, that is convolved with the input. If `batch_norm_params` is None, a
  second variable called 'biases' is added to the result of the convolution
  operation.

  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_filters_out: the number of output filters.
    kernel_size: a 2-D list comprising of the height and width of the filters.
    stride: the stride in height and width of the convolution.
    padding: one of 'VALID' or 'SAME'.
    activation: activation function.
    stddev: standard deviation of the truncated guassian weight distribution.
    bias: the initial value of the biases.
    weight_decay: the weight decay.
    batch_norm_params: parameters for the batch_norm. If is None don't use it.
    is_training: whether or not the model is in training mode.
    trainable: whether or not the variables should be trainable or not.
    restore: whether or not the variables should be marked for restore.
    scope: Optional scope for variable_op_scope.

  Returns:
    a tensor representing the output of the operation.

  Raises:
    ValueError: if 'kernel_size' is not a 2-D list.
  """
  if len(kernel_size) != 2:
    raise ValueError('kernel_size must be a 2-D list.')
  with tf.variable_op_scope([inputs], scope, 'Conv'):
    num_filters_in = inputs.get_shape()[-1]
    weights_shape = [kernel_size[0], kernel_size[1],
                     num_filters_in, num_filters_out]
    weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
    l2_regularizer = lambda t: losses.l2_loss(t, weight_decay)
    weights = variables.variable('weights',
                                 shape=weights_shape,
                                 initializer=weights_initializer,
                                 regularizer=l2_regularizer,
                                 trainable=trainable,
                                 restore=restore)
    conv = tf.nn.conv2d(inputs, weights, [1, stride, stride, 1],
                        padding=padding)
    if batch_norm_params is not None:
      with scopes.arg_scope([batch_norm], is_training=is_training,
                            trainable=trainable, restore=restore):
        outputs = batch_norm(conv, **batch_norm_params)
    else:
      bias_shape = [num_filters_out,]
      bias_initializer = tf.constant_initializer(bias)
      biases = variables.variable('biases',
                                  shape=bias_shape,
                                  initializer=bias_initializer,
                                  trainable=trainable,
                                  restore=restore)
      outputs = tf.nn.bias_add(conv, biases)
    if activation:
      outputs = activation(outputs)
    return outputs


@scopes.add_arg_scope
def fc(inputs,
       num_units_out,
       activation=tf.nn.relu,
       stddev=0.01,
       bias=0.0,
       weight_decay=0,
       batch_norm_params=None,
       is_training=True,
       trainable=True,
       restore=True,
       scope=None):
  """Adds a fully connected layer followed by an optional batch_norm layer.

  FC creates a variable called 'weights', representing the fully connected
  weight matrix, that is multiplied by the input. If `batch_norm` is None, a
  second variable called 'biases' is added to the result of the initial
  vector-matrix multiplication.

  Args:
    inputs: a [B x N] tensor where B is the batch size and N is the number of
            input units in the layer.
    num_units_out: the number of output units in the layer.
    activation: activation function.
    stddev: the standard deviation for the weights.
    bias: the initial value of the biases.
    weight_decay: the weight decay.
    batch_norm_params: parameters for the batch_norm. If is None don't use it.
    is_training: whether or not the model is in training mode.
    trainable: whether or not the variables should be trainable or not.
    restore: whether or not the variables should be marked for restore.
    scope: Optional scope for variable_op_scope.

  Returns:
     the tensor variable representing the result of the series of operations.
  """
  with tf.variable_op_scope([inputs], scope, 'FC'):
    num_units_in = inputs.get_shape()[1]
    weights_shape = [num_units_in, num_units_out]
    weights_initializer = tf.truncated_normal_initializer(stddev=stddev)
    l2_regularizer = lambda t: losses.l2_loss(t, weight_decay)
    weights = variables.variable('weights',
                                 shape=weights_shape,
                                 initializer=weights_initializer,
                                 regularizer=l2_regularizer,
                                 trainable=trainable,
                                 restore=restore)
    if batch_norm_params is not None:
      outputs = tf.matmul(inputs, weights)
      with scopes.arg_scope([batch_norm], is_training=is_training,
                            trainable=trainable, restore=restore):
        outputs = batch_norm(outputs, **batch_norm_params)
    else:
      bias_shape = [num_units_out,]
      bias_initializer = tf.constant_initializer(bias)
      biases = variables.variable('biases',
                                  shape=bias_shape,
                                  initializer=bias_initializer,
                                  trainable=trainable,
                                  restore=restore)
      outputs = tf.nn.xw_plus_b(inputs, weights, biases)
    if activation:
      outputs = activation(outputs)
    return outputs


def one_hot_encoding(labels, num_classes, scope=None):
  """Transform numeric labels into onehot_labels.

  Args:
    labels: [batch_size] target labels.
    num_classes: total number of classes.
    scope: Optional scope for op_scope.
  Returns:
    one hot encoding of the labels.
  """
  with tf.op_scope([labels], scope, 'OneHotEncoding'):
    batch_size = labels.get_shape()[0]
    indices = tf.expand_dims(tf.range(0, batch_size), 1)
    labels = tf.cast(tf.expand_dims(labels, 1), indices.dtype)
    concated = tf.concat(1, [indices, labels])
    onehot_labels = tf.sparse_to_dense(
        concated, tf.pack([batch_size, num_classes]), 1.0, 0.0)
    onehot_labels.set_shape([batch_size, num_classes])
    return onehot_labels


@scopes.add_arg_scope
def max_pool(inputs, kernel_size, stride=2, padding='VALID', scope=None):
  """Adds a Max Pooling layer.

  It is assumed by the wrapper that the pooling is only done per image and not
  in depth or batch.

  Args:
    inputs: a tensor of size [batch_size, height, width, depth].
    kernel_size: the size of the pooling kernel over which the op is computed.
    stride: the stride in height and width of the convolution.
    padding: the padding method, either 'VALID' or 'SAME'.
    scope: Optional scope for op_scope.

  Returns:
    a tensor representing the results of the pooling operation.
  Raises:
    ValueError: if 'kernel_size' is not a 2-D list
  """
  if len(kernel_size) != 2:
    raise ValueError('kernel_size must be a 2-D list.')
  with tf.op_scope([inputs], scope, 'MaxPool'):
    return tf.nn.max_pool(inputs,
                          ksize=[1, kernel_size[0], kernel_size[1], 1],
                          strides=[1, stride, stride, 1],
                          padding=padding)


@scopes.add_arg_scope
def avg_pool(inputs, kernel_size, stride=2, padding='VALID', scope=None):
  """Adds a Avg Pooling layer.

  It is assumed by the wrapper that the pooling is only done per image and not
  in depth or batch.

  Args:
    inputs: a tensor of size [batch_size, height, width, depth].
    kernel_size: the size of the pooling kernel over which the op is computed.
    stride: the stride in height and width of the convolution.
    padding: the padding method, either 'VALID' or 'SAME'.
    scope: Optional scope for op_scope.

  Returns:
    a tensor representing the results of the pooling operation.
  Raises:
    ValueError: if 'kernel_size' is not a 2-D list
  """
  if len(kernel_size) != 2:
    raise ValueError('kernel_size must be a 2-D list.')
  with tf.op_scope([inputs], scope, 'AvgPool'):
    return tf.nn.avg_pool(inputs,
                          ksize=[1, kernel_size[0], kernel_size[1], 1],
                          strides=[1, stride, stride, 1],
                          padding=padding)


@scopes.add_arg_scope
def dropout(inputs, keep_prob=0.5, is_training=True, scope=None):
  """Returns a dropout layer applied to the input.

  Args:
    inputs: the tensor to pass to the Dropout layer.
    keep_prob: the probability of dropping each input unit.
    is_training: whether or not the model is in training mode. If so, dropout is
    applied and values scaled. Otherwise, inputs is returned.
    scope: Optional scope for op_scope.

  Returns:
    a tensor representing the output of the operation.
  """
  if is_training and keep_prob > 0:
    with tf.op_scope([inputs], scope, 'Dropout'):
      return tf.nn.dropout(inputs, keep_prob)
  else:
    return inputs


def flatten(inputs, scope=None):
  """Flattens the input while maintaining the batch_size.

    Assumes that the first dimension represents the batch.

  Args:
    inputs: a tensor of size [batch_size, ...].
    scope: Optional scope for op_scope.

  Returns:
    a flattened tensor with shape [batch_size, k].
  Raises:
    ValueError: if inputs.shape is wrong.
  """
  if len(inputs.get_shape()) < 2:
    raise ValueError('Inputs must be have a least 2 dimensions')
  dims = inputs.get_shape()[1:]
  k = dims.num_elements()
  with tf.op_scope([inputs], scope, 'Flatten'):
    return tf.reshape(inputs, [-1, k])


def repeat_op(repetitions, inputs, op, *args, **kwargs):
  """Build a sequential Tower starting from inputs by using an op repeatedly.

  It creates new scopes for each operation by increasing the counter.
  Example: given repeat_op(3, _, ops.conv2d, 64, [3, 3], scope='conv1')
    it will repeat the given op under the following variable_scopes:
      conv1/Conv
      conv1/Conv_1
      conv1/Conv_2

  Args:
    repetitions: number or repetitions.
    inputs: a tensor of size [batch_size, height, width, channels].
    op: an operation.
    *args: args for the op.
    **kwargs: kwargs for the op.

  Returns:
    a tensor result of applying the operation op, num times.
  Raises:
    ValueError: if the op is unknown or wrong.
  """
  scope = kwargs.pop('scope', None)
  with tf.variable_op_scope([inputs], scope, 'RepeatOp'):
    tower = inputs
    for _ in range(repetitions):
      tower = op(tower, *args, **kwargs)
    return tower
