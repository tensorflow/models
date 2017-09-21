# Copyright 2017 Google Inc.
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

"""Task towers for PixelDA model."""
import tensorflow as tf

slim = tf.contrib.slim


def add_task_specific_model(images,
                            hparams,
                            num_classes=10,
                            is_training=False,
                            reuse_private=False,
                            private_scope=None,
                            reuse_shared=False,
                            shared_scope=None):
  """Create a classifier for the given images.

  The classifier is composed of a few 'private' layers followed by a few
  'shared' layers. This lets us account for different image 'style', while
  sharing the last few layers as 'content' layers.

  Args:
    images: A `Tensor` of size [batch_size, height, width, 3].
    hparams: model hparams
    num_classes: The number of output classes.
    is_training: whether model is training
    reuse_private: Whether or not to reuse the private weights, which are the
      first few layers in the classifier
    private_scope: The name of the variable_scope for the private (unshared)
      components of the classifier.
    reuse_shared: Whether or not to reuse the shared weights, which are the last
      few layers in the classifier
    shared_scope: The name of the variable_scope for the shared components of
      the classifier.

  Returns:
    The logits, a `Tensor` of shape [batch_size, num_classes].

  Raises:
    ValueError: If hparams.task_classifier is an unknown value
  """

  model = hparams.task_tower
  # Make sure the classifier name shows up in graph
  shared_scope = shared_scope or (model + '_shared')
  kwargs = {
      'num_classes': num_classes,
      'is_training': is_training,
      'reuse_private': reuse_private,
      'reuse_shared': reuse_shared,
  }

  if private_scope:
    kwargs['private_scope'] = private_scope
  if shared_scope:
    kwargs['shared_scope'] = shared_scope

  quaternion_pred = None
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      activation_fn=tf.nn.relu,
      weights_regularizer=tf.contrib.layers.l2_regularizer(
          hparams.weight_decay_task_classifier)):
    with slim.arg_scope([slim.conv2d], padding='SAME'):
      if model == 'doubling_pose_estimator':
        logits, quaternion_pred = doubling_cnn_class_and_quaternion(
            images, num_private_layers=hparams.num_private_layers, **kwargs)
      elif model == 'mnist':
        logits, _ = mnist_classifier(images, **kwargs)
      elif model == 'svhn':
        logits, _ = svhn_classifier(images, **kwargs)
      elif model == 'gtsrb':
        logits, _ = gtsrb_classifier(images, **kwargs)
      elif model == 'pose_mini':
        logits, quaternion_pred = pose_mini_tower(images, **kwargs)
      else:
        raise ValueError('Unknown task classifier %s' % model)

  return logits, quaternion_pred


#####################################
# Classifiers used in the DSN paper #
#####################################


def mnist_classifier(images,
                     is_training=False,
                     num_classes=10,
                     reuse_private=False,
                     private_scope='mnist',
                     reuse_shared=False,
                     shared_scope='task_model'):
  """Creates the convolutional MNIST model from the gradient reversal paper.

  Note that since the output is a set of 'logits', the values fall in the
  interval of (-infinity, infinity). Consequently, to convert the outputs to a
  probability distribution over the characters, one will need to convert them
  using the softmax function:
        logits, endpoints = conv_mnist(images, is_training=False)
        predictions = tf.nn.softmax(logits)

  Args:
    images: the MNIST digits, a tensor of size [batch_size, 28, 28, 1].
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    num_classes: the number of output classes to use.

  Returns:
    the output logits, a tensor of size [batch_size, num_classes].
    a dictionary with key/values the layer names and tensors.
  """

  net = {}

  with tf.variable_scope(private_scope, reuse=reuse_private):
    net['conv1'] = slim.conv2d(images, 32, [5, 5], scope='conv1')
    net['pool1'] = slim.max_pool2d(net['conv1'], [2, 2], 2, scope='pool1')

  with tf.variable_scope(shared_scope, reuse=reuse_shared):
    net['conv2'] = slim.conv2d(net['pool1'], 48, [5, 5], scope='conv2')
    net['pool2'] = slim.max_pool2d(net['conv2'], [2, 2], 2, scope='pool2')
    net['fc3'] = slim.fully_connected(
        slim.flatten(net['pool2']), 100, scope='fc3')
    net['fc4'] = slim.fully_connected(
        slim.flatten(net['fc3']), 100, scope='fc4')
    logits = slim.fully_connected(
        net['fc4'], num_classes, activation_fn=None, scope='fc5')
  return logits, net


def svhn_classifier(images,
                    is_training=False,
                    num_classes=10,
                    reuse_private=False,
                    private_scope=None,
                    reuse_shared=False,
                    shared_scope='task_model'):
  """Creates the convolutional SVHN model from the gradient reversal paper.

  Note that since the output is a set of 'logits', the values fall in the
  interval of (-infinity, infinity). Consequently, to convert the outputs to a
  probability distribution over the characters, one will need to convert them
  using the softmax function:
        logits = mnist.Mnist(images, is_training=False)
        predictions = tf.nn.softmax(logits)

  Args:
    images: the SVHN digits, a tensor of size [batch_size, 40, 40, 3].
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    num_classes: the number of output classes to use.

  Returns:
    the output logits, a tensor of size [batch_size, num_classes].
    a dictionary with key/values the layer names and tensors.
  """

  net = {}

  with tf.variable_scope(private_scope, reuse=reuse_private):
    net['conv1'] = slim.conv2d(images, 64, [5, 5], scope='conv1')
    net['pool1'] = slim.max_pool2d(net['conv1'], [3, 3], 2, scope='pool1')

  with tf.variable_scope(shared_scope, reuse=reuse_shared):
    net['conv2'] = slim.conv2d(net['pool1'], 64, [5, 5], scope='conv2')
    net['pool2'] = slim.max_pool2d(net['conv2'], [3, 3], 2, scope='pool2')
    net['conv3'] = slim.conv2d(net['pool2'], 128, [5, 5], scope='conv3')

    net['fc3'] = slim.fully_connected(
        slim.flatten(net['conv3']), 3072, scope='fc3')
    net['fc4'] = slim.fully_connected(
        slim.flatten(net['fc3']), 2048, scope='fc4')

    logits = slim.fully_connected(
        net['fc4'], num_classes, activation_fn=None, scope='fc5')

  return logits, net


def gtsrb_classifier(images,
                     is_training=False,
                     num_classes=43,
                     reuse_private=False,
                     private_scope='gtsrb',
                     reuse_shared=False,
                     shared_scope='task_model'):
  """Creates the convolutional GTSRB model from the gradient reversal paper.

  Note that since the output is a set of 'logits', the values fall in the
  interval of (-infinity, infinity). Consequently, to convert the outputs to a
  probability distribution over the characters, one will need to convert them
  using the softmax function:
        logits = mnist.Mnist(images, is_training=False)
        predictions = tf.nn.softmax(logits)

  Args:
    images: the SVHN digits, a tensor of size [batch_size, 40, 40, 3].
    is_training: specifies whether or not we're currently training the model.
      This variable will determine the behaviour of the dropout layer.
    num_classes: the number of output classes to use.
    reuse_private: Whether or not to reuse the private components of the model.
    private_scope: The name of the private scope.
    reuse_shared: Whether or not to reuse the shared components of the model.
    shared_scope: The name of the shared scope.

  Returns:
    the output logits, a tensor of size [batch_size, num_classes].
    a dictionary with key/values the layer names and tensors.
  """

  net = {}

  with tf.variable_scope(private_scope, reuse=reuse_private):
    net['conv1'] = slim.conv2d(images, 96, [5, 5], scope='conv1')
    net['pool1'] = slim.max_pool2d(net['conv1'], [2, 2], 2, scope='pool1')
  with tf.variable_scope(shared_scope, reuse=reuse_shared):
    net['conv2'] = slim.conv2d(net['pool1'], 144, [3, 3], scope='conv2')
    net['pool2'] = slim.max_pool2d(net['conv2'], [2, 2], 2, scope='pool2')
    net['conv3'] = slim.conv2d(net['pool2'], 256, [5, 5], scope='conv3')
    net['pool3'] = slim.max_pool2d(net['conv3'], [2, 2], 2, scope='pool3')

    net['fc3'] = slim.fully_connected(
        slim.flatten(net['pool3']), 512, scope='fc3')
    logits = slim.fully_connected(
        net['fc3'], num_classes, activation_fn=None, scope='fc4')

    return logits, net


#########################
# pose_mini task towers #
#########################


def pose_mini_tower(images,
                    num_classes=11,
                    is_training=False,
                    reuse_private=False,
                    private_scope='pose_mini',
                    reuse_shared=False,
                    shared_scope='task_model'):
  """Task tower for the pose_mini dataset."""

  with tf.variable_scope(private_scope, reuse=reuse_private):
    net = slim.conv2d(images, 32, [5, 5], scope='conv1')
    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool1')
  with tf.variable_scope(shared_scope, reuse=reuse_shared):
    net = slim.conv2d(net, 64, [5, 5], scope='conv2')
    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')
    net = slim.flatten(net)

    net = slim.fully_connected(net, 128, scope='fc3')
    net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')
    with tf.variable_scope('quaternion_prediction'):
      quaternion_pred = slim.fully_connected(
          net, 4, activation_fn=tf.tanh, scope='fc_q')
      quaternion_pred = tf.nn.l2_normalize(quaternion_pred, 1)

    logits = slim.fully_connected(
        net, num_classes, activation_fn=None, scope='fc4')

    return logits, quaternion_pred


def doubling_cnn_class_and_quaternion(images,
                                      num_private_layers=1,
                                      num_classes=10,
                                      is_training=False,
                                      reuse_private=False,
                                      private_scope='doubling_cnn',
                                      reuse_shared=False,
                                      shared_scope='task_model'):
  """Alternate conv, pool while doubling filter count."""
  net = images
  depth = 32
  layer_id = 1

  with tf.variable_scope(private_scope, reuse=reuse_private):
    while num_private_layers > 0 and net.shape.as_list()[1] > 5:
      net = slim.conv2d(net, depth, [3, 3], scope='conv%s' % layer_id)
      net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool%s' % layer_id)
      depth *= 2
      layer_id += 1
      num_private_layers -= 1

  with tf.variable_scope(shared_scope, reuse=reuse_shared):
    while net.shape.as_list()[1] > 5:
      net = slim.conv2d(net, depth, [3, 3], scope='conv%s' % layer_id)
      net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool%s' % layer_id)
      depth *= 2
      layer_id += 1

    net = slim.flatten(net)
    net = slim.fully_connected(net, 100, scope='fc1')
    net = slim.dropout(net, 0.5, is_training=is_training, scope='dropout')
    quaternion_pred = slim.fully_connected(
        net, 4, activation_fn=tf.tanh, scope='fc_q')
    quaternion_pred = tf.nn.l2_normalize(quaternion_pred, 1)

    logits = slim.fully_connected(
        net, num_classes, activation_fn=None, scope='fc_logits')

    return logits, quaternion_pred
