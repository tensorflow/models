# Copyright 2018 MLBenchmark Group. All Rights Reserved.
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

"""Defines DualNet model, the architecture of the policy and value network.

Both the policy and value networks share ResNet as a majority of their
architecture. The neural network consists of a residual tower (ResNet) and two
separate "heads" for computing the policy and value respectively.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def _batch_norm(inputs, training, center=True, scale=True):
  """Performs a batch normalization using a standard set of parameters."""
  return tf.layers.batch_normalization(
      inputs=inputs, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
      center=center, scale=scale, fused=True, training=training)


def _conv2d(inputs, filters, kernel_size):
  """Performs 2D convolution with a standard set of parameters."""
  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size,
      padding='same')


def _conv_block(inputs, filters, kernel_size, training):
  """A convolutional block.

  Args:
    inputs: A tensor representing a batch of input features with shape
      [BATCH_SIZE, go.N, go.N, features.NEW_FEATURES_PLANES].
    filters: The number of filters for network layers in residual tower.
    kernel_size: The kernel to be used in conv2d.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.

  Returns:
    The output tensor of the convolutional block layer.
  """
  conv = _conv2d(inputs, filters, kernel_size)
  batchn = _batch_norm(conv, training)

  output = tf.nn.relu(batchn)

  return output


def _res_block(inputs, filters, kernel_size, training):
  """A residual block.

  Args:
    inputs: A tensor representing a batch of input features with shape
      [BATCH_SIZE, go.N, go.N, features.NEW_FEATURES_PLANES].
    filters: The number of filters for network layers in residual tower.
    kernel_size: The kernel to be used in conv2d.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.

  Returns:
    The output tensor of the residual block layer.
  """

  initial_output = _conv_block(inputs, filters, kernel_size, training)

  int_layer2_conv = _conv2d(initial_output, filters, kernel_size)
  int_layer2_batchn = _batch_norm(int_layer2_conv, training)

  output = tf.nn.relu(inputs + int_layer2_batchn)

  return output


class Model(object):
  """Base class for building the DualNet Model."""

  def __init__(self, num_filters, num_shared_layers, fc_width, go_size):
    """Initialize a model for computing the policy and value in RL.

    Args:
      num_filters: Number of filters (AlphaGoZero used 256). We use 128 by
        default for a 19x19 go board, and 32 for 9x9 size.
      num_shared_layers: Number of shared residual blocks.  AGZ used both 19
        and 39. Here we use 19 for 19x19 size and 9 for 9x9 size because it's
        faster to train.
      fc_width: Dimensionality of the fully connected linear layer.
      go_size: A single integer for the board size.
    """
    self.num_filters = num_filters
    self.num_shared_layers = num_shared_layers
    self.fc_width = fc_width
    self.go_size = go_size
    self.kernel_size = [3, 3]  # kernel size is from AGZ paper

  def __call__(self, inputs, training):
    """Add operations to classify a batch of input Go features.

    Args:
      inputs: A Tensor representing a batch of input Go features with shape
        [BATCH_SIZE, go_size, go_size, features.NEW_FEATURES_PLANES]
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      policy_logits: A vector of size self.go_size * self.go_size + 1
        corresponding to the policy logit probabilities for all intersections
        and the pass move.
      value_logits: A scalar for the value logits output
    """
    initial_output = _conv_block(
        inputs=inputs, filters=self.num_filters,
        kernel_size=self.kernel_size, training=training)
    # the shared stack
    shared_output = initial_output
    for _ in range(self.num_shared_layers):
      shared_output = _res_block(
          inputs=shared_output, filters=self.num_filters,
          kernel_size=self.kernel_size, training=training)

    # policy head
    policy_conv2d = _conv2d(inputs=shared_output, filters=2, kernel_size=[1, 1])
    policy_batchn = _batch_norm(inputs=policy_conv2d, training=training,
                                center=False, scale=False)
    policy_relu = tf.nn.relu(policy_batchn)
    policy_logits = tf.layers.dense(
        tf.reshape(policy_relu, [-1, self.go_size * self.go_size * 2]),
        self.go_size * self.go_size + 1)

    # value head
    value_conv2d = _conv2d(shared_output, filters=1, kernel_size=[1, 1])
    value_batchn = _batch_norm(value_conv2d, training,
                               center=False, scale=False)
    value_relu = tf.nn.relu(value_batchn)
    value_fc_hidden = tf.nn.relu(tf.layers.dense(
        tf.reshape(value_relu, [-1, self.go_size * self.go_size]),
        self.fc_width))
    value_logits = tf.reshape(tf.layers.dense(value_fc_hidden, 1), [-1])

    return policy_logits, value_logits


def model_fn(features, labels, mode, params, config=None):  # pylint: disable=unused-argument
  """DualNet model function.

  Args:
    features: tensor with shape
      [BATCH_SIZE, self.go_size, self.go_size, features.NEW_FEATURES_PLANES]
    labels: dict from string to tensor with shape
      'pi_tensor': [BATCH_SIZE, self.go_size * self.go_size + 1]
      'value_tensor': [BATCH_SIZE]
    mode: a tf.estimator.ModeKeys (batchnorm params update for TRAIN only)
    params: a dict of hyperparams
    config: ignored; is required by Estimator API.
  Returns:
    EstimatorSpec parameterized according to the input params and the current
    mode.
  """
  model = Model(params.num_filters, params.num_shared_layers, params.fc_width,
                params.go_size)
  policy_logits, value_logits = model(
      features, mode == tf.estimator.ModeKeys.TRAIN)

  policy_output = tf.nn.softmax(policy_logits, name='policy_output')
  value_output = tf.nn.tanh(value_logits, name='value_output')

  # Calculate model loss (combined_cost)
  policy_cost = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
          logits=policy_logits, labels=labels['pi_tensor']))
  value_cost = tf.reduce_mean(
      tf.square(value_output - labels['value_tensor']))
  l2_cost = params.l2_strength * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()
       if 'bias' not in v.name])
  combined_cost = policy_cost + value_cost + l2_cost

  # For logging purpose
  policy_entropy = -tf.reduce_mean(tf.reduce_sum(
      policy_output * tf.log(policy_output), axis=1))

  # Get model train ops
  global_step = tf.train.get_or_create_global_step()
  boundaries = [int(1e6), int(2e6)]
  values = [1e-2, 1e-3, 1e-4]
  learning_rate = tf.train.piecewise_constant(
      global_step, boundaries, values)
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = tf.train.MomentumOptimizer(
        learning_rate, params.momentum).minimize(
            combined_cost, global_step=global_step)

  # Create multiple tensors for logging purpose
  metric_ops = {
      'accuracy': tf.metrics.accuracy(labels=labels['pi_tensor'],
                                      predictions=policy_output,
                                      name='accuracy_op'),
      'policy_cost': tf.metrics.mean(policy_cost),
      'value_cost': tf.metrics.mean(value_cost),
      'l2_cost': tf.metrics.mean(l2_cost),
      'policy_entropy': tf.metrics.mean(policy_entropy),
      'combined_cost': tf.metrics.mean(combined_cost),
  }
  for metric_name, metric_op in metric_ops.items():
    tf.summary.scalar(metric_name, metric_op[1])

  # Return tf.estimator.EstimatorSpec
  return tf.estimator.EstimatorSpec(
      mode=mode,
      predictions={
          'policy_output': policy_output,
          'value_output': value_output,
      },
      loss=combined_cost,
      train_op=train_op,
      eval_metric_ops=metric_ops)
