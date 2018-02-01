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
"""Domain Adaptation Loss Functions.

The following domain adaptation loss functions are defined:

- Maximum Mean Discrepancy (MMD).
  Relevant paper:
    Gretton, Arthur, et al.,
    "A kernel two-sample test."
    The Journal of Machine Learning Research, 2012

- Correlation Loss on a batch.
"""
from functools import partial
import tensorflow as tf

import grl_op_grads  # pylint: disable=unused-import
import grl_op_shapes  # pylint: disable=unused-import
import grl_ops
import utils
slim = tf.contrib.slim


################################################################################
# SIMILARITY LOSS
################################################################################
def maximum_mean_discrepancy(x, y, kernel=utils.gaussian_kernel_matrix):
  r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.

  Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
  the distributions of x and y. Here we use the kernel two sample estimate
  using the empirical mean of the two distributions.

  MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },

  where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.

  Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.

  Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
  """
  with tf.name_scope('MaximumMeanDiscrepancy'):
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))

    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
  return cost


def mmd_loss(source_samples, target_samples, weight, scope=None):
  """Adds a similarity loss term, the MMD between two representations.

  This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
  different Gaussian kernels.

  Args:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the MMD loss.
    scope: optional name scope for summary tags.

  Returns:
    a scalar tensor representing the MMD loss value.
  """
  sigmas = [
      1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
      1e3, 1e4, 1e5, 1e6
  ]
  gaussian_kernel = partial(
      utils.gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

  loss_value = maximum_mean_discrepancy(
      source_samples, target_samples, kernel=gaussian_kernel)
  loss_value = tf.maximum(1e-4, loss_value) * weight
  assert_op = tf.Assert(tf.is_finite(loss_value), [loss_value])
  with tf.control_dependencies([assert_op]):
    tag = 'MMD Loss'
    if scope:
      tag = scope + tag
    tf.summary.scalar(tag, loss_value)
    tf.losses.add_loss(loss_value)

  return loss_value


def correlation_loss(source_samples, target_samples, weight, scope=None):
  """Adds a similarity loss term, the correlation between two representations.

  Args:
    source_samples: a tensor of shape [num_samples, num_features]
    target_samples: a tensor of shape [num_samples, num_features]
    weight: a scalar weight for the loss.
    scope: optional name scope for summary tags.

  Returns:
    a scalar tensor representing the correlation loss value.
  """
  with tf.name_scope('corr_loss'):
    source_samples -= tf.reduce_mean(source_samples, 0)
    target_samples -= tf.reduce_mean(target_samples, 0)

    source_samples = tf.nn.l2_normalize(source_samples, 1)
    target_samples = tf.nn.l2_normalize(target_samples, 1)

    source_cov = tf.matmul(tf.transpose(source_samples), source_samples)
    target_cov = tf.matmul(tf.transpose(target_samples), target_samples)

    corr_loss = tf.reduce_mean(tf.square(source_cov - target_cov)) * weight

  assert_op = tf.Assert(tf.is_finite(corr_loss), [corr_loss])
  with tf.control_dependencies([assert_op]):
    tag = 'Correlation Loss'
    if scope:
      tag = scope + tag
    tf.summary.scalar(tag, corr_loss)
    tf.losses.add_loss(corr_loss)

  return corr_loss


def dann_loss(source_samples, target_samples, weight, scope=None):
  """Adds the domain adversarial (DANN) loss.

  Args:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the loss.
    scope: optional name scope for summary tags.

  Returns:
    a scalar tensor representing the correlation loss value.
  """
  with tf.variable_scope('dann'):
    batch_size = tf.shape(source_samples)[0]
    samples = tf.concat(axis=0, values=[source_samples, target_samples])
    samples = slim.flatten(samples)

    domain_selection_mask = tf.concat(
        axis=0, values=[tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))])

    # Perform the gradient reversal and be careful with the shape.
    grl = grl_ops.gradient_reversal(samples)
    grl = tf.reshape(grl, (-1, samples.get_shape().as_list()[1]))

    grl = slim.fully_connected(grl, 100, scope='fc1')
    logits = slim.fully_connected(grl, 1, activation_fn=None, scope='fc2')

    domain_predictions = tf.sigmoid(logits)

  domain_loss = tf.losses.log_loss(
      domain_selection_mask, domain_predictions, weights=weight)

  domain_accuracy = utils.accuracy(
      tf.round(domain_predictions), domain_selection_mask)

  assert_op = tf.Assert(tf.is_finite(domain_loss), [domain_loss])
  with tf.control_dependencies([assert_op]):
    tag_loss = 'losses/domain_loss'
    tag_accuracy = 'losses/domain_accuracy'
    if scope:
      tag_loss = scope + tag_loss
      tag_accuracy = scope + tag_accuracy

    tf.summary.scalar(tag_loss, domain_loss)
    tf.summary.scalar(tag_accuracy, domain_accuracy)

  return domain_loss


################################################################################
# DIFFERENCE LOSS
################################################################################
def difference_loss(private_samples, shared_samples, weight=1.0, name=''):
  """Adds the difference loss between the private and shared representations.

  Args:
    private_samples: a tensor of shape [num_samples, num_features].
    shared_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the incoherence loss.
    name: the name of the tf summary.
  """
  private_samples -= tf.reduce_mean(private_samples, 0)
  shared_samples -= tf.reduce_mean(shared_samples, 0)

  private_samples = tf.nn.l2_normalize(private_samples, 1)
  shared_samples = tf.nn.l2_normalize(shared_samples, 1)

  correlation_matrix = tf.matmul(
      private_samples, shared_samples, transpose_a=True)

  cost = tf.reduce_mean(tf.square(correlation_matrix)) * weight
  cost = tf.where(cost > 0, cost, 0, name='value')

  tf.summary.scalar('losses/Difference Loss {}'.format(name),
                                       cost)
  assert_op = tf.Assert(tf.is_finite(cost), [cost])
  with tf.control_dependencies([assert_op]):
    tf.losses.add_loss(cost)


################################################################################
# TASK LOSS
################################################################################
def log_quaternion_loss_batch(predictions, labels, params):
  """A helper function to compute the error between quaternions.

  Args:
    predictions: A Tensor of size [batch_size, 4].
    labels: A Tensor of size [batch_size, 4].
    params: A dictionary of parameters. Expecting 'use_logging', 'batch_size'.

  Returns:
    A Tensor of size [batch_size], denoting the error between the quaternions.
  """
  use_logging = params['use_logging']
  assertions = []
  if use_logging:
    assertions.append(
        tf.Assert(
            tf.reduce_all(
                tf.less(
                    tf.abs(tf.reduce_sum(tf.square(predictions), [1]) - 1),
                    1e-4)),
            ['The l2 norm of each prediction quaternion vector should be 1.']))
    assertions.append(
        tf.Assert(
            tf.reduce_all(
                tf.less(
                    tf.abs(tf.reduce_sum(tf.square(labels), [1]) - 1), 1e-4)),
            ['The l2 norm of each label quaternion vector should be 1.']))

  with tf.control_dependencies(assertions):
    product = tf.multiply(predictions, labels)
  internal_dot_products = tf.reduce_sum(product, [1])

  if use_logging:
    internal_dot_products = tf.Print(
        internal_dot_products,
        [internal_dot_products, tf.shape(internal_dot_products)],
        'internal_dot_products:')

  logcost = tf.log(1e-4 + 1 - tf.abs(internal_dot_products))
  return logcost


def log_quaternion_loss(predictions, labels, params):
  """A helper function to compute the mean error between batches of quaternions.

  The caller is expected to add the loss to the graph.

  Args:
    predictions: A Tensor of size [batch_size, 4].
    labels: A Tensor of size [batch_size, 4].
    params: A dictionary of parameters. Expecting 'use_logging', 'batch_size'.

  Returns:
    A Tensor of size 1, denoting the mean error between batches of quaternions.
  """
  use_logging = params['use_logging']
  logcost = log_quaternion_loss_batch(predictions, labels, params)
  logcost = tf.reduce_sum(logcost, [0])
  batch_size = params['batch_size']
  logcost = tf.multiply(logcost, 1.0 / batch_size, name='log_quaternion_loss')
  if use_logging:
    logcost = tf.Print(
        logcost, [logcost], '[logcost]', name='log_quaternion_loss_print')
  return logcost
