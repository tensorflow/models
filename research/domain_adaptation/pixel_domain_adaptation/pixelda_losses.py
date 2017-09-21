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

"""Defines the various loss functions in use by the PIXELDA model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow as tf

slim = tf.contrib.slim


def add_domain_classifier_losses(end_points, hparams):
  """Adds losses related to the domain-classifier.

  Args:
    end_points: A map of network end point names to `Tensors`.
    hparams: The hyperparameters struct.

  Returns:
    loss: A `Tensor` representing the total task-classifier loss.
  """
  if hparams.domain_loss_weight == 0:
    tf.logging.info(
        'Domain classifier loss weight is 0, so not creating losses.')
    return 0

  # The domain prediction loss is minimized with respect to the domain
  # classifier features only. Its aim is to predict the domain of the images.
  # Note: 1 = 'real image' label, 0 = 'fake image' label
  transferred_domain_loss = tf.losses.sigmoid_cross_entropy(
      multi_class_labels=tf.zeros_like(end_points['transferred_domain_logits']),
      logits=end_points['transferred_domain_logits'])
  tf.summary.scalar('Domain_loss_transferred', transferred_domain_loss)

  target_domain_loss = tf.losses.sigmoid_cross_entropy(
      multi_class_labels=tf.ones_like(end_points['target_domain_logits']),
      logits=end_points['target_domain_logits'])
  tf.summary.scalar('Domain_loss_target', target_domain_loss)

  # Compute the total domain loss:
  total_domain_loss = transferred_domain_loss + target_domain_loss
  total_domain_loss *= hparams.domain_loss_weight
  tf.summary.scalar('Domain_loss_total', total_domain_loss)

  return total_domain_loss

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
    internal_dot_products = tf.Print(internal_dot_products, [
        internal_dot_products,
        tf.shape(internal_dot_products)
    ], 'internal_dot_products:')

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

def _quaternion_loss(labels, predictions, weight, batch_size, domain,
                     add_summaries):
  """Creates a Quaternion Loss.

  Args:
    labels: The true quaternions.
    predictions: The predicted quaternions.
    weight: A scalar weight.
    batch_size: The size of the batches.
    domain: The name of the domain from which the labels were taken.
    add_summaries: Whether or not to add summaries for the losses.

  Returns:
    A `Tensor` representing the loss.
  """
  assert domain in ['Source', 'Transferred']

  params = {'use_logging': False, 'batch_size': batch_size}
  loss = weight * log_quaternion_loss(labels, predictions, params)

  if add_summaries:
    assert_op = tf.Assert(tf.is_finite(loss), [loss])
    with tf.control_dependencies([assert_op]):
      tf.summary.histogram(
          'Log_Quaternion_Loss_%s' % domain, loss, collections='losses')
      tf.summary.scalar(
          'Task_Quaternion_Loss_%s' % domain, loss, collections='losses')

  return loss


def _add_task_specific_losses(end_points, source_labels, num_classes, hparams,
                              add_summaries=False):
  """Adds losses related to the task-classifier.

  Args:
    end_points: A map of network end point names to `Tensors`.
    source_labels: A dictionary of output labels to `Tensors`.
    num_classes: The number of classes used by the classifier.
    hparams: The hyperparameters struct.
    add_summaries: Whether or not to add the summaries.

  Returns:
    loss: A `Tensor` representing the total task-classifier loss.
  """
  # TODO(ddohan): Make sure the l2 regularization is added to the loss

  one_hot_labels = slim.one_hot_encoding(source_labels['class'], num_classes)
  total_loss = 0

  if 'source_task_logits' in end_points:
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=one_hot_labels,
        logits=end_points['source_task_logits'],
        weights=hparams.source_task_loss_weight)
    if add_summaries:
      tf.summary.scalar('Task_Classifier_Loss_Source', loss)
    total_loss += loss

  if 'transferred_task_logits' in end_points:
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=one_hot_labels,
        logits=end_points['transferred_task_logits'],
        weights=hparams.transferred_task_loss_weight)
    if add_summaries:
      tf.summary.scalar('Task_Classifier_Loss_Transferred', loss)
    total_loss += loss

  #########################
  # Pose specific losses. #
  #########################
  if 'quaternion' in source_labels:
    total_loss += _quaternion_loss(
        source_labels['quaternion'],
        end_points['source_quaternion'],
        hparams.source_pose_weight,
        hparams.batch_size,
        'Source',
        add_summaries)

    total_loss += _quaternion_loss(
        source_labels['quaternion'],
        end_points['transferred_quaternion'],
        hparams.transferred_pose_weight,
        hparams.batch_size,
        'Transferred',
        add_summaries)

  if add_summaries:
    tf.summary.scalar('Task_Loss_Total', total_loss)

  return total_loss


def _transferred_similarity_loss(reconstructions,
                                 source_images,
                                 weight=1.0,
                                 method='mse',
                                 max_diff=0.4,
                                 name='similarity'):
  """Computes a loss encouraging similarity between source and transferred.

  Args:
    reconstructions: A `Tensor` of shape [batch_size, height, width, channels]
    source_images: A `Tensor` of shape [batch_size, height, width, channels].
    weight: Multiple similarity loss by this weight before returning
    method: One of:
      mpse = Mean Pairwise Squared Error
      mse = Mean Squared Error
      hinged_mse = Computes the mean squared error using squared differences
        greater than hparams.transferred_similarity_max_diff
      hinged_mae = Computes the mean absolute error using absolute
        differences greater than hparams.transferred_similarity_max_diff.
    max_diff: Maximum unpenalized difference for hinged losses
    name: Identifying name to use for creating summaries


  Returns:
    A `Tensor` representing the transferred similarity loss.

  Raises:
    ValueError: if `method` is not recognized.
  """
  if weight == 0:
    return 0

  source_channels = source_images.shape.as_list()[-1]
  reconstruction_channels = reconstructions.shape.as_list()[-1]

  # Convert grayscale source to RGB if target is RGB
  if source_channels == 1 and reconstruction_channels != 1:
    source_images = tf.tile(source_images, [1, 1, 1, reconstruction_channels])
  if reconstruction_channels == 1 and source_channels != 1:
    reconstructions = tf.tile(reconstructions, [1, 1, 1, source_channels])

  if method == 'mpse':
    reconstruction_similarity_loss_fn = (
        tf.contrib.losses.mean_pairwise_squared_error)
  elif method == 'masked_mpse':

    def masked_mpse(predictions, labels, weight):
      """Masked mpse assuming we have a depth to create a mask from."""
      assert labels.shape.as_list()[-1] == 4
      mask = tf.to_float(tf.less(labels[:, :, :, 3:4], 0.99))
      mask = tf.tile(mask, [1, 1, 1, 4])
      predictions *= mask
      labels *= mask
      tf.image_summary('masked_pred', predictions)
      tf.image_summary('masked_label', labels)
      return tf.contrib.losses.mean_pairwise_squared_error(
          predictions, labels, weight)

    reconstruction_similarity_loss_fn = masked_mpse
  elif method == 'mse':
    reconstruction_similarity_loss_fn = tf.contrib.losses.mean_squared_error
  elif method == 'hinged_mse':

    def hinged_mse(predictions, labels, weight):
      diffs = tf.square(predictions - labels)
      diffs = tf.maximum(0.0, diffs - max_diff)
      return tf.reduce_mean(diffs) * weight

    reconstruction_similarity_loss_fn = hinged_mse
  elif method == 'hinged_mae':

    def hinged_mae(predictions, labels, weight):
      diffs = tf.abs(predictions - labels)
      diffs = tf.maximum(0.0, diffs - max_diff)
      return tf.reduce_mean(diffs) * weight

    reconstruction_similarity_loss_fn = hinged_mae
  else:
    raise ValueError('Unknown reconstruction loss %s' % method)

  reconstruction_similarity_loss = reconstruction_similarity_loss_fn(
      reconstructions, source_images, weight)

  name = '%s_Similarity_(%s)' % (name, method)
  tf.summary.scalar(name, reconstruction_similarity_loss)
  return reconstruction_similarity_loss


def g_step_loss(source_images, source_labels, end_points, hparams, num_classes):
  """Configures the loss function which runs during the g-step.

  Args:
    source_images: A `Tensor` of shape [batch_size, height, width, channels].
    source_labels: A dictionary of `Tensors` of shape [batch_size]. Valid keys
      are 'class' and 'quaternion'.
    end_points: A map of the network end points.
    hparams: The hyperparameters struct.
    num_classes: Number of classes for classifier loss

  Returns:
    A `Tensor` representing a loss function.

  Raises:
    ValueError: if hparams.transferred_similarity_loss_weight is non-zero but
      hparams.transferred_similarity_loss is invalid.
  """
  generator_loss = 0

  ################################################################
  # Adds a loss which encourages the discriminator probabilities #
  # to be high (near one).
  ################################################################

  # As per the GAN paper, maximize the log probs, instead of minimizing
  # log(1-probs). Since we're minimizing, we'll minimize -log(probs) which is
  # the same thing.
  style_transfer_loss = tf.losses.sigmoid_cross_entropy(
      logits=end_points['transferred_domain_logits'],
      multi_class_labels=tf.ones_like(end_points['transferred_domain_logits']),
      weights=hparams.style_transfer_loss_weight)
  tf.summary.scalar('Style_transfer_loss', style_transfer_loss)
  generator_loss += style_transfer_loss

  # Optimizes the style transfer network to produce transferred images similar
  # to the source images.
  generator_loss += _transferred_similarity_loss(
      end_points['transferred_images'],
      source_images,
      weight=hparams.transferred_similarity_loss_weight,
      method=hparams.transferred_similarity_loss,
      name='transferred_similarity')

  # Optimizes the style transfer network to maximize classification accuracy.
  if source_labels is not None and hparams.task_tower_in_g_step:
    generator_loss += _add_task_specific_losses(
        end_points, source_labels, num_classes,
        hparams) * hparams.task_loss_in_g_weight

  return generator_loss


def d_step_loss(end_points, source_labels, num_classes, hparams):
  """Configures the losses during the D-Step.

  Note that during the D-step, the model optimizes both the domain (binary)
  classifier and the task classifier.

  Args:
    end_points: A map of the network end points.
    source_labels: A dictionary of output labels to `Tensors`.
    num_classes: The number of classes used by the classifier.
    hparams: The hyperparameters struct.

  Returns:
    A `Tensor` representing the value of the D-step loss.
  """
  domain_classifier_loss = add_domain_classifier_losses(end_points, hparams)

  task_classifier_loss = 0
  if source_labels is not None:
    task_classifier_loss = _add_task_specific_losses(
        end_points, source_labels, num_classes, hparams, add_summaries=True)

  return domain_classifier_loss + task_classifier_loss
