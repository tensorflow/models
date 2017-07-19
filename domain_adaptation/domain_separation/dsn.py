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
"""Functions to create a DSN model and add the different losses to it.

Specifically, in this file we define the:
  - Shared Encoding Similarity Loss Module, with:
    - The MMD Similarity method
    - The Correlation Similarity method
    - The Gradient Reversal (Domain-Adversarial) method
  - Difference Loss Module
  - Reconstruction Loss Module
  - Task Loss Module
"""
from functools import partial

import tensorflow as tf

import losses
import models
import utils

slim = tf.contrib.slim


################################################################################
# HELPER FUNCTIONS
################################################################################
def dsn_loss_coefficient(params):
  """The global_step-dependent weight that specifies when to kick in DSN losses.

  Args:
    params: A dictionary of parameters. Expecting 'domain_separation_startpoint'

  Returns:
    A weight to that effectively enables or disables the DSN-related losses,
    i.e. similarity, difference, and reconstruction losses.
  """
  return tf.where(
      tf.less(slim.get_or_create_global_step(),
              params['domain_separation_startpoint']), 1e-10, 1.0)


################################################################################
# MODEL CREATION
################################################################################
def create_model(source_images, source_labels, domain_selection_mask,
                 target_images, target_labels, similarity_loss, params,
                 basic_tower_name):
  """Creates a DSN model.

  Args:
    source_images: images from the source domain, a tensor of size
      [batch_size, height, width, channels]
    source_labels: a dictionary with the name, tensor pairs. 'classes' is one-
      hot for the number of classes.
    domain_selection_mask: a boolean tensor of size [batch_size, ] which denotes
      the labeled images that belong to the source domain.
    target_images: images from the target domain, a tensor of size
      [batch_size, height width, channels].
    target_labels: a dictionary with the name, tensor pairs.
    similarity_loss: The type of method to use for encouraging
      the codes from the shared encoder to be similar.
    params: A dictionary of parameters. Expecting 'weight_decay',
      'layers_to_regularize', 'use_separation', 'domain_separation_startpoint',
      'alpha_weight', 'beta_weight', 'gamma_weight', 'recon_loss_name',
      'decoder_name', 'encoder_name'
    basic_tower_name: the name of the tower to use for the shared encoder.

  Raises:
    ValueError: if the arch is not one of the available architectures.
  """
  network = getattr(models, basic_tower_name)
  num_classes = source_labels['classes'].get_shape().as_list()[1]

  # Make sure we are using the appropriate number of classes.
  network = partial(network, num_classes=num_classes)

  # Add the classification/pose estimation loss to the source domain.
  source_endpoints = add_task_loss(source_images, source_labels, network,
                                   params)

  if similarity_loss == 'none':
    # No domain adaptation, we can stop here.
    return

  with tf.variable_scope('towers', reuse=True):
    target_logits, target_endpoints = network(
        target_images, weight_decay=params['weight_decay'], prefix='target')

  # Plot target accuracy of the train set.
  target_accuracy = utils.accuracy(
      tf.argmax(target_logits, 1), tf.argmax(target_labels['classes'], 1))

  if 'quaternions' in target_labels:
    target_quaternion_loss = losses.log_quaternion_loss(
        target_labels['quaternions'], target_endpoints['quaternion_pred'],
        params)
    tf.summary.scalar('eval/Target quaternions', target_quaternion_loss)

  tf.summary.scalar('eval/Target accuracy', target_accuracy)

  source_shared = source_endpoints[params['layers_to_regularize']]
  target_shared = target_endpoints[params['layers_to_regularize']]

  # When using the semisupervised model we include labeled target data in the
  # source classifier. We do not want to include these target domain when
  # we use the similarity loss.
  indices = tf.range(0, source_shared.get_shape().as_list()[0])
  indices = tf.boolean_mask(indices, domain_selection_mask)
  add_similarity_loss(similarity_loss,
                      tf.gather(source_shared, indices),
                      tf.gather(target_shared, indices), params)

  if params['use_separation']:
    add_autoencoders(
        source_images,
        source_shared,
        target_images,
        target_shared,
        params=params,)


def add_similarity_loss(method_name,
                        source_samples,
                        target_samples,
                        params,
                        scope=None):
  """Adds a loss encouraging the shared encoding from each domain to be similar.

  Args:
    method_name: the name of the encoding similarity method to use. Valid
      options include `dann_loss', `mmd_loss' or `correlation_loss'.
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    params: a dictionary of parameters. Expecting 'gamma_weight'.
    scope: optional name scope for summary tags.
  Raises:
    ValueError: if `method_name` is not recognized.
  """
  weight = dsn_loss_coefficient(params) * params['gamma_weight']
  method = getattr(losses, method_name)
  method(source_samples, target_samples, weight, scope)


def add_reconstruction_loss(recon_loss_name, images, recons, weight, domain):
  """Adds a reconstruction loss.

  Args:
    recon_loss_name: The name of the reconstruction loss.
    images: A `Tensor` of size [batch_size, height, width, 3].
    recons: A `Tensor` whose size matches `images`.
    weight: A scalar coefficient for the loss.
    domain: The name of the domain being reconstructed.

  Raises:
    ValueError: If `recon_loss_name` is not recognized.
  """
  if recon_loss_name == 'sum_of_pairwise_squares':
    loss_fn = tf.contrib.losses.mean_pairwise_squared_error
  elif recon_loss_name == 'sum_of_squares':
    loss_fn = tf.contrib.losses.mean_squared_error
  else:
    raise ValueError('recon_loss_name value [%s] not recognized.' %
                     recon_loss_name)

  loss = loss_fn(recons, images, weight)
  assert_op = tf.Assert(tf.is_finite(loss), [loss])
  with tf.control_dependencies([assert_op]):
    tf.summary.scalar('losses/%s Recon Loss' % domain, loss)


def add_autoencoders(source_data, source_shared, target_data, target_shared,
                     params):
  """Adds the encoders/decoders for our domain separation model w/ incoherence.

  Args:
    source_data: images from the source domain, a tensor of size
      [batch_size, height, width, channels]
    source_shared: a tensor with first dimension batch_size
    target_data: images from the target domain, a tensor of size
      [batch_size, height, width, channels]
    target_shared: a tensor with first dimension batch_size
    params: A dictionary of parameters. Expecting 'layers_to_regularize',
      'beta_weight', 'alpha_weight', 'recon_loss_name', 'decoder_name',
      'encoder_name', 'weight_decay'
  """

  def normalize_images(images):
    images -= tf.reduce_min(images)
    return images / tf.reduce_max(images)

  def concat_operation(shared_repr, private_repr):
    return shared_repr + private_repr

  mu = dsn_loss_coefficient(params)

  # The layer to concatenate the networks at.
  concat_layer = params['layers_to_regularize']

  # The coefficient for modulating the private/shared difference loss.
  difference_loss_weight = params['beta_weight'] * mu

  # The reconstruction weight.
  recon_loss_weight = params['alpha_weight'] * mu

  # The reconstruction loss to use.
  recon_loss_name = params['recon_loss_name']

  # The decoder/encoder to use.
  decoder_name = params['decoder_name']
  encoder_name = params['encoder_name']

  _, height, width, _ = source_data.get_shape().as_list()
  code_size = source_shared.get_shape().as_list()[-1]
  weight_decay = params['weight_decay']

  encoder_fn = getattr(models, encoder_name)
  # Target Auto-encoding.
  with tf.variable_scope('source_encoder'):
    source_endpoints = encoder_fn(
        source_data, code_size, weight_decay=weight_decay)

  with tf.variable_scope('target_encoder'):
    target_endpoints = encoder_fn(
        target_data, code_size, weight_decay=weight_decay)

  decoder_fn = getattr(models, decoder_name)

  decoder = partial(
      decoder_fn,
      height=height,
      width=width,
      channels=source_data.get_shape().as_list()[-1],
      weight_decay=weight_decay)

  # Source Auto-encoding.
  source_private = source_endpoints[concat_layer]
  target_private = target_endpoints[concat_layer]
  with tf.variable_scope('decoder'):
    source_recons = decoder(concat_operation(source_shared, source_private))

  with tf.variable_scope('decoder', reuse=True):
    source_private_recons = decoder(
        concat_operation(tf.zeros_like(source_private), source_private))
    source_shared_recons = decoder(
        concat_operation(source_shared, tf.zeros_like(source_shared)))

  with tf.variable_scope('decoder', reuse=True):
    target_recons = decoder(concat_operation(target_shared, target_private))
    target_shared_recons = decoder(
        concat_operation(target_shared, tf.zeros_like(target_shared)))
    target_private_recons = decoder(
        concat_operation(tf.zeros_like(target_private), target_private))

  losses.difference_loss(
      source_private,
      source_shared,
      weight=difference_loss_weight,
      name='Source')
  losses.difference_loss(
      target_private,
      target_shared,
      weight=difference_loss_weight,
      name='Target')

  add_reconstruction_loss(recon_loss_name, source_data, source_recons,
                          recon_loss_weight, 'source')
  add_reconstruction_loss(recon_loss_name, target_data, target_recons,
                          recon_loss_weight, 'target')

  # Add summaries
  source_reconstructions = tf.concat(
      axis=2,
      values=map(normalize_images, [
          source_data, source_recons, source_shared_recons,
          source_private_recons
      ]))
  target_reconstructions = tf.concat(
      axis=2,
      values=map(normalize_images, [
          target_data, target_recons, target_shared_recons,
          target_private_recons
      ]))
  tf.summary.image(
      'Source Images:Recons:RGB',
      source_reconstructions[:, :, :, :3],
      max_outputs=10)
  tf.summary.image(
      'Target Images:Recons:RGB',
      target_reconstructions[:, :, :, :3],
      max_outputs=10)

  if source_reconstructions.get_shape().as_list()[3] == 4:
    tf.summary.image(
        'Source Images:Recons:Depth',
        source_reconstructions[:, :, :, 3:4],
        max_outputs=10)
    tf.summary.image(
        'Target Images:Recons:Depth',
        target_reconstructions[:, :, :, 3:4],
        max_outputs=10)


def add_task_loss(source_images, source_labels, basic_tower, params):
  """Adds a classification and/or pose estimation loss to the model.

  Args:
    source_images: images from the source domain, a tensor of size
      [batch_size, height, width, channels]
    source_labels: labels from the source domain, a tensor of size [batch_size].
      or a tuple of (quaternions, class_labels)
    basic_tower: a function that creates the single tower of the model.
    params: A dictionary of parameters. Expecting 'weight_decay', 'pose_weight'.
  Returns:
    The source endpoints.

  Raises:
    RuntimeError: if basic tower does not support pose estimation.
  """
  with tf.variable_scope('towers'):
    source_logits, source_endpoints = basic_tower(
        source_images, weight_decay=params['weight_decay'], prefix='Source')

  if 'quaternions' in source_labels:  # We have pose estimation as well
    if 'quaternion_pred' not in source_endpoints:
      raise RuntimeError('Please use a model for estimation e.g. pose_mini')

    loss = losses.log_quaternion_loss(source_labels['quaternions'],
                                      source_endpoints['quaternion_pred'],
                                      params)

    assert_op = tf.Assert(tf.is_finite(loss), [loss])
    with tf.control_dependencies([assert_op]):
      quaternion_loss = loss
      tf.summary.histogram('log_quaternion_loss_hist', quaternion_loss)
    slim.losses.add_loss(quaternion_loss * params['pose_weight'])
    tf.summary.scalar('losses/quaternion_loss', quaternion_loss)

  classification_loss = tf.losses.softmax_cross_entropy(
      source_labels['classes'], source_logits)

  tf.summary.scalar('losses/classification_loss', classification_loss)
  return source_endpoints
