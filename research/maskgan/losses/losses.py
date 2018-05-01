# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Losses for Generator and Discriminator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def discriminator_loss(predictions, labels, missing_tokens):
  """Discriminator loss based on predictions and labels.

  Args:
    predictions:  Discriminator linear predictions Tensor of shape [batch_size,
      sequence_length]
    labels: Labels for predictions, Tensor of shape [batch_size,
      sequence_length]
    missing_tokens:  Indicator for the missing tokens.  Evaluate the loss only
      on the tokens that were missing.

  Returns:
    loss:  Scalar tf.float32 loss.

  """
  loss = tf.losses.sigmoid_cross_entropy(labels,
                                         predictions,
                                         weights=missing_tokens)
  loss = tf.Print(
      loss, [loss, labels, missing_tokens],
      message='loss, labels, missing_tokens',
      summarize=25,
      first_n=25)
  return loss


def cross_entropy_loss_matrix(gen_labels, gen_logits):
  """Computes the cross entropy loss for G.

  Args:
    gen_labels:  Labels for the correct token.
    gen_logits: Generator logits.

  Returns:
    loss_matrix:  Loss matrix of shape [batch_size, sequence_length].
  """
  cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=gen_labels, logits=gen_logits)
  return cross_entropy_loss


def GAN_loss_matrix(dis_predictions):
  """Computes the cross entropy loss for G.

  Args:
    dis_predictions:  Discriminator predictions.

  Returns:
    loss_matrix: Loss matrix of shape [batch_size, sequence_length].
  """
  eps = tf.constant(1e-7, tf.float32)
  gan_loss_matrix = -tf.log(dis_predictions + eps)
  return gan_loss_matrix


def generator_GAN_loss(predictions):
  """Generator GAN loss based on Discriminator predictions."""
  return -tf.log(tf.reduce_mean(predictions))


def generator_blended_forward_loss(gen_logits, gen_labels, dis_predictions,
                                   is_real_input):
  """Computes the masked-loss for G.  This will be a blend of cross-entropy
  loss where the true label is known and GAN loss where the true label has been
  masked.

  Args:
    gen_logits: Generator logits.
    gen_labels:  Labels for the correct token.
    dis_predictions:  Discriminator predictions.
    is_real_input:  Tensor indicating whether the label is present.

  Returns:
    loss: Scalar tf.float32 total loss.
  """
  cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=gen_labels, logits=gen_logits)
  gan_loss = -tf.log(dis_predictions)
  loss_matrix = tf.where(is_real_input, cross_entropy_loss, gan_loss)
  return tf.reduce_mean(loss_matrix)


def wasserstein_generator_loss(gen_logits, gen_labels, dis_values,
                               is_real_input):
  """Computes the masked-loss for G.  This will be a blend of cross-entropy
  loss where the true label is known and GAN loss where the true label is
  missing.

  Args:
    gen_logits:  Generator logits.
    gen_labels:  Labels for the correct token.
    dis_values:  Discriminator values Tensor of shape [batch_size,
      sequence_length].
    is_real_input:  Tensor indicating whether the label is present.

  Returns:
    loss: Scalar tf.float32 total loss.
  """
  cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=gen_labels, logits=gen_logits)
  # Maximize the dis_values (minimize the negative)
  gan_loss = -dis_values
  loss_matrix = tf.where(is_real_input, cross_entropy_loss, gan_loss)
  loss = tf.reduce_mean(loss_matrix)
  return loss


def wasserstein_discriminator_loss(real_values, fake_values):
  """Wasserstein discriminator loss.

  Args:
    real_values: Value given by the Wasserstein Discriminator to real data.
    fake_values: Value given by the Wasserstein Discriminator to fake data.

  Returns:
    loss:  Scalar tf.float32 loss.

  """
  real_avg = tf.reduce_mean(real_values)
  fake_avg = tf.reduce_mean(fake_values)

  wasserstein_loss = real_avg - fake_avg
  return wasserstein_loss


def wasserstein_discriminator_loss_intrabatch(values, is_real_input):
  """Wasserstein discriminator loss.  This is an odd variant where the value
  difference is between the real tokens and the fake tokens within a single
  batch.

  Args:
    values: Value given by the Wasserstein Discriminator of shape [batch_size,
      sequence_length] to an imputed batch (real and fake).
    is_real_input: tf.bool Tensor of shape [batch_size, sequence_length]. If
      true, it indicates that the label is known.

  Returns:
    wasserstein_loss:  Scalar tf.float32 loss.

  """
  zero_tensor = tf.constant(0., dtype=tf.float32, shape=[])

  present = tf.cast(is_real_input, tf.float32)
  missing = tf.cast(1 - present, tf.float32)

  # Counts for real and fake tokens.
  real_count = tf.reduce_sum(present)
  fake_count = tf.reduce_sum(missing)

  # Averages for real and fake token values.
  real = tf.mul(values, present)
  fake = tf.mul(values, missing)
  real_avg = tf.reduce_sum(real) / real_count
  fake_avg = tf.reduce_sum(fake) / fake_count

  # If there are no real or fake entries in the batch, we assign an average
  # value of zero.
  real_avg = tf.where(tf.equal(real_count, 0), zero_tensor, real_avg)
  fake_avg = tf.where(tf.equal(fake_count, 0), zero_tensor, fake_avg)

  wasserstein_loss = real_avg - fake_avg
  return wasserstein_loss
