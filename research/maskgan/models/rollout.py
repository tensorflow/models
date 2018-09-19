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

"""Rollout RNN model definitions which call rnn_zaremba code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from six.moves import xrange
import tensorflow as tf

from losses import losses
from model_utils import helper
from model_utils import model_construction
from model_utils import model_losses
from model_utils import model_optimization

FLAGS = tf.app.flags.FLAGS


def create_rollout_MaskGAN(hparams, is_training):
  """Create the MaskGAN model.

  Args:
    hparams:  Hyperparameters for the MaskGAN.
    is_training:  Boolean indicating operational mode (train/inference).
      evaluated with a teacher forcing regime.

  Return:
    model:  Namedtuple for specifying the MaskGAN."""
  global_step = tf.Variable(0, name='global_step', trainable=False)

  new_learning_rate = tf.placeholder(tf.float32, [], name='new_learning_rate')
  learning_rate = tf.Variable(0.0, name='learning_rate', trainable=False)
  learning_rate_update = tf.assign(learning_rate, new_learning_rate)

  new_rate = tf.placeholder(tf.float32, [], name='new_rate')
  percent_real_var = tf.Variable(0.0, trainable=False)
  percent_real_update = tf.assign(percent_real_var, new_rate)

  ## Placeholders.
  inputs = tf.placeholder(
      tf.int32, shape=[FLAGS.batch_size, FLAGS.sequence_length])
  present = tf.placeholder(
      tf.bool, shape=[FLAGS.batch_size, FLAGS.sequence_length])
  inv_present = tf.placeholder(
      tf.bool, shape=[FLAGS.batch_size, FLAGS.sequence_length])

  ## Rollout Generator.
  fwd_gen_rollouts = rollout_generator(
      hparams, inputs, present, is_training=is_training, is_validating=False)
  inv_gen_rollouts = rollout_generator(
      hparams,
      inputs,
      inv_present,
      is_training=is_training,
      is_validating=False,
      reuse=True)

  ## Rollout Discriminator.
  fwd_dis_rollouts = rollout_discriminator(
      hparams, fwd_gen_rollouts, is_training=is_training)
  inv_dis_rollouts = rollout_discriminator(
      hparams, inv_gen_rollouts, is_training=is_training, reuse=True)

  ## Discriminator Loss.
  [dis_loss, dis_loss_pred, dis_loss_inv_pred] = rollout_discriminator_loss(
      fwd_dis_rollouts, present, inv_dis_rollouts, inv_present)

  ## Average log-perplexity for only missing words.  However, to do this,
  # the logits are still computed using teacher forcing, that is, the ground
  # truth tokens are fed in at each time point to be valid.
  # TODO(liamfedus): Fix the naming convention.
  with tf.variable_scope('gen_rollout'):
    _, fwd_eval_logits, _ = model_construction.create_generator(
        hparams,
        inputs,
        present,
        is_training=False,
        is_validating=True,
        reuse=True)

  avg_log_perplexity = model_losses.calculate_log_perplexity(
      fwd_eval_logits, inputs, present)

  ## Generator Loss.
  # 1.  Cross Entropy losses on missing tokens.
  [fwd_cross_entropy_losses,
   inv_cross_entropy_losses] = rollout_masked_cross_entropy_loss(
       inputs, present, inv_present, fwd_gen_rollouts, inv_gen_rollouts)

  # 2.  GAN losses on missing tokens.
  [fwd_RL_loss,
   fwd_RL_statistics, fwd_averages_op] = rollout_reinforce_objective(
       hparams, fwd_gen_rollouts, fwd_dis_rollouts, present)
  [inv_RL_loss,
   inv_RL_statistics, inv_averages_op] = rollout_reinforce_objective(
       hparams, inv_gen_rollouts, inv_dis_rollouts, inv_present)

  # TODO(liamfedus):  Generalize this to use all logs.
  [fwd_sequence, fwd_logits, fwd_log_probs] = fwd_gen_rollouts[-1]
  [inv_sequence, inv_logits, inv_log_probs] = inv_gen_rollouts[-1]

  # TODO(liamfedus):  Generalize this to use all logs.
  fwd_predictions = fwd_dis_rollouts[-1]
  inv_predictions = inv_dis_rollouts[-1]

  # TODO(liamfedus):  Generalize this to use all logs.
  [fwd_log_probs, fwd_rewards, fwd_advantages,
   fwd_baselines] = fwd_RL_statistics[-1]
  [inv_log_probs, inv_rewards, inv_advantages,
   inv_baselines] = inv_RL_statistics[-1]

  ## Pre-training.
  if FLAGS.gen_pretrain_steps:
    # TODO(liamfedus): Rewrite this.
    fwd_cross_entropy_loss = tf.reduce_mean(fwd_cross_entropy_losses)
    gen_pretrain_op = model_optimization.create_gen_pretrain_op(
        hparams, fwd_cross_entropy_loss, global_step)
  else:
    gen_pretrain_op = tf.no_op('gen_pretrain_no_op')
  if FLAGS.dis_pretrain_steps:
    dis_pretrain_op = model_optimization.create_dis_pretrain_op(
        hparams, dis_loss, global_step)
  else:
    dis_pretrain_op = tf.no_op('dis_pretrain_no_op')

  ##  Generator Train Op.
  # 1.  Cross-Entropy.
  if FLAGS.gen_training_strategy == 'cross_entropy':
    gen_loss = tf.reduce_mean(
        fwd_cross_entropy_losses + inv_cross_entropy_losses) / 2.
    [gen_train_op, gen_grads,
     gen_vars] = model_optimization.create_gen_train_op(
         hparams, learning_rate, gen_loss, global_step, mode='MINIMIZE')

  # 2.  GAN (REINFORCE)
  elif FLAGS.gen_training_strategy == 'reinforce':
    gen_loss = (fwd_RL_loss + inv_RL_loss) / 2.
    [gen_train_op, gen_grads,
     gen_vars] = model_optimization.create_reinforce_gen_train_op(
         hparams, learning_rate, gen_loss, fwd_averages_op, inv_averages_op,
         global_step)

  else:
    raise NotImplementedError

  ## Discriminator Train Op.
  dis_train_op, dis_grads, dis_vars = model_optimization.create_dis_train_op(
      hparams, dis_loss, global_step)

  ## Summaries.
  with tf.name_scope('general'):
    tf.summary.scalar('percent_real', percent_real_var)
    tf.summary.scalar('learning_rate', learning_rate)

  with tf.name_scope('generator_losses'):
    tf.summary.scalar('gen_loss', tf.reduce_mean(gen_loss))
    tf.summary.scalar('gen_loss_fwd_cross_entropy',
                      tf.reduce_mean(fwd_cross_entropy_losses))
    tf.summary.scalar('gen_loss_inv_cross_entropy',
                      tf.reduce_mean(inv_cross_entropy_losses))

  with tf.name_scope('REINFORCE'):
    with tf.name_scope('objective'):
      tf.summary.scalar('fwd_RL_loss', tf.reduce_mean(fwd_RL_loss))
      tf.summary.scalar('inv_RL_loss', tf.reduce_mean(inv_RL_loss))

    with tf.name_scope('rewards'):
      helper.variable_summaries(fwd_rewards, 'fwd_rewards')
      helper.variable_summaries(inv_rewards, 'inv_rewards')

    with tf.name_scope('advantages'):
      helper.variable_summaries(fwd_advantages, 'fwd_advantages')
      helper.variable_summaries(inv_advantages, 'inv_advantages')

    with tf.name_scope('baselines'):
      helper.variable_summaries(fwd_baselines, 'fwd_baselines')
      helper.variable_summaries(inv_baselines, 'inv_baselines')

    with tf.name_scope('log_probs'):
      helper.variable_summaries(fwd_log_probs, 'fwd_log_probs')
      helper.variable_summaries(inv_log_probs, 'inv_log_probs')

  with tf.name_scope('discriminator_losses'):
    tf.summary.scalar('dis_loss', dis_loss)
    tf.summary.scalar('dis_loss_fwd_sequence', dis_loss_pred)
    tf.summary.scalar('dis_loss_inv_sequence', dis_loss_inv_pred)

  with tf.name_scope('logits'):
    helper.variable_summaries(fwd_logits, 'fwd_logits')
    helper.variable_summaries(inv_logits, 'inv_logits')

  for v, g in zip(gen_vars, gen_grads):
    helper.variable_summaries(v, v.op.name)
    helper.variable_summaries(g, 'grad/' + v.op.name)

  for v, g in zip(dis_vars, dis_grads):
    helper.variable_summaries(v, v.op.name)
    helper.variable_summaries(g, 'grad/' + v.op.name)

  merge_summaries_op = tf.summary.merge_all()

  # Model saver.
  saver = tf.train.Saver(keep_checkpoint_every_n_hours=1, max_to_keep=5)

  # Named tuple that captures elements of the MaskGAN model.
  Model = collections.namedtuple('Model', [
      'inputs', 'present', 'inv_present', 'percent_real_update', 'new_rate',
      'fwd_sequence', 'fwd_logits', 'fwd_rewards', 'fwd_advantages',
      'fwd_log_probs', 'fwd_predictions', 'fwd_cross_entropy_losses',
      'inv_sequence', 'inv_logits', 'inv_rewards', 'inv_advantages',
      'inv_log_probs', 'inv_predictions', 'inv_cross_entropy_losses',
      'avg_log_perplexity', 'dis_loss', 'gen_loss', 'dis_train_op',
      'gen_train_op', 'gen_pretrain_op', 'dis_pretrain_op',
      'merge_summaries_op', 'global_step', 'new_learning_rate',
      'learning_rate_update', 'saver'
  ])

  model = Model(
      inputs, present, inv_present, percent_real_update, new_rate, fwd_sequence,
      fwd_logits, fwd_rewards, fwd_advantages, fwd_log_probs, fwd_predictions,
      fwd_cross_entropy_losses, inv_sequence, inv_logits, inv_rewards,
      inv_advantages, inv_log_probs, inv_predictions, inv_cross_entropy_losses,
      avg_log_perplexity, dis_loss, gen_loss, dis_train_op, gen_train_op,
      gen_pretrain_op, dis_pretrain_op, merge_summaries_op, global_step,
      new_learning_rate, learning_rate_update, saver)
  return model


def rollout_generator(hparams,
                      inputs,
                      input_present,
                      is_training,
                      is_validating,
                      reuse=None):
  """Define the Generator graph which does rollouts.

    G will now impute tokens that have been masked from the input seqeunce.
  """
  rollouts = []

  with tf.variable_scope('gen_rollout'):
    for n in xrange(FLAGS.num_rollouts):
      if n > 0:
        # TODO(liamfedus): Why is it necessary here to manually set reuse?
        reuse = True
        tf.get_variable_scope().reuse_variables()

      [sequence, logits, log_probs] = model_construction.create_generator(
          hparams,
          inputs,
          input_present,
          is_training,
          is_validating,
          reuse=reuse)

      rollouts.append([sequence, logits, log_probs])

  # Length assertion.
  assert len(rollouts) == FLAGS.num_rollouts

  return rollouts


def rollout_discriminator(hparams, gen_rollouts, is_training, reuse=None):
  """Define the Discriminator graph which does rollouts.

    G will now impute tokens that have been masked from the input seqeunce.
  """
  rollout_predictions = []

  with tf.variable_scope('dis_rollout'):
    for n, rollout in enumerate(gen_rollouts):
      if n > 0:
        # TODO(liamfedus): Why is it necessary here to manually set reuse?
        reuse = True
        tf.get_variable_scope().reuse_variables()

      [sequence, _, _] = rollout

      predictions = model_construction.create_discriminator(
          hparams, sequence, is_training=is_training, reuse=reuse)

      # Predictions for each rollout.
      rollout_predictions.append(predictions)

  # Length assertion.
  assert len(rollout_predictions) == FLAGS.num_rollouts

  return rollout_predictions


def rollout_reinforce_objective(hparams, gen_rollouts, dis_rollouts, present):
  cumulative_gen_objective = 0.
  cumulative_averages_op = []
  cumulative_statistics = []

  assert len(gen_rollouts) == len(dis_rollouts)

  for gen_rollout, dis_rollout in zip(gen_rollouts, dis_rollouts):
    [_, _, log_probs] = gen_rollout
    dis_predictions = dis_rollout

    [
        final_gen_objective, log_probs, rewards, advantages, baselines,
        maintain_averages_op
    ] = model_losses.calculate_reinforce_objective(hparams, log_probs,
                                                   dis_predictions, present)

    # Accumulate results.
    cumulative_gen_objective += final_gen_objective
    cumulative_averages_op.append(maintain_averages_op)
    cumulative_statistics.append([log_probs, rewards, advantages, baselines])

  # Group all the averaging operations.
  cumulative_averages_op = tf.group(*cumulative_averages_op)
  cumulative_gen_objective /= FLAGS.num_rollouts
  [log_probs, rewards, advantages, baselines] = cumulative_statistics[-1]

  # Length assertion.
  assert len(cumulative_statistics) == FLAGS.num_rollouts

  return [
      cumulative_gen_objective, cumulative_statistics, cumulative_averages_op
  ]


def rollout_masked_cross_entropy_loss(inputs, present, inv_present,
                                      fwd_rollouts, inv_rollouts):
  cumulative_fwd_cross_entropy_losses = tf.zeros(
      shape=[FLAGS.batch_size, FLAGS.sequence_length])
  cumulative_inv_cross_entropy_losses = tf.zeros(
      shape=[FLAGS.batch_size, FLAGS.sequence_length])

  for fwd_rollout, inv_rollout in zip(fwd_rollouts, inv_rollouts):
    [_, fwd_logits, _] = fwd_rollout
    [_, inv_logits, _] = inv_rollout

    [fwd_cross_entropy_losses,
     inv_cross_entropy_losses] = model_losses.create_masked_cross_entropy_loss(
         inputs, present, inv_present, fwd_logits, inv_logits)

    cumulative_fwd_cross_entropy_losses = tf.add(
        cumulative_fwd_cross_entropy_losses, fwd_cross_entropy_losses)
    cumulative_inv_cross_entropy_losses = tf.add(
        cumulative_inv_cross_entropy_losses, inv_cross_entropy_losses)

  return [
      cumulative_fwd_cross_entropy_losses, cumulative_inv_cross_entropy_losses
  ]


def rollout_discriminator_loss(fwd_rollouts, present, inv_rollouts,
                               inv_present):

  dis_loss = 0
  dis_loss_pred = 0
  dis_loss_inv_pred = 0

  for fwd_predictions, inv_predictions in zip(fwd_rollouts, inv_rollouts):
    dis_loss_pred += losses.discriminator_loss(fwd_predictions, present)
    dis_loss_inv_pred += losses.discriminator_loss(inv_predictions, inv_present)

  dis_loss_pred /= FLAGS.num_rollouts
  dis_loss_inv_pred /= FLAGS.num_rollouts

  dis_loss = (dis_loss_pred + dis_loss_inv_pred) / 2.
  return [dis_loss, dis_loss_pred, dis_loss_inv_pred]
