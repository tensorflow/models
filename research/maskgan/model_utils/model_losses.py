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

"""Model loss construction."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from six.moves import xrange
import tensorflow as tf

# Useful for REINFORCE baseline.
from losses import losses

FLAGS = tf.app.flags.FLAGS


def create_dis_loss(fake_predictions, real_predictions, targets_present):
  """Compute Discriminator loss across real/fake."""

  missing = tf.cast(targets_present, tf.int32)
  missing = 1 - missing
  missing = tf.cast(missing, tf.bool)

  real_labels = tf.ones([FLAGS.batch_size, FLAGS.sequence_length])
  dis_loss_real = tf.losses.sigmoid_cross_entropy(
      real_labels, real_predictions, weights=missing)
  dis_loss_fake = tf.losses.sigmoid_cross_entropy(
      targets_present, fake_predictions, weights=missing)

  dis_loss = (dis_loss_fake + dis_loss_real) / 2.
  return dis_loss, dis_loss_fake, dis_loss_real


def create_critic_loss(cumulative_rewards, estimated_values, present):
  """Compute Critic loss in estimating the value function.  This should be an
  estimate only for the missing elements."""
  missing = tf.cast(present, tf.int32)
  missing = 1 - missing
  missing = tf.cast(missing, tf.bool)

  loss = tf.losses.mean_squared_error(
      labels=cumulative_rewards, predictions=estimated_values, weights=missing)
  return loss


def create_masked_cross_entropy_loss(targets, present, logits):
  """Calculate the cross entropy loss matrices for the masked tokens."""
  cross_entropy_losses = losses.cross_entropy_loss_matrix(targets, logits)

  # Zeros matrix.
  zeros_losses = tf.zeros(
      shape=[FLAGS.batch_size, FLAGS.sequence_length], dtype=tf.float32)

  missing_ce_loss = tf.where(present, zeros_losses, cross_entropy_losses)

  return missing_ce_loss


def calculate_reinforce_objective(hparams,
                                  log_probs,
                                  dis_predictions,
                                  present,
                                  estimated_values=None):
  """Calculate the REINFORCE objectives.  The REINFORCE objective should
  only be on the tokens that were missing.  Specifically, the final Generator
  reward should be based on the Discriminator predictions on missing tokens.
  The log probaibilities should be only for missing tokens and the baseline
  should be calculated only on the missing tokens.

  For this model, we optimize the reward is the log of the *conditional*
  probability the Discriminator assigns to the distribution.  Specifically, for
  a Discriminator D which outputs probability of real, given the past context,

    r_t = log D(x_t|x_0,x_1,...x_{t-1})

  And the policy for Generator G is the log-probability of taking action x2
  given the past context.


  Args:
    hparams:  MaskGAN hyperparameters.
    log_probs:  tf.float32 Tensor of log probailities of the tokens selected by
      the Generator.  Shape [batch_size, sequence_length].
    dis_predictions:  tf.float32 Tensor of the predictions from the
      Discriminator.  Shape [batch_size, sequence_length].
    present:  tf.bool Tensor indicating which tokens are present.  Shape
      [batch_size, sequence_length].
    estimated_values:  tf.float32 Tensor of estimated state values of tokens.
      Shape [batch_size, sequence_length]

  Returns:
    final_gen_objective:  Final REINFORCE objective for the sequence.
    rewards:  tf.float32 Tensor of rewards for sequence of shape [batch_size,
      sequence_length]
    advantages: tf.float32 Tensor of advantages for sequence of shape
      [batch_size, sequence_length]
    baselines:  tf.float32 Tensor of baselines for sequence of shape
      [batch_size, sequence_length]
    maintain_averages_op:  ExponentialMovingAverage apply average op to
      maintain the baseline.
  """
  # Final Generator objective.
  final_gen_objective = 0.
  gamma = hparams.rl_discount_rate
  eps = 1e-7

  # Generator rewards are log-probabilities.
  eps = tf.constant(1e-7, tf.float32)
  dis_predictions = tf.nn.sigmoid(dis_predictions)
  rewards = tf.log(dis_predictions + eps)

  # Apply only for missing elements.
  zeros = tf.zeros_like(present, dtype=tf.float32)
  log_probs = tf.where(present, zeros, log_probs)
  rewards = tf.where(present, zeros, rewards)

  # Unstack Tensors into lists.
  rewards_list = tf.unstack(rewards, axis=1)
  log_probs_list = tf.unstack(log_probs, axis=1)
  missing = 1. - tf.cast(present, tf.float32)
  missing_list = tf.unstack(missing, axis=1)

  # Cumulative Discounted Returns.  The true value function V*(s).
  cumulative_rewards = []
  for t in xrange(FLAGS.sequence_length):
    cum_value = tf.zeros(shape=[FLAGS.batch_size])
    for s in xrange(t, FLAGS.sequence_length):
      cum_value += missing_list[s] * np.power(gamma, (s - t)) * rewards_list[s]
    cumulative_rewards.append(cum_value)
  cumulative_rewards = tf.stack(cumulative_rewards, axis=1)

  ## REINFORCE with different baselines.
  # We create a separate critic functionality for the Discriminator.  This
  # will need to operate unidirectionally and it may take in the past context.
  if FLAGS.baseline_method == 'critic':

    # Critic loss calculated from the estimated value function \hat{V}(s)
    # versus the true value function V*(s).
    critic_loss = create_critic_loss(cumulative_rewards, estimated_values,
                                     present)

    # Baselines are coming from the critic's estimated state values.
    baselines = tf.unstack(estimated_values, axis=1)

    ## Calculate the Advantages, A(s,a) = Q(s,a) - \hat{V}(s).
    advantages = []
    for t in xrange(FLAGS.sequence_length):
      log_probability = log_probs_list[t]
      cum_advantage = tf.zeros(shape=[FLAGS.batch_size])

      for s in xrange(t, FLAGS.sequence_length):
        cum_advantage += missing_list[s] * np.power(gamma,
                                                    (s - t)) * rewards_list[s]
      cum_advantage -= baselines[t]
      # Clip advantages.
      cum_advantage = tf.clip_by_value(cum_advantage, -FLAGS.advantage_clipping,
                                       FLAGS.advantage_clipping)
      advantages.append(missing_list[t] * cum_advantage)
      final_gen_objective += tf.multiply(
          log_probability, missing_list[t] * tf.stop_gradient(cum_advantage))

    maintain_averages_op = None
    baselines = tf.stack(baselines, axis=1)
    advantages = tf.stack(advantages, axis=1)

  # Split the batch into half.  Use half for MC estimates for REINFORCE.
  # Use the other half to establish a baseline.
  elif FLAGS.baseline_method == 'dis_batch':
    # TODO(liamfedus):  Recheck.
    [rewards_half, baseline_half] = tf.split(
        rewards, num_or_size_splits=2, axis=0)
    [log_probs_half, _] = tf.split(log_probs, num_or_size_splits=2, axis=0)
    [reward_present_half, baseline_present_half] = tf.split(
        present, num_or_size_splits=2, axis=0)

    # Unstack to lists.
    baseline_list = tf.unstack(baseline_half, axis=1)
    baseline_missing = 1. - tf.cast(baseline_present_half, tf.float32)
    baseline_missing_list = tf.unstack(baseline_missing, axis=1)

    baselines = []
    for t in xrange(FLAGS.sequence_length):
      # Calculate baseline only for missing tokens.
      num_missing = tf.reduce_sum(baseline_missing_list[t])

      avg_baseline = tf.reduce_sum(
          baseline_missing_list[t] * baseline_list[t], keep_dims=True) / (
              num_missing + eps)
      baseline = tf.tile(avg_baseline, multiples=[FLAGS.batch_size / 2])
      baselines.append(baseline)

    # Unstack to lists.
    rewards_list = tf.unstack(rewards_half, axis=1)
    log_probs_list = tf.unstack(log_probs_half, axis=1)
    reward_missing = 1. - tf.cast(reward_present_half, tf.float32)
    reward_missing_list = tf.unstack(reward_missing, axis=1)

    ## Calculate the Advantages, A(s,a) = Q(s,a) - \hat{V}(s).
    advantages = []
    for t in xrange(FLAGS.sequence_length):
      log_probability = log_probs_list[t]
      cum_advantage = tf.zeros(shape=[FLAGS.batch_size / 2])

      for s in xrange(t, FLAGS.sequence_length):
        cum_advantage += reward_missing_list[s] * np.power(gamma, (s - t)) * (
            rewards_list[s] - baselines[s])
      # Clip advantages.
      cum_advantage = tf.clip_by_value(cum_advantage, -FLAGS.advantage_clipping,
                                       FLAGS.advantage_clipping)
      advantages.append(reward_missing_list[t] * cum_advantage)
      final_gen_objective += tf.multiply(
          log_probability,
          reward_missing_list[t] * tf.stop_gradient(cum_advantage))

    # Cumulative Discounted Returns.  The true value function V*(s).
    cumulative_rewards = []
    for t in xrange(FLAGS.sequence_length):
      cum_value = tf.zeros(shape=[FLAGS.batch_size / 2])
      for s in xrange(t, FLAGS.sequence_length):
        cum_value += reward_missing_list[s] * np.power(gamma, (
            s - t)) * rewards_list[s]
      cumulative_rewards.append(cum_value)
    cumulative_rewards = tf.stack(cumulative_rewards, axis=1)

    rewards = rewards_half
    critic_loss = None
    maintain_averages_op = None
    baselines = tf.stack(baselines, axis=1)
    advantages = tf.stack(advantages, axis=1)

  # Exponential Moving Average baseline.
  elif FLAGS.baseline_method == 'ema':
    # TODO(liamfedus): Recheck.
    # Lists of rewards and Log probabilities of the actions taken only for
    # missing tokens.
    ema = tf.train.ExponentialMovingAverage(decay=hparams.baseline_decay)
    maintain_averages_op = ema.apply(rewards_list)

    baselines = []
    for r in rewards_list:
      baselines.append(ema.average(r))

    ## Calculate the Advantages, A(s,a) = Q(s,a) - \hat{V}(s).
    advantages = []
    for t in xrange(FLAGS.sequence_length):
      log_probability = log_probs_list[t]

      # Calculate the forward advantage only on the missing tokens.
      cum_advantage = tf.zeros(shape=[FLAGS.batch_size])
      for s in xrange(t, FLAGS.sequence_length):
        cum_advantage += missing_list[s] * np.power(gamma, (s - t)) * (
            rewards_list[s] - baselines[s])
      # Clip advantages.
      cum_advantage = tf.clip_by_value(cum_advantage, -FLAGS.advantage_clipping,
                                       FLAGS.advantage_clipping)
      advantages.append(missing_list[t] * cum_advantage)
      final_gen_objective += tf.multiply(
          log_probability, missing_list[t] * tf.stop_gradient(cum_advantage))

    critic_loss = None
    baselines = tf.stack(baselines, axis=1)
    advantages = tf.stack(advantages, axis=1)

  elif FLAGS.baseline_method is None:
    num_missing = tf.reduce_sum(missing)
    final_gen_objective += tf.reduce_sum(rewards) / (num_missing + eps)
    baselines = tf.zeros_like(rewards)
    critic_loss = None
    maintain_averages_op = None
    advantages = cumulative_rewards

  else:
    raise NotImplementedError

  return [
      final_gen_objective, log_probs, rewards, advantages, baselines,
      maintain_averages_op, critic_loss, cumulative_rewards
  ]


def calculate_log_perplexity(logits, targets, present):
  """Calculate the average log perplexity per *missing* token.

  Args:
    logits:  tf.float32 Tensor of the logits of shape [batch_size,
      sequence_length, vocab_size].
    targets:  tf.int32 Tensor of the sequence target of shape [batch_size,
      sequence_length].
    present:  tf.bool Tensor indicating the presence or absence of the token
      of shape [batch_size, sequence_length].

  Returns:
    avg_log_perplexity:  Scalar indicating the average log perplexity per
      missing token in the batch.
  """
  # logits = tf.Print(logits, [logits], message='logits:', summarize=50)
  # targets = tf.Print(targets, [targets], message='targets:', summarize=50)
  eps = 1e-12
  logits = tf.reshape(logits, [-1, FLAGS.vocab_size])

  # Only calculate log-perplexity on missing tokens.
  weights = tf.cast(present, tf.float32)
  weights = 1. - weights
  weights = tf.reshape(weights, [-1])
  num_missing = tf.reduce_sum(weights)

  log_perplexity = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
      [logits], [tf.reshape(targets, [-1])], [weights])

  avg_log_perplexity = tf.reduce_sum(log_perplexity) / (num_missing + eps)
  return avg_log_perplexity
