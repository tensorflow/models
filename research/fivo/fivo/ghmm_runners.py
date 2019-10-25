# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

"""Creates and runs Gaussian HMM-related graphs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

from fivo import smc
from fivo import bounds
from fivo.data import datasets
from fivo.models import ghmm


def run_train(config):
  """Runs training for a Gaussian HMM setup."""

  def create_logging_hook(step, bound_value, likelihood, bound_gap):
    """Creates a logging hook that prints the bound value periodically."""
    bound_label = config.bound + "/t"
    def summary_formatter(log_dict):
      string = ("Step {step}, %s: {value:.3f}, "
                "likelihood: {ll:.3f}, gap: {gap:.3e}") % bound_label
      return string.format(**log_dict)
    logging_hook = tf.train.LoggingTensorHook(
        {"step": step, "value": bound_value,
         "ll": likelihood, "gap": bound_gap},
        every_n_iter=config.summarize_every,
        formatter=summary_formatter)
    return logging_hook

  def create_losses(model, observations, lengths):
    """Creates the loss to be optimized.

    Args:
      model: A Trainable GHMM model.
      observations: A set of observations.
      lengths: The lengths of each sequence in the observations.
    Returns:
      loss: A float Tensor that when differentiated yields the gradients
         to apply to the model. Should be optimized via gradient descent.
      bound: A float Tensor containing the value of the bound that is
         being optimized.
      true_ll: The true log-likelihood of the data under the model.
      bound_gap: The gap between the bound and the true log-likelihood.
    """
    # Compute lower bounds on the log likelihood.
    if config.bound == "elbo":
      ll_per_seq, _, _ = bounds.iwae(
          model, observations, lengths, num_samples=1,
          parallel_iterations=config.parallel_iterations
      )
    elif config.bound == "iwae":
      ll_per_seq, _, _ = bounds.iwae(
          model, observations, lengths, num_samples=config.num_samples,
          parallel_iterations=config.parallel_iterations
      )
    elif config.bound == "fivo":
      if config.resampling_type == "relaxed":
        ll_per_seq, _, _, _ = bounds.fivo(
            model,
            observations,
            lengths,
            num_samples=config.num_samples,
            resampling_criterion=smc.ess_criterion,
            resampling_type=config.resampling_type,
            relaxed_resampling_temperature=config.
            relaxed_resampling_temperature,
            random_seed=config.random_seed,
            parallel_iterations=config.parallel_iterations)
      else:
        ll_per_seq, _, _, _ = bounds.fivo(
            model, observations, lengths,
            num_samples=config.num_samples,
            resampling_criterion=smc.ess_criterion,
            resampling_type=config.resampling_type,
            random_seed=config.random_seed,
            parallel_iterations=config.parallel_iterations
        )
    ll_per_t = tf.reduce_mean(ll_per_seq / tf.to_float(lengths))
    # Compute the data's true likelihood under the model and the bound gap.
    true_ll_per_seq = model.likelihood(tf.squeeze(observations))
    true_ll_per_t = tf.reduce_mean(true_ll_per_seq / tf.to_float(lengths))
    bound_gap = true_ll_per_seq - ll_per_seq
    bound_gap = tf.reduce_mean(bound_gap/ tf.to_float(lengths))
    tf.summary.scalar("train_ll_bound", ll_per_t)
    tf.summary.scalar("train_true_ll", true_ll_per_t)
    tf.summary.scalar("bound_gap", bound_gap)
    return -ll_per_t, ll_per_t, true_ll_per_t, bound_gap

  def create_graph():
    """Creates the training graph."""
    global_step = tf.train.get_or_create_global_step()
    xs, lengths = datasets.create_chain_graph_dataset(
        config.batch_size,
        config.num_timesteps,
        steps_per_observation=1,
        state_size=1,
        transition_variance=config.variance,
        observation_variance=config.variance)
    model = ghmm.TrainableGaussianHMM(
        config.num_timesteps,
        config.proposal_type,
        transition_variances=config.variance,
        emission_variances=config.variance,
        random_seed=config.random_seed)
    loss, bound, true_ll, gap = create_losses(model, xs, lengths)
    opt = tf.train.AdamOptimizer(config.learning_rate)
    grads = opt.compute_gradients(loss, var_list=tf.trainable_variables())
    train_op = opt.apply_gradients(grads, global_step=global_step)
    return bound, true_ll, gap, train_op, global_step

  with tf.Graph().as_default():
    if config.random_seed:
      tf.set_random_seed(config.random_seed)
      np.random.seed(config.random_seed)
    bound, true_ll, gap, train_op, global_step = create_graph()
    log_hook = create_logging_hook(global_step, bound, true_ll, gap)
    with tf.train.MonitoredTrainingSession(
        master="",
        hooks=[log_hook],
        checkpoint_dir=config.logdir,
        save_checkpoint_secs=120,
        save_summaries_steps=config.summarize_every,
        log_step_count_steps=config.summarize_every*20) as sess:
      cur_step = -1
      while cur_step <= config.max_steps and not sess.should_stop():
        cur_step = sess.run(global_step)
        _, cur_step = sess.run([train_op, global_step])


def run_eval(config):
  """Evaluates a Gaussian HMM using the given config."""

  def create_bound(model, xs, lengths):
    """Creates the bound to be evaluated."""
    if config.bound == "elbo":
      ll_per_seq, log_weights, _ = bounds.iwae(
          model, xs, lengths, num_samples=1,
          parallel_iterations=config.parallel_iterations
      )
    elif config.bound == "iwae":
      ll_per_seq, log_weights, _ = bounds.iwae(
          model, xs, lengths, num_samples=config.num_samples,
          parallel_iterations=config.parallel_iterations
      )
    elif config.bound == "fivo":
      ll_per_seq, log_weights, resampled, _ = bounds.fivo(
          model, xs, lengths,
          num_samples=config.num_samples,
          resampling_criterion=smc.ess_criterion,
          resampling_type=config.resampling_type,
          random_seed=config.random_seed,
          parallel_iterations=config.parallel_iterations
      )
    # Compute bound scaled by number of timesteps.
    bound_per_t = ll_per_seq / tf.to_float(lengths)
    if config.bound == "fivo":
      return bound_per_t, log_weights, resampled
    else:
      return bound_per_t, log_weights

  def create_graph():
    """Creates the dataset, model, and bound."""
    xs, lengths = datasets.create_chain_graph_dataset(
        config.batch_size,
        config.num_timesteps,
        steps_per_observation=1,
        state_size=1,
        transition_variance=config.variance,
        observation_variance=config.variance)
    model = ghmm.TrainableGaussianHMM(
        config.num_timesteps,
        config.proposal_type,
        transition_variances=config.variance,
        emission_variances=config.variance,
        random_seed=config.random_seed)
    true_likelihood = tf.reduce_mean(
        model.likelihood(tf.squeeze(xs)) / tf.to_float(lengths))
    outs = [true_likelihood]
    outs.extend(list(create_bound(model, xs, lengths)))
    return outs

  with tf.Graph().as_default():
    if config.random_seed:
      tf.set_random_seed(config.random_seed)
      np.random.seed(config.random_seed)
    graph_outs = create_graph()
    with tf.train.SingularMonitoredSession(
        checkpoint_dir=config.logdir) as sess:
      outs = sess.run(graph_outs)
      likelihood = outs[0]
      avg_bound = np.mean(outs[1])
      std = np.std(outs[1])
      log_weights = outs[2]
      log_weight_variances = np.var(log_weights, axis=2)
      avg_log_weight_variance = np.var(log_weight_variances, axis=1)
      avg_log_weight = np.mean(log_weights, axis=(1, 2))
      data = {"mean": avg_bound, "std": std, "log_weights": log_weights,
              "log_weight_means": avg_log_weight,
              "log_weight_variances": avg_log_weight_variance}
      if len(outs) == 4:
        data["resampled"] = outs[3]
        data["avg_resampled"] = np.mean(outs[3], axis=1)
      # Log some useful statistics.
      tf.logging.info("Evaled bound %s with batch_size: %d, num_samples: %d."
                      % (config.bound, config.batch_size, config.num_samples))
      tf.logging.info("mean: %f, std: %f" % (avg_bound, std))
      tf.logging.info("true likelihood: %s" % likelihood)
      tf.logging.info("avg log weight: %s" % avg_log_weight)
      tf.logging.info("log weight variance: %s" % avg_log_weight_variance)
      if len(outs) == 4:
        tf.logging.info("avg resamples per t: %s" % data["avg_resampled"])
      if not tf.gfile.Exists(config.logdir):
        tf.gfile.MakeDirs(config.logdir)
      with tf.gfile.Open(os.path.join(config.logdir, "out.npz"), "w") as fout:
        np.save(fout, data)
