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

"""Utils for plotting and summarizing.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy

import tensorflow as tf

import models


def summarize_ess(weights, only_last_timestep=False):
  """Plots the effective sample size.

  Args:
    weights: List of length num_timesteps Tensors of shape
    [num_samples, batch_size]
  """
  num_timesteps = len(weights)
  batch_size = tf.cast(tf.shape(weights[0])[1], dtype=tf.float64)
  for i in range(num_timesteps):
    if only_last_timestep and i < num_timesteps-1: continue

    w = tf.nn.softmax(weights[i], dim=0)
    centered_weights = w - tf.reduce_mean(w, axis=0, keepdims=True)
    variance = tf.reduce_sum(tf.square(centered_weights))/(batch_size-1)
    ess = 1./tf.reduce_mean(tf.reduce_sum(tf.square(w), axis=0))
    tf.summary.scalar("ess/%d" % i, ess)
    tf.summary.scalar("ese/%d" % i, ess / batch_size)
    tf.summary.scalar("weight_variance/%d" % i, variance)


def summarize_particles(states, weights, observation, model):
  """Plots particle locations and weights.

  Args:
    states: List of length num_timesteps Tensors of shape
      [batch_size*num_particles, state_size].
    weights: List of length num_timesteps Tensors of shape [num_samples,
      batch_size]
    observation: Tensor of shape [batch_size*num_samples, state_size]
  """
  num_timesteps = len(weights)
  num_samples, batch_size = weights[0].get_shape().as_list()
  # get q0 information for plotting
  q0_dist = model.q.q_zt(observation, tf.zeros_like(states[0]), 0)
  q0_loc = q0_dist.loc[0:batch_size, 0]
  q0_scale = q0_dist.scale[0:batch_size, 0]
  # get posterior information for plotting
  post = (model.p.mixing_coeff, model.p.prior_mode_mean, model.p.variance,
          tf.reduce_sum(model.p.bs), model.p.num_timesteps)

  # Reshape states and weights to be [time, num_samples, batch_size]
  states = tf.stack(states)
  weights = tf.stack(weights)
  # normalize the weights over the sample dimension
  weights = tf.nn.softmax(weights, dim=1)
  states = tf.reshape(states, tf.shape(weights))

  ess = 1./tf.reduce_sum(tf.square(weights), axis=1)

  def _plot_states(states_batch, weights_batch, observation_batch, ess_batch, q0, post):
    """
    states: [time, num_samples, batch_size]
    weights [time, num_samples, batch_size]
    observation: [batch_size, 1]
    q0: ([batch_size], [batch_size])
    post: ...
    """
    num_timesteps, _, batch_size = states_batch.shape
    plots = []
    for i in range(batch_size):
      states = states_batch[:,:,i]
      weights = weights_batch[:,:,i]
      observation = observation_batch[i]
      ess = ess_batch[:,i]
      q0_loc = q0[0][i]
      q0_scale = q0[1][i]

      fig = plt.figure(figsize=(7, (num_timesteps + 1) * 2))
      # Each timestep gets two plots -- a bar plot and a histogram of state locs.
      # The bar plot will be bar_rows rows tall.
      # The histogram will be 1 row tall.
      # There is also 1 extra plot at the top showing the posterior and q.
      bar_rows = 8
      num_rows = (num_timesteps + 1) * (bar_rows + 1)
      gs = gridspec.GridSpec(num_rows, 1)

      # Figure out how wide to make the plot
      prior_lims = (post[1] * -2, post[1] * 2)
      q_lims = (scipy.stats.norm.ppf(0.01, loc=q0_loc, scale=q0_scale),
                scipy.stats.norm.ppf(0.99, loc=q0_loc, scale=q0_scale))
      state_width = states.max() - states.min()
      state_lims = (states.min() - state_width * 0.15,
                    states.max() + state_width * 0.15)

      lims = (min(prior_lims[0], q_lims[0], state_lims[0]),
              max(prior_lims[1], q_lims[1], state_lims[1]))
      # plot the posterior
      z0 = np.arange(lims[0], lims[1], 0.1)
      alpha, pos_mu, sigma_sq, B, T = post
      neg_mu = -pos_mu
      scale = np.sqrt((T + 1) * sigma_sq)
      p_zn = (
          alpha * scipy.stats.norm.pdf(
              observation, loc=pos_mu + B, scale=scale) + (1 - alpha) *
          scipy.stats.norm.pdf(observation, loc=neg_mu + B, scale=scale))
      p_z0 = (
          alpha * scipy.stats.norm.pdf(z0, loc=pos_mu, scale=np.sqrt(sigma_sq))
          + (1 - alpha) * scipy.stats.norm.pdf(
              z0, loc=neg_mu, scale=np.sqrt(sigma_sq)))
      p_zn_given_z0 = scipy.stats.norm.pdf(
          observation, loc=z0 + B, scale=np.sqrt(T * sigma_sq))
      post_z0 = (p_z0 * p_zn_given_z0) / p_zn
      # plot q
      q_z0 = scipy.stats.norm.pdf(z0, loc=q0_loc, scale=q0_scale)
      ax = plt.subplot(gs[0:bar_rows, :])
      ax.plot(z0, q_z0, color="blue")
      ax.plot(z0, post_z0, color="green")
      ax.plot(z0, p_z0, color="red")
      ax.legend(("q", "posterior", "prior"), loc="best", prop={"size": 10})

      ax.set_xticks([])
      ax.set_xlim(*lims)

      # plot the states
      for t in range(num_timesteps):
        start = (t + 1) * (bar_rows + 1)
        ax1 = plt.subplot(gs[start:start + bar_rows, :])
        ax2 = plt.subplot(gs[start + bar_rows:start + bar_rows + 1, :])
        # plot the states barplot
        # ax1.hist(
        #     states[t, :],
        #     weights=weights[t, :],
        #     bins=50,
        #     edgecolor="none",
        #     alpha=0.2)
        ax1.bar(states[t,:], weights[t,:], width=0.02, alpha=0.2, edgecolor = "none")
        ax1.set_ylabel("t=%d" % t)
        ax1.set_xticks([])
        ax1.grid(True, which="both")
        ax1.set_xlim(*lims)
        # plot the observation
        ax1.axvline(x=observation, color="red", linestyle="dashed")
        # add the ESS
        ax1.text(0.1, 0.9, "ESS: %0.2f" % ess[t],
                 ha='center', va='center', transform=ax1.transAxes)

        # plot the state location histogram
        ax2.hist2d(
            states[t, :], np.zeros_like(states[t, :]), bins=[50, 1], cmap="Greys")
        ax2.grid(False)
        ax2.set_yticks([])
        ax2.set_xlim(*lims)
        if t != num_timesteps - 1:
          ax2.set_xticks([])

      fig.canvas.draw()
      p = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
      plots.append(p.reshape(fig.canvas.get_width_height()[::-1] + (3,)))
      plt.close(fig)
    return np.stack(plots)

  plots = tf.py_func(_plot_states,
                     [states, weights, observation, ess, (q0_loc, q0_scale), post],
                     [tf.uint8])[0]
  tf.summary.image("states", plots, 5, collections=["infrequent_summaries"])


def plot_weights(weights, resampled=None):
  """Plots the weights and effective sample size from an SMC rollout.

  Args:
    weights: [num_timesteps, num_samples, batch_size] importance weights
    resampled: [num_timesteps] 0/1 indicating if resampling ocurred
  """
  weights = tf.convert_to_tensor(weights)

  def _make_plots(weights, resampled):
    num_timesteps, num_samples, batch_size = weights.shape
    plots = []
    for i in range(batch_size):
      fig, axes = plt.subplots(nrows=1, sharex=True, figsize=(8, 4))
      axes.stackplot(np.arange(num_timesteps), np.transpose(weights[:, :, i]))
      axes.set_title("Weights")
      axes.set_xlabel("Steps")
      axes.set_ylim([0, 1])
      axes.set_xlim([0, num_timesteps - 1])
      for j in np.where(resampled > 0)[0]:
        axes.axvline(x=j, color="red", linestyle="dashed", ymin=0.0, ymax=1.0)
      fig.canvas.draw()
      data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
      data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      plots.append(data)
      plt.close(fig)
    return np.stack(plots, axis=0)

  if resampled is None:
    num_timesteps, _, batch_size = weights.get_shape().as_list()
    resampled = tf.zeros([num_timesteps], dtype=tf.float32)
  plots = tf.py_func(_make_plots,
                     [tf.nn.softmax(weights, dim=1),
                      tf.to_float(resampled)], [tf.uint8])[0]
  batch_size = weights.get_shape().as_list()[-1]
  tf.summary.image(
      "weights", plots, batch_size, collections=["infrequent_summaries"])


def summarize_weights(weights, num_timesteps, num_samples):
  # weights is [num_timesteps, num_samples, batch_size]
  weights = tf.convert_to_tensor(weights)
  mean = tf.reduce_mean(weights, axis=1, keepdims=True)
  squared_diff = tf.square(weights - mean)
  variances = tf.reduce_sum(squared_diff, axis=1) / (num_samples - 1)
  # average the variance over the batch
  variances = tf.reduce_mean(variances, axis=1)
  avg_magnitude = tf.reduce_mean(tf.abs(weights), axis=[1, 2])
  for t in xrange(num_timesteps):
    tf.summary.scalar("weights/variance_%d" % t, variances[t])
    tf.summary.scalar("weights/magnitude_%d" % t, avg_magnitude[t])
    tf.summary.histogram("weights/step_%d" % t, weights[t])


def summarize_learning_signal(rewards, tag):
  num_resampling_events, _ = rewards.get_shape().as_list()
  mean = tf.reduce_mean(rewards, axis=1)
  avg_magnitude = tf.reduce_mean(tf.abs(rewards), axis=1)
  reward_square = tf.reduce_mean(tf.square(rewards), axis=1)
  for t in xrange(num_resampling_events):
    tf.summary.scalar("%s/mean_%d" % (tag, t), mean[t])
    tf.summary.scalar("%s/magnitude_%d" % (tag, t), avg_magnitude[t])
    tf.summary.scalar("%s/squared_%d" % (tag, t), reward_square[t])
    tf.summary.histogram("%s/step_%d" % (tag, t), rewards[t])


def summarize_qs(model, observation, states):
  model.q.summarize_weights()
  if hasattr(model.p, "posterior") and callable(getattr(model.p, "posterior")):
    states = [tf.zeros_like(states[0])] + states[:-1]
    for t, prev_state in enumerate(states):
      p = model.p.posterior(observation, prev_state, t)
      q = model.q.q_zt(observation, prev_state, t)
      kl = tf.reduce_mean(tf.contrib.distributions.kl_divergence(p, q))
      tf.summary.scalar("kl_q/%d" % t, tf.reduce_mean(kl))
      mean_diff = q.loc - p.loc
      mean_abs_err = tf.abs(mean_diff)
      mean_rel_err = tf.abs(mean_diff / p.loc)
      tf.summary.scalar("q_mean_convergence/absolute_error_%d" % t,
                        tf.reduce_mean(mean_abs_err))
      tf.summary.scalar("q_mean_convergence/relative_error_%d" % t,
                        tf.reduce_mean(mean_rel_err))
      sigma_diff = tf.square(q.scale) - tf.square(p.scale)
      sigma_abs_err = tf.abs(sigma_diff)
      sigma_rel_err = tf.abs(sigma_diff / tf.square(p.scale))
      tf.summary.scalar("q_variance_convergence/absolute_error_%d" % t,
                        tf.reduce_mean(sigma_abs_err))
      tf.summary.scalar("q_variance_convergence/relative_error_%d" % t,
                        tf.reduce_mean(sigma_rel_err))


def summarize_rs(model, states):
  model.r.summarize_weights()
  for t, state in enumerate(states):
    true_r = model.p.lookahead(state, t)
    r = model.r.r_xn(state, t)
    kl = tf.reduce_mean(tf.contrib.distributions.kl_divergence(true_r, r))
    tf.summary.scalar("kl_r/%d" % t, tf.reduce_mean(kl))
    mean_diff = true_r.loc - r.loc
    mean_abs_err = tf.abs(mean_diff)
    mean_rel_err = tf.abs(mean_diff / true_r.loc)
    tf.summary.scalar("r_mean_convergence/absolute_error_%d" % t,
                      tf.reduce_mean(mean_abs_err))
    tf.summary.scalar("r_mean_convergence/relative_error_%d" % t,
                      tf.reduce_mean(mean_rel_err))
    sigma_diff = tf.square(r.scale) - tf.square(true_r.scale)
    sigma_abs_err = tf.abs(sigma_diff)
    sigma_rel_err = tf.abs(sigma_diff / tf.square(true_r.scale))
    tf.summary.scalar("r_variance_convergence/absolute_error_%d" % t,
                      tf.reduce_mean(sigma_abs_err))
    tf.summary.scalar("r_variance_convergence/relative_error_%d" % t,
                      tf.reduce_mean(sigma_rel_err))


def summarize_model(model, true_bs, observation, states, bound, summarize_r=True):
  if hasattr(model.p, "bs"):
    model_b = tf.reduce_sum(model.p.bs, axis=0)
    true_b = tf.reduce_sum(true_bs, axis=0)
    abs_err = tf.abs(model_b - true_b)
    rel_err = abs_err / true_b
    tf.summary.scalar("sum_of_bs/data_generating_process", tf.reduce_mean(true_b))
    tf.summary.scalar("sum_of_bs/model", tf.reduce_mean(model_b))
    tf.summary.scalar("sum_of_bs/absolute_error", tf.reduce_mean(abs_err))
    tf.summary.scalar("sum_of_bs/relative_error", tf.reduce_mean(rel_err))
  #summarize_qs(model, observation, states)
  #if bound == "fivo-aux" and summarize_r:
  #  summarize_rs(model, states)


def summarize_grads(grads, loss_name):
  grad_ema = tf.train.ExponentialMovingAverage(decay=0.99)
  vectorized_grads = tf.concat(
      [tf.reshape(g, [-1]) for g, _ in grads if g is not None], axis=0)
  new_second_moments = tf.square(vectorized_grads)
  new_first_moments = vectorized_grads
  maintain_grad_ema_op = grad_ema.apply([new_first_moments, new_second_moments])
  first_moments = grad_ema.average(new_first_moments)
  second_moments = grad_ema.average(new_second_moments)
  variances = second_moments - tf.square(first_moments)
  tf.summary.scalar("grad_variance/%s" % loss_name, tf.reduce_mean(variances))
  tf.summary.histogram("grad_variance/%s" % loss_name, variances)
  tf.summary.histogram("grad_mean/%s" % loss_name, first_moments)
  return maintain_grad_ema_op
