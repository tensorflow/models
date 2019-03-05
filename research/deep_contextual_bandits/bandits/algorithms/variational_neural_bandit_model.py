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

"""Bayesian NN using factorized VI (Bayes By Backprop. Blundell et al. 2014).

See https://arxiv.org/abs/1505.05424 for details.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from absl import flags
from bandits.core.bayesian_nn import BayesianNN

FLAGS = flags.FLAGS


def log_gaussian(x, mu, sigma, reduce_sum=True):
  """Returns log Gaussian pdf."""
  res = (-0.5 * np.log(2 * np.pi) - tf.log(sigma) - tf.square(x - mu) /
         (2 * tf.square(sigma)))
  if reduce_sum:
    return tf.reduce_sum(res)
  else:
    return res


def analytic_kl(mu_1, sigma_1, mu_2, sigma_2):
  """KL for two Gaussian distributions with diagonal covariance matrix."""
  sigma_1_sq = tf.square(sigma_1)
  sigma_2_sq = tf.square(sigma_2)

  t1 = tf.square(mu_1 - mu_2) / (2. * sigma_2_sq)
  t2 = (sigma_1_sq/sigma_2_sq - 1. - tf.log(sigma_1_sq) + tf.log(sigma_2_sq))/2.
  return tf.reduce_sum(t1 + t2)


class VariationalNeuralBanditModel(BayesianNN):
  """Implements an approximate Bayesian NN using Variational Inference."""

  def __init__(self, hparams, name="BBBNN"):

    self.name = name
    self.hparams = hparams

    self.n_in = self.hparams.context_dim
    self.n_out = self.hparams.num_actions
    self.layers = self.hparams.layer_sizes
    self.init_scale = self.hparams.init_scale
    self.f_num_points = None
    if "f_num_points" in hparams:
      self.f_num_points = self.hparams.f_num_points

    self.cleared_times_trained = self.hparams.cleared_times_trained
    self.initial_training_steps = self.hparams.initial_training_steps
    self.training_schedule = np.linspace(self.initial_training_steps,
                                         self.hparams.training_epochs,
                                         self.cleared_times_trained)
    self.verbose = getattr(self.hparams, "verbose", True)

    self.weights_m = {}
    self.weights_std = {}
    self.biases_m = {}
    self.biases_std = {}

    self.times_trained = 0

    if self.hparams.use_sigma_exp_transform:
      self.sigma_transform = tf.exp
      self.inverse_sigma_transform = np.log
    else:
      self.sigma_transform = tf.nn.softplus
      self.inverse_sigma_transform = lambda y: y + np.log(1. - np.exp(-y))

    # Whether to use the local reparameterization trick to compute the loss.
    # See details in https://arxiv.org/abs/1506.02557
    self.use_local_reparameterization = True

    self.build_graph()

  def build_mu_variable(self, shape):
    """Returns a mean variable initialized as N(0, 0.05)."""
    return tf.Variable(tf.random_normal(shape, 0.0, 0.05))

  def build_sigma_variable(self, shape, init=-5.):
    """Returns a sigma variable initialized as N(init, 0.05)."""
    # Initialize sigma to be very small initially to encourage MAP opt first
    return tf.Variable(tf.random_normal(shape, init, 0.05))

  def build_layer(self, input_x, input_x_local, shape,
                  layer_id, activation_fn=tf.nn.relu):
    """Builds a variational layer, and computes KL term.

    Args:
      input_x: Input to the variational layer.
      input_x_local: Input when the local reparameterization trick was applied.
      shape: [number_inputs, number_outputs] for the layer.
      layer_id: Number of layer in the architecture.
      activation_fn: Activation function to apply.

    Returns:
      output_h: Output of the variational layer.
      output_h_local: Output when local reparameterization trick was applied.
      neg_kl: Negative KL term for the layer.
    """

    w_mu = self.build_mu_variable(shape)
    w_sigma = self.sigma_transform(self.build_sigma_variable(shape))
    w_noise = tf.random_normal(shape)
    w = w_mu + w_sigma * w_noise

    b_mu = self.build_mu_variable([1, shape[1]])
    b_sigma = self.sigma_transform(self.build_sigma_variable([1, shape[1]]))
    b = b_mu

    # Store means and stds
    self.weights_m[layer_id] = w_mu
    self.weights_std[layer_id] = w_sigma
    self.biases_m[layer_id] = b_mu
    self.biases_std[layer_id] = b_sigma

    # Create outputs
    output_h = activation_fn(tf.matmul(input_x, w) + b)

    if self.use_local_reparameterization:
      # Use analytic KL divergence wrt the prior
      neg_kl = -analytic_kl(w_mu, w_sigma,
                            0., tf.to_float(np.sqrt(2./shape[0])))
    else:
      # Create empirical KL loss terms
      log_p = log_gaussian(w, 0., tf.to_float(np.sqrt(2./shape[0])))
      log_q = log_gaussian(w, tf.stop_gradient(w_mu), tf.stop_gradient(w_sigma))
      neg_kl = log_p - log_q

    # Apply local reparameterization trick: sample activations pre nonlinearity
    m_h = tf.matmul(input_x_local, w_mu) + b
    v_h = tf.matmul(tf.square(input_x_local), tf.square(w_sigma))
    output_h_local = m_h + tf.sqrt(v_h + 1e-6) * tf.random_normal(tf.shape(v_h))
    output_h_local = activation_fn(output_h_local)

    return output_h, output_h_local, neg_kl

  def build_action_noise(self):
    """Defines a model for additive noise per action, and its KL term."""

    # Define mean and std variables (log-normal dist) for each action.
    noise_sigma_mu = (self.build_mu_variable([1, self.n_out])
                      + self.inverse_sigma_transform(self.hparams.noise_sigma))
    noise_sigma_sigma = self.sigma_transform(
        self.build_sigma_variable([1, self.n_out]))

    pre_noise_sigma = (noise_sigma_mu
                       + tf.random_normal([1, self.n_out]) * noise_sigma_sigma)
    self.noise_sigma = self.sigma_transform(pre_noise_sigma)

    # Compute KL for additive noise sigma terms.
    if getattr(self.hparams, "infer_noise_sigma", False):
      neg_kl_term = log_gaussian(
          pre_noise_sigma,
          self.inverse_sigma_transform(self.hparams.noise_sigma),
          self.hparams.prior_sigma
      )
      neg_kl_term -= log_gaussian(pre_noise_sigma,
                                  noise_sigma_mu,
                                  noise_sigma_sigma)
    else:
      neg_kl_term = 0.

    return neg_kl_term

  def build_model(self, activation_fn=tf.nn.relu):
    """Defines the actual NN model with fully connected layers.

    The loss is computed for partial feedback settings (bandits), so only
    the observed outcome is backpropagated (see weighted loss).
    Selects the optimizer and, finally, it also initializes the graph.

    Args:
      activation_fn: the activation function used in the nn layers.
    """

    if self.verbose:
      print("Initializing model {}.".format(self.name))
    neg_kl_term, l_number = 0, 0
    use_local_reparameterization = self.use_local_reparameterization

    # Compute model additive noise for each action with log-normal distribution
    neg_kl_term += self.build_action_noise()

    # Build network.
    input_x = self.x
    input_local = self.x
    n_in = self.n_in

    for l_number, n_nodes in enumerate(self.layers):
      if n_nodes > 0:
        h, h_local, neg_kl = self.build_layer(input_x, input_local,
                                              [n_in, n_nodes], l_number)

        neg_kl_term += neg_kl
        input_x, input_local = h, h_local
        n_in = n_nodes

    # Create last linear layer
    h, h_local, neg_kl = self.build_layer(input_x, input_local,
                                          [n_in, self.n_out],
                                          l_number + 1,
                                          activation_fn=lambda x: x)
    neg_kl_term += neg_kl

    self.y_pred = h
    self.y_pred_local = h_local

    # Compute log likelihood (with learned or fixed noise level)
    if getattr(self.hparams, "infer_noise_sigma", False):
      log_likelihood = log_gaussian(
          self.y, self.y_pred_local, self.noise_sigma, reduce_sum=False)
    else:
      y_hat = self.y_pred_local if use_local_reparameterization else self.y_pred
      log_likelihood = log_gaussian(
          self.y, y_hat, self.hparams.noise_sigma, reduce_sum=False)

    # Only take into account observed outcomes (bandits setting)
    batch_size = tf.to_float(tf.shape(self.x)[0])
    weighted_log_likelihood = tf.reduce_sum(
        log_likelihood * self.weights) / batch_size

    # The objective is 1/n * (\sum_i log_like_i - KL); neg_kl_term estimates -KL
    elbo = weighted_log_likelihood + (neg_kl_term / self.n)

    self.loss = -elbo
    self.global_step = tf.train.get_or_create_global_step()
    self.train_op = tf.train.AdamOptimizer(self.hparams.initial_lr).minimize(
        self.loss, global_step=self.global_step)

    # Create tensorboard metrics
    self.create_summaries()
    self.summary_writer = tf.summary.FileWriter(
        "{}/graph_{}".format(FLAGS.logdir, self.name), self.sess.graph)

  def build_graph(self):
    """Defines graph, session, placeholders, and model.

    Placeholders are: n (size of the dataset), x and y (context and observed
    reward for each action), and weights (one-hot encoding of selected action
    for each context, i.e., only possibly non-zero element in each y).
    """

    self.graph = tf.Graph()
    with self.graph.as_default():

      self.sess = tf.Session()

      self.n = tf.placeholder(shape=[], dtype=tf.float32)

      self.x = tf.placeholder(shape=[None, self.n_in], dtype=tf.float32)
      self.y = tf.placeholder(shape=[None, self.n_out], dtype=tf.float32)
      self.weights = tf.placeholder(shape=[None, self.n_out], dtype=tf.float32)

      self.build_model()
      self.sess.run(tf.global_variables_initializer())

  def create_summaries(self):
    """Defines summaries including mean loss, and global step."""

    with self.graph.as_default():
      with tf.name_scope(self.name + "_summaries"):
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("global_step", self.global_step)
        self.summary_op = tf.summary.merge_all()

  def assign_lr(self):
    """Resets the learning rate in dynamic schedules for subsequent trainings.

    In bandits settings, we do expand our dataset over time. Then, we need to
    re-train the network with the new data. The algorithms that do not keep
    the step constant, can reset it at the start of each *training* process.
    """

    decay_steps = 1
    if self.hparams.activate_decay:
      current_gs = self.sess.run(self.global_step)
      with self.graph.as_default():
        self.lr = tf.train.inverse_time_decay(self.hparams.initial_lr,
                                              self.global_step - current_gs,
                                              decay_steps,
                                              self.hparams.lr_decay_rate)

  def train(self, data, num_steps):
    """Trains the BNN for num_steps, using the data in 'data'.

    Args:
      data: ContextualDataset object that provides the data.
      num_steps: Number of minibatches to train the network for.

    Returns:
      losses: Loss history during training.
    """

    if self.times_trained < self.cleared_times_trained:
      num_steps = int(self.training_schedule[self.times_trained])
    self.times_trained += 1

    losses = []

    with self.graph.as_default():

      if self.verbose:
        print("Training {} for {} steps...".format(self.name, num_steps))

      for step in range(num_steps):
        x, y, weights = data.get_batch_with_weights(self.hparams.batch_size)
        _, summary, global_step, loss = self.sess.run(
            [self.train_op, self.summary_op, self.global_step, self.loss],
            feed_dict={
                self.x: x,
                self.y: y,
                self.weights: weights,
                self.n: data.num_points(self.f_num_points),
            })

        losses.append(loss)

        if step % self.hparams.freq_summary == 0:
          if self.hparams.show_training:
            print("{} | step: {}, loss: {}".format(
                self.name, global_step, loss))
          self.summary_writer.add_summary(summary, global_step)

    return losses
