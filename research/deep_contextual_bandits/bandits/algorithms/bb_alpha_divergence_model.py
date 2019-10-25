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

"""Bayesian NN using expectation propagation (Black-Box Alpha-Divergence).

See https://arxiv.org/abs/1511.03243 for details.
All formulas used in this implementation are derived in:
https://www.overleaf.com/12837696kwzjxkyhdytk#/49028744/.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
from absl import flags

from bandits.core.bayesian_nn import BayesianNN


FLAGS = flags.FLAGS
tfd = tf.contrib.distributions  # update to: tensorflow_probability.distributions


def log_gaussian(x, mu, sigma, reduce_sum=True):
  res = tfd.Normal(mu, sigma).log_prob(x)
  if reduce_sum:
    return tf.reduce_sum(res)
  else:
    return res


class BBAlphaDivergence(BayesianNN):
  """Implements an approximate Bayesian NN via Black-Box Alpha-Divergence."""

  def __init__(self, hparams, name):

    self.name = name
    self.hparams = hparams

    self.alpha = getattr(self.hparams, 'alpha', 1.0)
    self.num_mc_nn_samples = getattr(self.hparams, 'num_mc_nn_samples', 10)

    self.n_in = self.hparams.context_dim
    self.n_out = self.hparams.num_actions
    self.layers = self.hparams.layer_sizes
    self.batch_size = self.hparams.batch_size

    self.show_training = self.hparams.show_training
    self.freq_summary = self.hparams.freq_summary
    self.verbose = getattr(self.hparams, 'verbose', True)

    self.cleared_times_trained = self.hparams.cleared_times_trained
    self.initial_training_steps = self.hparams.initial_training_steps
    self.training_schedule = np.linspace(self.initial_training_steps,
                                         self.hparams.training_epochs,
                                         self.cleared_times_trained)

    self.times_trained = 0
    self.initialize_model()

  def initialize_model(self):
    """Builds and initialize the model."""

    self.num_w = 0
    self.num_b = 0

    self.weights_m = {}
    self.weights_std = {}
    self.biases_m = {}
    self.biases_std = {}

    self.h_max_var = []

    if self.hparams.use_sigma_exp_transform:
      self.sigma_transform = tfd.bijectors.Exp()
    else:
      self.sigma_transform = tfd.bijectors.Softplus()

    # Build the graph corresponding to the Bayesian NN instance.
    self.graph = tf.Graph()

    with self.graph.as_default():

      self.sess = tf.Session()
      self.x = tf.placeholder(shape=[None, self.n_in],
                              dtype=tf.float32, name='x')
      self.y = tf.placeholder(shape=[None, self.n_out],
                              dtype=tf.float32, name='y')
      self.weights = tf.placeholder(shape=[None, self.n_out],
                                    dtype=tf.float32, name='w')
      self.data_size = tf.placeholder(tf.float32, shape=(), name='data_size')

      self.prior_variance = self.hparams.prior_variance
      if self.prior_variance < 0:
        # if not fixed, we learn the prior.
        self.prior_variance = self.sigma_transform.forward(
            self.build_mu_variable([1, 1]))

      self.build_model()
      self.sess.run(tf.global_variables_initializer())

  def build_mu_variable(self, shape):
    """Returns a mean variable initialized as N(0, 0.05)."""
    return tf.Variable(tf.random_normal(shape, 0.0, 0.05))

  def build_sigma_variable(self, shape, init=-5.):
    """Returns a sigma variable initialized as N(init, 0.05)."""
    # Initialize sigma to be very small initially to encourage MAP opt first
    return tf.Variable(tf.random_normal(shape, init, 0.05))

  def build_layer(self, input_x, shape, layer_id, activation_fn=tf.nn.relu):
    """Builds a layer with N(mean, std) for each weight, and samples from it."""

    w_mu = self.build_mu_variable(shape)
    w_sigma = self.sigma_transform.forward(self.build_sigma_variable(shape))

    w_noise = tf.random_normal(shape)
    w = w_mu + w_sigma * w_noise

    b_mu = self.build_mu_variable([1, shape[1]])
    b_sigma = self.sigma_transform.forward(
        self.build_sigma_variable([1, shape[1]]))

    b_noise = tf.random_normal([1, shape[1]])
    b = b_mu + b_sigma * b_noise

    # Create outputs
    output_h = activation_fn(tf.matmul(input_x, w) + b)

    # Store means and stds
    self.weights_m[layer_id] = w_mu
    self.weights_std[layer_id] = w_sigma
    self.biases_m[layer_id] = b_mu
    self.biases_std[layer_id] = b_sigma

    return output_h

  def sample_neural_network(self, activation_fn=tf.nn.relu):
    """Samples a nn from posterior, computes data log lk and log f factor."""

    with self.graph.as_default():

      log_f = 0
      n = self.data_size
      input_x = self.x

      for layer_id in range(self.total_layers):

        # load mean and std of each weight
        w_mu = self.weights_m[layer_id]
        w_sigma = self.weights_std[layer_id]
        b_mu = self.biases_m[layer_id]
        b_sigma = self.biases_std[layer_id]

        # sample weights from Gaussian distribution
        shape = w_mu.shape
        w_noise = tf.random_normal(shape)
        b_noise = tf.random_normal([1, int(shape[1])])
        w = w_mu + w_sigma * w_noise
        b = b_mu + b_sigma * b_noise

        # compute contribution to log_f
        t1 = w * w_mu / (n * w_sigma ** 2)
        t2 = (0.5 * w ** 2 / n) * (1 / self.prior_variance - 1 / w_sigma ** 2)
        log_f += tf.reduce_sum(t1 + t2)

        t1 = b * b_mu / (n * b_sigma ** 2)
        t2 = (0.5 * b ** 2 / n) * (1 / self.prior_variance - 1 / b_sigma ** 2)
        log_f += tf.reduce_sum(t1 + t2)

        if layer_id < self.total_layers - 1:
          output_h = activation_fn(tf.matmul(input_x, w) + b)
        else:
          output_h = tf.matmul(input_x, w) + b

        input_x = output_h

      # compute log likelihood of the observed reward under the sampled nn
      log_likelihood = log_gaussian(
          self.y, output_h, self.noise_sigma, reduce_sum=False)
      weighted_log_likelihood = tf.reduce_sum(log_likelihood * self.weights, -1)

    return log_f, weighted_log_likelihood

  def log_z_q(self):
    """Computes log-partition function of current posterior parameters."""

    with self.graph.as_default():

      log_z_q = 0

      for layer_id in range(self.total_layers):

        w_mu = self.weights_m[layer_id]
        w_sigma = self.weights_std[layer_id]
        b_mu = self.biases_m[layer_id]
        b_sigma = self.biases_std[layer_id]

        w_term = 0.5 * tf.reduce_sum(w_mu ** 2 / w_sigma ** 2)
        w_term += 0.5 * tf.reduce_sum(tf.log(2 * np.pi) + 2 * tf.log(w_sigma))

        b_term = 0.5 * tf.reduce_sum(b_mu ** 2 / b_sigma ** 2)
        b_term += 0.5 * tf.reduce_sum(tf.log(2 * np.pi) + 2 * tf.log(b_sigma))

        log_z_q += w_term + b_term

      return log_z_q

  def log_z_prior(self):
    """Computes log-partition function of the prior parameters."""
    num_params = self.num_w + self.num_b
    return num_params * 0.5 * tf.log(2 * np.pi * self.prior_variance)

  def log_alpha_likelihood_ratio(self, activation_fn=tf.nn.relu):

    # each nn sample returns (log f, log likelihoods)
    nn_samples = [
        self.sample_neural_network(activation_fn)
        for _ in range(self.num_mc_nn_samples)
    ]
    nn_log_f_samples = [elt[0] for elt in nn_samples]
    nn_log_lk_samples = [elt[1] for elt in nn_samples]

    # we stack the (log f, log likelihoods) from the k nn samples
    nn_log_f_stack = tf.stack(nn_log_f_samples)      # k x 1
    nn_log_lk_stack = tf.stack(nn_log_lk_samples)    # k x N
    nn_f_tile = tf.tile(nn_log_f_stack, [self.batch_size])
    nn_f_tile = tf.reshape(nn_f_tile,
                           [self.num_mc_nn_samples, self.batch_size])

    # now both the log f and log likelihood terms have shape: k x N
    # apply formula in https://www.overleaf.com/12837696kwzjxkyhdytk#/49028744/
    nn_log_ratio = nn_log_lk_stack - nn_f_tile
    nn_log_ratio = self.alpha * tf.transpose(nn_log_ratio)
    logsumexp_value = tf.reduce_logsumexp(nn_log_ratio, -1)
    log_k_scalar = tf.log(tf.cast(self.num_mc_nn_samples, tf.float32))
    log_k = log_k_scalar * tf.ones([self.batch_size])

    return tf.reduce_sum(logsumexp_value - log_k, -1)

  def build_model(self, activation_fn=tf.nn.relu):
    """Defines the actual NN model with fully connected layers.

    Args:
      activation_fn: Activation function for the neural network.

    The loss is computed for partial feedback settings (bandits), so only
    the observed outcome is backpropagated (see weighted loss).
    Selects the optimizer and, finally, it also initializes the graph.
    """

    print('Initializing model {}.'.format(self.name))

    # Build terms for the noise sigma estimation for each action.
    noise_sigma_mu = (self.build_mu_variable([1, self.n_out])
                      + self.sigma_transform.inverse(self.hparams.noise_sigma))
    noise_sigma_sigma = self.sigma_transform.forward(
        self.build_sigma_variable([1, self.n_out]))

    pre_noise_sigma = noise_sigma_mu + tf.random_normal(
        [1, self.n_out]) * noise_sigma_sigma
    self.noise_sigma = self.sigma_transform.forward(pre_noise_sigma)

    # Build network
    input_x = self.x
    n_in = self.n_in
    self.total_layers = len(self.layers) + 1
    if self.layers[0] == 0:
      self.total_layers = 1

    for l_number, n_nodes in enumerate(self.layers):
      if n_nodes > 0:
        h = self.build_layer(input_x, [n_in, n_nodes], l_number)
        input_x = h
        n_in = n_nodes
        self.num_w += n_in * n_nodes
        self.num_b += n_nodes

    self.y_pred = self.build_layer(input_x, [n_in, self.n_out],
                                   self.total_layers - 1,
                                   activation_fn=lambda x: x)

    # Compute energy function based on sampled nn's
    log_coeff = self.data_size / (self.batch_size * self.alpha)
    log_ratio = log_coeff * self.log_alpha_likelihood_ratio(activation_fn)
    logzprior = self.log_z_prior()
    logzq = self.log_z_q()
    energy = logzprior - logzq - log_ratio

    self.loss = energy
    self.global_step = tf.train.get_or_create_global_step()
    self.train_op = tf.train.AdamOptimizer(self.hparams.initial_lr).minimize(
        self.loss, global_step=self.global_step)

    # Useful for debugging
    sq_loss = tf.squared_difference(self.y_pred, self.y)
    weighted_sq_loss = self.weights * sq_loss
    self.cost = tf.reduce_sum(weighted_sq_loss) / self.batch_size

    # Create tensorboard metrics
    self.create_summaries()
    self.summary_writer = tf.summary.FileWriter('{}/graph_{}'.format(
        FLAGS.logdir, self.name), self.sess.graph)

  def create_summaries(self):
    tf.summary.scalar('loss', self.loss)
    tf.summary.scalar('cost', self.cost)
    self.summary_op = tf.summary.merge_all()

  def assign_lr(self):
    """Resets the learning rate in dynamic schedules for subsequent trainings.

    In bandits settings, we do expand our dataset over time. Then, we need to
    re-train the network with the new data. Those algorithms that do not keep
    the step constant, can reset it at the start of each training process.
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
    """

    if self.times_trained < self.cleared_times_trained:
      num_steps = int(self.training_schedule[self.times_trained])
    self.times_trained += 1

    if self.verbose:
      print('Training {} for {} steps...'.format(self.name, num_steps))

    with self.graph.as_default():

      for step in range(num_steps):
        x, y, w = data.get_batch_with_weights(self.hparams.batch_size)
        _, summary, global_step, loss = self.sess.run(
            [self.train_op, self.summary_op, self.global_step, self.loss],
            feed_dict={self.x: x, self.y: y, self.weights: w,
                       self.data_size: data.num_points()})

        weights_l = self.sess.run(self.weights_std[0])
        self.h_max_var.append(np.max(weights_l))

        if step % self.freq_summary == 0:
          if self.show_training:
            print('step: {}, loss: {}'.format(step, loss))
            sys.stdout.flush()
          self.summary_writer.add_summary(summary, global_step)
