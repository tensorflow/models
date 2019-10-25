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

"""A Multitask Gaussian process."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import logging

import numpy as np
import tensorflow as tf
from bandits.core.bayesian_nn import BayesianNN

FLAGS = flags.FLAGS
tfd = tf.contrib.distributions

class MultitaskGP(BayesianNN):
  """Implements a Gaussian process with multi-task outputs.

  Optimizes the hyperparameters over the log marginal likelihood.
  Uses a Matern 3/2 + linear covariance and returns
  sampled predictions for test inputs.  The outputs are optionally
  correlated where the correlation structure is learned through latent
  embeddings of the tasks.
  """

  def __init__(self, hparams):
    self.name = "MultiTaskGP"
    self.hparams = hparams

    self.n_in = self.hparams.context_dim
    self.n_out = self.hparams.num_outputs
    self.keep_fixed_after_max_obs = self.hparams.keep_fixed_after_max_obs

    self._show_training = self.hparams.show_training
    self._freq_summary = self.hparams.freq_summary

    # Dimensionality of the latent task vectors
    self.task_latent_dim = self.hparams.task_latent_dim

    # Maximum number of observations to include
    self.max_num_points = self.hparams.max_num_points

    if self.hparams.learn_embeddings:
      self.learn_embeddings = self.hparams.learn_embeddings
    else:
      self.learn_embeddings = False

    # create the graph corresponding to the BNN instance
    self.graph = tf.Graph()
    with self.graph.as_default():
      # store a new session for the graph
      self.sess = tf.Session()

      with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
        self.n = tf.placeholder(shape=[], dtype=tf.float64)
        self.x = tf.placeholder(shape=[None, self.n_in], dtype=tf.float64)
        self.x_in = tf.placeholder(shape=[None, self.n_in], dtype=tf.float64)
        self.y = tf.placeholder(shape=[None, self.n_out], dtype=tf.float64)
        self.weights = tf.placeholder(shape=[None, self.n_out],
                                      dtype=tf.float64)

        self.build_model()
      self.sess.run(tf.global_variables_initializer())

  def atleast_2d(self, x, dims):
    return tf.reshape(tf.expand_dims(x, axis=0), (-1, dims))

  def sq_dist(self, x, x2):
    a2 = tf.reduce_sum(tf.square(x), 1)
    b2 = tf.reduce_sum(tf.square(x2), 1)
    sqdists = tf.expand_dims(a2, 1) + b2 - 2.0 * tf.matmul(x, tf.transpose(x2))
    return sqdists

  # Covariance between outputs
  def task_cov(self, x, x2):
    """Squared Exponential Covariance Kernel over latent task embeddings."""
    # Index into latent task vectors
    x_vecs = tf.gather(self.task_vectors, tf.argmax(x, axis=1), axis=0)
    x2_vecs = tf.gather(self.task_vectors, tf.argmax(x2, axis=1), axis=0)
    r = self.sq_dist(self.atleast_2d(x_vecs, self.task_latent_dim),
                     self.atleast_2d(x2_vecs, self.task_latent_dim))
    return tf.exp(-r)

  def cov(self, x, x2):
    """Matern 3/2 + Linear Gaussian Process Covariance Function."""
    ls = tf.clip_by_value(self.length_scales, -5.0, 5.0)
    ls_lin = tf.clip_by_value(self.length_scales_lin, -5.0, 5.0)
    r = self.sq_dist(self.atleast_2d(x, self.n_in)/tf.nn.softplus(ls),
                     self.atleast_2d(x2, self.n_in)/tf.nn.softplus(ls))
    r = tf.clip_by_value(r, 0, 1e8)

    # Matern 3/2 Covariance
    matern = (1.0 + tf.sqrt(3.0*r + 1e-16)) * tf.exp(-tf.sqrt(3.0*r + 1e-16))
    # Linear Covariance
    lin = tf.matmul(x / tf.nn.softplus(ls_lin),
                    x2 / tf.nn.softplus(ls_lin), transpose_b=True)
    return (tf.nn.softplus(self.amplitude) * matern +
            tf.nn.softplus(self.amplitude_linear) * lin)

  def build_model(self):
    """Defines the GP model.

    The loss is computed for partial feedback settings (bandits), so only
    the observed outcome is backpropagated (see weighted loss).
    Selects the optimizer and, finally, it also initializes the graph.
    """

    logging.info("Initializing model %s.", self.name)
    self.global_step = tf.train.get_or_create_global_step()

    # Define state for the model (inputs, etc.)
    self.x_train = tf.get_variable(
        "training_data",
        initializer=tf.ones(
            [self.hparams.batch_size, self.n_in], dtype=tf.float64),
        validate_shape=False,
        trainable=False)
    self.y_train = tf.get_variable(
        "training_labels",
        initializer=tf.zeros([self.hparams.batch_size, 1], dtype=tf.float64),
        validate_shape=False,
        trainable=False)
    self.weights_train = tf.get_variable(
        "weights_train",
        initializer=tf.ones(
            [self.hparams.batch_size, self.n_out], dtype=tf.float64),
        validate_shape=False,
        trainable=False)
    self.input_op = tf.assign(self.x_train, self.x_in, validate_shape=False)
    self.input_w_op = tf.assign(
        self.weights_train, self.weights, validate_shape=False)

    self.input_std = tf.get_variable(
        "data_standard_deviation",
        initializer=tf.ones([1, self.n_out], dtype=tf.float64),
        dtype=tf.float64,
        trainable=False)
    self.input_mean = tf.get_variable(
        "data_mean",
        initializer=tf.zeros([1, self.n_out], dtype=tf.float64),
        dtype=tf.float64,
        trainable=True)

    # GP Hyperparameters
    self.noise = tf.get_variable(
        "noise", initializer=tf.cast(0.0, dtype=tf.float64))
    self.amplitude = tf.get_variable(
        "amplitude", initializer=tf.cast(1.0, dtype=tf.float64))
    self.amplitude_linear = tf.get_variable(
        "linear_amplitude", initializer=tf.cast(1.0, dtype=tf.float64))
    self.length_scales = tf.get_variable(
        "length_scales", initializer=tf.zeros([1, self.n_in], dtype=tf.float64))
    self.length_scales_lin = tf.get_variable(
        "length_scales_linear",
        initializer=tf.zeros([1, self.n_in], dtype=tf.float64))

    # Latent embeddings of the different outputs for task covariance
    self.task_vectors = tf.get_variable(
        "latent_task_vectors",
        initializer=tf.random_normal(
            [self.n_out, self.task_latent_dim], dtype=tf.float64))

    # Normalize outputs across each dimension
    # Since we have different numbers of observations across each task, we
    # normalize by their respective counts.
    index_counts = self.atleast_2d(tf.reduce_sum(self.weights, axis=0),
                                   self.n_out)
    index_counts = tf.where(index_counts > 0, index_counts,
                            tf.ones(tf.shape(index_counts), dtype=tf.float64))
    self.mean_op = tf.assign(self.input_mean,
                             tf.reduce_sum(self.y, axis=0) / index_counts)
    self.var_op = tf.assign(
        self.input_std, tf.sqrt(1e-4 + tf.reduce_sum(tf.square(
            self.y - tf.reduce_sum(self.y, axis=0) / index_counts), axis=0)
                                / index_counts))

    with tf.control_dependencies([self.var_op]):
      y_normed = self.atleast_2d(
          (self.y - self.input_mean) / self.input_std, self.n_out)
      y_normed = self.atleast_2d(tf.boolean_mask(y_normed, self.weights > 0), 1)
    self.out_op = tf.assign(self.y_train, y_normed, validate_shape=False)

    # Observation noise
    alpha = tf.nn.softplus(self.noise) + 1e-6

    # Covariance
    with tf.control_dependencies([self.input_op, self.input_w_op, self.out_op]):
      self.self_cov = (self.cov(self.x_in, self.x_in) *
                       self.task_cov(self.weights, self.weights) +
                       tf.eye(tf.shape(self.x_in)[0], dtype=tf.float64) * alpha)

    self.chol = tf.cholesky(self.self_cov)
    self.kinv = tf.cholesky_solve(self.chol, tf.eye(tf.shape(self.x_in)[0],
                                                    dtype=tf.float64))

    self.input_inv = tf.Variable(
        tf.eye(self.hparams.batch_size, dtype=tf.float64),
        validate_shape=False,
        trainable=False)
    self.input_cov_op = tf.assign(self.input_inv, self.kinv,
                                  validate_shape=False)

    # Log determinant by taking the singular values along the diagonal
    # of self.chol
    with tf.control_dependencies([self.input_cov_op]):
      logdet = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(self.chol) + 1e-16))

    # Log Marginal likelihood
    self.marginal_ll = -tf.reduce_sum(-0.5 * tf.matmul(
        tf.transpose(y_normed), tf.matmul(self.kinv, y_normed)) - 0.5 * logdet -
                                      0.5 * self.n * np.log(2 * np.pi))

    zero = tf.cast(0., dtype=tf.float64)
    one = tf.cast(1., dtype=tf.float64)
    standard_normal = tfd.Normal(loc=zero, scale=one)

    # Loss is marginal likelihood and priors
    self.loss = tf.reduce_sum(
        self.marginal_ll -
        (standard_normal.log_prob(self.amplitude) +
         standard_normal.log_prob(tf.exp(self.noise)) +
         standard_normal.log_prob(self.amplitude_linear) +
         tfd.Normal(loc=zero, scale=one * 10.).log_prob(
             self.task_vectors))
    )

    # Optimizer for hyperparameters
    optimizer = tf.train.AdamOptimizer(learning_rate=self.hparams.lr)
    vars_to_optimize = [
        self.amplitude, self.length_scales, self.length_scales_lin,
        self.amplitude_linear, self.noise, self.input_mean
    ]

    if self.learn_embeddings:
      vars_to_optimize.append(self.task_vectors)
    grads = optimizer.compute_gradients(self.loss, vars_to_optimize)
    self.train_op = optimizer.apply_gradients(grads,
                                              global_step=self.global_step)

    # Predictions for test data
    self.y_mean, self.y_pred = self.posterior_mean_and_sample(self.x)

    # create tensorboard metrics
    self.create_summaries()
    self.summary_writer = tf.summary.FileWriter("{}/graph_{}".format(
        FLAGS.logdir, self.name), self.sess.graph)
    self.check = tf.add_check_numerics_ops()

  def posterior_mean_and_sample(self, candidates):
    """Draw samples for test predictions.

    Given a Tensor of 'candidates' inputs, returns samples from the posterior
    and the posterior mean prediction for those inputs.

    Args:
      candidates: A (num-examples x num-dims) Tensor containing the inputs for
      which to return predictions.
    Returns:
      y_mean: The posterior mean prediction given these inputs
      y_sample: A sample from the posterior of the outputs given these inputs
    """
    # Cross-covariance for test predictions
    w = tf.identity(self.weights_train)
    inds = tf.squeeze(
        tf.reshape(
            tf.tile(
                tf.reshape(tf.range(self.n_out), (self.n_out, 1)),
                (1, tf.shape(candidates)[0])), (-1, 1)))

    cross_cov = self.cov(tf.tile(candidates, [self.n_out, 1]), self.x_train)
    cross_task_cov = self.task_cov(tf.one_hot(inds, self.n_out), w)
    cross_cov *= cross_task_cov

    # Test mean prediction
    y_mean = tf.matmul(cross_cov, tf.matmul(self.input_inv, self.y_train))

    # Test sample predictions
    # Note this can be done much more efficiently using Kronecker products
    # if all tasks are fully observed (which we won't assume)
    test_cov = (
        self.cov(tf.tile(candidates, [self.n_out, 1]),
                 tf.tile(candidates, [self.n_out, 1])) *
        self.task_cov(tf.one_hot(inds, self.n_out),
                      tf.one_hot(inds, self.n_out)) -
        tf.matmul(cross_cov,
                  tf.matmul(self.input_inv,
                            tf.transpose(cross_cov))))

    # Get the matrix square root through an SVD for drawing samples
    # This seems more numerically stable than the Cholesky
    s, _, v = tf.svd(test_cov, full_matrices=True)
    test_sqrt = tf.matmul(v, tf.matmul(tf.diag(s), tf.transpose(v)))

    y_sample = (
        tf.matmul(
            test_sqrt,
            tf.random_normal([tf.shape(test_sqrt)[0], 1], dtype=tf.float64)) +
        y_mean)

    y_sample = (
        tf.transpose(tf.reshape(y_sample,
                                (self.n_out, -1))) * self.input_std +
        self.input_mean)

    return y_mean, y_sample

  def create_summaries(self):
    with self.graph.as_default():
      tf.summary.scalar("loss", self.loss)
      tf.summary.scalar("log_noise", self.noise)
      tf.summary.scalar("log_amp", self.amplitude)
      tf.summary.scalar("log_amp_lin", self.amplitude_linear)
      tf.summary.histogram("length_scales", self.length_scales)
      tf.summary.histogram("length_scales_lin", self.length_scales_lin)
      self.summary_op = tf.summary.merge_all()

  def train(self, data, num_steps):
    """Trains the GP for num_steps, using the data in 'data'.

    Args:
      data: ContextualDataset object that provides the data.
      num_steps: Number of minibatches to train the network for.
    """

    logging.info("Training %s for %d steps...", self.name, num_steps)
    for step in range(num_steps):
      numpts = min(data.num_points(None), self.max_num_points)
      if numpts >= self.max_num_points and self.keep_fixed_after_max_obs:
        x = data.contexts[:numpts, :]
        y = data.rewards[:numpts, :]
        weights = np.zeros((x.shape[0], self.n_out))
        for i, val in enumerate(data.actions[:numpts]):
          weights[i, val] = 1.0
      else:
        x, y, weights = data.get_batch_with_weights(numpts)

      ops = [
          self.global_step, self.summary_op, self.loss, self.noise,
          self.amplitude, self.amplitude_linear, self.length_scales,
          self.length_scales_lin, self.input_cov_op, self.input_op, self.var_op,
          self.input_w_op, self.out_op, self.train_op
      ]

      res = self.sess.run(ops,
                          feed_dict={self.x: x,
                                     self.x_in: x,
                                     self.y: y,
                                     self.weights: weights,
                                     self.n: numpts,
                                    })

      if step % self._freq_summary == 0:
        if self._show_training:
          logging.info("step: %d, loss: %g noise: %f amp: %f amp_lin: %f",
                       step, res[2], res[3], res[4], res[5])
      summary = res[1]
      global_step = res[0]
      self.summary_writer.add_summary(summary, global_step=global_step)
