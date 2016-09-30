# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Defines hyperparameters for training the def model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import tensorflow as tf

from ops import tf_lib


flags = tf.flags

# Dataset options
flags.DEFINE_enum('dataset', 'MNIST', ['MNIST', 'alternating'],
                  'Dataset to use. mnist or synthetic bernoulli data')
flags.DEFINE_integer('n_timesteps', 28, 'Number of timesteps per example')
flags.DEFINE_integer('timestep_dim', 28, 'Dimensionality of each timestep')
flags.DEFINE_integer('n_examples', 50000, 'Number of examples to use from the '
                     'dataset.')

# Model options
flags.DEFINE_integer('z_dim', 2, 'Latent dimensionality')
flags.DEFINE_float('p_z_sigma', 1., 'Prior variance for latent variables')
flags.DEFINE_float('p_w_mu_sigma', 1., 'Prior variance for weights for mean')
flags.DEFINE_float('p_w_sigma_sigma', 1., 'Prior variance for weights for '
                   'standard deviation')
flags.DEFINE_boolean('fixed_p_z_sigma', True, 'Whether to have the variance '
                     'depend recurrently across timesteps')

# Variational family options
flags.DEFINE_float('init_q_sigma_scale', 0.1, 'Factor by which to scale prior'
                   ' variances to use as initialization for variational stddev')
flags.DEFINE_boolean('fixed_q_z_sigma', False, 'Whether to learn variational '
                     'variance parameters for latents')
flags.DEFINE_boolean('fixed_q_w_mu_sigma', False, 'Whether to learn variational'
                     'variance parameters for weights for mean')
flags.DEFINE_boolean('fixed_q_w_sigma_sigma', False, 'Whether to learn '
                     'variance parameters for weights for variance')
flags.DEFINE_boolean('fixed_q_w_0_sigma', False, 'Whether to learn '
                     'variance parameters for weights for observations')

# Training options
flags.DEFINE_enum('optimizer', 'Adam', ['Adam', 'RMSProp', 'SGD', 'Adagrad'],
                  'Optimizer to use')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
flags.DEFINE_float('momentum', 0., 'Momentum for optimizer')
flags.DEFINE_integer('batch_size', 10, 'Batch size')

FLAGS = tf.flags.FLAGS


def h_params():
  """Returns hyperparameters defaulting to the corresponding flag values."""
  try:
    hparams = tf.HParams
  except AttributeError:
    hparams = tf_lib.HParams
  return hparams(
      dataset=FLAGS.dataset,
      z_dim=FLAGS.z_dim,
      timestep_dim=FLAGS.timestep_dim,
      n_timesteps=FLAGS.n_timesteps,
      batch_size=FLAGS.batch_size,
      learning_rate=FLAGS.learning_rate,
      n_examples=FLAGS.n_examples,
      momentum=FLAGS.momentum,
      p_z_sigma=FLAGS.p_z_sigma,
      p_w_mu_sigma=FLAGS.p_w_mu_sigma,
      p_w_sigma_sigma=FLAGS.p_w_sigma_sigma,
      init_q_sigma_scale=FLAGS.init_q_sigma_scale,
      fixed_p_z_sigma=FLAGS.fixed_p_z_sigma,
      fixed_q_z_sigma=FLAGS.fixed_q_z_sigma,
      fixed_q_w_mu_sigma=FLAGS.fixed_q_w_mu_sigma,
      fixed_q_w_sigma_sigma=FLAGS.fixed_q_w_sigma_sigma,
      fixed_q_w_0_sigma=FLAGS.fixed_q_w_0_sigma,
      dtype='float32')
