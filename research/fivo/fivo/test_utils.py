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

"""Utilities for testing FIVO.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from fivo.models import base
from fivo.models import srnn
from fivo.models import vrnn


def create_vrnn(generative_class=base.ConditionalNormalDistribution,
                batch_size=2, data_size=3, rnn_hidden_size=4,
                latent_size=5, fcnet_hidden_size=7, encoded_data_size=9,
                encoded_latent_size=11, num_timesteps=7, data_lengths=(7, 4),
                use_tilt=False, random_seed=None):
  """Creates a VRNN and some dummy data to feed it for testing purposes.

  Args:
    generative_class: The class of the generative distribution.
    batch_size: The number of elements per batch.
    data_size: The dimension of the vectors that make up the data sequences.
    rnn_hidden_size: The hidden state dimension of the RNN that forms the
      deterministic part of this VRNN.
    latent_size: The size of the stochastic latent state of the VRNN.
    fcnet_hidden_size: The size of the hidden layer of the fully connected
      networks that parameterize the conditional probability distributions
      of the VRNN.
    encoded_data_size: The size of the output of the data encoding network.
    encoded_latent_size: The size of the output of the latent state encoding
      network.
    num_timesteps: The maximum number of timesteps in the data.
    data_lengths: A tuple of size batch_size that contains the desired lengths
      of each sequence in the dummy data.
    use_tilt: Use a tilting function.
    random_seed: A random seed to feed the VRNN, mainly useful for testing
      purposes.

  Returns:
    model: A VRNN object.
    inputs: A Tensor of shape [num_timesteps, batch_size, data_size], the inputs
      to the model, also known as the observations.
    targets: A Tensor of shape [num_timesteps, batch_size, data_size], the
      desired outputs of the model.
    lengths: A Tensor of shape [batch_size], the lengths of the sequences in the
      batch.
  """

  fcnet_hidden_sizes = [fcnet_hidden_size]
  initializers = {"w": tf.contrib.layers.xavier_initializer(seed=random_seed),
                  "b": tf.zeros_initializer()}
  model = vrnn.create_vrnn(
      data_size,
      latent_size,
      generative_class,
      rnn_hidden_size=rnn_hidden_size,
      fcnet_hidden_sizes=fcnet_hidden_sizes,
      encoded_data_size=encoded_data_size,
      encoded_latent_size=encoded_latent_size,
      use_tilt=use_tilt,
      initializers=initializers,
      random_seed=random_seed)
  inputs = tf.random_uniform([num_timesteps, batch_size, data_size],
                             seed=random_seed, dtype=tf.float32)
  targets = tf.random_uniform([num_timesteps, batch_size, data_size],
                              seed=random_seed, dtype=tf.float32)
  lengths = tf.constant(data_lengths, dtype=tf.int32)
  return model, inputs, targets, lengths


def create_srnn(generative_class=base.ConditionalNormalDistribution,
                batch_size=2, data_size=3, rnn_hidden_size=4,
                latent_size=5, fcnet_hidden_size=7, encoded_data_size=3,
                encoded_latent_size=2, num_timesteps=7, data_lengths=(7, 4),
                use_tilt=False, random_seed=None):
  """Creates a SRNN and some dummy data to feed it for testing purposes.

  Args:
    generative_class: The class of the generative distribution.
    batch_size: The number of elements per batch.
    data_size: The dimension of the vectors that make up the data sequences.
    rnn_hidden_size: The hidden state dimension of the RNN that forms the
      deterministic part of this SRNN.
    latent_size: The size of the stochastic latent state of the SRNN.
    fcnet_hidden_size: The size of the hidden layer of the fully connected
      networks that parameterize the conditional probability distributions
      of the SRNN.
    encoded_data_size: The size of the output of the data encoding network.
    encoded_latent_size: The size of the output of the latent state encoding
      network.
    num_timesteps: The maximum number of timesteps in the data.
    data_lengths: A tuple of size batch_size that contains the desired lengths
      of each sequence in the dummy data.
    use_tilt: Use a tilting function.
    random_seed: A random seed to feed the SRNN, mainly useful for testing
      purposes.

  Returns:
    model: A SRNN object.
    inputs: A Tensor of shape [num_timesteps, batch_size, data_size], the inputs
      to the model, also known as the observations.
    targets: A Tensor of shape [num_timesteps, batch_size, data_size], the
      desired outputs of the model.
    lengths: A Tensor of shape [batch_size], the lengths of the sequences in the
      batch.
  """

  fcnet_hidden_sizes = [fcnet_hidden_size]
  initializers = {"w": tf.contrib.layers.xavier_initializer(seed=random_seed),
                  "b": tf.zeros_initializer()}
  model = srnn.create_srnn(
      data_size,
      latent_size,
      generative_class,
      rnn_hidden_size=rnn_hidden_size,
      fcnet_hidden_sizes=fcnet_hidden_sizes,
      encoded_data_size=encoded_data_size,
      encoded_latent_size=encoded_latent_size,
      use_tilt=use_tilt,
      initializers=initializers,
      random_seed=random_seed)
  inputs = tf.random_uniform([num_timesteps, batch_size, data_size],
                             seed=random_seed, dtype=tf.float32)
  targets = tf.random_uniform([num_timesteps, batch_size, data_size],
                              seed=random_seed, dtype=tf.float32)
  lengths = tf.constant(data_lengths, dtype=tf.int32)
  return model, inputs, targets, lengths
