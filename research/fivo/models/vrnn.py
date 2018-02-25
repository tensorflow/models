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

"""VRNN classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf


class VRNNCell(snt.AbstractModule):
  """Implementation of a Variational Recurrent Neural Network (VRNN).

  Introduced in "A Recurrent Latent Variable Model for Sequential data"
  by Chung et al. https://arxiv.org/pdf/1506.02216.pdf.

  The VRNN is a sequence model similar to an RNN that uses stochastic latent
  variables to improve its representational power. It can be thought of as a
  sequential analogue to the variational auto-encoder (VAE).

  The VRNN has a deterministic RNN as its backbone, represented by the
  sequence of RNN hidden states h_t. At each timestep, the RNN hidden state h_t
  is conditioned on the previous sequence element, x_{t-1}, as well as the
  latent state from the previous timestep, z_{t-1}.

  In this implementation of the VRNN the latent state z_t is Gaussian. The
  model's prior over z_t is distributed as Normal(mu_t, diag(sigma_t^2)) where
  mu_t and sigma_t are the mean and standard deviation output from a fully
  connected network that accepts the rnn hidden state h_t as input.

  The approximate posterior (also known as q or the encoder in the VAE
  framework) is similar to the prior except that it is conditioned on the
  current target, x_t, as well as h_t via a fully connected network.

  This implementation uses the 'res_q' parameterization of the approximate
  posterior, meaning that instead of directly predicting the mean of z_t, the
  approximate posterior predicts the 'residual' from the prior's mean. This is
  explored more in section 3.3 of https://arxiv.org/pdf/1605.07571.pdf.

  During training, the latent state z_t is sampled from the approximate
  posterior and the reparameterization trick is used to provide low-variance
  gradients.

  The generative distribution p(x_t|z_t, h_t) is conditioned on the latent state
  z_t as well as the current RNN hidden state h_t via a fully connected network.

  To increase the modeling power of the VRNN, two additional networks are
  used to extract features from the data and the latent state. Those networks
  are called data_feat_extractor and latent_feat_extractor respectively.

  There are a few differences between this exposition and the paper.
  First, the indexing scheme for h_t is different than the paper's -- what the
  paper calls h_t we call h_{t+1}. This is the same notation used by Fraccaro
  et al. to describe the VRNN in the paper linked above. Also, the VRNN paper
  uses VAE terminology to refer to the different internal networks, so it
  refers to the approximate posterior as the encoder and the generative
  distribution as the decoder. This implementation also renamed the functions
  phi_x and phi_z in the paper to data_feat_extractor and latent_feat_extractor.
  """

  def __init__(self,
               rnn_cell,
               data_feat_extractor,
               latent_feat_extractor,
               prior,
               approx_posterior,
               generative,
               random_seed=None,
               name="vrnn"):
    """Creates a VRNN cell.

    Args:
      rnn_cell: A subclass of tf.nn.rnn_cell.RNNCell that will form the
        deterministic backbone of the VRNN. The inputs to the RNN will be the
        encoded latent state of the previous timestep with shape
        [batch_size, encoded_latent_size] as well as the encoded input of the
        current timestep, a Tensor of shape [batch_size, encoded_data_size].
      data_feat_extractor: A callable that accepts a batch of data x_t and
        'encodes' it, e.g. runs it through a fully connected network. Must
        accept as argument the inputs x_t, a Tensor of the shape
        [batch_size, data_size] and return a Tensor of shape
        [batch_size, encoded_data_size]. This callable will be called multiple
        times in the VRNN cell so if scoping is not handled correctly then
        multiple copies of the variables in this network could be made. It is
        recommended to use a snt.nets.MLP module, which takes care of this for
        you.
      latent_feat_extractor: A callable that accepts a latent state z_t and
        'encodes' it, e.g. runs it through a fully connected network. Must
        accept as argument a Tensor of shape [batch_size, latent_size] and
        return a Tensor of shape [batch_size, encoded_latent_size].
        This callable must also have the property 'output_size' defined,
        returning encoded_latent_size.
      prior: A callable that implements the prior p(z_t|h_t). Must accept as
        argument the previous RNN hidden state and return a
        tf.contrib.distributions.Normal distribution conditioned on the input.
      approx_posterior: A callable that implements the approximate posterior
        q(z_t|h_t,x_t). Must accept as arguments the encoded target of the
        current timestep and the previous RNN hidden state. Must return
        a tf.contrib.distributions.Normal distribution conditioned on the
        inputs.
      generative: A callable that implements the generative distribution
        p(x_t|z_t, h_t). Must accept as arguments the encoded latent state
        and the RNN hidden state and return a subclass of
        tf.contrib.distributions.Distribution that can be used to evaluate
        the logprob of the targets.
      random_seed: The seed for the random ops. Used mainly for testing.
      name: The name of this VRNN.
    """
    super(VRNNCell, self).__init__(name=name)
    self.rnn_cell = rnn_cell
    self.data_feat_extractor = data_feat_extractor
    self.latent_feat_extractor = latent_feat_extractor
    self.prior = prior
    self.approx_posterior = approx_posterior
    self.generative = generative
    self.random_seed = random_seed
    self.encoded_z_size = latent_feat_extractor.output_size
    self.state_size = (self.rnn_cell.state_size, self.encoded_z_size)

  def zero_state(self, batch_size, dtype):
    """The initial state of the VRNN.

    Contains the initial state of the RNN as well as a vector of zeros
    corresponding to z_0.
    Args:
      batch_size: The batch size.
      dtype: The data type of the VRNN.
    Returns:
      zero_state: The initial state of the VRNN.
    """
    return (self.rnn_cell.zero_state(batch_size, dtype),
            tf.zeros([batch_size, self.encoded_z_size], dtype=dtype))

  def _build(self, observations, state, mask):
    """Computes one timestep of the VRNN.

    Args:
      observations: The observations at the current timestep, a tuple
        containing the model inputs and targets as Tensors of shape
        [batch_size, data_size].
      state: The current state of the VRNN
      mask: Tensor of shape [batch_size], 1.0 if the current timestep is active
        active, 0.0 if it is not active.

    Returns:
      log_q_z: The logprob of the latent state according to the approximate
        posterior.
      log_p_z: The logprob of the latent state according to the prior.
      log_p_x_given_z: The conditional log-likelihood, i.e. logprob of the
        observation according to the generative distribution.
      kl: The analytic kl divergence from q(z) to p(z).
      state: The new state of the VRNN.
    """
    inputs, targets = observations
    rnn_state, prev_latent_encoded = state
    # Encode the data.
    inputs_encoded = self.data_feat_extractor(inputs)
    targets_encoded = self.data_feat_extractor(targets)
    # Run the RNN cell.
    rnn_inputs = tf.concat([inputs_encoded, prev_latent_encoded], axis=1)
    rnn_out, new_rnn_state = self.rnn_cell(rnn_inputs, rnn_state)
    # Create the prior and approximate posterior distributions.
    latent_dist_prior = self.prior(rnn_out)
    latent_dist_q = self.approx_posterior(rnn_out, targets_encoded,
                                          prior_mu=latent_dist_prior.loc)
    # Sample the new latent state z and encode it.
    latent_state = latent_dist_q.sample(seed=self.random_seed)
    latent_encoded = self.latent_feat_extractor(latent_state)
    # Calculate probabilities of the latent state according to the prior p
    # and approximate posterior q.
    log_q_z = tf.reduce_sum(latent_dist_q.log_prob(latent_state), axis=-1)
    log_p_z = tf.reduce_sum(latent_dist_prior.log_prob(latent_state), axis=-1)
    analytic_kl = tf.reduce_sum(
        tf.contrib.distributions.kl_divergence(
            latent_dist_q, latent_dist_prior),
        axis=-1)
    # Create the generative dist. and calculate the logprob of the targets.
    generative_dist = self.generative(latent_encoded, rnn_out)
    log_p_x_given_z = tf.reduce_sum(generative_dist.log_prob(targets), axis=-1)
    return (log_q_z, log_p_z, log_p_x_given_z, analytic_kl,
            (new_rnn_state, latent_encoded))

_DEFAULT_INITIALIZERS = {"w": tf.contrib.layers.xavier_initializer(),
                         "b": tf.zeros_initializer()}


def create_vrnn(
    data_size,
    latent_size,
    generative_class,
    rnn_hidden_size=None,
    fcnet_hidden_sizes=None,
    encoded_data_size=None,
    encoded_latent_size=None,
    sigma_min=0.0,
    raw_sigma_bias=0.25,
    generative_bias_init=0.0,
    initializers=None,
    random_seed=None):
  """A factory method for creating VRNN cells.

  Args:
    data_size: The dimension of the vectors that make up the data sequences.
    latent_size: The size of the stochastic latent state of the VRNN.
    generative_class: The class of the generative distribution. Can be either
      ConditionalNormalDistribution or ConditionalBernoulliDistribution.
    rnn_hidden_size: The hidden state dimension of the RNN that forms the
      deterministic part of this VRNN. If None, then it defaults
      to latent_size.
    fcnet_hidden_sizes: A list of python integers, the size of the hidden
      layers of the fully connected networks that parameterize the conditional
      distributions of the VRNN. If None, then it defaults to one hidden
      layer of size latent_size.
    encoded_data_size: The size of the output of the data encoding network. If
      None, defaults to latent_size.
    encoded_latent_size: The size of the output of the latent state encoding
      network. If None, defaults to latent_size.
    sigma_min: The minimum value that the standard deviation of the
      distribution over the latent state can take.
    raw_sigma_bias: A scalar that is added to the raw standard deviation
      output from the neural networks that parameterize the prior and
      approximate posterior. Useful for preventing standard deviations close
      to zero.
    generative_bias_init: A bias to added to the raw output of the fully
      connected network that parameterizes the generative distribution. Useful
      for initalizing the mean of the distribution to a sensible starting point
      such as the mean of the training data. Only used with Bernoulli generative
      distributions.
    initializers: The variable intitializers to use for the fully connected
      networks and RNN cell. Must be a dictionary mapping the keys 'w' and 'b'
      to the initializers for the weights and biases. Defaults to xavier for
      the weights and zeros for the biases when initializers is None.
    random_seed: A random seed for the VRNN resampling operations.
  Returns:
    model: A VRNNCell object.
  """
  if rnn_hidden_size is None:
    rnn_hidden_size = latent_size
  if fcnet_hidden_sizes is None:
    fcnet_hidden_sizes = [latent_size]
  if encoded_data_size is None:
    encoded_data_size = latent_size
  if encoded_latent_size is None:
    encoded_latent_size = latent_size
  if initializers is None:
    initializers = _DEFAULT_INITIALIZERS
  data_feat_extractor = snt.nets.MLP(
      output_sizes=fcnet_hidden_sizes + [encoded_data_size],
      initializers=initializers,
      name="data_feat_extractor")
  latent_feat_extractor = snt.nets.MLP(
      output_sizes=fcnet_hidden_sizes + [encoded_latent_size],
      initializers=initializers,
      name="latent_feat_extractor")
  prior = ConditionalNormalDistribution(
      size=latent_size,
      hidden_layer_sizes=fcnet_hidden_sizes,
      sigma_min=sigma_min,
      raw_sigma_bias=raw_sigma_bias,
      initializers=initializers,
      name="prior")
  approx_posterior = NormalApproximatePosterior(
      size=latent_size,
      hidden_layer_sizes=fcnet_hidden_sizes,
      sigma_min=sigma_min,
      raw_sigma_bias=raw_sigma_bias,
      initializers=initializers,
      name="approximate_posterior")
  if generative_class == ConditionalBernoulliDistribution:
    generative = ConditionalBernoulliDistribution(
        size=data_size,
        hidden_layer_sizes=fcnet_hidden_sizes,
        initializers=initializers,
        bias_init=generative_bias_init,
        name="generative")
  else:
    generative = ConditionalNormalDistribution(
        size=data_size,
        hidden_layer_sizes=fcnet_hidden_sizes,
        initializers=initializers,
        name="generative")
  rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_hidden_size,
                                     initializer=initializers["w"])
  return VRNNCell(rnn_cell, data_feat_extractor, latent_feat_extractor,
                  prior, approx_posterior, generative, random_seed=random_seed)


class ConditionalNormalDistribution(object):
  """A Normal distribution conditioned on Tensor inputs via a fc network."""

  def __init__(self, size, hidden_layer_sizes, sigma_min=0.0,
               raw_sigma_bias=0.25, hidden_activation_fn=tf.nn.relu,
               initializers=None, name="conditional_normal_distribution"):
    """Creates a conditional Normal distribution.

    Args:
      size: The dimension of the random variable.
      hidden_layer_sizes: The sizes of the hidden layers of the fully connected
        network used to condition the distribution on the inputs.
      sigma_min: The minimum standard deviation allowed, a scalar.
      raw_sigma_bias: A scalar that is added to the raw standard deviation
        output from the fully connected network. Set to 0.25 by default to
        prevent standard deviations close to 0.
      hidden_activation_fn: The activation function to use on the hidden layers
        of the fully connected network.
      initializers: The variable intitializers to use for the fully connected
        network. The network is implemented using snt.nets.MLP so it must
        be a dictionary mapping the keys 'w' and 'b' to the initializers for
        the weights and biases. Defaults to xavier for the weights and zeros
        for the biases when initializers is None.
      name: The name of this distribution, used for sonnet scoping.
    """
    self.sigma_min = sigma_min
    self.raw_sigma_bias = raw_sigma_bias
    self.name = name
    if initializers is None:
      initializers = _DEFAULT_INITIALIZERS
    self.fcnet = snt.nets.MLP(
        output_sizes=hidden_layer_sizes + [2*size],
        activation=hidden_activation_fn,
        initializers=initializers,
        activate_final=False,
        use_bias=True,
        name=name + "_fcnet")

  def condition(self, tensor_list, **unused_kwargs):
    """Computes the parameters of a normal distribution based on the inputs."""
    inputs = tf.concat(tensor_list, axis=1)
    outs = self.fcnet(inputs)
    mu, sigma = tf.split(outs, 2, axis=1)
    sigma = tf.maximum(tf.nn.softplus(sigma + self.raw_sigma_bias),
                       self.sigma_min)
    return mu, sigma

  def __call__(self, *args, **kwargs):
    """Creates a normal distribution conditioned on the inputs."""
    mu, sigma = self.condition(args, **kwargs)
    return tf.contrib.distributions.Normal(loc=mu, scale=sigma)


class ConditionalBernoulliDistribution(object):
  """A Bernoulli distribution conditioned on Tensor inputs via a fc net."""

  def __init__(self, size, hidden_layer_sizes, hidden_activation_fn=tf.nn.relu,
               initializers=None, bias_init=0.0,
               name="conditional_bernoulli_distribution"):
    """Creates a conditional Bernoulli distribution.

    Args:
      size: The dimension of the random variable.
      hidden_layer_sizes: The sizes of the hidden layers of the fully connected
        network used to condition the distribution on the inputs.
      hidden_activation_fn: The activation function to use on the hidden layers
        of the fully connected network.
      initializers: The variable intiializers to use for the fully connected
        network. The network is implemented using snt.nets.MLP so it must
        be a dictionary mapping the keys 'w' and 'b' to the initializers for
        the weights and biases. Defaults to xavier for the weights and zeros
        for the biases when initializers is None.
      bias_init: A scalar or vector Tensor that is added to the output of the
        fully-connected network that parameterizes the mean of this
        distribution.
      name: The name of this distribution, used for sonnet scoping.
    """
    self.bias_init = bias_init
    if initializers is None:
      initializers = _DEFAULT_INITIALIZERS
    self.fcnet = snt.nets.MLP(
        output_sizes=hidden_layer_sizes + [size],
        activation=hidden_activation_fn,
        initializers=initializers,
        activate_final=False,
        use_bias=True,
        name=name + "_fcnet")

  def condition(self, tensor_list):
    """Computes the p parameter of the Bernoulli distribution."""
    inputs = tf.concat(tensor_list, axis=1)
    return self.fcnet(inputs) + self.bias_init

  def __call__(self, *args):
    p = self.condition(args)
    return tf.contrib.distributions.Bernoulli(logits=p)


class NormalApproximatePosterior(ConditionalNormalDistribution):
  """A Normally-distributed approx. posterior with res_q parameterization."""

  def condition(self, tensor_list, prior_mu):
    """Generates the mean and variance of the normal distribution.

    Args:
      tensor_list: The list of Tensors to condition on. Will be concatenated and
        fed through a fully connected network.
      prior_mu: The mean of the prior distribution associated with this
        approximate posterior. Will be added to the mean produced by
        this approximate posterior, in res_q fashion.
    Returns:
      mu: The mean of the approximate posterior.
      sigma: The standard deviation of the approximate posterior.
    """
    mu, sigma = super(NormalApproximatePosterior, self).condition(tensor_list)
    return mu + prior_mu, sigma
