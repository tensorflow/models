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

"""VRNN classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import functools

import sonnet as snt
import tensorflow as tf

from fivo.models import base


VRNNState = namedtuple("VRNNState", "rnn_state latent_encoded")


class VRNN(object):
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
  model's prior over z_t (also called the transition distribution) is
  distributed as Normal(mu_t, diag(sigma_t^2)) where mu_t and sigma_t are the
  mean and standard deviation output from a fully connected network that accepts
  the rnn hidden state h_t as input.

  The emission distribution p(x_t|z_t, h_t) is conditioned on the latent state
  z_t as well as the current RNN hidden state h_t via a fully connected network.

  To increase the modeling power of the VRNN, two additional networks are
  used to extract features from the data and the latent state. Those networks
  are called data_encoder and latent_encoder respectively.

  For an example of how to call the VRNN's methods see sample_step.

  There are a few differences between this exposition and the paper.
  First, the indexing scheme for h_t is different than the paper's -- what the
  paper calls h_t we call h_{t+1}. This is the same notation used by Fraccaro
  et al. to describe the VRNN in the paper linked above. Also, the VRNN paper
  uses VAE terminology to refer to the different internal networks, so it
  refers to the emission distribution as the decoder. This implementation also
  renames the functions phi_x and phi_z in the paper to data_encoder and
  latent_encoder.
  """

  def __init__(self,
               rnn_cell,
               data_encoder,
               latent_encoder,
               transition,
               emission,
               random_seed=None):
    """Create a VRNN.

    Args:
      rnn_cell: A subclass of tf.nn.rnn_cell.RNNCell that will form the
        deterministic backbone of the VRNN. The inputs to the RNN will be the
        encoded latent state of the previous timestep with shape
        [batch_size, encoded_latent_size] as well as the encoded input of the
        current timestep, a Tensor of shape [batch_size, encoded_data_size].
      data_encoder: A callable that accepts a batch of data x_t and
        'encodes' it, e.g. runs it through a fully connected network. Must
        accept as argument the inputs x_t, a Tensor of the shape
        [batch_size, data_size] and return a Tensor of shape
        [batch_size, encoded_data_size]. This callable will be called multiple
        times in the VRNN cell so if scoping is not handled correctly then
        multiple copies of the variables in this network could be made. It is
        recommended to use a snt.nets.MLP module, which takes care of this for
        you.
      latent_encoder: A callable that accepts a latent state z_t and
        'encodes' it, e.g. runs it through a fully connected network. Must
        accept as argument a Tensor of shape [batch_size, latent_size] and
        return a Tensor of shape [batch_size, encoded_latent_size].
        This callable must also have the property 'output_size' defined,
        returning encoded_latent_size.
      transition: A callable that implements the transition distribution
        p(z_t|h_t). Must accept as argument the previous RNN hidden state and
        return a tf.distributions.Normal distribution conditioned on the input.
      emission: A callable that implements the emission distribution
        p(x_t|z_t, h_t). Must accept as arguments the encoded latent state
        and the RNN hidden state and return a subclass of
        tf.distributions.Distribution that can be used to evaluate the logprob
        of the targets.
      random_seed: The seed for the random ops. Sets the seed for sample_step.
    """
    self.random_seed = random_seed
    self.rnn_cell = rnn_cell
    self.data_encoder = data_encoder
    self.latent_encoder = latent_encoder
    self.encoded_z_size = latent_encoder.output_size
    self.state_size = (self.rnn_cell.state_size)
    self._transition = transition
    self._emission = emission

  def zero_state(self, batch_size, dtype):
    """The initial state of the VRNN.

    Contains the initial state of the RNN and the inital encoded latent.

    Args:
      batch_size: The batch size.
      dtype: The data type of the VRNN.
    Returns:
      zero_state: The initial state of the VRNN.
    """
    return VRNNState(
        rnn_state=self.rnn_cell.zero_state(batch_size, dtype),
        latent_encoded=tf.zeros(
            [batch_size, self.latent_encoder.output_size], dtype=dtype))

  def run_rnn(self, prev_rnn_state, prev_latent_encoded, inputs):
    """Runs the deterministic RNN for one step.

    Args:
      prev_rnn_state: The state of the RNN from the previous timestep.
      prev_latent_encoded: Float Tensor of shape
        [batch_size, encoded_latent_size], the previous latent state z_{t-1}
        run through latent_encoder.
      inputs: A Tensor of shape [batch_size, data_size], the current inputs to
        the model. Most often this is x_{t-1}, the previous token in the
        observation sequence.
    Returns:
      rnn_out: The output of the RNN.
      rnn_state: The new state of the RNN.
    """
    inputs_encoded = self.data_encoder(tf.to_float(inputs))
    rnn_inputs = tf.concat([inputs_encoded, prev_latent_encoded], axis=1)
    rnn_out, rnn_state = self.rnn_cell(rnn_inputs, prev_rnn_state)
    return rnn_out, rnn_state

  def transition(self, rnn_out):
    """Computes the transition distribution p(z_t|h_t).

    Note that p(z_t | h_t) = p(z_t| z_{1:t-1}, x_{1:t-1})

    Args:
      rnn_out: The output of the rnn for the current timestep.
    Returns:
      p(z_t | h_t): A normal distribution with event shape
        [batch_size, latent_size].
    """
    return self._transition(rnn_out)

  def emission(self, latent, rnn_out):
    """Computes the emission distribution p(x_t | z_t, h_t).

    Note that p(x_t | z_t, h_t) = p(x_t | z_{1:t}, x_{1:t-1}).

    Args:
      latent: The stochastic latent state z_t.
      rnn_out: The output of the rnn for the current timestep.
    Returns:
      p(x_t | z_t, h_t): A distribution with event shape
        [batch_size, data_size].
      latent_encoded: The latent state encoded with latent_encoder. Should be
        passed to run_rnn on the next timestep.
    """
    latent_encoded = self.latent_encoder(latent)
    return self._emission(latent_encoded, rnn_out), latent_encoded

  def sample_step(self, prev_state, inputs, unused_t):
    """Samples one output from the model.

    Args:
      prev_state: The previous state of the model, a VRNNState containing the
        previous rnn state and the previous encoded latent.
      inputs: A Tensor of shape [batch_size, data_size], the current inputs to
        the model. Most often this is x_{t-1}, the previous token in the
        observation sequence.
      unused_t: The current timestep. Not used currently.
    Returns:
      new_state: The next state of the model, a VRNNState.
      xt: A float Tensor of shape [batch_size, data_size], an output sampled
        from the emission distribution.
    """
    rnn_out, rnn_state = self.run_rnn(prev_state.rnn_state,
                                      prev_state.latent_encoded,
                                      inputs)
    p_zt = self.transition(rnn_out)
    zt = p_zt.sample(seed=self.random_seed)
    p_xt_given_zt, latent_encoded = self.emission(zt, rnn_out)
    xt = p_xt_given_zt.sample(seed=self.random_seed)
    new_state = VRNNState(rnn_state=rnn_state, latent_encoded=latent_encoded)
    return new_state, tf.to_float(xt)

# pylint: disable=invalid-name
# pylint thinks this is a top-level constant.
TrainableVRNNState = namedtuple("TrainableVRNNState",
                                VRNNState._fields + ("rnn_out",))
# pylint: enable=g-invalid-name


class TrainableVRNN(VRNN, base.ELBOTrainableSequenceModel):
  """A VRNN subclass with proposals and methods for training and evaluation.

  This class adds proposals used for training with importance-sampling based
  methods such as the ELBO. The model can be configured to propose from one
  of three proposals: a learned filtering proposal, a learned smoothing
  proposal, or the prior (i.e. the transition distribution).

  As described in the VRNN paper, the learned filtering proposal is
  parameterized by a fully connected neural network that accepts as input the
  current target x_t and the current rnn output h_t. The learned smoothing
  proposal is also given the hidden state of an RNN run in reverse over the
  inputs, so as to incorporate information about future observations. This
  smoothing proposal is not described in the VRNN paper.

  All learned proposals use the 'res_q' parameterization, meaning that instead
  of directly producing the mean of z_t, the proposal network predicts the
  'residual' from the prior's mean. This is explored more in section 3.3 of
  https://arxiv.org/pdf/1605.07571.pdf.

  During training, the latent state z_t is sampled from the proposal and the
  reparameterization trick is used to provide low-variance gradients.

  Note that the VRNN paper uses VAE terminology to refer to the different
  internal networks, so the proposal is referred to as the encoder.
  """

  def __init__(self,
               rnn_cell,
               data_encoder,
               latent_encoder,
               transition,
               emission,
               proposal_type,
               proposal=None,
               rev_rnn_cell=None,
               tilt=None,
               random_seed=None):
    """Create a trainable RNN.

    Args:
      rnn_cell: A subclass of tf.nn.rnn_cell.RNNCell that will form the
        deterministic backbone of the VRNN. The inputs to the RNN will be the
        encoded latent state of the previous timestep with shape
        [batch_size, encoded_latent_size] as well as the encoded input of the
        current timestep, a Tensor of shape [batch_size, encoded_data_size].
      data_encoder: A callable that accepts a batch of data x_t and
        'encodes' it, e.g. runs it through a fully connected network. Must
        accept as argument the inputs x_t, a Tensor of the shape
        [batch_size, data_size] and return a Tensor of shape
        [batch_size, encoded_data_size]. This callable will be called multiple
        times in the VRNN cell so if scoping is not handled correctly then
        multiple copies of the variables in this network could be made. It is
        recommended to use a snt.nets.MLP module, which takes care of this for
        you.
      latent_encoder: A callable that accepts a latent state z_t and
        'encodes' it, e.g. runs it through a fully connected network. Must
        accept as argument a Tensor of shape [batch_size, latent_size] and
        return a Tensor of shape [batch_size, encoded_latent_size].
        This callable must also have the property 'output_size' defined,
        returning encoded_latent_size.
      transition: A callable that implements the transition distribution
        p(z_t|h_t). Must accept as argument the previous RNN hidden state and
        return a tf.distributions.Normal distribution conditioned on the input.
      emission: A callable that implements the emission distribution
        p(x_t|z_t, h_t). Must accept as arguments the encoded latent state
        and the RNN hidden state and return a subclass of
        tf.distributions.Distribution that can be used to evaluate the logprob
        of the targets.
      proposal_type: A string indicating the type of proposal to use. Can
        be either "filtering", "smoothing", or "prior". When proposal_type is
        "filtering" or "smoothing", proposal must be provided. When
        proposal_type is "smoothing", rev_rnn_cell must also be provided.
      proposal: A callable that implements the proposal q(z_t| h_t, x_{1:T}).
        If proposal_type is "filtering" then proposal must accept as arguments
        the current rnn output, the encoded target of the current timestep,
        and the mean of the prior. If proposal_type is "smoothing" then
        in addition to the current rnn output and the mean of the prior
        proposal must accept as arguments the output of the reverse rnn.
        proposal should return a tf.distributions.Normal distribution
        conditioned on its inputs. If proposal_type is "prior" this argument is
        ignored.
      rev_rnn_cell: A subclass of tf.nn.rnn_cell.RNNCell that will aggregate
        observation statistics in the reverse direction. The inputs to the RNN
        will be the encoded reverse input of the current timestep, a Tensor of
        shape [batch_size, encoded_data_size].
      tilt: A callable that implements the log of a positive tilting function
        (ideally approximating log p(x_{t+1}|z_t, h_t). Must accept as arguments
        the encoded latent state and the RNN hidden state and return a subclass
        of tf.distributions.Distribution that can be used to evaluate the
        logprob of x_{t+1}. Optionally, None and then no tilt is used.
      random_seed: The seed for the random ops. Sets the seed for sample_step
        and __call__.
    """
    super(TrainableVRNN, self).__init__(
        rnn_cell, data_encoder, latent_encoder,
        transition, emission, random_seed=random_seed)
    self.rev_rnn_cell = rev_rnn_cell
    self._tilt = tilt
    assert proposal_type in ["filtering", "smoothing", "prior"]
    self._proposal = proposal
    self.proposal_type = proposal_type
    if proposal_type != "prior":
      assert proposal, "If not proposing from the prior, must provide proposal."
    if proposal_type == "smoothing":
      assert rev_rnn_cell, "Must provide rev_rnn_cell for smoothing proposal."

  def zero_state(self, batch_size, dtype):
    super_state = super(TrainableVRNN, self).zero_state(batch_size, dtype)
    return TrainableVRNNState(
        rnn_out=tf.zeros([batch_size, self.rnn_cell.output_size], dtype=dtype),
        **super_state._asdict())

  def set_observations(self, observations, seq_lengths):
    """Stores the model's observations.

    Stores the observations (inputs and targets) in TensorArrays and precomputes
    things for later like the reverse RNN output and encoded targets.

    Args:
      observations: The observations of the model, a tuple containing two
        Tensors of shape [max_seq_len, batch_size, data_size]. The Tensors
        should be the inputs and targets, respectively.
      seq_lengths: An int Tensor of shape [batch_size] containing the length
        of each sequence in observations.
    """
    inputs, targets = observations
    self.seq_lengths = seq_lengths
    self.max_seq_len = tf.reduce_max(seq_lengths)
    self.inputs_ta = base.ta_for_tensor(inputs, clear_after_read=False)
    self.targets_ta = base.ta_for_tensor(targets, clear_after_read=False)
    targets_encoded = base.encode_all(targets, self.data_encoder)
    self.targets_encoded_ta = base.ta_for_tensor(targets_encoded,
                                                 clear_after_read=False)
    if self.rev_rnn_cell:
      reverse_targets_encoded = tf.reverse_sequence(
          targets_encoded, seq_lengths, seq_axis=0, batch_axis=1)
      # Compute the reverse rnn over the targets.
      reverse_rnn_out, _ = tf.nn.dynamic_rnn(self.rev_rnn_cell,
                                             reverse_targets_encoded,
                                             time_major=True,
                                             dtype=tf.float32)
      reverse_rnn_out = tf.reverse_sequence(reverse_rnn_out, seq_lengths,
                                            seq_axis=0, batch_axis=1)
      self.reverse_rnn_ta = base.ta_for_tensor(reverse_rnn_out,
                                               clear_after_read=False)

  def _filtering_proposal(self, rnn_out, prior, t):
    """Computes the filtering proposal distribution."""
    return self._proposal(rnn_out,
                          self.targets_encoded_ta.read(t),
                          prior_mu=prior.mean())

  def _smoothing_proposal(self, rnn_out, prior, t):
    """Computes the smoothing proposal distribution."""
    return self._proposal(rnn_out,
                          smoothing_tensors=[self.reverse_rnn_ta.read(t)],
                          prior_mu=prior.mean())

  def proposal(self, rnn_out, prior, t):
    """Computes the proposal distribution specified by proposal_type.

    Args:
      rnn_out: The output of the rnn for the current timestep.
      prior: A tf.distributions.Normal distribution representing the prior
        over z_t, p(z_t | z_{1:t-1}, x_{1:t-1}). Used for 'res_q'.
      t: A scalar int Tensor, the current timestep.
    """
    if self.proposal_type == "filtering":
      return self._filtering_proposal(rnn_out, prior, t)
    elif self.proposal_type == "smoothing":
      return self._smoothing_proposal(rnn_out, prior, t)
    elif self.proposal_type == "prior":
      return self.transition(rnn_out)

  def tilt(self, rnn_out, latent_encoded, targets):
    r_func = self._tilt(rnn_out, latent_encoded)
    return tf.reduce_sum(r_func.log_prob(targets), axis=-1)

  def propose_and_weight(self, state, t):
    """Runs the model and computes importance weights for one timestep.

    Runs the model and computes importance weights, sampling from the proposal
    instead of the transition/prior.

    Args:
      state: The previous state of the model, a TrainableVRNNState containing
        the previous rnn state, the previous rnn outs, and the previous encoded
        latent.
      t: A scalar integer Tensor, the current timestep.
    Returns:
      weights: A float Tensor of shape [batch_size].
      new_state: The new state of the model.
    """
    inputs = self.inputs_ta.read(t)
    targets = self.targets_ta.read(t)
    rnn_out, next_rnn_state = self.run_rnn(state.rnn_state,
                                           state.latent_encoded,
                                           inputs)
    p_zt = self.transition(rnn_out)
    q_zt = self.proposal(rnn_out, p_zt, t)
    zt = q_zt.sample(seed=self.random_seed)
    p_xt_given_zt, latent_encoded = self.emission(zt, rnn_out)
    log_p_xt_given_zt = tf.reduce_sum(p_xt_given_zt.log_prob(targets), axis=-1)
    log_p_zt = tf.reduce_sum(p_zt.log_prob(zt), axis=-1)
    log_q_zt = tf.reduce_sum(q_zt.log_prob(zt), axis=-1)
    weights = log_p_zt + log_p_xt_given_zt - log_q_zt
    if self._tilt:
      prev_log_r = tf.cond(
          tf.greater(t, 0),
          lambda: self.tilt(state.rnn_out, state.latent_encoded, targets),
          lambda: 0.)  # On the first step, prev_log_r = 0.
      log_r = tf.cond(
          tf.less(t + 1, self.max_seq_len),
          lambda: self.tilt(rnn_out, latent_encoded, self.targets_ta.read(t+1)),
          lambda: 0.)
      # On the last step, log_r = 0.
      log_r *= tf.to_float(t < self.seq_lengths - 1)
      weights += log_r - prev_log_r
    new_state = TrainableVRNNState(rnn_state=next_rnn_state,
                                   rnn_out=rnn_out,
                                   latent_encoded=latent_encoded)
    return weights, new_state


_DEFAULT_INITIALIZERS = {"w": tf.contrib.layers.xavier_initializer(),
                         "b": tf.zeros_initializer()}


def create_vrnn(
    data_size,
    latent_size,
    emission_class,
    rnn_hidden_size=None,
    fcnet_hidden_sizes=None,
    encoded_data_size=None,
    encoded_latent_size=None,
    sigma_min=0.0,
    raw_sigma_bias=0.25,
    emission_bias_init=0.0,
    use_tilt=False,
    proposal_type="filtering",
    initializers=None,
    random_seed=None):
  """A factory method for creating VRNN cells.

  Args:
    data_size: The dimension of the vectors that make up the data sequences.
    latent_size: The size of the stochastic latent state of the VRNN.
    emission_class: The class of the emission distribution. Can be either
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
    emission_bias_init: A bias to added to the raw output of the fully
      connected network that parameterizes the emission distribution. Useful
      for initalizing the mean of the distribution to a sensible starting point
      such as the mean of the training data. Only used with Bernoulli generative
      distributions.
    use_tilt: If true, create a VRNN with a tilting function.
    proposal_type: The type of proposal to use. Can be "filtering", "smoothing",
      or "prior".
    initializers: The variable intitializers to use for the fully connected
      networks and RNN cell. Must be a dictionary mapping the keys 'w' and 'b'
      to the initializers for the weights and biases. Defaults to xavier for
      the weights and zeros for the biases when initializers is None.
    random_seed: A random seed for the VRNN resampling operations.
  Returns:
    model: A TrainableVRNN object.
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
  data_encoder = snt.nets.MLP(
      output_sizes=fcnet_hidden_sizes + [encoded_data_size],
      initializers=initializers,
      name="data_encoder")
  latent_encoder = snt.nets.MLP(
      output_sizes=fcnet_hidden_sizes + [encoded_latent_size],
      initializers=initializers,
      name="latent_encoder")
  transition = base.ConditionalNormalDistribution(
      size=latent_size,
      hidden_layer_sizes=fcnet_hidden_sizes,
      sigma_min=sigma_min,
      raw_sigma_bias=raw_sigma_bias,
      initializers=initializers,
      name="prior")
  # Construct the emission distribution.
  if emission_class == base.ConditionalBernoulliDistribution:
    # For Bernoulli distributed outputs, we initialize the bias so that the
    # network generates on average the mean from the training set.
    emission_dist = functools.partial(base.ConditionalBernoulliDistribution,
                                      bias_init=emission_bias_init)
  else:
    emission_dist = base.ConditionalNormalDistribution
  emission = emission_dist(
      size=data_size,
      hidden_layer_sizes=fcnet_hidden_sizes,
      initializers=initializers,
      name="generative")
  # Construct the proposal distribution.
  if proposal_type in ["filtering", "smoothing"]:
    proposal = base.NormalApproximatePosterior(
        size=latent_size,
        hidden_layer_sizes=fcnet_hidden_sizes,
        sigma_min=sigma_min,
        raw_sigma_bias=raw_sigma_bias,
        initializers=initializers,
        smoothing=(proposal_type == "smoothing"),
        name="approximate_posterior")
  else:
    proposal = None

  if use_tilt:
    tilt = emission_dist(
        size=data_size,
        hidden_layer_sizes=fcnet_hidden_sizes,
        initializers=initializers,
        name="tilt")
  else:
    tilt = None

  rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_hidden_size,
                                     initializer=initializers["w"])
  rev_rnn_cell = tf.nn.rnn_cell.LSTMCell(rnn_hidden_size,
                                         initializer=initializers["w"])
  return TrainableVRNN(
      rnn_cell, data_encoder, latent_encoder, transition,
      emission, proposal_type, proposal=proposal, rev_rnn_cell=rev_rnn_cell,
      tilt=tilt, random_seed=random_seed)
