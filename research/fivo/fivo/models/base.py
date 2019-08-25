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

"""Reusable model classes for FIVO."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf

from fivo import nested_utils as nested

tfd = tf.contrib.distributions


class ELBOTrainableSequenceModel(object):
  """An abstract class for ELBO-trainable sequence models to extend.

  Because the ELBO, IWAE, and FIVO bounds all accept the same arguments,
  any model that is ELBO-trainable is also IWAE- and FIVO-trainable.
  """

  def zero_state(self, batch_size, dtype):
    """Returns the initial state of the model as a Tensor or tuple of Tensors.

    Args:
      batch_size: The batch size.
      dtype: The datatype to use for the state.
    """
    raise NotImplementedError("zero_state not yet implemented.")

  def set_observations(self, observations, seq_lengths):
    """Sets the observations for the model.

    This method provides the model with all observed variables including both
    inputs and targets. It will be called before running any computations with
    the model that require the observations, e.g. training the model or
    computing bounds, and should be used to run any necessary preprocessing
    steps.

    Args:
      observations: A potentially nested set of Tensors containing
        all observations for the model, both inputs and targets. Typically
        a set of Tensors with shape [max_seq_len, batch_size, data_size].
      seq_lengths: A [batch_size] Tensor of ints encoding the length of each
        sequence in the batch (sequences can be padded to a common length).
    """
    self.observations = observations
    self.max_seq_len = tf.reduce_max(seq_lengths)
    self.observations_ta = nested.tas_for_tensors(
        observations, self.max_seq_len, clear_after_read=False)
    self.seq_lengths = seq_lengths

  def propose_and_weight(self, state, t):
    """Propogates model state one timestep and computes log weights.

    This method accepts the current state of the model and computes the state
    for the next timestep as well as the incremental log weight of each
    element in the batch.

    Args:
      state: The current state of the model.
      t: A scalar integer Tensor representing the current timestep.
    Returns:
      next_state: The state of the model after one timestep.
      log_weights: A [batch_size] Tensor containing the incremental log weights.
    """
    raise NotImplementedError("propose_and_weight not yet implemented.")

DEFAULT_INITIALIZERS = {"w": tf.contrib.layers.xavier_initializer(),
                        "b": tf.zeros_initializer()}


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
    self.size = size
    if initializers is None:
      initializers = DEFAULT_INITIALIZERS
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
    self.size = size
    if initializers is None:
      initializers = DEFAULT_INITIALIZERS
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

  def __init__(self, size, hidden_layer_sizes, sigma_min=0.0,
               raw_sigma_bias=0.25, hidden_activation_fn=tf.nn.relu,
               initializers=None, smoothing=False,
               name="conditional_normal_distribution"):
    super(NormalApproximatePosterior, self).__init__(
        size, hidden_layer_sizes, sigma_min=sigma_min,
        raw_sigma_bias=raw_sigma_bias,
        hidden_activation_fn=hidden_activation_fn, initializers=initializers,
        name=name)
    self.smoothing = smoothing

  def condition(self, tensor_list, prior_mu, smoothing_tensors=None):
    """Generates the mean and variance of the normal distribution.

    Args:
      tensor_list: The list of Tensors to condition on. Will be concatenated and
        fed through a fully connected network.
      prior_mu: The mean of the prior distribution associated with this
        approximate posterior. Will be added to the mean produced by
        this approximate posterior, in res_q fashion.
      smoothing_tensors: A list of Tensors. If smoothing is True, these Tensors
        will be concatenated with the tensors in tensor_list.
    Returns:
      mu: The mean of the approximate posterior.
      sigma: The standard deviation of the approximate posterior.
    """
    if self.smoothing:
      tensor_list.extend(smoothing_tensors)
    mu, sigma = super(NormalApproximatePosterior, self).condition(tensor_list)
    return mu + prior_mu, sigma


class NonstationaryLinearDistribution(object):
  """A set of loc-scale distributions that are linear functions of inputs.

  This class defines a series of location-scale distributions such that
  the means are learnable linear functions of the inputs and the log variances
  are learnable constants. The functions and log variances are different across
  timesteps, allowing the distributions to be nonstationary.
  """

  def __init__(self,
               num_timesteps,
               inputs_per_timestep=None,
               outputs_per_timestep=None,
               initializers=None,
               variance_min=0.0,
               output_distribution=tfd.Normal,
               dtype=tf.float32):
    """Creates a NonstationaryLinearDistribution.

    Args:
      num_timesteps: The number of timesteps, i.e. the number of distributions.
      inputs_per_timestep: A list of python ints, the dimension of inputs to the
        linear function at each timestep. If not provided, the dimension at each
        timestep is assumed to be 1.
      outputs_per_timestep: A list of python ints, the dimension of the output
        distribution at each timestep. If not provided, the dimension at each
        timestep is assumed to be 1.
      initializers: A dictionary containing intializers for the variables. The
        initializer under the key 'w' is used for the weights in the linear
        function and the initializer under the key 'b' is used for the biases.
        Defaults to xavier initialization for the weights and zeros for the
        biases.
      variance_min: Python float, the minimum variance of each distribution.
      output_distribution: A locatin-scale subclass of tfd.Distribution that
        defines the output distribution, e.g. Normal.
      dtype: The dtype of the weights and biases.
    """
    if not initializers:
      initializers = DEFAULT_INITIALIZERS
    if not inputs_per_timestep:
      inputs_per_timestep = [1] * num_timesteps
    if not outputs_per_timestep:
      outputs_per_timestep = [1] * num_timesteps
    self.num_timesteps = num_timesteps
    self.variance_min = variance_min
    self.initializers = initializers
    self.dtype = dtype
    self.output_distribution = output_distribution

    def _get_variables_ta(shapes, name, initializer, trainable=True):
      """Creates a sequence of variables and stores them in a TensorArray."""
      # Infer shape if all shapes are equal.
      first_shape = shapes[0]
      infer_shape = all(shape == first_shape for shape in shapes)
      ta = tf.TensorArray(
          dtype=dtype, size=len(shapes), dynamic_size=False,
          clear_after_read=False, infer_shape=infer_shape)
      for t, shape in enumerate(shapes):
        var = tf.get_variable(
            name % t, shape=shape, initializer=initializer, trainable=trainable)
        ta = ta.write(t, var)
      return ta

    bias_shapes = [[num_outputs] for num_outputs in outputs_per_timestep]
    self.log_variances = _get_variables_ta(
        bias_shapes, "proposal_log_variance_%d", initializers["b"])
    self.mean_biases = _get_variables_ta(
        bias_shapes, "proposal_b_%d", initializers["b"])
    weight_shapes = zip(inputs_per_timestep, outputs_per_timestep)
    self.mean_weights = _get_variables_ta(
        weight_shapes, "proposal_w_%d", initializers["w"])
    self.shapes = tf.TensorArray(
        dtype=tf.int32, size=num_timesteps,
        dynamic_size=False, clear_after_read=False).unstack(weight_shapes)

  def __call__(self, t, inputs):
    """Computes the distribution at timestep t.

    Args:
      t: Scalar integer Tensor, the current timestep. Must be in
        [0, num_timesteps).
      inputs: The inputs to the linear function parameterizing the mean of
        the current distribution. A Tensor of shape [batch_size, num_inputs_t].
    Returns:
      A tfd.Distribution subclass representing the distribution at timestep t.
    """
    b = self.mean_biases.read(t)
    w = self.mean_weights.read(t)
    shape = self.shapes.read(t)
    w = tf.reshape(w, shape)
    b = tf.reshape(b, [shape[1], 1])
    log_variance = self.log_variances.read(t)
    scale = tf.sqrt(tf.maximum(tf.exp(log_variance), self.variance_min))
    loc = tf.matmul(w, inputs, transpose_a=True) + b
    return self.output_distribution(loc=loc, scale=scale)


def encode_all(inputs, encoder):
  """Encodes a timeseries of inputs with a time independent encoder.

  Args:
    inputs: A [time, batch, feature_dimensions] tensor.
    encoder: A network that takes a [batch, features_dimensions] input and
      encodes the input.
  Returns:
    A [time, batch, encoded_feature_dimensions] output tensor.
  """
  input_shape = tf.shape(inputs)
  num_timesteps, batch_size = input_shape[0], input_shape[1]
  reshaped_inputs = tf.reshape(inputs, [-1, inputs.shape[-1]])
  inputs_encoded = encoder(reshaped_inputs)
  inputs_encoded = tf.reshape(inputs_encoded,
                              [num_timesteps, batch_size, encoder.output_size])
  return inputs_encoded


def ta_for_tensor(x, **kwargs):
  """Creates a TensorArray for the input tensor."""
  return tf.TensorArray(
      x.dtype, tf.shape(x)[0], dynamic_size=False, **kwargs).unstack(x)
