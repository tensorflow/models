# Copyright 2017 Google Inc. All Rights Reserved.
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
#
# ==============================================================================
"""
LFADS - Latent Factor Analysis via Dynamical Systems.

LFADS is an unsupervised method to decompose time series data into
various factors, such as an initial condition, a generative
dynamical system, control inputs to that generator, and a low
dimensional description of the observed data, called the factors.
Additionally, the observations have a noise model (in this case
Poisson), so a denoised version of the observations is also created
(e.g. underlying rates of a Poisson distribution given the observed
event counts).

The main data structure being passed around is a dataset.  This is a dictionary
of data dictionaries.

DATASET: The top level dictionary is simply name (string -> dictionary).
The nested dictionary is the DATA DICTIONARY, which has the following keys:
  'train_data' and 'valid_data', whose values are the corresponding training
    and validation data with shape
    ExTxD, E - # examples, T - # time steps, D - # dimensions in data.
  The data dictionary also has a few more keys:
    'train_ext_input' and 'valid_ext_input', if there are know external inputs
      to the system being modeled, these take on dimensions:
      ExTxI, E - # examples, T - # time steps, I = # dimensions in input.
   'alignment_matrix_cxf' - If you are using multiple days data, it's possible
     that one can align the channels (see manuscript).  If so each dataset will
     contain this matrix, which will be used for both the input adapter and the
     output adapter for each dataset. These matrices, if provided, must be of
     size [data_dim x factors] where data_dim is the number of neurons recorded
     on that day, and factors is chosen and set through the '--factors' flag.
   'alignment_bias_c' - See alignment_matrix_cxf.  This bias will used to
     the offset for the alignment transformation.  It will *subtract* off the
     bias from the data, so pca style inits can align factors across sessions.


  If one runs LFADS on data where the true rates are known for some trials,
  (say simulated, testing data, as in the example shipped with the paper), then
  one can add three more fields for plotting purposes.  These are 'train_truth'
  and 'valid_truth', and 'conversion_factor'.  These have the same dimensions as
  'train_data', and 'valid_data' but represent the underlying rates of the
  observations.  Finally, if one needs to convert scale for plotting the true
  underlying firing rates, there is the 'conversion_factor' key.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import os
import tensorflow as tf
from distributions import LearnableDiagonalGaussian, DiagonalGaussianFromInput
from distributions import diag_gaussian_log_likelihood
from distributions import KLCost_GaussianGaussian, Poisson
from distributions import LearnableAutoRegressive1Prior
from distributions import KLCost_GaussianGaussianProcessSampled

from utils import init_linear, linear, list_t_bxn_to_tensor_bxtxn, write_data
from utils import log_sum_exp, flatten
from plot_lfads import plot_lfads


class GRU(object):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

  """
  def __init__(self, num_units, forget_bias=1.0, weight_scale=1.0,
               clip_value=np.inf, collections=None):
    """Create a GRU object.

    Args:
      num_units: Number of units in the GRU
      forget_bias (optional): Hack to help learning.
      weight_scale (optional): weights are scaled by ws/sqrt(#inputs), with
       ws being the weight scale.
      clip_value (optional): if the recurrent values grow above this value,
        clip them.
      collections (optional): List of additonal collections variables should
        belong to.
    """
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._weight_scale = weight_scale
    self._clip_value = clip_value
    self._collections = collections

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_multiplier(self):
    return 1

  def output_from_state(self, state):
    """Return the output portion of the state."""
    return state

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) function.

    Args:
      inputs: A 2D batch x input_dim tensor of inputs.
      state: The previous state from the last time step.
      scope (optional): TF variable scope for defined GRU variables.

    Returns:
      A tuple (state, state), where state is the newly computed state at time t.
      It is returned twice to respect an interface that works for LSTMs.
    """

    x = inputs
    h = state
    if inputs is not None:
      xh = tf.concat(axis=1, values=[x, h])
    else:
      xh = h

    with tf.variable_scope(scope or type(self).__name__):  # "GRU"
      with tf.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        r, u = tf.split(axis=1, num_or_size_splits=2, value=linear(xh,
                                     2 * self._num_units,
                                     alpha=self._weight_scale,
                                     name="xh_2_ru",
                                     collections=self._collections))
        r, u = tf.sigmoid(r), tf.sigmoid(u + self._forget_bias)
      with tf.variable_scope("Candidate"):
        xrh = tf.concat(axis=1, values=[x, r * h])
        c = tf.tanh(linear(xrh, self._num_units, name="xrh_2_c",
                           collections=self._collections))
      new_h = u * h + (1 - u) * c
      new_h = tf.clip_by_value(new_h, -self._clip_value, self._clip_value)

    return new_h, new_h


class GenGRU(object):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

  This version is specialized for the generator, but isn't as fast, so
  we have two.  Note this allows for l2 regularization on the recurrent
  weights, but also implicitly rescales the inputs via the 1/sqrt(input)
  scaling in the linear helper routine to be large magnitude, if there are
  fewer inputs than recurrent state.

  """
  def __init__(self, num_units, forget_bias=1.0,
               input_weight_scale=1.0, rec_weight_scale=1.0, clip_value=np.inf,
               input_collections=None, recurrent_collections=None):
    """Create a GRU object.

    Args:
      num_units: Number of units in the GRU
      forget_bias (optional): Hack to help learning.
      input_weight_scale (optional): weights are scaled ws/sqrt(#inputs), with
        ws being the weight scale.
      rec_weight_scale (optional): weights are scaled ws/sqrt(#inputs),
        with ws being the weight scale.
      clip_value (optional): if the recurrent values grow above this value,
        clip them.
      input_collections (optional): List of additonal collections variables
        that input->rec weights should belong to.
      recurrent_collections (optional): List of additonal collections variables
        that rec->rec weights should belong to.
    """
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._input_weight_scale = input_weight_scale
    self._rec_weight_scale = rec_weight_scale
    self._clip_value = clip_value
    self._input_collections = input_collections
    self._rec_collections = recurrent_collections

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  @property
  def state_multiplier(self):
    return 1

  def output_from_state(self, state):
    """Return the output portion of the state."""
    return state

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) function.

    Args:
      inputs: A 2D batch x input_dim tensor of inputs.
      state: The previous state from the last time step.
      scope (optional): TF variable scope for defined GRU variables.

    Returns:
      A tuple (state, state), where state is the newly computed state at time t.
      It is returned twice to respect an interface that works for LSTMs.
    """

    x = inputs
    h = state
    with tf.variable_scope(scope or type(self).__name__):  # "GRU"
      with tf.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        r_x = u_x = 0.0
        if x is not None:
          r_x, u_x = tf.split(axis=1, num_or_size_splits=2, value=linear(x,
                                           2 * self._num_units,
                                           alpha=self._input_weight_scale,
                                           do_bias=False,
                                           name="x_2_ru",
                                           normalized=False,
                                           collections=self._input_collections))

        r_h, u_h = tf.split(axis=1, num_or_size_splits=2, value=linear(h,
                                         2 * self._num_units,
                                         do_bias=True,
                                         alpha=self._rec_weight_scale,
                                         name="h_2_ru",
                                         collections=self._rec_collections))
        r = r_x + r_h
        u = u_x + u_h
        r, u = tf.sigmoid(r), tf.sigmoid(u + self._forget_bias)

      with tf.variable_scope("Candidate"):
        c_x = 0.0
        if x is not None:
          c_x = linear(x, self._num_units, name="x_2_c", do_bias=False,
                       alpha=self._input_weight_scale,
                       normalized=False,
                       collections=self._input_collections)
        c_rh = linear(r*h, self._num_units, name="rh_2_c", do_bias=True,
                     alpha=self._rec_weight_scale,
                     collections=self._rec_collections)
        c = tf.tanh(c_x + c_rh)

      new_h = u * h + (1 - u) * c
      new_h = tf.clip_by_value(new_h, -self._clip_value, self._clip_value)

    return new_h, new_h


class LFADS(object):
  """LFADS - Latent Factor Analysis via Dynamical Systems.

  LFADS is an unsupervised method to decompose time series data into
  various factors, such as an initial condition, a generative
  dynamical system, inferred inputs to that generator, and a low
  dimensional description of the observed data, called the factors.
  Additoinally, the observations have a noise model (in this case
  Poisson), so a denoised version of the observations is also created
  (e.g. underlying rates of a Poisson distribution given the observed
  event counts).
  """

  def __init__(self, hps, kind="train", datasets=None):
    """Create an LFADS model.

       train - a model for training, sampling of posteriors is used
       posterior_sample_and_average - sample from the posterior, this is used
         for evaluating the expected value of the outputs of LFADS, given a
         specific input, by averaging over multiple samples from the approx
         posterior.  Also used for the lower bound on the negative
         log-likelihood using IWAE error (Importance Weighed Auto-encoder).
         This is the denoising operation.
       prior_sample - a model for generation - sampling from priors is used

    Args:
      hps: The dictionary of hyper parameters.
      kind: the type of model to build (see above).
      datasets: a dictionary of named data_dictionaries, see top of lfads.py
    """
    print("Building graph...")
    all_kinds = ['train', 'posterior_sample_and_average', 'posterior_push_mean',
                 'prior_sample']
    assert kind in all_kinds, 'Wrong kind'
    if hps.feedback_factors_or_rates == "rates":
      assert len(hps.dataset_names) == 1, \
      "Multiple datasets not supported for rate feedback."
    num_steps = hps.num_steps
    ic_dim = hps.ic_dim
    co_dim = hps.co_dim
    ext_input_dim = hps.ext_input_dim
    cell_class = GRU
    gen_cell_class = GenGRU

    def makelambda(v):          # Used with tf.case
      return lambda: v

    # Define the data placeholder, and deal with all parts of the graph
    # that are dataset dependent.
    self.dataName = tf.placeholder(tf.string, shape=())
    # The batch_size to be inferred from data, as normal.
    # Additionally, the data_dim will be inferred as well, allowing for a
    # single placeholder for all datasets, regardless of data dimension.
    if hps.output_dist == 'poisson':
      # Enforce correct dtype
      assert np.issubdtype(
          datasets[hps.dataset_names[0]]['train_data'].dtype, int), \
          "Data dtype must be int for poisson output distribution"
      data_dtype = tf.int32
    elif hps.output_dist == 'gaussian':
      assert np.issubdtype(
          datasets[hps.dataset_names[0]]['train_data'].dtype, float), \
          "Data dtype must be float for gaussian output dsitribution"
      data_dtype = tf.float32
    else:
      assert False, "NIY"
    self.dataset_ph = dataset_ph = tf.placeholder(data_dtype,
                                                  [None, num_steps, None],
                                                  name="data")
    self.train_step = tf.get_variable("global_step", [], tf.int64,
                                      tf.zeros_initializer(),
                                      trainable=False)
    self.hps = hps
    ndatasets = hps.ndatasets
    factors_dim = hps.factors_dim
    self.preds = preds = [None] * ndatasets
    self.fns_in_fac_Ws = fns_in_fac_Ws = [None] * ndatasets
    self.fns_in_fatcor_bs = fns_in_fac_bs = [None] * ndatasets
    self.fns_out_fac_Ws = fns_out_fac_Ws = [None] * ndatasets
    self.fns_out_fac_bs = fns_out_fac_bs = [None] * ndatasets
    self.datasetNames = dataset_names = hps.dataset_names
    self.ext_inputs = ext_inputs = None

    if len(dataset_names) == 1:  # single session
      if 'alignment_matrix_cxf' in datasets[dataset_names[0]].keys():
        used_in_factors_dim = factors_dim
        in_identity_if_poss = False
      else:
        used_in_factors_dim = hps.dataset_dims[dataset_names[0]]
        in_identity_if_poss = True
    else:  # multisession
      used_in_factors_dim = factors_dim
      in_identity_if_poss = False

    for d, name in enumerate(dataset_names):
      data_dim = hps.dataset_dims[name]
      in_mat_cxf = None
      in_bias_1xf = None
      align_bias_1xc = None

      if datasets and 'alignment_matrix_cxf' in datasets[name].keys():
        dataset = datasets[name]
        if hps.do_train_readin:
            print("Initializing trainable readin matrix with alignment matrix" \
                  " provided for dataset:", name)
        else:
            print("Setting non-trainable readin matrix to alignment matrix" \
                  " provided for dataset:", name)
        in_mat_cxf = dataset['alignment_matrix_cxf'].astype(np.float32)
        if in_mat_cxf.shape != (data_dim, factors_dim):
          raise ValueError("""Alignment matrix must have dimensions %d x %d
          (data_dim x factors_dim), but currently has %d x %d."""%
                           (data_dim, factors_dim, in_mat_cxf.shape[0],
                            in_mat_cxf.shape[1]))
      if datasets and 'alignment_bias_c' in datasets[name].keys():
        dataset = datasets[name]
        if hps.do_train_readin:
          print("Initializing trainable readin bias with alignment bias " \
                "provided for dataset:", name)
        else:
          print("Setting non-trainable readin bias to alignment bias " \
                "provided for dataset:", name)
        align_bias_c = dataset['alignment_bias_c'].astype(np.float32)
        align_bias_1xc = np.expand_dims(align_bias_c, axis=0)
        if align_bias_1xc.shape[1] != data_dim:
          raise ValueError("""Alignment bias must have dimensions %d
          (data_dim), but currently has %d."""%
                           (data_dim, in_mat_cxf.shape[0]))
        if in_mat_cxf is not None and align_bias_1xc is not None:
          # (data - alignment_bias) * W_in
          # data * W_in - alignment_bias * W_in
          # So b = -alignment_bias * W_in to accommodate PCA style offset.
          in_bias_1xf = -np.dot(align_bias_1xc, in_mat_cxf)

      if hps.do_train_readin:
          # only add to IO transformations collection only if we want it to be
          # learnable, because IO_transformations collection will be trained
          # when do_train_io_only
          collections_readin=['IO_transformations']
      else:
          collections_readin=None

      in_fac_lin = init_linear(data_dim, used_in_factors_dim,
                               do_bias=True,
                               mat_init_value=in_mat_cxf,
                               bias_init_value=in_bias_1xf,
                               identity_if_possible=in_identity_if_poss,
                               normalized=False, name="x_2_infac_"+name,
                               collections=collections_readin,
                               trainable=hps.do_train_readin)
      in_fac_W, in_fac_b = in_fac_lin
      fns_in_fac_Ws[d] = makelambda(in_fac_W)
      fns_in_fac_bs[d] = makelambda(in_fac_b)

    with tf.variable_scope("glm"):
      out_identity_if_poss = False
      if len(dataset_names) == 1 and \
          factors_dim == hps.dataset_dims[dataset_names[0]]:
        out_identity_if_poss = True
      for d, name in enumerate(dataset_names):
        data_dim = hps.dataset_dims[name]
        in_mat_cxf = None
        if datasets and 'alignment_matrix_cxf' in datasets[name].keys():
          dataset = datasets[name]
          in_mat_cxf = dataset['alignment_matrix_cxf'].astype(np.float32)

        if datasets and 'alignment_bias_c' in datasets[name].keys():
          dataset = datasets[name]
          align_bias_c = dataset['alignment_bias_c'].astype(np.float32)
          align_bias_1xc = np.expand_dims(align_bias_c, axis=0)

        out_mat_fxc = None
        out_bias_1xc = None
        if in_mat_cxf is not None:
            out_mat_fxc = in_mat_cxf.T
        if align_bias_1xc is not None:
          out_bias_1xc = align_bias_1xc

        if hps.output_dist == 'poisson':
          out_fac_lin = init_linear(factors_dim, data_dim, do_bias=True,
                                    mat_init_value=out_mat_fxc,
                                    bias_init_value=out_bias_1xc,
                                    identity_if_possible=out_identity_if_poss,
                                    normalized=False,
                                    name="fac_2_logrates_"+name,
                                    collections=['IO_transformations'])
          out_fac_W, out_fac_b = out_fac_lin

        elif hps.output_dist == 'gaussian':
          out_fac_lin_mean = \
              init_linear(factors_dim, data_dim, do_bias=True,
                          mat_init_value=out_mat_fxc,
                          bias_init_value=out_bias_1xc,
                          normalized=False,
                          name="fac_2_means_"+name,
                          collections=['IO_transformations'])
          out_fac_W_mean, out_fac_b_mean = out_fac_lin_mean

          mat_init_value = np.zeros([factors_dim, data_dim]).astype(np.float32)
          bias_init_value = np.ones([1, data_dim]).astype(np.float32)
          out_fac_lin_logvar = \
              init_linear(factors_dim, data_dim, do_bias=True,
                          mat_init_value=mat_init_value,
                          bias_init_value=bias_init_value,
                          normalized=False,
                          name="fac_2_logvars_"+name,
                          collections=['IO_transformations'])
          out_fac_W_mean, out_fac_b_mean = out_fac_lin_mean
          out_fac_W_logvar, out_fac_b_logvar = out_fac_lin_logvar
          out_fac_W = tf.concat(
              axis=1, values=[out_fac_W_mean, out_fac_W_logvar])
          out_fac_b = tf.concat(
              axis=1, values=[out_fac_b_mean, out_fac_b_logvar])
        else:
          assert False, "NIY"

        preds[d] = tf.equal(tf.constant(name), self.dataName)
        data_dim = hps.dataset_dims[name]
        fns_out_fac_Ws[d] = makelambda(out_fac_W)
        fns_out_fac_bs[d] =  makelambda(out_fac_b)

    pf_pairs_in_fac_Ws = zip(preds, fns_in_fac_Ws)
    pf_pairs_in_fac_bs = zip(preds, fns_in_fac_bs)
    pf_pairs_out_fac_Ws = zip(preds, fns_out_fac_Ws)
    pf_pairs_out_fac_bs = zip(preds, fns_out_fac_bs)

    this_in_fac_W = tf.case(pf_pairs_in_fac_Ws, exclusive=True)
    this_in_fac_b = tf.case(pf_pairs_in_fac_bs, exclusive=True)
    this_out_fac_W = tf.case(pf_pairs_out_fac_Ws, exclusive=True)
    this_out_fac_b = tf.case(pf_pairs_out_fac_bs, exclusive=True)

    # External inputs (not changing by dataset, by definition).
    if hps.ext_input_dim > 0:
      self.ext_input = tf.placeholder(tf.float32,
                                      [None, num_steps, ext_input_dim],
                                      name="ext_input")
    else:
      self.ext_input = None
    ext_input_bxtxi = self.ext_input

    self.keep_prob = keep_prob = tf.placeholder(tf.float32, [], "keep_prob")
    self.batch_size = batch_size = int(hps.batch_size)
    self.learning_rate = tf.Variable(float(hps.learning_rate_init),
                                     trainable=False, name="learning_rate")
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * hps.learning_rate_decay_factor)

    # Dropout the data.
    dataset_do_bxtxd = tf.nn.dropout(tf.to_float(dataset_ph), keep_prob)
    if hps.ext_input_dim > 0:
      ext_input_do_bxtxi = tf.nn.dropout(ext_input_bxtxi, keep_prob)
    else:
      ext_input_do_bxtxi = None

    # ENCODERS
    def encode_data(dataset_bxtxd, enc_cell, name, forward_or_reverse,
                num_steps_to_encode):
      """Encode data for LFADS
      Args:
        dataset_bxtxd - the data to encode, as a 3 tensor, with dims
          time x batch x data dims.
        enc_cell: encoder cell
        name: name of encoder
        forward_or_reverse: string, encode in forward or reverse direction
        num_steps_to_encode: number of steps to  encode, 0:num_steps_to_encode
      Returns:
        encoded data as a list with num_steps_to_encode items, in order
      """
      if forward_or_reverse == "forward":
        dstr = "_fwd"
        time_fwd_or_rev = range(num_steps_to_encode)
      else:
        dstr = "_rev"
        time_fwd_or_rev = reversed(range(num_steps_to_encode))

      with tf.variable_scope(name+"_enc"+dstr, reuse=False):
        enc_state = tf.tile(
            tf.Variable(tf.zeros([1, enc_cell.state_size]),
                        name=name+"_enc_t0"+dstr), tf.stack([batch_size, 1]))
        enc_state.set_shape([None, enc_cell.state_size]) # tile loses shape

      enc_outs = [None] * num_steps_to_encode
      for i, t in enumerate(time_fwd_or_rev):
        with tf.variable_scope(name+"_enc"+dstr, reuse=True if i > 0 else None):
          dataset_t_bxd = dataset_bxtxd[:,t,:]
          in_fac_t_bxf = tf.matmul(dataset_t_bxd, this_in_fac_W) + this_in_fac_b
          in_fac_t_bxf.set_shape([None, used_in_factors_dim])
          if ext_input_dim > 0 and not hps.inject_ext_input_to_gen:
            ext_input_t_bxi = ext_input_do_bxtxi[:,t,:]
            enc_input_t_bxfpe = tf.concat(
                axis=1, values=[in_fac_t_bxf, ext_input_t_bxi])
          else:
            enc_input_t_bxfpe = in_fac_t_bxf
          enc_out, enc_state = enc_cell(enc_input_t_bxfpe, enc_state)
          enc_outs[t] = enc_out

      return enc_outs

    # Encode initial condition means and variances
    # ([x_T, x_T-1, ... x_0] and [x_0, x_1, ... x_T] -> g0/c0)
    self.ic_enc_fwd = [None] * num_steps
    self.ic_enc_rev = [None] * num_steps
    if ic_dim > 0:
      enc_ic_cell = cell_class(hps.ic_enc_dim,
                               weight_scale=hps.cell_weight_scale,
                               clip_value=hps.cell_clip_value)
      ic_enc_fwd = encode_data(dataset_do_bxtxd, enc_ic_cell,
                               "ic", "forward",
                               hps.num_steps_for_gen_ic)
      ic_enc_rev = encode_data(dataset_do_bxtxd, enc_ic_cell,
                               "ic", "reverse",
                               hps.num_steps_for_gen_ic)
      self.ic_enc_fwd = ic_enc_fwd
      self.ic_enc_rev = ic_enc_rev

    # Encoder control input means and variances, bi-directional encoding so:
    # ([x_T, x_T-1, ..., x_0] and [x_0, x_1 ... x_T] -> u_t)
    self.ci_enc_fwd = [None] * num_steps
    self.ci_enc_rev = [None] * num_steps
    if co_dim > 0:
      enc_ci_cell = cell_class(hps.ci_enc_dim,
                               weight_scale=hps.cell_weight_scale,
                               clip_value=hps.cell_clip_value)
      ci_enc_fwd = encode_data(dataset_do_bxtxd, enc_ci_cell,
                               "ci", "forward",
                               hps.num_steps)
      if hps.do_causal_controller:
        ci_enc_rev = None
      else:
        ci_enc_rev = encode_data(dataset_do_bxtxd, enc_ci_cell,
                                 "ci", "reverse",
                                 hps.num_steps)
      self.ci_enc_fwd = ci_enc_fwd
      self.ci_enc_rev = ci_enc_rev

    # STOCHASTIC LATENT VARIABLES, priors and posteriors
    # (initial conditions g0, and control inputs, u_t)
    # Note that zs represent all the stochastic latent variables.
    with tf.variable_scope("z", reuse=False):
      self.prior_zs_g0 = None
      self.posterior_zs_g0 = None
      self.g0s_val = None
      if ic_dim > 0:
        self.prior_zs_g0 = \
            LearnableDiagonalGaussian(batch_size, ic_dim, name="prior_g0",
                                      mean_init=0.0,
                                      var_min=hps.ic_prior_var_min,
                                      var_init=hps.ic_prior_var_scale,
                                      var_max=hps.ic_prior_var_max)
        ic_enc = tf.concat(axis=1, values=[ic_enc_fwd[-1], ic_enc_rev[0]])
        ic_enc = tf.nn.dropout(ic_enc, keep_prob)
        self.posterior_zs_g0 = \
            DiagonalGaussianFromInput(ic_enc, ic_dim, "ic_enc_2_post_g0",
                                      var_min=hps.ic_post_var_min)
        if kind in ["train", "posterior_sample_and_average",
                    "posterior_push_mean"]:
          zs_g0 = self.posterior_zs_g0
        else:
          zs_g0 = self.prior_zs_g0
        if kind in ["train", "posterior_sample_and_average", "prior_sample"]:
          self.g0s_val = zs_g0.sample
        else:
          self.g0s_val = zs_g0.mean

      # Priors for controller, 'co' for controller output
      self.prior_zs_co = prior_zs_co = [None] * num_steps
      self.posterior_zs_co = posterior_zs_co = [None] * num_steps
      self.zs_co = zs_co = [None] * num_steps
      self.prior_zs_ar_con = None
      if co_dim > 0:
        # Controller outputs
        autocorrelation_taus = [hps.prior_ar_atau for x in range(hps.co_dim)]
        noise_variances = [hps.prior_ar_nvar for x in range(hps.co_dim)]
        self.prior_zs_ar_con = prior_zs_ar_con = \
            LearnableAutoRegressive1Prior(batch_size, hps.co_dim,
                                          autocorrelation_taus,
                                          noise_variances,
                                          hps.do_train_prior_ar_atau,
                                          hps.do_train_prior_ar_nvar,
                                          num_steps, "u_prior_ar1")

    # CONTROLLER -> GENERATOR -> RATES
    # (u(t) -> gen(t) -> factors(t) -> rates(t) -> p(x_t|z_t) )
    self.controller_outputs = u_t = [None] * num_steps
    self.con_ics = con_state = None
    self.con_states = con_states = [None] * num_steps
    self.con_outs = con_outs = [None] * num_steps
    self.gen_inputs = gen_inputs = [None] * num_steps
    if co_dim > 0:
      # gen_cell_class here for l2 penalty recurrent weights
      # didn't split the cell_weight scale here, because I doubt it matters
      con_cell = gen_cell_class(hps.con_dim,
                                input_weight_scale=hps.cell_weight_scale,
                                rec_weight_scale=hps.cell_weight_scale,
                                clip_value=hps.cell_clip_value,
                                recurrent_collections=['l2_con_reg'])
      with tf.variable_scope("con", reuse=False):
        self.con_ics = tf.tile(
            tf.Variable(tf.zeros([1, hps.con_dim*con_cell.state_multiplier]),
                        name="c0"),
            tf.stack([batch_size, 1]))
        self.con_ics.set_shape([None, con_cell.state_size]) # tile loses shape
        con_states[-1] = self.con_ics

    gen_cell = gen_cell_class(hps.gen_dim,
                              input_weight_scale=hps.gen_cell_input_weight_scale,
                              rec_weight_scale=hps.gen_cell_rec_weight_scale,
                              clip_value=hps.cell_clip_value,
                              recurrent_collections=['l2_gen_reg'])
    with tf.variable_scope("gen", reuse=False):
      if ic_dim == 0:
        self.gen_ics = tf.tile(
              tf.Variable(tf.zeros([1, gen_cell.state_size]), name="g0"),
              tf.stack([batch_size, 1]))
      else:
        self.gen_ics = linear(self.g0s_val, gen_cell.state_size,
                              identity_if_possible=True,
                              name="g0_2_gen_ic")

      self.gen_states = gen_states = [None] * num_steps
      self.gen_outs = gen_outs = [None] * num_steps
      gen_states[-1] = self.gen_ics
      gen_outs[-1] = gen_cell.output_from_state(gen_states[-1])
      self.factors = factors = [None] * num_steps
      factors[-1] = linear(gen_outs[-1], factors_dim, do_bias=False,
                           normalized=True, name="gen_2_fac")

    self.rates = rates = [None] * num_steps
    # rates[-1] is collected to potentially feed back to controller
    with tf.variable_scope("glm", reuse=False):
      if hps.output_dist == 'poisson':
        log_rates_t0 = tf.matmul(factors[-1], this_out_fac_W) + this_out_fac_b
        log_rates_t0.set_shape([None, None])
        rates[-1] = tf.exp(log_rates_t0) # rate
        rates[-1].set_shape([None, hps.dataset_dims[hps.dataset_names[0]]])
      elif hps.output_dist == 'gaussian':
        mean_n_logvars = tf.matmul(factors[-1],this_out_fac_W) + this_out_fac_b
        mean_n_logvars.set_shape([None, None])
        means_t_bxd, logvars_t_bxd = tf.split(axis=1, num_or_size_splits=2,
                                              value=mean_n_logvars)
        rates[-1] = means_t_bxd
      else:
        assert False, "NIY"

    # We support multiple output distributions, for example Poisson, and also
    # Gaussian. In these two cases respectively, there are one and two
    # parameters (rates vs. mean and variance).  So the output_dist_params
    # tensor will variable sizes via tf.concat and tf.split, along the 1st
    # dimension. So in the case of gaussian, for example, it'll be
    # batch x (D+D), where each D dims is the mean, and then variances,
    # respectively. For a distribution with 3 parameters, it would be
    # batch x (D+D+D).
    self.output_dist_params = dist_params = [None] * num_steps
    self.log_p_xgz_b = log_p_xgz_b = 0.0  # log P(x|z)
    for t in range(num_steps):
      # Controller
      if co_dim > 0:
        # Build inputs for controller
        tlag = t - hps.controller_input_lag
        if tlag < 0:
          con_in_f_t = tf.zeros_like(ci_enc_fwd[0])
        else:
          con_in_f_t = ci_enc_fwd[tlag]
        if hps.do_causal_controller:
          # If controller is causal (wrt to data generation process), then it
          # cannot see future data.  Thus, excluding ci_enc_rev[t] is obvious.
          # Less obvious is the need to exclude factors[t-1].  This arises
          # because information flows from g0 through factors to the controller
          # input.  The g0 encoding is backwards, so we must necessarily exclude
          # the factors in order to keep the controller input purely from a
          # forward encoding (however unlikely it is that
          # g0->factors->controller channel might actually be used in this way).
          con_in_list_t = [con_in_f_t]
        else:
          tlag_rev = t + hps.controller_input_lag
          if tlag_rev >= num_steps:
            # better than zeros
            con_in_r_t = tf.zeros_like(ci_enc_rev[0])
          else:
            con_in_r_t = ci_enc_rev[tlag_rev]
          con_in_list_t = [con_in_f_t, con_in_r_t]

        if hps.do_feed_factors_to_controller:
          if hps.feedback_factors_or_rates == "factors":
            con_in_list_t.append(factors[t-1])
          elif hps.feedback_factors_or_rates == "rates":
            con_in_list_t.append(rates[t-1])
          else:
            assert False, "NIY"

        con_in_t = tf.concat(axis=1, values=con_in_list_t)
        con_in_t = tf.nn.dropout(con_in_t, keep_prob)
        with tf.variable_scope("con", reuse=True if t > 0 else None):
          con_outs[t], con_states[t] = con_cell(con_in_t, con_states[t-1])
          posterior_zs_co[t] = \
            DiagonalGaussianFromInput(con_outs[t], co_dim,
                                      name="con_to_post_co")
        if kind == "train":
          u_t[t] = posterior_zs_co[t].sample
        elif kind == "posterior_sample_and_average":
          u_t[t] = posterior_zs_co[t].sample
        elif kind == "posterior_push_mean":
          u_t[t] = posterior_zs_co[t].mean
        else:
          u_t[t] = prior_zs_ar_con.samples_t[t]

      # Inputs to the generator (controller output + external input)
      if ext_input_dim > 0 and hps.inject_ext_input_to_gen:
        ext_input_t_bxi = ext_input_do_bxtxi[:,t,:]
        if co_dim > 0:
          gen_inputs[t] = tf.concat(axis=1, values=[u_t[t], ext_input_t_bxi])
        else:
          gen_inputs[t] = ext_input_t_bxi
      else:
        gen_inputs[t] = u_t[t]

      # Generator
      data_t_bxd = dataset_ph[:,t,:]
      with tf.variable_scope("gen", reuse=True if t > 0 else None):
        gen_outs[t], gen_states[t] = gen_cell(gen_inputs[t], gen_states[t-1])
        gen_outs[t] = tf.nn.dropout(gen_outs[t], keep_prob)
      with tf.variable_scope("gen", reuse=True): # ic defined it above
        factors[t] = linear(gen_outs[t], factors_dim, do_bias=False,
                            normalized=True, name="gen_2_fac")
      with tf.variable_scope("glm", reuse=True if t > 0 else None):
        if hps.output_dist == 'poisson':
          log_rates_t = tf.matmul(factors[t], this_out_fac_W) + this_out_fac_b
          log_rates_t.set_shape([None, None])
          rates[t] = dist_params[t] = tf.exp(tf.clip_by_value(log_rates_t, -hps._clip_value, hps._clip_value)) # rates feed back
          rates[t].set_shape([None, hps.dataset_dims[hps.dataset_names[0]]])
          loglikelihood_t = Poisson(log_rates_t).logp(data_t_bxd)

        elif hps.output_dist == 'gaussian':
          mean_n_logvars = tf.matmul(factors[t],this_out_fac_W) + this_out_fac_b
          mean_n_logvars.set_shape([None, None])
          means_t_bxd, logvars_t_bxd = tf.split(axis=1, num_or_size_splits=2,
                                                value=mean_n_logvars)
          rates[t] = means_t_bxd # rates feed back to controller
          dist_params[t] = tf.concat(
              axis=1, values=[means_t_bxd, tf.exp(tf.clip_by_value(logvars_t_bxd, -hps._clip_value, hps._clip_value))])
          loglikelihood_t = \
              diag_gaussian_log_likelihood(data_t_bxd,
                                           means_t_bxd, logvars_t_bxd)
        else:
          assert False, "NIY"

        log_p_xgz_b += tf.reduce_sum(loglikelihood_t, [1])

    # Correlation of inferred inputs cost.
    self.corr_cost = tf.constant(0.0)
    if hps.co_mean_corr_scale > 0.0:
      all_sum_corr = []
      for i in range(hps.co_dim):
        for j in range(i+1, hps.co_dim):
          sum_corr_ij = tf.constant(0.0)
          for t in range(num_steps):
            u_mean_t = posterior_zs_co[t].mean
            sum_corr_ij += u_mean_t[:,i]*u_mean_t[:,j]
          all_sum_corr.append(0.5 * tf.square(sum_corr_ij))
      self.corr_cost = tf.reduce_mean(all_sum_corr) # div by batch and by n*(n-1)/2 pairs

    # Variational Lower Bound on posterior, p(z|x), plus reconstruction cost.
    # KL and reconstruction costs are normalized only by batch size, not by
    # dimension, or by time steps.
    kl_cost_g0_b = tf.zeros_like(batch_size, dtype=tf.float32)
    kl_cost_co_b = tf.zeros_like(batch_size, dtype=tf.float32)
    self.kl_cost = tf.constant(0.0) # VAE KL cost
    self.recon_cost = tf.constant(0.0) # VAE reconstruction cost
    self.nll_bound_vae = tf.constant(0.0)
    self.nll_bound_iwae = tf.constant(0.0) # for eval with IWAE cost.
    if kind in ["train", "posterior_sample_and_average", "posterior_push_mean"]:
      kl_cost_g0_b = 0.0
      kl_cost_co_b = 0.0
      if ic_dim > 0:
        g0_priors = [self.prior_zs_g0]
        g0_posts = [self.posterior_zs_g0]
        kl_cost_g0_b = KLCost_GaussianGaussian(g0_posts, g0_priors).kl_cost_b
        kl_cost_g0_b = hps.kl_ic_weight * kl_cost_g0_b
      if co_dim > 0:
        kl_cost_co_b = \
            KLCost_GaussianGaussianProcessSampled(
                posterior_zs_co, prior_zs_ar_con).kl_cost_b
        kl_cost_co_b = hps.kl_co_weight * kl_cost_co_b

      # L = -KL + log p(x|z), to maximize bound on likelihood
      # -L = KL - log p(x|z), to minimize bound on NLL
      # so 'reconstruction cost' is negative log likelihood
      self.recon_cost = - tf.reduce_mean(log_p_xgz_b)
      self.kl_cost = tf.reduce_mean(kl_cost_g0_b + kl_cost_co_b)

      lb_on_ll_b = log_p_xgz_b - kl_cost_g0_b - kl_cost_co_b

      # VAE error averages outside the log
      self.nll_bound_vae = -tf.reduce_mean(lb_on_ll_b)

      # IWAE error averages inside the log
      k = tf.cast(tf.shape(log_p_xgz_b)[0], tf.float32)
      iwae_lb_on_ll = -tf.log(k) + log_sum_exp(lb_on_ll_b)
      self.nll_bound_iwae = -iwae_lb_on_ll

    # L2 regularization on the generator, normalized by number of parameters.
    self.l2_cost = tf.constant(0.0)
    if self.hps.l2_gen_scale > 0.0 or self.hps.l2_con_scale > 0.0:
      l2_costs = []
      l2_numels = []
      l2_reg_var_lists = [tf.get_collection('l2_gen_reg'),
                          tf.get_collection('l2_con_reg')]
      l2_reg_scales = [self.hps.l2_gen_scale, self.hps.l2_con_scale]
      for l2_reg_vars, l2_scale in zip(l2_reg_var_lists, l2_reg_scales):
        for v in l2_reg_vars:
          numel = tf.reduce_prod(tf.concat(axis=0, values=tf.shape(v)))
          numel_f = tf.cast(numel, tf.float32)
          l2_numels.append(numel_f)
          v_l2 = tf.reduce_sum(v*v)
          l2_costs.append(0.5 * l2_scale * v_l2)
      self.l2_cost = tf.add_n(l2_costs) / tf.add_n(l2_numels)

    # Compute the cost for training, part of the graph regardless.
    # The KL cost can be problematic at the beginning of optimization,
    # so we allow an exponential increase in weighting the KL from 0
    # to 1.
    self.kl_decay_step = tf.maximum(self.train_step - hps.kl_start_step, 0)
    self.l2_decay_step = tf.maximum(self.train_step - hps.l2_start_step, 0)
    kl_decay_step_f = tf.cast(self.kl_decay_step, tf.float32)
    l2_decay_step_f = tf.cast(self.l2_decay_step, tf.float32)
    kl_increase_steps_f = tf.cast(hps.kl_increase_steps, tf.float32)
    l2_increase_steps_f = tf.cast(hps.l2_increase_steps, tf.float32)
    self.kl_weight = kl_weight = \
        tf.minimum(kl_decay_step_f / kl_increase_steps_f, 1.0)
    self.l2_weight = l2_weight = \
        tf.minimum(l2_decay_step_f / l2_increase_steps_f, 1.0)

    self.timed_kl_cost = kl_weight * self.kl_cost
    self.timed_l2_cost = l2_weight * self.l2_cost
    self.weight_corr_cost = hps.co_mean_corr_scale * self.corr_cost
    self.cost = self.recon_cost + self.timed_kl_cost + \
        self.timed_l2_cost + self.weight_corr_cost

    if kind != "train":
      # save every so often
      self.seso_saver = tf.train.Saver(tf.global_variables(),
                                       max_to_keep=hps.max_ckpt_to_keep)
      # lowest validation error
      self.lve_saver = tf.train.Saver(tf.global_variables(),
                                      max_to_keep=hps.max_ckpt_to_keep_lve)

      return

    # OPTIMIZATION
    # train the io matrices only
    if self.hps.do_train_io_only:
      self.train_vars = tvars = \
        tf.get_collection('IO_transformations',
                          scope=tf.get_variable_scope().name)
    # train the encoder only
    elif self.hps.do_train_encoder_only:
      tvars1 = \
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                          scope='LFADS/ic_enc_*')
      tvars2 = \
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                          scope='LFADS/z/ic_enc_*')

      self.train_vars = tvars = tvars1 + tvars2
    # train all variables
    else:
      self.train_vars = tvars = \
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                          scope=tf.get_variable_scope().name)
    print("done.")
    print("Model Variables (to be optimized): ")
    total_params = 0
    for i in range(len(tvars)):
      shape = tvars[i].get_shape().as_list()
      print("    ", i, tvars[i].name, shape)
      total_params += np.prod(shape)
    print("Total model parameters: ", total_params)

    grads = tf.gradients(self.cost, tvars)
    grads, grad_global_norm = tf.clip_by_global_norm(grads, hps.max_grad_norm)
    opt = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999,
                                 epsilon=1e-01)
    self.grads = grads
    self.grad_global_norm = grad_global_norm
    self.train_op = opt.apply_gradients(
        zip(grads, tvars), global_step=self.train_step)

    self.seso_saver = tf.train.Saver(tf.global_variables(),
                                     max_to_keep=hps.max_ckpt_to_keep)

    # lowest validation error
    self.lve_saver = tf.train.Saver(tf.global_variables(),
                                    max_to_keep=hps.max_ckpt_to_keep)

    # SUMMARIES, used only during training.
    # example summary
    self.example_image = tf.placeholder(tf.float32, shape=[1,None,None,3],
                                        name='image_tensor')
    self.example_summ = tf.summary.image("LFADS example", self.example_image,
                                         collections=["example_summaries"])

    # general training summaries
    self.lr_summ = tf.summary.scalar("Learning rate", self.learning_rate)
    self.kl_weight_summ = tf.summary.scalar("KL weight", self.kl_weight)
    self.l2_weight_summ = tf.summary.scalar("L2 weight", self.l2_weight)
    self.corr_cost_summ = tf.summary.scalar("Corr cost", self.weight_corr_cost)
    self.grad_global_norm_summ = tf.summary.scalar("Gradient global norm",
                                                   self.grad_global_norm)
    if hps.co_dim > 0:
      self.atau_summ = [None] * hps.co_dim
      self.pvar_summ = [None] * hps.co_dim
      for c in range(hps.co_dim):
        self.atau_summ[c] = \
            tf.summary.scalar("AR Autocorrelation taus " + str(c),
                              tf.exp(self.prior_zs_ar_con.logataus_1xu[0,c]))
        self.pvar_summ[c] = \
            tf.summary.scalar("AR Variances " + str(c),
                              tf.exp(self.prior_zs_ar_con.logpvars_1xu[0,c]))

    # cost summaries, separated into different collections for
    # training vs validation.  We make placeholders for these, because
    # even though the graph computes these costs on a per-batch basis,
    # we want to report the more reliable metric of per-epoch cost.
    kl_cost_ph = tf.placeholder(tf.float32, shape=[], name='kl_cost_ph')
    self.kl_t_cost_summ = tf.summary.scalar("KL cost (train)", kl_cost_ph,
                                            collections=["train_summaries"])
    self.kl_v_cost_summ = tf.summary.scalar("KL cost (valid)", kl_cost_ph,
                                            collections=["valid_summaries"])
    l2_cost_ph = tf.placeholder(tf.float32, shape=[], name='l2_cost_ph')
    self.l2_cost_summ = tf.summary.scalar("L2 cost", l2_cost_ph,
                                          collections=["train_summaries"])

    recon_cost_ph = tf.placeholder(tf.float32, shape=[], name='recon_cost_ph')
    self.recon_t_cost_summ = tf.summary.scalar("Reconstruction cost (train)",
                                               recon_cost_ph,
                                               collections=["train_summaries"])
    self.recon_v_cost_summ = tf.summary.scalar("Reconstruction cost (valid)",
                                               recon_cost_ph,
                                               collections=["valid_summaries"])

    total_cost_ph = tf.placeholder(tf.float32, shape=[], name='total_cost_ph')
    self.cost_t_summ = tf.summary.scalar("Total cost (train)", total_cost_ph,
                                         collections=["train_summaries"])
    self.cost_v_summ = tf.summary.scalar("Total cost (valid)", total_cost_ph,
                                         collections=["valid_summaries"])

    self.kl_cost_ph = kl_cost_ph
    self.l2_cost_ph = l2_cost_ph
    self.recon_cost_ph = recon_cost_ph
    self.total_cost_ph = total_cost_ph

    # Merged summaries, for easy coding later.
    self.merged_examples = tf.summary.merge_all(key="example_summaries")
    self.merged_generic = tf.summary.merge_all() # default key is 'summaries'
    self.merged_train = tf.summary.merge_all(key="train_summaries")
    self.merged_valid = tf.summary.merge_all(key="valid_summaries")

    session = tf.get_default_session()
    self.logfile = os.path.join(hps.lfads_save_dir, "lfads_log")
    self.writer = tf.summary.FileWriter(self.logfile)

  def build_feed_dict(self, train_name, data_bxtxd, ext_input_bxtxi=None,
                      keep_prob=None):
    """Build the feed dictionary, handles cases where there is no value defined.

    Args:
      train_name: The key into the datasets, to set the tf.case statement for
        the proper readin / readout matrices.
      data_bxtxd: The data tensor
      ext_input_bxtxi (optional): The external input tensor
      keep_prob: The drop out keep probability.

    Returns:
      The feed dictionary with TF tensors as keys and data as values, for use
      with tf.Session.run()

    """
    feed_dict = {}
    B, T, _ = data_bxtxd.shape
    feed_dict[self.dataName] = train_name
    feed_dict[self.dataset_ph] = data_bxtxd

    if self.ext_input is not None and ext_input_bxtxi is not None:
      feed_dict[self.ext_input] = ext_input_bxtxi

    if keep_prob is None:
      feed_dict[self.keep_prob] = self.hps.keep_prob
    else:
      feed_dict[self.keep_prob] = keep_prob

    return feed_dict

  @staticmethod
  def get_batch(data_extxd, ext_input_extxi=None, batch_size=None,
                example_idxs=None):
    """Get a batch of data, either randomly chosen, or specified directly.

    Args:
      data_extxd: The data to model, numpy tensors with shape:
        # examples x # time steps x # dimensions
      ext_input_extxi (optional): The external inputs, numpy tensor with shape:
        # examples x # time steps x # external input dimensions
      batch_size:  The size of the batch to return
      example_idxs (optional): The example indices used to select examples.

    Returns:
      A tuple with two parts:
        1. Batched data numpy tensor with shape:
        batch_size x # time steps x # dimensions
        2. Batched external input numpy tensor with shape:
        batch_size x # time steps x # external input dims
    """
    assert batch_size is not None or example_idxs is not None, "Problems"
    E, T, D = data_extxd.shape
    if example_idxs is None:
      example_idxs = np.random.choice(E, batch_size)

    ext_input_bxtxi = None
    if ext_input_extxi is not None:
      ext_input_bxtxi = ext_input_extxi[example_idxs,:,:]

    return data_extxd[example_idxs,:,:], ext_input_bxtxi

  @staticmethod
  def example_idxs_mod_batch_size(nexamples, batch_size):
    """Given a number of examples, E, and a batch_size, B, generate indices
    [0, 1, 2, ... B-1;
    [B, B+1, ... 2*B-1;
    ...
    ]
    returning those indices as a 2-dim tensor shaped like E/B x B.  Note that
    shape is only correct if E % B == 0.  If not, then an extra row is generated
    so that the remainder of examples is included. The extra examples are
    explicitly to to the zero index (see randomize_example_idxs_mod_batch_size)
    for randomized behavior.

    Args:
      nexamples: The number of examples to batch up.
      batch_size: The size of the batch.
    Returns:
      2-dim tensor as described above.
    """
    bmrem = batch_size - (nexamples % batch_size)
    bmrem_examples = []
    if bmrem < batch_size:
      #bmrem_examples = np.zeros(bmrem, dtype=np.int32)
      ridxs = np.random.permutation(nexamples)[0:bmrem].astype(np.int32)
      bmrem_examples = np.sort(ridxs)
    example_idxs = range(nexamples) + list(bmrem_examples)
    example_idxs_e_x_edivb = np.reshape(example_idxs, [-1, batch_size])
    return example_idxs_e_x_edivb, bmrem

  @staticmethod
  def randomize_example_idxs_mod_batch_size(nexamples, batch_size):
    """Indices 1:nexamples, randomized, in 2D form of
    shape = (nexamples / batch_size) x batch_size.  The remainder
    is managed by drawing randomly from 1:nexamples.

    Args:
      nexamples: number of examples to randomize
      batch_size: number of elements in batch

    Returns:
      The randomized, properly shaped indicies.
    """
    assert nexamples > batch_size, "Problems"
    bmrem = batch_size - nexamples % batch_size
    bmrem_examples = []
    if bmrem < batch_size:
      bmrem_examples = np.random.choice(range(nexamples),
                                        size=bmrem, replace=False)
    example_idxs = range(nexamples) + list(bmrem_examples)
    mixed_example_idxs = np.random.permutation(example_idxs)
    example_idxs_e_x_edivb = np.reshape(mixed_example_idxs, [-1, batch_size])
    return example_idxs_e_x_edivb, bmrem

  def shuffle_spikes_in_time(self, data_bxtxd):
    """Shuffle the spikes in the temporal dimension.  This is useful to
    help the LFADS system avoid overfitting to individual spikes or fast
    oscillations found in the data that are irrelevant to behavior. A
    pure 'tabula rasa' approach would avoid this, but LFADS is sensitive
    enough to pick up dynamics that you may not want.

    Args:
      data_bxtxd: numpy array of spike count data to be shuffled.
    Returns:
    S_bxtxd, a numpy array with the same dimensions and contents as
      data_bxtxd, but shuffled appropriately.

    """

    B, T, N = data_bxtxd.shape
    w = self.hps.temporal_spike_jitter_width

    if w == 0:
      return data_bxtxd

    max_counts = np.max(data_bxtxd)
    S_bxtxd = np.zeros([B,T,N])

    # Intuitively, shuffle spike occurances, 0 or 1, but since we have counts,
    # Do it over and over again up to the max count.
    for mc in range(1,max_counts+1):
      idxs = np.nonzero(data_bxtxd >= mc)

      data_ones = np.zeros_like(data_bxtxd)
      data_ones[data_bxtxd >= mc] = 1

      nfound = len(idxs[0])
      shuffles_incrs_in_time = np.random.randint(-w, w, size=nfound)

      shuffle_tidxs = idxs[1].copy()
      shuffle_tidxs += shuffles_incrs_in_time

      # Reflect on the boundaries to not lose mass.
      shuffle_tidxs[shuffle_tidxs < 0] = -shuffle_tidxs[shuffle_tidxs < 0]
      shuffle_tidxs[shuffle_tidxs > T-1] = \
          (T-1)-(shuffle_tidxs[shuffle_tidxs > T-1] -(T-1))

      for iii in zip(idxs[0], shuffle_tidxs, idxs[2]):
        S_bxtxd[iii] += 1

    return S_bxtxd

  def shuffle_and_flatten_datasets(self, datasets, kind='train'):
    """Since LFADS supports multiple datasets in the same dynamical model,
    we have to be careful to use all the data in a single training epoch.  But
    since the datasets my have different data dimensionality, we cannot batch
    examples from data dictionaries together.  Instead, we generate random
    batches within each data dictionary, and then randomize these batches
    while holding onto the dataname, so that when it's time to feed
    the graph, the correct in/out matrices can be selected, per batch.

    Args:
      datasets: A dict of data dicts.  The dataset dict is simply a
        name(string)-> data dictionary mapping (See top of lfads.py).
      kind: 'train' or 'valid'

    Returns:
      A flat list, in which each element is a pair ('name', indices).
    """
    batch_size = self.hps.batch_size
    ndatasets = len(datasets)
    random_example_idxs = {}
    epoch_idxs = {}
    all_name_example_idx_pairs = []
    kind_data = kind + '_data'
    for name, data_dict in datasets.items():
      nexamples, ntime, data_dim = data_dict[kind_data].shape
      epoch_idxs[name] = 0
      random_example_idxs, _ = \
        self.randomize_example_idxs_mod_batch_size(nexamples, batch_size)

      epoch_size = random_example_idxs.shape[0]
      names = [name] * epoch_size
      all_name_example_idx_pairs += zip(names, random_example_idxs)

    np.random.shuffle(all_name_example_idx_pairs) # shuffle in place

    return all_name_example_idx_pairs

  def train_epoch(self, datasets, batch_size=None, do_save_ckpt=True):
    """Train the model through the entire dataset once.

    Args:
      datasets: A dict of data dicts.  The dataset dict is simply a
        name(string)-> data dictionary mapping (See top of lfads.py).
      batch_size (optional):  The batch_size to use
      do_save_ckpt (optional): Should the routine save a checkpoint on this
        training epoch?

    Returns:
    A tuple with 6 float values:
      (total cost of the epoch, epoch reconstruction cost,
       epoch kl cost, KL weight used this training epoch,
       total l2 cost on generator, and the corresponding weight).
    """
    ops_to_eval = [self.cost, self.recon_cost,
                   self.kl_cost, self.kl_weight,
                   self.l2_cost, self.l2_weight,
                   self.train_op]
    collected_op_values = self.run_epoch(datasets, ops_to_eval, kind="train")

    total_cost = total_recon_cost = total_kl_cost = 0.0
    # normalizing by batch done in distributions.py
    epoch_size = len(collected_op_values)
    for op_values in collected_op_values:
      total_cost += op_values[0]
      total_recon_cost += op_values[1]
      total_kl_cost += op_values[2]

    kl_weight = collected_op_values[-1][3]
    l2_cost = collected_op_values[-1][4]
    l2_weight = collected_op_values[-1][5]

    epoch_total_cost = total_cost / epoch_size
    epoch_recon_cost = total_recon_cost / epoch_size
    epoch_kl_cost = total_kl_cost / epoch_size

    if do_save_ckpt:
      session = tf.get_default_session()
      checkpoint_path = os.path.join(self.hps.lfads_save_dir,
                                     self.hps.checkpoint_name + '.ckpt')
      self.seso_saver.save(session, checkpoint_path,
                           global_step=self.train_step)

    return epoch_total_cost, epoch_recon_cost, epoch_kl_cost, \
        kl_weight, l2_cost, l2_weight


  def run_epoch(self, datasets, ops_to_eval, kind="train", batch_size=None,
                do_collect=True, keep_prob=None):
    """Run the model through the entire dataset once.

    Args:
      datasets: A dict of data dicts.  The dataset dict is simply a
        name(string)-> data dictionary mapping (See top of lfads.py).
      ops_to_eval: A list of tensorflow operations that will be evaluated in
        the tf.session.run() call.
      batch_size (optional):  The batch_size to use
      do_collect (optional): Should the routine collect all session.run
        output as a list, and return it?
      keep_prob (optional): The dropout keep probability.

    Returns:
      A list of lists, the internal list is the return for the ops for each
      session.run() call.  The outer list collects over the epoch.
    """
    hps = self.hps
    all_name_example_idx_pairs = \
        self.shuffle_and_flatten_datasets(datasets, kind)

    kind_data = kind + '_data'
    kind_ext_input = kind + '_ext_input'

    total_cost = total_recon_cost = total_kl_cost = 0.0
    session = tf.get_default_session()
    epoch_size = len(all_name_example_idx_pairs)
    evaled_ops_list = []
    for name, example_idxs in all_name_example_idx_pairs:
      data_dict = datasets[name]
      data_extxd = data_dict[kind_data]
      if hps.output_dist == 'poisson' and hps.temporal_spike_jitter_width > 0:
        data_extxd = self.shuffle_spikes_in_time(data_extxd)

      ext_input_extxi = data_dict[kind_ext_input]
      data_bxtxd, ext_input_bxtxi = self.get_batch(data_extxd, ext_input_extxi,
                                                   example_idxs=example_idxs)

      feed_dict = self.build_feed_dict(name, data_bxtxd, ext_input_bxtxi,
                                       keep_prob=keep_prob)
      evaled_ops_np = session.run(ops_to_eval, feed_dict=feed_dict)
      if do_collect:
        evaled_ops_list.append(evaled_ops_np)

    return evaled_ops_list

  def summarize_all(self, datasets, summary_values):
    """Plot and summarize stuff in tensorboard.

    Note that everything done in the current function is otherwise done on
    a single, randomly selected dataset (except for summary_values, which are
    passed in.)

    Args:
      datasets, the dictionary of datasets used in the study.
      summary_values:  These summary values are created from the training loop,
      and so summarize the entire set of datasets.
    """
    hps = self.hps
    tr_kl_cost = summary_values['tr_kl_cost']
    tr_recon_cost = summary_values['tr_recon_cost']
    tr_total_cost = summary_values['tr_total_cost']
    kl_weight = summary_values['kl_weight']
    l2_weight = summary_values['l2_weight']
    l2_cost = summary_values['l2_cost']
    has_any_valid_set = summary_values['has_any_valid_set']
    i = summary_values['nepochs']

    session = tf.get_default_session()
    train_summ, train_step = session.run([self.merged_train,
                                          self.train_step],
                             feed_dict={self.l2_cost_ph:l2_cost,
                                        self.kl_cost_ph:tr_kl_cost,
                                        self.recon_cost_ph:tr_recon_cost,
                                        self.total_cost_ph:tr_total_cost})
    self.writer.add_summary(train_summ, train_step)
    if has_any_valid_set:
      ev_kl_cost = summary_values['ev_kl_cost']
      ev_recon_cost = summary_values['ev_recon_cost']
      ev_total_cost = summary_values['ev_total_cost']
      eval_summ = session.run(self.merged_valid,
                              feed_dict={self.kl_cost_ph:ev_kl_cost,
                                         self.recon_cost_ph:ev_recon_cost,
                                         self.total_cost_ph:ev_total_cost})
      self.writer.add_summary(eval_summ, train_step)
      print("Epoch:%d, step:%d (TRAIN, VALID): total: %.2f, %.2f\
      recon: %.2f, %.2f,     kl: %.2f, %.2f,     l2: %.5f,\
      kl weight: %.2f, l2 weight: %.2f" % \
            (i, train_step, tr_total_cost, ev_total_cost,
             tr_recon_cost, ev_recon_cost, tr_kl_cost, ev_kl_cost,
             l2_cost, kl_weight, l2_weight))

      csv_outstr = "epoch,%d, step,%d, total,%.2f,%.2f, \
      recon,%.2f,%.2f, kl,%.2f,%.2f, l2,%.5f, \
      klweight,%.2f, l2weight,%.2f\n"% \
      (i, train_step, tr_total_cost, ev_total_cost,
       tr_recon_cost, ev_recon_cost, tr_kl_cost, ev_kl_cost,
       l2_cost, kl_weight, l2_weight)

    else:
      print("Epoch:%d, step:%d TRAIN: total: %.2f     recon: %.2f, kl: %.2f,\
      l2: %.5f,    kl weight: %.2f, l2 weight: %.2f" % \
            (i, train_step, tr_total_cost, tr_recon_cost, tr_kl_cost,
             l2_cost, kl_weight, l2_weight))
      csv_outstr = "epoch,%d, step,%d, total,%.2f, recon,%.2f, kl,%.2f, \
      l2,%.5f, klweight,%.2f, l2weight,%.2f\n"% \
      (i, train_step, tr_total_cost, tr_recon_cost,
       tr_kl_cost, l2_cost, kl_weight, l2_weight)

    if self.hps.csv_log:
      csv_file = os.path.join(self.hps.lfads_save_dir, self.hps.csv_log+'.csv')
      with open(csv_file, "a") as myfile:
        myfile.write(csv_outstr)


  def plot_single_example(self, datasets):
    """Plot an image relating to a randomly chosen, specific example.  We use
    posterior sample and average by taking one example, and filling a whole
    batch with that example, sample from the posterior, and then average the
    quantities.

    """
    hps = self.hps
    all_data_names = datasets.keys()
    data_name = np.random.permutation(all_data_names)[0]
    data_dict = datasets[data_name]
    has_valid_set = True if data_dict['valid_data'] is not None else False
    cf = 1.0                  # plotting concern

    # posterior sample and average here
    E, _, _ = data_dict['train_data'].shape
    eidx = np.random.choice(E)
    example_idxs = eidx * np.ones(hps.batch_size, dtype=np.int32)

    train_data_bxtxd, train_ext_input_bxtxi = \
        self.get_batch(data_dict['train_data'], data_dict['train_ext_input'],
                       example_idxs=example_idxs)

    truth_train_data_bxtxd = None
    if 'train_truth' in data_dict and data_dict['train_truth'] is not None:
      truth_train_data_bxtxd, _ = self.get_batch(data_dict['train_truth'],
                                                 example_idxs=example_idxs)
      cf = data_dict['conversion_factor']

    # plotter does averaging
    train_model_values = self.eval_model_runs_batch(data_name,
                                                    train_data_bxtxd,
                                                    train_ext_input_bxtxi,
                                                    do_average_batch=False)

    train_step = train_model_values['train_steps']
    feed_dict = self.build_feed_dict(data_name, train_data_bxtxd,
                                     train_ext_input_bxtxi, keep_prob=1.0)

    session = tf.get_default_session()
    generic_summ = session.run(self.merged_generic, feed_dict=feed_dict)
    self.writer.add_summary(generic_summ, train_step)

    valid_data_bxtxd = valid_model_values = valid_ext_input_bxtxi = None
    truth_valid_data_bxtxd = None
    if has_valid_set:
      E, _, _ = data_dict['valid_data'].shape
      eidx = np.random.choice(E)
      example_idxs = eidx * np.ones(hps.batch_size, dtype=np.int32)
      valid_data_bxtxd, valid_ext_input_bxtxi = \
          self.get_batch(data_dict['valid_data'],
                         data_dict['valid_ext_input'],
                         example_idxs=example_idxs)
      if 'valid_truth' in data_dict and data_dict['valid_truth'] is not None:
        truth_valid_data_bxtxd, _ = self.get_batch(data_dict['valid_truth'],
                                                   example_idxs=example_idxs)
      else:
        truth_valid_data_bxtxd = None

      # plotter does averaging
      valid_model_values = self.eval_model_runs_batch(data_name,
                                                      valid_data_bxtxd,
                                                      valid_ext_input_bxtxi,
                                                      do_average_batch=False)

    example_image = plot_lfads(train_bxtxd=train_data_bxtxd,
                               train_model_vals=train_model_values,
                               train_ext_input_bxtxi=train_ext_input_bxtxi,
                               train_truth_bxtxd=truth_train_data_bxtxd,
                               valid_bxtxd=valid_data_bxtxd,
                               valid_model_vals=valid_model_values,
                               valid_ext_input_bxtxi=valid_ext_input_bxtxi,
                               valid_truth_bxtxd=truth_valid_data_bxtxd,
                               bidx=None, cf=cf, output_dist=hps.output_dist)
    example_image = np.expand_dims(example_image, axis=0)
    example_summ = session.run(self.merged_examples,
                               feed_dict={self.example_image : example_image})
    self.writer.add_summary(example_summ)

  def train_model(self, datasets):
    """Train the model, print per-epoch information, and save checkpoints.

    Loop over training epochs. The function that actually does the
    training is train_epoch.  This function iterates over the training
    data, one epoch at a time.  The learning rate schedule is such
    that it will stay the same until the cost goes up in comparison to
    the last few values, then it will drop.

    Args:
      datasets: A dict of data dicts.  The dataset dict is simply a
        name(string)-> data dictionary mapping (See top of lfads.py).
    """
    hps = self.hps
    has_any_valid_set = False
    for data_dict in datasets.values():
      if data_dict['valid_data'] is not None:
        has_any_valid_set = True
        break

    session = tf.get_default_session()
    lr = session.run(self.learning_rate)
    lr_stop = hps.learning_rate_stop
    i = -1
    train_costs = []
    valid_costs = []
    ev_total_cost = ev_recon_cost = ev_kl_cost = 0.0
    lowest_ev_cost = np.Inf
    while True:
      i += 1
      do_save_ckpt = True if i % 10 ==0 else False
      tr_total_cost, tr_recon_cost, tr_kl_cost, kl_weight, l2_cost, l2_weight = \
                self.train_epoch(datasets, do_save_ckpt=do_save_ckpt)

      # Evaluate the validation cost, and potentially save.  Note that this
      # routine will not save a validation checkpoint until the kl weight and
      # l2 weights are equal to 1.0.
      if has_any_valid_set:
        ev_total_cost, ev_recon_cost, ev_kl_cost = \
            self.eval_cost_epoch(datasets, kind='valid')
        valid_costs.append(ev_total_cost)

        # > 1 may give more consistent results, but not the actual lowest vae.
        # == 1 gives the lowest vae seen so far.
        n_lve = 1
        run_avg_lve = np.mean(valid_costs[-n_lve:])

        # conditions for saving checkpoints:
        #   KL weight must have finished stepping (>=1.0), AND
        #   L2 weight must have finished stepping OR L2 is not being used, AND
        #   the current run has a lower LVE than previous runs AND
        #     len(valid_costs > n_lve) (not sure what that does)
        if kl_weight >= 1.0 and \
          (l2_weight >= 1.0 or \
           (self.hps.l2_gen_scale == 0.0 and self.hps.l2_con_scale == 0.0)) \
           and (len(valid_costs) > n_lve and run_avg_lve < lowest_ev_cost):

          lowest_ev_cost = run_avg_lve
          checkpoint_path = os.path.join(self.hps.lfads_save_dir,
                                         self.hps.checkpoint_name + '_lve.ckpt')
          self.lve_saver.save(session, checkpoint_path,
                              global_step=self.train_step,
                              latest_filename='checkpoint_lve')

      # Plot and summarize.
      values = {'nepochs':i, 'has_any_valid_set': has_any_valid_set,
                'tr_total_cost':tr_total_cost, 'ev_total_cost':ev_total_cost,
                'tr_recon_cost':tr_recon_cost, 'ev_recon_cost':ev_recon_cost,
                'tr_kl_cost':tr_kl_cost, 'ev_kl_cost':ev_kl_cost,
                'l2_weight':l2_weight, 'kl_weight':kl_weight,
                'l2_cost':l2_cost}
      self.summarize_all(datasets, values)
      self.plot_single_example(datasets)

      # Manage learning rate.
      train_res = tr_total_cost
      n_lr = hps.learning_rate_n_to_compare
      if len(train_costs) > n_lr and train_res > np.max(train_costs[-n_lr:]):
        _ = session.run(self.learning_rate_decay_op)
        lr = session.run(self.learning_rate)
        print("     Decreasing learning rate to %f." % lr)
        # Force the system to run n_lr times while at this lr.
        train_costs.append(np.inf)
      else:
        train_costs.append(train_res)

      if lr < lr_stop:
        print("Stopping optimization based on learning rate criteria.")
        break

  def eval_cost_epoch(self, datasets, kind='train', ext_input_extxi=None,
                      batch_size=None):
    """Evaluate the cost of the epoch.

    Args:
      data_dict: The dictionary of data (training and validation) used for
        training and evaluation of the model, respectively.

    Returns:
      a 3 tuple of costs:
        (epoch total cost, epoch reconstruction cost, epoch KL cost)
    """
    ops_to_eval = [self.cost, self.recon_cost, self.kl_cost]
    collected_op_values = self.run_epoch(datasets, ops_to_eval, kind=kind,
                                         keep_prob=1.0)

    total_cost = total_recon_cost = total_kl_cost = 0.0
    # normalizing by batch done in distributions.py
    epoch_size = len(collected_op_values)
    for op_values in collected_op_values:
      total_cost += op_values[0]
      total_recon_cost += op_values[1]
      total_kl_cost += op_values[2]

    epoch_total_cost = total_cost / epoch_size
    epoch_recon_cost = total_recon_cost / epoch_size
    epoch_kl_cost = total_kl_cost / epoch_size

    return epoch_total_cost, epoch_recon_cost, epoch_kl_cost

  def eval_model_runs_batch(self, data_name, data_bxtxd, ext_input_bxtxi=None,
                            do_eval_cost=False, do_average_batch=False):
    """Returns all the goodies for the entire model, per batch.

    If data_bxtxd and ext_input_bxtxi can have fewer than batch_size along dim 1
    in which case this handles the padding and truncating automatically

    Args:
      data_name: The name of the data dict, to select which in/out matrices
        to use.
      data_bxtxd:  Numpy array training data with shape:
        batch_size x # time steps x # dimensions
      ext_input_bxtxi: Numpy array training external input with shape:
        batch_size x # time steps x # external input dims
      do_eval_cost (optional): If true, the IWAE (Importance Weighted
         Autoencoder) log likeihood bound, instead of the VAE version.
      do_average_batch (optional): average over the batch, useful for getting
      good IWAE costs, and model outputs for a single data point.

    Returns:
      A dictionary with the outputs of the model decoder, namely:
        prior g0 mean, prior g0 variance, approx. posterior mean, approx
        posterior mean, the generator initial conditions, the control inputs (if
        enabled), the state of the generator, the factors, and the rates.
    """
    session = tf.get_default_session()

    # if fewer than batch_size provided, pad to batch_size
    hps = self.hps
    batch_size = hps.batch_size
    E, _, _ = data_bxtxd.shape
    if E < hps.batch_size:
      data_bxtxd = np.pad(data_bxtxd, ((0, hps.batch_size-E), (0, 0), (0, 0)),
                          mode='constant', constant_values=0)
      if ext_input_bxtxi is not None:
        ext_input_bxtxi = np.pad(ext_input_bxtxi,
                                 ((0, hps.batch_size-E), (0, 0), (0, 0)),
                                 mode='constant', constant_values=0)

    feed_dict = self.build_feed_dict(data_name, data_bxtxd,
                                     ext_input_bxtxi, keep_prob=1.0)

    # Non-temporal signals will be batch x dim.
    # Temporal signals are list length T with elements batch x dim.
    tf_vals = [self.gen_ics, self.gen_states, self.factors,
               self.output_dist_params]
    tf_vals.append(self.cost)
    tf_vals.append(self.nll_bound_vae)
    tf_vals.append(self.nll_bound_iwae)
    tf_vals.append(self.train_step) # not train_op!
    if self.hps.ic_dim > 0:
      tf_vals += [self.prior_zs_g0.mean, self.prior_zs_g0.logvar,
                  self.posterior_zs_g0.mean, self.posterior_zs_g0.logvar]
    if self.hps.co_dim > 0:
      tf_vals.append(self.controller_outputs)
    tf_vals_flat, fidxs = flatten(tf_vals)

    np_vals_flat = session.run(tf_vals_flat, feed_dict=feed_dict)

    ff = 0
    gen_ics = [np_vals_flat[f] for f in fidxs[ff]]; ff += 1
    gen_states = [np_vals_flat[f] for f in fidxs[ff]]; ff += 1
    factors = [np_vals_flat[f] for f in fidxs[ff]]; ff += 1
    out_dist_params = [np_vals_flat[f] for f in fidxs[ff]]; ff += 1
    costs = [np_vals_flat[f] for f in fidxs[ff]]; ff += 1
    nll_bound_vaes = [np_vals_flat[f] for f in fidxs[ff]]; ff += 1
    nll_bound_iwaes = [np_vals_flat[f] for f in fidxs[ff]]; ff +=1
    train_steps = [np_vals_flat[f] for f in fidxs[ff]]; ff +=1
    if self.hps.ic_dim > 0:
      prior_g0_mean = [np_vals_flat[f] for f in fidxs[ff]]; ff +=1
      prior_g0_logvar = [np_vals_flat[f] for f in fidxs[ff]]; ff += 1
      post_g0_mean = [np_vals_flat[f] for f in fidxs[ff]]; ff += 1
      post_g0_logvar = [np_vals_flat[f] for f in fidxs[ff]]; ff += 1
    if self.hps.co_dim > 0:
      controller_outputs = [np_vals_flat[f] for f in fidxs[ff]]; ff += 1

    # [0] are to take out the non-temporal items from lists
    gen_ics = gen_ics[0]
    costs = costs[0]
    nll_bound_vaes = nll_bound_vaes[0]
    nll_bound_iwaes = nll_bound_iwaes[0]
    train_steps = train_steps[0]

    # Convert to full tensors, not lists of tensors in time dim.
    gen_states = list_t_bxn_to_tensor_bxtxn(gen_states)
    factors = list_t_bxn_to_tensor_bxtxn(factors)
    out_dist_params = list_t_bxn_to_tensor_bxtxn(out_dist_params)
    if self.hps.ic_dim > 0:
      # select first time point
      prior_g0_mean = prior_g0_mean[0]
      prior_g0_logvar = prior_g0_logvar[0]
      post_g0_mean = post_g0_mean[0]
      post_g0_logvar = post_g0_logvar[0]
    if self.hps.co_dim > 0:
      controller_outputs = list_t_bxn_to_tensor_bxtxn(controller_outputs)

    # slice out the trials in case < batch_size provided
    if E < hps.batch_size:
      idx = np.arange(E)
      gen_ics = gen_ics[idx, :]
      gen_states = gen_states[idx, :]
      factors = factors[idx, :, :]
      out_dist_params = out_dist_params[idx, :, :]
      if self.hps.ic_dim > 0:
        prior_g0_mean = prior_g0_mean[idx, :]
        prior_g0_logvar = prior_g0_logvar[idx, :]
        post_g0_mean = post_g0_mean[idx, :]
        post_g0_logvar = post_g0_logvar[idx, :]
      if self.hps.co_dim > 0:
        controller_outputs = controller_outputs[idx, :, :]

    if do_average_batch:
      gen_ics = np.mean(gen_ics, axis=0)
      gen_states = np.mean(gen_states, axis=0)
      factors = np.mean(factors, axis=0)
      out_dist_params = np.mean(out_dist_params, axis=0)
      if self.hps.ic_dim > 0:
        prior_g0_mean = np.mean(prior_g0_mean, axis=0)
        prior_g0_logvar = np.mean(prior_g0_logvar, axis=0)
        post_g0_mean = np.mean(post_g0_mean, axis=0)
        post_g0_logvar = np.mean(post_g0_logvar, axis=0)
      if self.hps.co_dim > 0:
        controller_outputs = np.mean(controller_outputs, axis=0)

    model_vals = {}
    model_vals['gen_ics'] = gen_ics
    model_vals['gen_states'] = gen_states
    model_vals['factors'] = factors
    model_vals['output_dist_params'] = out_dist_params
    model_vals['costs'] = costs
    model_vals['nll_bound_vaes'] = nll_bound_vaes
    model_vals['nll_bound_iwaes'] = nll_bound_iwaes
    model_vals['train_steps'] = train_steps
    if self.hps.ic_dim > 0:
      model_vals['prior_g0_mean'] = prior_g0_mean
      model_vals['prior_g0_logvar'] = prior_g0_logvar
      model_vals['post_g0_mean'] = post_g0_mean
      model_vals['post_g0_logvar'] = post_g0_logvar
    if self.hps.co_dim > 0:
      model_vals['controller_outputs'] = controller_outputs

    return model_vals

  def eval_model_runs_avg_epoch(self, data_name, data_extxd,
                                ext_input_extxi=None):
    """Returns all the expected value for goodies for the entire model.

    The expected value is taken over hidden (z) variables, namely the initial
    conditions and the control inputs.  The expected value is approximate, and
    accomplished via sampling (batch_size) samples for every examples.

    Args:
      data_name: The name of the data dict, to select which in/out matrices
        to use.
      data_extxd:  Numpy array training data with shape:
        # examples x # time steps x # dimensions
      ext_input_extxi (optional): Numpy array training external input with
        shape: # examples x # time steps x # external input dims

    Returns:
      A dictionary with the averaged outputs of the model decoder, namely:
        prior g0 mean, prior g0 variance, approx. posterior mean, approx
        posterior mean, the generator initial conditions, the control inputs (if
        enabled), the state of the generator, the factors, and the output
        distribution parameters, e.g. (rates or mean and variances).
    """
    hps = self.hps
    batch_size = hps.batch_size
    E, T, D  = data_extxd.shape
    E_to_process = hps.ps_nexamples_to_process
    if E_to_process > E:
      E_to_process = E

    if hps.ic_dim > 0:
      prior_g0_mean = np.zeros([E_to_process, hps.ic_dim])
      prior_g0_logvar = np.zeros([E_to_process, hps.ic_dim])
      post_g0_mean = np.zeros([E_to_process, hps.ic_dim])
      post_g0_logvar = np.zeros([E_to_process, hps.ic_dim])

    if hps.co_dim > 0:
      controller_outputs = np.zeros([E_to_process, T, hps.co_dim])
    gen_ics = np.zeros([E_to_process, hps.gen_dim])
    gen_states = np.zeros([E_to_process, T, hps.gen_dim])
    factors = np.zeros([E_to_process, T, hps.factors_dim])

    if hps.output_dist == 'poisson':
      out_dist_params = np.zeros([E_to_process, T, D])
    elif hps.output_dist == 'gaussian':
      out_dist_params = np.zeros([E_to_process, T, D+D])
    else:
      assert False, "NIY"

    costs = np.zeros(E_to_process)
    nll_bound_vaes = np.zeros(E_to_process)
    nll_bound_iwaes = np.zeros(E_to_process)
    train_steps = np.zeros(E_to_process)
    for es_idx in range(E_to_process):
      print("Running %d of %d." % (es_idx+1, E_to_process))
      example_idxs = es_idx * np.ones(batch_size, dtype=np.int32)
      data_bxtxd, ext_input_bxtxi = self.get_batch(data_extxd,
                                                   ext_input_extxi,
                                                   batch_size=batch_size,
                                                   example_idxs=example_idxs)
      model_values = self.eval_model_runs_batch(data_name, data_bxtxd,
                                                ext_input_bxtxi,
                                                do_eval_cost=True,
                                                do_average_batch=True)

      if self.hps.ic_dim > 0:
        prior_g0_mean[es_idx,:] = model_values['prior_g0_mean']
        prior_g0_logvar[es_idx,:] = model_values['prior_g0_logvar']
        post_g0_mean[es_idx,:] = model_values['post_g0_mean']
        post_g0_logvar[es_idx,:] = model_values['post_g0_logvar']
      gen_ics[es_idx,:] = model_values['gen_ics']

      if self.hps.co_dim > 0:
        controller_outputs[es_idx,:,:] = model_values['controller_outputs']
      gen_states[es_idx,:,:] = model_values['gen_states']
      factors[es_idx,:,:] = model_values['factors']
      out_dist_params[es_idx,:,:] = model_values['output_dist_params']
      costs[es_idx] = model_values['costs']
      nll_bound_vaes[es_idx] = model_values['nll_bound_vaes']
      nll_bound_iwaes[es_idx] = model_values['nll_bound_iwaes']
      train_steps[es_idx] = model_values['train_steps']
      print('bound nll(vae): %.3f, bound nll(iwae): %.3f' \
            % (nll_bound_vaes[es_idx], nll_bound_iwaes[es_idx]))

    model_runs = {}
    if self.hps.ic_dim > 0:
      model_runs['prior_g0_mean'] = prior_g0_mean
      model_runs['prior_g0_logvar'] = prior_g0_logvar
      model_runs['post_g0_mean'] = post_g0_mean
      model_runs['post_g0_logvar'] = post_g0_logvar
    model_runs['gen_ics'] = gen_ics

    if self.hps.co_dim > 0:
      model_runs['controller_outputs'] = controller_outputs
    model_runs['gen_states'] = gen_states
    model_runs['factors'] = factors
    model_runs['output_dist_params'] = out_dist_params
    model_runs['costs'] = costs
    model_runs['nll_bound_vaes'] = nll_bound_vaes
    model_runs['nll_bound_iwaes'] = nll_bound_iwaes
    model_runs['train_steps'] = train_steps
    return model_runs

  def eval_model_runs_push_mean(self, data_name, data_extxd,
                                ext_input_extxi=None):
    """Returns values of interest for the  model by pushing the means through

    The mean values for both initial conditions and the control inputs are
    pushed through the model instead of sampling (as is done in
    eval_model_runs_avg_epoch).
    This is a quick and approximate version of estimating these values instead
    of sampling from the posterior many times and then averaging those values of
    interest.

    Internally, a total of batch_size trials are run through the model at once.

    Args:
      data_name: The name of the data dict, to select which in/out matrices
        to use.
      data_extxd:  Numpy array training data with shape:
        # examples x # time steps x # dimensions
      ext_input_extxi (optional): Numpy array training external input with
        shape: # examples x # time steps x # external input dims

    Returns:
      A dictionary with the estimated outputs of the model decoder, namely:
        prior g0 mean, prior g0 variance, approx. posterior mean, approx
        posterior mean, the generator initial conditions, the control inputs (if
        enabled), the state of the generator, the factors, and the output
        distribution parameters, e.g. (rates or mean and variances).
    """
    hps = self.hps
    batch_size = hps.batch_size
    E, T, D  = data_extxd.shape
    E_to_process = hps.ps_nexamples_to_process
    if E_to_process > E:
      print("Setting number of posterior samples to process to : ", E)
      E_to_process = E

    if hps.ic_dim > 0:
      prior_g0_mean = np.zeros([E_to_process, hps.ic_dim])
      prior_g0_logvar = np.zeros([E_to_process, hps.ic_dim])
      post_g0_mean = np.zeros([E_to_process, hps.ic_dim])
      post_g0_logvar = np.zeros([E_to_process, hps.ic_dim])

    if hps.co_dim > 0:
      controller_outputs = np.zeros([E_to_process, T, hps.co_dim])
    gen_ics = np.zeros([E_to_process, hps.gen_dim])
    gen_states = np.zeros([E_to_process, T, hps.gen_dim])
    factors = np.zeros([E_to_process, T, hps.factors_dim])

    if hps.output_dist == 'poisson':
      out_dist_params = np.zeros([E_to_process, T, D])
    elif hps.output_dist == 'gaussian':
      out_dist_params = np.zeros([E_to_process, T, D+D])
    else:
      assert False, "NIY"

    costs = np.zeros(E_to_process)
    nll_bound_vaes = np.zeros(E_to_process)
    nll_bound_iwaes = np.zeros(E_to_process)
    train_steps = np.zeros(E_to_process)

    # generator that will yield 0:N in groups of per items, e.g.
    # (0:per-1), (per:2*per-1), ..., with the last group containing <= per items
    # this will be used to feed per=batch_size trials into the model at a time
    def trial_batches(N, per):
      for i in range(0, N, per):
        yield np.arange(i, min(i+per, N), dtype=np.int32)

    for batch_idx, es_idx in enumerate(trial_batches(E_to_process,
                                                     hps.batch_size)):
      print("Running trial batch %d with %d trials" % (batch_idx+1,
                                                       len(es_idx)))
      data_bxtxd, ext_input_bxtxi = self.get_batch(data_extxd,
                                                   ext_input_extxi,
                                                   batch_size=batch_size,
                                                   example_idxs=es_idx)
      model_values = self.eval_model_runs_batch(data_name, data_bxtxd,
                                                ext_input_bxtxi,
                                                do_eval_cost=True,
                                                do_average_batch=False)

      if self.hps.ic_dim > 0:
        prior_g0_mean[es_idx,:] = model_values['prior_g0_mean']
        prior_g0_logvar[es_idx,:] = model_values['prior_g0_logvar']
        post_g0_mean[es_idx,:] = model_values['post_g0_mean']
        post_g0_logvar[es_idx,:] = model_values['post_g0_logvar']
      gen_ics[es_idx,:] = model_values['gen_ics']

      if self.hps.co_dim > 0:
        controller_outputs[es_idx,:,:] = model_values['controller_outputs']
      gen_states[es_idx,:,:] = model_values['gen_states']
      factors[es_idx,:,:] = model_values['factors']
      out_dist_params[es_idx,:,:] = model_values['output_dist_params']

      # TODO
      # model_values['costs'] and other costs come out as scalars, summed over
      # all the trials in the batch. what we want is the per-trial costs
      costs[es_idx] = model_values['costs']
      nll_bound_vaes[es_idx] = model_values['nll_bound_vaes']
      nll_bound_iwaes[es_idx] = model_values['nll_bound_iwaes']

      train_steps[es_idx] = model_values['train_steps']

    model_runs = {}
    if self.hps.ic_dim > 0:
      model_runs['prior_g0_mean'] = prior_g0_mean
      model_runs['prior_g0_logvar'] = prior_g0_logvar
      model_runs['post_g0_mean'] = post_g0_mean
      model_runs['post_g0_logvar'] = post_g0_logvar
    model_runs['gen_ics'] = gen_ics

    if self.hps.co_dim > 0:
      model_runs['controller_outputs'] = controller_outputs
    model_runs['gen_states'] = gen_states
    model_runs['factors'] = factors
    model_runs['output_dist_params'] = out_dist_params

    # You probably do not want the LL associated values when pushing the mean
    # instead of sampling.
    model_runs['costs'] = costs
    model_runs['nll_bound_vaes'] = nll_bound_vaes
    model_runs['nll_bound_iwaes'] = nll_bound_iwaes
    model_runs['train_steps'] = train_steps
    return model_runs

  def write_model_runs(self, datasets, output_fname=None, push_mean=False):
    """Run the model on the data in data_dict, and save the computed values.

    LFADS generates a number of outputs for each examples, and these are all
    saved.  They are:
      The mean and variance of the prior of g0.
      The mean and variance of approximate posterior of g0.
      The control inputs (if enabled)
      The initial conditions, g0, for all examples.
      The generator states for all time.
      The factors for all time.
      The output distribution parameters (e.g. rates) for all time.

    Args:
      datasets: a dictionary of named data_dictionaries, see top of lfads.py
      output_fname: a file name stem for the output files.
      push_mean: if False (default), generates batch_size samples for each trial
        and averages the results. if True, runs each trial once without noise,
        pushing the posterior mean initial conditions and control inputs through
        the trained model. False is used for posterior_sample_and_average, True
        is used for posterior_push_mean.
    """
    hps = self.hps
    kind = hps.kind

    for data_name, data_dict in datasets.items():
      data_tuple = [('train', data_dict['train_data'],
                     data_dict['train_ext_input']),
                    ('valid', data_dict['valid_data'],
                     data_dict['valid_ext_input'])]
      for data_kind, data_extxd, ext_input_extxi in data_tuple:
        if not output_fname:
          fname = "model_runs_" + data_name + '_' + data_kind + '_' + kind
        else:
          fname = output_fname + data_name + '_' + data_kind + '_' + kind

        print("Writing data for %s data and kind %s." % (data_name, data_kind))
        if push_mean:
          model_runs = self.eval_model_runs_push_mean(data_name, data_extxd,
                                                      ext_input_extxi)
        else:
          model_runs = self.eval_model_runs_avg_epoch(data_name, data_extxd,
                                                      ext_input_extxi)
        full_fname = os.path.join(hps.lfads_save_dir, fname)
        write_data(full_fname, model_runs, compression='gzip')
        print("Done.")

  def write_model_samples(self, dataset_name, output_fname=None):
    """Use the prior distribution to generate batch_size number of samples
    from the model.

    LFADS generates a number of outputs for each sample, and these are all
    saved.  They are:
      The mean and variance of the prior of g0.
      The control inputs (if enabled)
      The initial conditions, g0, for all examples.
      The generator states for all time.
      The factors for all time.
      The output distribution parameters (e.g. rates) for all time.

    Args:
      dataset_name: The name of the dataset to grab the factors -> rates
      alignment matrices from.
      output_fname: The name of the file in which to save the generated
        samples.
    """
    hps = self.hps
    batch_size = hps.batch_size

    print("Generating %d samples" % (batch_size))
    tf_vals = [self.factors, self.gen_states, self.gen_ics,
               self.cost, self.output_dist_params]
    if hps.ic_dim > 0:
      tf_vals += [self.prior_zs_g0.mean, self.prior_zs_g0.logvar]
    if hps.co_dim > 0:
      tf_vals += [self.prior_zs_ar_con.samples_t]
    tf_vals_flat, fidxs = flatten(tf_vals)

    session = tf.get_default_session()
    feed_dict = {}
    feed_dict[self.dataName] = dataset_name
    feed_dict[self.keep_prob] = 1.0

    np_vals_flat = session.run(tf_vals_flat, feed_dict=feed_dict)

    ff = 0
    factors = [np_vals_flat[f] for f in fidxs[ff]]; ff += 1
    gen_states = [np_vals_flat[f] for f in fidxs[ff]]; ff += 1
    gen_ics = [np_vals_flat[f] for f in fidxs[ff]]; ff += 1
    costs = [np_vals_flat[f] for f in fidxs[ff]]; ff += 1
    output_dist_params = [np_vals_flat[f] for f in fidxs[ff]]; ff += 1
    if hps.ic_dim > 0:
      prior_g0_mean = [np_vals_flat[f] for f in fidxs[ff]]; ff += 1
      prior_g0_logvar = [np_vals_flat[f] for f in fidxs[ff]]; ff += 1
    if hps.co_dim > 0:
      prior_zs_ar_con = [np_vals_flat[f] for f in fidxs[ff]]; ff += 1

    # [0] are to take out the non-temporal items from lists
    gen_ics = gen_ics[0]
    costs = costs[0]

    # Convert to full tensors, not lists of tensors in time dim.
    gen_states = list_t_bxn_to_tensor_bxtxn(gen_states)
    factors = list_t_bxn_to_tensor_bxtxn(factors)
    output_dist_params = list_t_bxn_to_tensor_bxtxn(output_dist_params)
    if hps.ic_dim > 0:
      prior_g0_mean = prior_g0_mean[0]
      prior_g0_logvar = prior_g0_logvar[0]
    if hps.co_dim > 0:
      prior_zs_ar_con = list_t_bxn_to_tensor_bxtxn(prior_zs_ar_con)

    model_vals = {}
    model_vals['gen_ics'] = gen_ics
    model_vals['gen_states'] = gen_states
    model_vals['factors'] = factors
    model_vals['output_dist_params'] = output_dist_params
    model_vals['costs'] = costs.reshape(1)
    if hps.ic_dim > 0:
      model_vals['prior_g0_mean'] = prior_g0_mean
      model_vals['prior_g0_logvar'] = prior_g0_logvar
    if hps.co_dim > 0:
      model_vals['prior_zs_ar_con'] = prior_zs_ar_con

    full_fname = os.path.join(hps.lfads_save_dir, output_fname)
    write_data(full_fname, model_vals, compression='gzip')
    print("Done.")

  @staticmethod
  def eval_model_parameters(use_nested=True, include_strs=None):
    """Evaluate and return all of the TF variables in the model.

    Args:
    use_nested (optional): For returning values, use a nested dictoinary, based
      on variable scoping, or return all variables in a flat dictionary.
    include_strs (optional): A list of strings to use as a filter, to reduce the
      number of variables returned.  A variable name must contain at least one
      string in include_strs as a sub-string in order to be returned.

    Returns:
      The parameters of the model.  This can be in a flat
      dictionary, or a nested dictionary, where the nesting is by variable
      scope.
    """
    all_tf_vars = tf.global_variables()
    session = tf.get_default_session()
    all_tf_vars_eval = session.run(all_tf_vars)
    vars_dict = {}
    strs = ["LFADS"]
    if include_strs:
      strs += include_strs

    for i, (var, var_eval) in enumerate(zip(all_tf_vars, all_tf_vars_eval)):
      if any(s in include_strs for s in var.name):
        if not isinstance(var_eval, np.ndarray): # for H5PY
          print(var.name, """ is not numpy array, saving as numpy array
                with value: """, var_eval, type(var_eval))
          e = np.array(var_eval)
          print(e, type(e))
        else:
          e = var_eval
        vars_dict[var.name] = e

    if not use_nested:
      return vars_dict

    var_names = vars_dict.keys()
    nested_vars_dict = {}
    current_dict = nested_vars_dict
    for v, var_name in enumerate(var_names):
      var_split_name_list = var_name.split('/')
      split_name_list_len = len(var_split_name_list)
      current_dict = nested_vars_dict
      for p, part in enumerate(var_split_name_list):
        if p < split_name_list_len - 1:
          if part in current_dict:
            current_dict = current_dict[part]
          else:
            current_dict[part] = {}
            current_dict = current_dict[part]
        else:
          current_dict[part] = vars_dict[var_name]

    return nested_vars_dict

  @staticmethod
  def spikify_rates(rates_bxtxd):
    """Randomly spikify underlying rates according a Poisson distribution

    Args:
      rates_bxtxd: a numpy tensor with shape:

    Returns:
      A numpy array with the same shape as rates_bxtxd, but with the event
      counts.
    """

    B,T,N = rates_bxtxd.shape
    assert all([B > 0, N > 0]), "problems"

    # Because the rates are changing, there is nesting
    spikes_bxtxd = np.zeros([B,T,N], dtype=np.int32)
    for b in range(B):
      for t in range(T):
        for n in range(N):
          rate = rates_bxtxd[b,t,n]
          count = np.random.poisson(rate)
          spikes_bxtxd[b,t,n] = count

    return spikes_bxtxd
