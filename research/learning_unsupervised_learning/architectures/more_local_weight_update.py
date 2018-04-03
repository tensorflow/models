# Copyright 2018 Google, Inc. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import sonnet as snt
import tensorflow as tf

from learning_unsupervised_learning.architectures import common
from learning_unsupervised_learning import optimizers
from learning_unsupervised_learning import utils
from learning_unsupervised_learning import summary_utils

OptState = collections.namedtuple('OptState',
                                  ['variables', 'opt_state', 'index'])

BaseModelOutputs = collections.namedtuple(
    'BaseModelOutputs', ['xs', 'zs', 'mods', 'batch', 'backward_mods'])


class GradChannelReadout(snt.AbstractModule):
  """Perform a linear readout and reshape from input 3 tensor."""

  def __init__(self,
               num_grad_channels,
               device,
               perm=(2, 0, 1),
               name='GradChannelReadout'):
    """Args:

      num_grad_channels: int
        number of channels to readout to.
      device: str or callable
        devicwe to place weights.
      perm: list or tuple
        transpose applied.
    """

    self.num_grad_channels = num_grad_channels
    self.device = device
    self.perm = perm
    super(GradChannelReadout, self).__init__(name=name)

  def _build(self, h):
    with tf.device(self.device):
      mod = snt.Linear(self.num_grad_channels)
      ret = snt.BatchApply(mod)(h)
      # return as [num_grad_channels] x [bs] x [num units]
      return tf.transpose(ret, perm=self.perm)


def get_weight_stats(x, axis):
  """ Compute weight statistics over the given axis.

  Args:
    x: tf.Tensor
      a batch of activations.
    axis: int
      axis to perform statistics over.
  Returns:
    tf.Tensor
      a 3-D tensor with statistics.
  """
  if x is None:
    return []

  stats = []
  l1 = tf.reduce_mean(tf.abs(x), axis=axis)
  l2 = tf.sqrt(tf.reduce_mean(x**2, axis=axis) + 1e-6)

  mean, var = tf.nn.moments(x, [axis])
  stats.extend([l1, l2, mean, tf.sqrt(var + 1e-8)])

  stats = [tf.reshape(s, [-1, 1, 1]) for s in stats]

  return stats


class AddUnitBatchStatistics(snt.AbstractModule):
  """Compute some number of statistics over units and concat them on."""

  def __init__(self, name='AddUnitBatchStatistics'):
    super(AddUnitBatchStatistics, self).__init__(name=name)

  def _build(self, x):
    # [channel, bs, 1]
    output = x
    for d in [0, 1]:
      stats = []
      l1 = tf.reduce_mean(tf.abs(x), axis=d, keepdims=True)
      l2 = tf.sqrt(tf.reduce_mean(x**2, axis=d, keepdims=True) + 1e-6)

      mean, var = tf.nn.moments(x, [d], keepdims=True)
      stats.extend([l1, l2, mean, tf.sqrt(var + 1e-8)])

      to_add = tf.concat(stats, axis=2)  # [channels/1, units/1, stats]
      output += snt.BatchApply(snt.Linear(x.shape.as_list()[2]))(to_add)
    return output


class ConcatUnitConv(snt.AbstractModule):
  """Do a small number of convolutions over units and concat / add them on."""

  def __init__(self, add=True):
    self.add = add
    super(ConcatUnitConv, self).__init__(name='ConcatUnitConv')

  def _build(self, x):
    # x is [units, bs, 1]
    net = tf.transpose(x, [1, 0, 2])  # now [bs x units x 1]
    channels = x.shape.as_list()[2]
    mod = snt.Conv1D(output_channels=channels, kernel_shape=[3])
    net = mod(net)
    net = snt.BatchNorm(axis=[0, 1])(net, is_training=False)
    net = tf.nn.relu(net)
    mod = snt.Conv1D(output_channels=channels, kernel_shape=[3])
    net = mod(net)
    net = snt.BatchNorm(axis=[0, 1])(net, is_training=False)
    net = tf.nn.relu(net)
    to_concat = tf.transpose(net, [1, 0, 2])
    if self.add:
      return x + to_concat
    else:
      return tf.concat([x, to_concat], 2)


class MoreLocalWeightUpdateProcess(snt.AbstractModule):

  def __init__(
      self,
      remote_device,
      local_device,
      top_delta_size=64,
      top_delta_layers=2,
      compute_h_size=64,
      compute_h_layers=1,
      delta_dim=32,
      num_grad_channels=4,
      normalize_epsilon=1.,
  ):
    self.local_device = local_device
    self.remote_device = remote_device
    self.top_delta_size = top_delta_size
    self.top_delta_layers = top_delta_layers
    self.compute_h_size = compute_h_size
    self.compute_h_layers = compute_h_layers
    self.delta_dim = delta_dim
    self.num_grad_channels = num_grad_channels
    self.normalize_epsilon = normalize_epsilon,

    with tf.device(local_device):
      self.opt = optimizers.UnrollableGradientDescentRollingOptimizer(
          learning_rate=1e-4)

    # lazily initialized for readouts
    self.readout_mods = {}

    super(MoreLocalWeightUpdateProcess,
          self).__init__(name='MoreLocalWeightUpdateProcess')

    with tf.device(remote_device):
      self()

  def normalize(self, change_w, normalize_epsilon=None):
    if normalize_epsilon is None:
      normalize_epsilon = self.normalize_epsilon

    # normalize the weights per receptive-field, rather than per-matrix
    var = tf.reduce_mean(tf.square(change_w), axis=0, keepdims=True)
    change_w = (change_w) / tf.sqrt(normalize_epsilon + var)
    return change_w

  def _build(self):
    pass

  @snt.reuse_variables
  def compute_top_delta(self, z):
    """ parameterization of topD. This converts the top level activation
    to an error signal.
    Args:
      z: tf.Tensor
        batch of final layer post activations
    Returns
      delta: tf.Tensor
        the error signal
    """
    s_idx = 0
    with tf.variable_scope('compute_top_delta'), tf.device(self.remote_device):
      # typically this takes [BS, length, input_channels],
      # We are applying this such that we convolve over the batch dimension.
      act = tf.expand_dims(tf.transpose(z, [1, 0]), 2)  # [channels, BS, 1]

      mod = snt.Conv1D(output_channels=self.top_delta_size, kernel_shape=[5])
      act = mod(act)

      act = snt.BatchNorm(axis=[0, 1])(act, is_training=False)
      act = tf.nn.relu(act)

      bs = act.shape.as_list()[0]
      act = tf.transpose(act, [2, 1, 0])
      act = snt.Conv1D(output_channels=bs, kernel_shape=[3])(act)
      act = snt.BatchNorm(axis=[0, 1])(act, is_training=False)
      act = tf.nn.relu(act)
      act = snt.Conv1D(output_channels=bs, kernel_shape=[3])(act)
      act = snt.BatchNorm(axis=[0, 1])(act, is_training=False)
      act = tf.nn.relu(act)
      act = tf.transpose(act, [2, 1, 0])

      prev_act = act
      for i in range(self.top_delta_layers):
        mod = snt.Conv1D(output_channels=self.top_delta_size, kernel_shape=[3])
        act = mod(act)

        act = snt.BatchNorm(axis=[0, 1])(act, is_training=False)
        act = tf.nn.relu(act)

        prev_act = act

      mod = snt.Conv1D(output_channels=self.delta_dim, kernel_shape=[3])
      act = mod(act)

      # [bs, feature_channels, delta_channels]
      act = tf.transpose(act, [1, 0, 2])
      return act

  @snt.reuse_variables
  def compute_h(self,
                x,
                z,
                d,
                bias,
                W_bot,
                W_top,
                compute_perc=1.0,
                compute_units=None):
    """z = [BS, n_units] a = [BS, n_units] b = [BS, n_units] d = [BS, n_units, delta_channels]

    """

    s_idx = 0
    if compute_perc != 1.0:
      assert compute_units is None

    with tf.device(self.remote_device):
      inp_feat = [x, z]
      inp_feat = [tf.transpose(f, [1, 0]) for f in inp_feat]

      units = x.shape.as_list()[1]
      bs = x.shape.as_list()[0]

      # add unit ID, to help the network differentiate units
      id_theta = tf.linspace(0., (4) * np.pi, units)
      assert bs is not None
      id_theta_bs = tf.reshape(id_theta, [-1, 1]) * tf.ones([1, bs])
      inp_feat += [tf.sin(id_theta_bs), tf.cos(id_theta_bs)]

      # list of [units, BS, 1]
      inp_feat = [tf.expand_dims(f, 2) for f in inp_feat]

      d_trans = tf.transpose(d, [1, 0, 2])

      if compute_perc != 1.0:
        compute_units = int(compute_perc * inp_feat.shape.as_list()[0])

      # add weight matrix statistics, both from above and below
      w_stats_bot = get_weight_stats(W_bot, 0)
      w_stats_top = get_weight_stats(W_top, 1)
      w_stats = w_stats_bot + w_stats_top
      if W_bot is None or W_top is None:
        # if it's an edge layer (top or bottom), just duplicate the stats for
        # the weight matrix that does exist
        w_stats = w_stats + w_stats
      w_stats = [tf.ones([1, x.shape[0], 1]) * ww for ww in w_stats]
      # w_stats is a list, with entries with shape UNITS x 1 x channels

      if compute_units is None:
        inp_feat_in = inp_feat
        d_trans_in = d_trans
        w_stats_in = w_stats
        bias_in = tf.transpose(bias)
      else:
        # only run on a subset of the activations.
        mask = tf.random_uniform(
            minval=0,
            maxval=1,
            dtype=tf.float32,
            shape=inp_feat[0].shape.as_list()[0:1])
        _, ind = tf.nn.top_k(mask, k=compute_units)
        ind = tf.reshape(ind, [-1, 1])

        inp_feat_in = [tf.gather_nd(xx, ind) for xx in inp_feat]
        w_stats_in = [tf.gather_nd(xx, ind) for xx in w_stats]
        d_trans_in = tf.gather_nd(d_trans, ind)
        bias_in = tf.gather_nd(tf.transpose(bias), ind)

      w_stats_in = tf.concat(w_stats_in, 2)
      w_stats_in_norm = w_stats_in * tf.rsqrt(
          tf.reduce_mean(w_stats_in**2) + 1e-6)

      act = tf.concat(inp_feat_in + [d_trans_in], 2)
      act = snt.BatchNorm(axis=[0, 1])(act, is_training=True)

      bias_dense = tf.reshape(bias_in, [-1, 1, 1]) * tf.ones([1, bs, 1])
      act = tf.concat([w_stats_in_norm, bias_dense, act], 2)

      mod = snt.Conv1D(output_channels=self.compute_h_size, kernel_shape=[3])
      act = mod(act)

      act = snt.BatchNorm(axis=[0, 1])(act, is_training=True)
      act = tf.nn.relu(act)

      act2 = ConcatUnitConv()(act)
      act = act2

      prev_act = act
      for i in range(self.compute_h_layers):
        mod = snt.Conv1D(output_channels=self.compute_h_size, kernel_shape=[3])
        act = mod(act)

        act = snt.BatchNorm(axis=[0, 1])(act, is_training=True)
        act = tf.nn.relu(act)

        act = ConcatUnitConv()(act)

        prev_act = act

      h = act
      if compute_units is not None:
        shape = inp_feat[0].shape.as_list()[:1] + h.shape.as_list()[1:]
        h = tf.scatter_nd(ind, h, shape=shape)

      h = tf.transpose(h, [1, 0, 2])  # [bs, units, channels]

      return h

  ## wrappers to allow forward and backward to have different variables
  @snt.reuse_variables
  def merge_change_w_forward(self, change_w_terms, global_prefix='', prefix=''):
    return self.merge_change_w(
        change_w_terms, global_prefix=global_prefix, prefix=prefix)

  @snt.reuse_variables
  def merge_change_w_backward(self, change_w_terms, global_prefix='',
                              prefix=''):
    return self.merge_change_w(
        change_w_terms, global_prefix=global_prefix, prefix=prefix)

  def merge_change_w(self, change_w_terms, global_prefix='', prefix=''):
    with tf.device(
        self.remote_device), tf.name_scope(global_prefix + '_merge_change_w'):
      w_base = change_w_terms['w_base']

      for kk in sorted(change_w_terms.keys()):
        name = global_prefix + 'change_w_plane_%s' % kk
        delta_w = change_w_terms[kk]
        mean, var = tf.nn.moments(delta_w, [0, 1])
        root_mean_square = tf.sqrt(tf.reduce_mean(delta_w**2) + 1e-6)

      for kk in sorted(change_w_terms.keys()):
        change_w_terms[kk] = self.normalize(change_w_terms[kk])

      initializers = {
          'w': tf.constant_initializer(0.1),
          'b': tf.zeros_initializer()
      }
      mod = snt.Linear(
          1,
          name=global_prefix + '_weight_readout_coeffs',
          initializers=initializers)

      change_w_terms_list = [
          change_w_terms[kk] for kk in sorted(change_w_terms.keys())
      ]
      stack_terms = tf.stack(change_w_terms_list, axis=-1)
      change_w = tf.squeeze(
          snt.BatchApply(mod)(stack_terms), axis=-1) / len(change_w_terms)

      # only allow perpendicular updates, or updates which grow length. don't
      # allow length to decay towards zero.
      ip = tf.reduce_mean(change_w * w_base)
      # zero out any updates that shrink length
      ip = tf.nn.relu(ip)
      change_w -= w_base * ip
      change_w /= tf.sqrt(len(change_w_terms) * 1.)

      change_w = self.normalize(change_w)

      # encourage the receptive field to not collapse to 0
      change_w -= w_base / 7.  # This is an arbitrary scale choice

      return tf.identity(change_w)

  @snt.reuse_variables
  def bias_readout(self, h):
    with tf.device(self.remote_device):
      mod = snt.Linear(1, name='bias_readout')
      ret = snt.BatchApply(mod)(h)
      return tf.squeeze(ret, 2)

  @snt.reuse_variables
  def next_delta(self, z, h, d):
    with tf.device(self.remote_device):
      return d * tf.expand_dims(tf.nn.sigmoid(z), 2) + self.to_delta_size(h)

  @utils.create_variables_in_class_scope
  def get_readout_mod(self, name):
    if name not in self.readout_mods:
      self.readout_mods[name] = GradChannelReadout(
          self.num_grad_channels, device=self.remote_device, name=name)

    return self.readout_mods[name]

  @utils.create_variables_in_class_scope
  def low_rank_readout(self, name, h1, h2, psd=False):
    BS = h1.shape.as_list()[0]
    r_t = self.get_readout_mod(name + '_top')(h1)
    if psd:
      r_b = r_t
    else:
      r_b = self.get_readout_mod(name + '_bottom')(h2)
    return tf.reduce_mean(tf.matmul(r_b, r_t, transpose_a=True), axis=0) / BS

  @snt.reuse_variables
  def to_delta_size(self, h):
    with tf.device(self.remote_device):
      mod = snt.Linear(self.delta_dim)
      return snt.BatchApply(mod)(h)

  @snt.reuse_variables
  def initial_state(self, variables):
    """The inner optimization state.

    Args:
      variables: list of tf.Variable
        list of variables to get the initial state of.
    Returns:
      opt_state: OptState
    """

    with tf.device(self.local_device):
      initial_opt_state = self.opt.get_state(variables)

    return OptState(
        variables=variables, opt_state=initial_opt_state, index=tf.constant(0))

  @snt.reuse_variables
  def compute_next_state(self, grads, learning_rate, cur_state,
                         cur_transformer):

    summaries = []
    with tf.device(self.local_device):
      with tf.control_dependencies(summaries):
        new_vars, new_state = self.opt.compute_updates(
            cur_state.variables, grads, learning_rate, cur_state.opt_state)
        pass

    return OptState(
        variables=tuple(new_vars),
        opt_state=new_state,
        index=cur_state.index + 1)

  def assign_state(self, base_model, next_state):
    var_ups = [
        v.assign(nv) for v, nv in utils.eqzip(base_model.get_variables(),
                                              next_state.variables)
    ]

    opt_ups = self.opt.assign_state(next_state.opt_state)

    return tf.group(opt_ups, *var_ups)

  def local_variables(self):
    return list(self.opt.get_variables())

  def remote_variables(self):
    train = list(
        snt.get_variables_in_module(self, tf.GraphKeys.TRAINABLE_VARIABLES))
    train += list(
        snt.get_variables_in_module(self,
                                    tf.GraphKeys.MOVING_AVERAGE_VARIABLES))
    return train


class MoreLocalWeightUpdateWLearner(snt.AbstractModule):
  """The BaseModel that the UnsupervisedUpdateRule acts on.
  """

  def __init__(self,
               remote_device,
               local_device,
               inner_size=128,
               output_size=32,
               n_layers=4,
               shuffle_input=True,
               activation_fn=tf.nn.relu,
               identical_updates=True,
               **kwargs):
    self.local_device = local_device
    self.remote_device = remote_device
    self.inner_size = inner_size
    self.n_layers = n_layers
    self.shuffle_input = shuffle_input
    self.activation_fn = activation_fn
    self.identical_updates = identical_updates

    self.output_size = output_size
    if output_size == None:
      self.output_size = inner_size

    self.shuffle_ind = None

    super(MoreLocalWeightUpdateWLearner, self).__init__(
        name='LocalWeightUpdateWLearner', **kwargs)

  @snt.reuse_variables
  def get_shuffle_ind(self, size):
    if self.shuffle_ind is None:
      # put the shuffle in tf memory to make the eval jobs
      # re-entrant.
      shuffle_ind_val = np.random.permutation(size)
      shuffle_ind = tf.get_variable(
          name='shuffle_ind', dtype=tf.int64, initializer=shuffle_ind_val)
      unshuffle_ind = tf.scatter_nd(
          tf.reshape(shuffle_ind, [-1, 1]), tf.range(size), [size])

    return shuffle_ind, unshuffle_ind

  def _build(self, batch):
    image = batch.image
    x0 = snt.BatchFlatten()(image)
    if self.shuffle_input:
      size = x0.shape.as_list()[1]
      shuffle_ind, unshuffle_ind = self.get_shuffle_ind(size)
      x0 = tf.gather(x0, shuffle_ind, axis=1)

    xs = [x0]
    mods = []
    zs = []
    init = {}

    for i in range(self.n_layers):
      mod = common.LinearBatchNorm(
          self.inner_size, activation_fn=self.activation_fn)
      z, x = mod(xs[i])
      xs.append(x)
      zs.append(z)
      mods.append(mod)

    mod = common.LinearBatchNorm(
        self.output_size, activation_fn=self.activation_fn)
    z, x = mod(xs[-1])
    mods.append(mod)

    xs.append(x)
    zs.append(z)

    embedding_x = xs[-1]

    # make a random set of backward mods
    backward_mods = []
    for i, (x, x_p1) in enumerate(zip(xs[0:-1], xs[1:])):
      m = common.LinearBatchNorm(
          x_p1.shape.as_list()[1], activation_fn=tf.identity)
      _ = m(x)
      backward_mods.append(m)

    shape = image.shape.as_list()[1:4]

    for mods_p, prefix in [(mods, 'forward'), (backward_mods, 'backward')]:
      if self.shuffle_input:
        unshuf_w = tf.gather(mods_p[0].w, unshuffle_ind, axis=0)
      else:
        unshuf_w = mods_p[0].w
      img = summary_utils.first_layer_weight_image(unshuf_w, shape)
      tf.summary.image(prefix + '_w0_receptive_field', img)

      for i, m in enumerate(mods_p[0:]):
        img = summary_utils.inner_layer_weight_image(m.w)
        tf.summary.image(prefix + '_w%d' % (i + 1), img)

    img = summary_utils.sorted_images(image, batch.label_onehot)
    tf.summary.image('inputs', img)

    # log out pre-activations and activations
    for all_vis, base_name in [(xs, 'x'), (zs, 'z')]:
      for i, x_vis in enumerate(all_vis):
        img = summary_utils.activation_image(x_vis, batch.label_onehot)
        tf.summary.image('%s%d' % (base_name, i), img)

    embedding_x = tf.identity(embedding_x)

    outputs = BaseModelOutputs(
        xs=xs, zs=zs, mods=mods, batch=batch, backward_mods=backward_mods)

    return embedding_x, outputs

  def compute_next_h_d(self, meta_opt, w_bot, w_top, bias, x, z, d, backward_w):
    """ Propogate error back down the network while computing hidden state.
    """
    if z is None:
      z = x

    h = meta_opt.compute_h(x, z, d, bias, w_bot,
                           w_top)  # [bs x 60 x h_channels]

    # compute the next d
    delta = meta_opt.next_delta(z, h, d)

    if backward_w is not None:

      def delta_matmul(w, delta):
        d = tf.transpose(delta, [0, 2, 1])  # [bs x delta_channels x n_units)
        d = snt.BatchApply(lambda x: tf.matmul(x, w, transpose_b=True))(d)
        d = tf.transpose(d, [0, 2, 1])
        return d

      # replace the "backward pass" with a random matrix.
      d = delta_matmul(backward_w, delta)  # [bs x 60 x delta_channels]
      var = tf.reduce_mean(tf.square(d), [2], keepdims=True)
      d = d * tf.rsqrt(1e-6 + var)

    return h, d

  def weight_change_for_layer(self, meta_opt, l_idx, w_base, b_base, upper_h,
                              lower_h, upper_x, lower_x, prefix, include_bias):
    """Compute the change in weights for each layer.
    This computes something roughly analagous to a gradient.
    """
    reduce_upper_h = upper_h
    reduce_lower_h = lower_h

    BS = lower_x.shape.as_list()[0]

    change_w_terms = dict()

    # initial weight value normalized
    # normalize the weights per receptive-field, rather than per-matrix
    weight_scale = tf.rsqrt(
        tf.reduce_mean(w_base**2, axis=0, keepdims=True) + 1e-6)
    w_base *= weight_scale

    change_w_terms['w_base'] = w_base

    # this will act to decay larger weights towards zero
    change_w_terms['large_decay'] = w_base**2 * tf.sign(w_base)

    # term based on activations
    ux0 = upper_x - tf.reduce_mean(upper_x, axis=0, keepdims=True)
    uxs0 = ux0 * tf.rsqrt(tf.reduce_mean(ux0**2, axis=0, keepdims=True) + 1e-6)
    change_U = tf.matmul(uxs0, uxs0, transpose_a=True) / BS
    change_U /= tf.sqrt(float(change_U.shape.as_list()[0]))

    cw = tf.matmul(w_base, change_U)
    cw_scale = tf.rsqrt(tf.reduce_mean(cw**2 + 1e-8))
    cw *= cw_scale
    change_w_terms['decorr_x'] = cw

    # hebbian term
    lx0 = lower_x - tf.reduce_mean(lower_x, axis=0, keepdims=True)
    lxs0 = lx0 * tf.rsqrt(tf.reduce_mean(lx0**2, axis=0, keepdims=True) + 1e-6)
    cw = tf.matmul(lxs0, uxs0, transpose_a=True) / BS
    change_w_terms['hebb'] = -cw

    # 0th order term
    w_term = meta_opt.low_rank_readout(prefix + 'weight_readout_0', upper_h,
                                       lower_h)
    change_w_terms['0_order'] = w_term

    # # rbf term (weight update scaled by distance from 0)
    w_term = meta_opt.low_rank_readout(prefix + 'weight_readout_rbf',
                                       reduce_upper_h, reduce_lower_h)
    change_w_terms['rbf'] = tf.exp(-w_base**2) * w_term

    # 1st order term (weight dependent update to weights)
    w_term = meta_opt.low_rank_readout(prefix + 'weight_readout_1',
                                       reduce_upper_h, reduce_lower_h)
    change_w_terms['1_order'] = w_base * w_term

    # more terms based on single layer readouts.
    for update_type in ['lin', 'sqr']:
      for h_source, h_source_name in [(reduce_upper_h, 'upper'),
                                      (reduce_lower_h, 'lower')]:
        structures = ['symm']
        if update_type == 'lin' and h_source_name == 'upper':
          structures += ['psd']
        for structure in structures:
          name = update_type + '_' + h_source_name + '_' + structure
          if structure == 'symm':
            change_U = meta_opt.low_rank_readout(prefix + name, h_source,
                                                 h_source)
            change_U = (change_U + tf.transpose(change_U)) / tf.sqrt(2.)
            change_U = tf.matrix_set_diag(change_U,
                                          tf.zeros(
                                              [change_U.shape.as_list()[0]]))
          elif structure == 'psd':
            change_U = meta_opt.low_rank_readout(
                prefix + name, h_source, None, psd=True)
          else:
            assert False
          change_U /= tf.sqrt(float(change_U.shape.as_list()[0]))

          if update_type == 'lin':
            sign_multiplier = tf.ones_like(w_base)
            w_base_l = w_base
          elif update_type == 'sqr':
            sign_multiplier = tf.sign(w_base)
            w_base_l = tf.sqrt(1. + w_base**2) - 1.

          if h_source_name == 'upper':
            cw = tf.matmul(w_base_l, change_U)  # [N^l-1 x N^l]
          elif h_source_name == 'lower':
            cw = tf.matmul(change_U, w_base_l)
          change_w_terms[name] = cw * sign_multiplier


    if prefix == 'forward':
      change_w = meta_opt.merge_change_w_forward(
          change_w_terms, global_prefix=prefix, prefix='l%d' % l_idx)
    elif prefix == 'backward':
      change_w = meta_opt.merge_change_w_backward(
          change_w_terms, global_prefix=prefix, prefix='l%d' % l_idx)
    else:
      assert (False)

    if not include_bias:
      return change_w

    change_b = tf.reduce_mean(meta_opt.bias_readout(upper_h), [0])

    # force nonlinearities to be exercised -- biases can't all be increased without bound
    change_b_mean = tf.reduce_mean(change_b)
    offset = -tf.nn.relu(-change_b_mean)
    change_b -= offset

    var = tf.reduce_mean(tf.square(change_b), [0], keepdims=True)
    change_b = (change_b) / tf.sqrt(0.5 + var)
    return change_w, change_b

  def compute_next_state(self, outputs, meta_opt, previous_state):
    zs = outputs.zs
    xs = outputs.xs
    batch = outputs.batch
    mods = outputs.mods
    backward_mods = outputs.backward_mods
    variables = self.get_variables()

    rev_mods = mods[::-1]
    rev_backward_mods = backward_mods[::-1]
    rev_xs = xs[::-1]
    rev_zs = zs[::-1] + [None]

    to_top = xs[-1]

    # variables that change in the loop
    hs = []
    d = meta_opt.compute_top_delta(to_top)  # [bs x 32 x delta_channels]

    iterator = utils.eqzip(rev_backward_mods + [None], rev_mods + [None],
                           [None] + rev_mods, rev_xs, rev_zs)
    for (backward_mod, lower_mod, upper_mod, x, z) in iterator:
      w_bot = None
      if not lower_mod is None:
        w_bot = previous_state.variables[variables.index(lower_mod.w)]
      w_top = None
      if not upper_mod is None:
        w_top = previous_state.variables[variables.index(upper_mod.w)]
      backward_w = None
      if backward_mod is not None:
        backward_w = previous_state.variables[variables.index(backward_mod.w)]
      if lower_mod is not None:
        bias = previous_state.variables[variables.index(lower_mod.b)]
      else:
        bias = tf.zeros([x.shape[1]])

      h, d = self.compute_next_h_d(
          meta_opt=meta_opt,
          w_bot=w_bot,
          w_top=w_top,
          bias=bias,
          backward_w=backward_w,
          x=x,
          z=z,
          d=d)
      hs.append(h)

    w_forward_var_idx = [variables.index(mod.w) for mod in rev_mods]
    w_backward_var_idx = [variables.index(mod.w) for mod in rev_backward_mods]
    b_var_idx = [variables.index(mod.b) for mod in rev_mods]

    # storage location for outputs of below loop
    grads = [None for _ in previous_state.variables]

    # over-ride learning rate for perturbation variables
    learning_rate = [None for _ in previous_state.variables]

    # This is a map -- no state is shared cross loop
    for l_idx, w_forward_idx, w_backward_idx, b_idx, upper_h, lower_h, lower_x, upper_x in utils.eqzip(
        range(len(w_forward_var_idx)), w_forward_var_idx, w_backward_var_idx,
        b_var_idx, hs[:-1], hs[1:], xs[::-1][1:], xs[::-1][:-1]):

      b_base = previous_state.variables[b_idx]
      change_w_forward, change_b = self.weight_change_for_layer(
          meta_opt=meta_opt,
          l_idx=l_idx,
          w_base=previous_state.variables[w_forward_idx],
          b_base=b_base,
          upper_h=upper_h,
          lower_h=lower_h,
          upper_x=upper_x,
          lower_x=lower_x,
          prefix='forward',
          include_bias=True)

      if self.identical_updates:
        change_w_backward = change_w_forward
      else:
        change_w_backward = self.weight_change_for_layer(
            meta_opt=meta_opt,
            l_idx=l_idx,
            w_base=previous_state.variables[w_backward_idx],
            b_base=b_base,
            upper_h=upper_h,
            lower_h=lower_h,
            upper_x=upper_x,
            lower_x=lower_x,
            prefix='backward',
            include_bias=False)

      grads[w_forward_idx] = change_w_forward

      grads[w_backward_idx] = change_w_backward

      grads[b_idx] = change_b

    cur_transformer = common.transformer_at_state(self,
                                                  previous_state.variables)
    next_state = meta_opt.compute_next_state(
        grads,
        learning_rate=learning_rate,
        cur_state=previous_state,
        cur_transformer=lambda x: cur_transformer(x)[0])
    return next_state

  def initial_state(self, meta_opt):
    return meta_opt.initial_state(self.get_variables())
