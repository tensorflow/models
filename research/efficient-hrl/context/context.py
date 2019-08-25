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

"""Context for Universal Value Function agents.

A context specifies a list of contextual variables, each with
  own sampling and reward computation methods.

Examples of contextual variables include
  goal states, reward combination vectors, etc.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tf_agents import specs
import gin.tf
from utils import utils as uvf_utils


@gin.configurable
class Context(object):
  """Base context."""
  VAR_NAME = 'action'

  def __init__(self,
               tf_env,
               context_ranges=None,
               context_shapes=None,
               state_indices=None,
               variable_indices=None,
               gamma_index=None,
               settable_context=False,
               timers=None,
               samplers=None,
               reward_weights=None,
               reward_fn=None,
               random_sampler_mode='random',
               normalizers=None,
               context_transition_fn=None,
               context_multi_transition_fn=None,
               meta_action_every_n=None):
    self._tf_env = tf_env
    self.variable_indices = variable_indices
    self.gamma_index = gamma_index
    self._settable_context = settable_context
    self.timers = timers
    self._context_transition_fn = context_transition_fn
    self._context_multi_transition_fn = context_multi_transition_fn
    self._random_sampler_mode = random_sampler_mode

    # assign specs
    self._obs_spec = self._tf_env.observation_spec()
    self._context_shapes = tuple([
        shape if shape is not None else self._obs_spec.shape
        for shape in context_shapes
    ])
    self.context_specs = tuple([
        specs.TensorSpec(dtype=self._obs_spec.dtype, shape=shape)
        for shape in self._context_shapes
    ])
    if context_ranges is not None:
      self.context_ranges = context_ranges
    else:
      self.context_ranges = [None] * len(self._context_shapes)

    self.context_as_action_specs = tuple([
        specs.BoundedTensorSpec(
            shape=shape,
            dtype=(tf.float32 if self._obs_spec.dtype in
                   [tf.float32, tf.float64] else self._obs_spec.dtype),
            minimum=context_range[0],
            maximum=context_range[-1])
        for shape, context_range in zip(self._context_shapes, self.context_ranges)
    ])

    if state_indices is not None:
      self.state_indices = state_indices
    else:
      self.state_indices = [None] * len(self._context_shapes)
    if self.variable_indices is not None and self.n != len(
        self.variable_indices):
      raise ValueError(
          'variable_indices (%s) must have the same length as contexts (%s).' %
          (self.variable_indices, self.context_specs))
    assert self.n == len(self.context_ranges)
    assert self.n == len(self.state_indices)

    # assign reward/sampler fns
    self._sampler_fns = dict()
    self._samplers = dict()
    self._reward_fns = dict()

    # assign reward fns
    self._add_custom_reward_fns()
    reward_weights = reward_weights or None
    self._reward_fn = self._make_reward_fn(reward_fn, reward_weights)

    # assign samplers
    self._add_custom_sampler_fns()
    for mode, sampler_fns in samplers.items():
      self._make_sampler_fn(sampler_fns, mode)

    # create normalizers
    if normalizers is None:
      self._normalizers = [None] * len(self.context_specs)
    else:
      self._normalizers = [
          normalizer(tf.zeros(shape=spec.shape, dtype=spec.dtype))
          if normalizer is not None else None
          for normalizer, spec in zip(normalizers, self.context_specs)
      ]
    assert self.n == len(self._normalizers)

    self.meta_action_every_n = meta_action_every_n

    # create vars
    self.context_vars = {}
    self.timer_vars = {}
    self.create_vars(self.VAR_NAME)
    self.t = tf.Variable(
        tf.zeros(shape=(), dtype=tf.int32), name='num_timer_steps')

  def _add_custom_reward_fns(self):
    pass

  def _add_custom_sampler_fns(self):
    pass

  def sample_random_contexts(self, batch_size):
    """Sample random batch contexts."""
    assert self._random_sampler_mode is not None
    return self.sample_contexts(self._random_sampler_mode, batch_size)[0]

  def sample_contexts(self, mode, batch_size, state=None, next_state=None,
                      **kwargs):
    """Sample a batch of contexts.

    Args:
      mode: A string representing the mode [`train`, `explore`, `eval`].
      batch_size: Batch size.
    Returns:
      Two lists of [batch_size, num_context_dims] contexts.
    """
    contexts, next_contexts = self._sampler_fns[mode](
        batch_size, state=state, next_state=next_state,
        **kwargs)
    self._validate_contexts(contexts)
    self._validate_contexts(next_contexts)
    return contexts, next_contexts

  def compute_rewards(self, mode, states, actions, rewards, next_states,
                      contexts):
    """Compute context-based rewards.

    Args:
      mode: A string representing the mode ['uvf', 'task'].
      states: A [batch_size, num_state_dims] tensor.
      actions: A [batch_size, num_action_dims] tensor.
      rewards: A [batch_size] tensor representing unmodified rewards.
      next_states: A [batch_size, num_state_dims] tensor.
      contexts: A list of [batch_size, num_context_dims] tensors.
    Returns:
      A [batch_size] tensor representing rewards.
    """
    return self._reward_fn(states, actions, rewards, next_states,
                           contexts)

  def _make_reward_fn(self, reward_fns_list, reward_weights):
    """Returns a fn that computes rewards.

    Args:
      reward_fns_list: A fn or a list of reward fns.
      mode: A string representing the operating mode.
      reward_weights: A list of reward weights.
    """
    if not isinstance(reward_fns_list, (list, tuple)):
      reward_fns_list = [reward_fns_list]
    if reward_weights is None:
      reward_weights = [1.0] * len(reward_fns_list)
    assert len(reward_fns_list) == len(reward_weights)

    reward_fns_list = [
        self._custom_reward_fns[fn] if isinstance(fn, (str,)) else fn
        for fn in reward_fns_list
    ]

    def reward_fn(*args, **kwargs):
      """Returns rewards, discounts."""
      reward_tuples = [
          reward_fn(*args, **kwargs) for reward_fn in reward_fns_list
      ]
      rewards_list = [reward_tuple[0] for reward_tuple in reward_tuples]
      discounts_list = [reward_tuple[1] for reward_tuple in reward_tuples]
      ndims = max([r.shape.ndims for r in rewards_list])
      if ndims > 1:  # expand reward shapes to allow broadcasting
        for i in range(len(rewards_list)):
          for _ in range(rewards_list[i].shape.ndims - ndims):
            rewards_list[i] = tf.expand_dims(rewards_list[i], axis=-1)
          for _ in range(discounts_list[i].shape.ndims - ndims):
            discounts_list[i] = tf.expand_dims(discounts_list[i], axis=-1)
      rewards = tf.add_n(
          [r * tf.to_float(w) for r, w in zip(rewards_list, reward_weights)])
      discounts = discounts_list[0]
      for d in discounts_list[1:]:
        discounts *= d

      return rewards, discounts

    return reward_fn

  def _make_sampler_fn(self, sampler_cls_list, mode):
    """Returns a fn that samples a list of context vars.

    Args:
      sampler_cls_list: A list of sampler classes.
      mode: A string representing the operating mode.
    """
    if not isinstance(sampler_cls_list, (list, tuple)):
      sampler_cls_list = [sampler_cls_list]

    self._samplers[mode] = []
    sampler_fns = []
    for spec, sampler in zip(self.context_specs, sampler_cls_list):
      if isinstance(sampler, (str,)):
        sampler_fn = self._custom_sampler_fns[sampler]
      else:
        sampler_fn = sampler(context_spec=spec)
        self._samplers[mode].append(sampler_fn)
      sampler_fns.append(sampler_fn)

    def batch_sampler_fn(batch_size, state=None, next_state=None, **kwargs):
      """Sampler fn."""
      contexts_tuples = [
          sampler(batch_size, state=state, next_state=next_state, **kwargs)
          for sampler in sampler_fns]
      contexts = [c[0] for c in contexts_tuples]
      next_contexts = [c[1] for c in contexts_tuples]
      contexts = [
          normalizer.update_apply(c) if normalizer is not None else c
          for normalizer, c in zip(self._normalizers, contexts)
      ]
      next_contexts = [
          normalizer.apply(c) if normalizer is not None else c
          for normalizer, c in zip(self._normalizers, next_contexts)
      ]
      return contexts, next_contexts

    self._sampler_fns[mode] = batch_sampler_fn

  def set_env_context_op(self, context, disable_unnormalizer=False):
    """Returns a TensorFlow op that sets the environment context.

    Args:
      context: A list of context Tensor variables.
      disable_unnormalizer: Disable unnormalization.
    Returns:
      A TensorFlow op that sets the environment context.
    """
    ret_val = np.array(1.0, dtype=np.float32)
    if not self._settable_context:
      return tf.identity(ret_val)

    if not disable_unnormalizer:
      context = [
          normalizer.unapply(tf.expand_dims(c, 0))[0]
          if normalizer is not None else c
          for normalizer, c in zip(self._normalizers, context)
      ]

    def set_context_func(*env_context_values):
      tf.logging.info('[set_env_context_op] Setting gym environment context.')
      # pylint: disable=protected-access
      self.gym_env.set_context(*env_context_values)
      return ret_val
      # pylint: enable=protected-access

    with tf.name_scope('set_env_context'):
      set_op = tf.py_func(set_context_func, context, tf.float32,
                          name='set_env_context_py_func')
      set_op.set_shape([])
    return set_op

  def set_replay(self, replay):
    """Set replay buffer for samplers.

    Args:
      replay: A replay buffer.
    """
    for _, samplers in self._samplers.items():
      for sampler in samplers:
        sampler.set_replay(replay)

  def get_clip_fns(self):
    """Returns a list of clip fns for contexts.

    Returns:
      A list of fns that clip context tensors.
    """
    clip_fns = []
    for context_range in self.context_ranges:
      def clip_fn(var_, range_=context_range):
        """Clip a tensor."""
        if range_ is None:
          clipped_var = tf.identity(var_)
        elif isinstance(range_[0], (int, long, float, list, np.ndarray)):
          clipped_var = tf.clip_by_value(
              var_,
              range_[0],
              range_[1],)
        else: raise NotImplementedError(range_)
        return clipped_var
      clip_fns.append(clip_fn)
    return clip_fns

  def _validate_contexts(self, contexts):
    """Validate if contexts have right specs.

    Args:
      contexts: A list of [batch_size, num_context_dim] tensors.
    Raises:
      ValueError: If shape or dtype mismatches that of spec.
    """
    for i, (context, spec) in enumerate(zip(contexts, self.context_specs)):
      if context[0].shape != spec.shape:
        raise ValueError('contexts[%d] has invalid shape %s wrt spec shape %s' %
                         (i, context[0].shape, spec.shape))
      if context.dtype != spec.dtype:
        raise ValueError('contexts[%d] has invalid dtype %s wrt spec dtype %s' %
                         (i, context.dtype, spec.dtype))

  def context_multi_transition_fn(self, contexts, **kwargs):
    """Returns multiple future contexts starting from a batch."""
    assert self._context_multi_transition_fn
    return self._context_multi_transition_fn(contexts, None, None, **kwargs)

  def step(self, mode, agent=None, action_fn=None, **kwargs):
    """Returns [next_contexts..., next_timer] list of ops.

    Args:
      mode: a string representing the mode=[train, explore, eval].
      **kwargs: kwargs for context_transition_fn.
    Returns:
      a list of ops that set the context.
    """
    if agent is None:
      ops = []
      if self._context_transition_fn is not None:
        def sampler_fn():
          samples = self.sample_contexts(mode, 1)[0]
          return [s[0] for s in samples]
        values = self._context_transition_fn(self.vars, self.t, sampler_fn, **kwargs)
        ops += [tf.assign(var, value) for var, value in zip(self.vars, values)]
      ops.append(tf.assign_add(self.t, 1))  # increment timer
      return ops
    else:
      ops = agent.tf_context.step(mode, **kwargs)
      state = kwargs['state']
      next_state = kwargs['next_state']
      state_repr = kwargs['state_repr']
      next_state_repr = kwargs['next_state_repr']
      with tf.control_dependencies(ops):  # Step high level context before computing low level one.
        # Get the context transition function output.
        values = self._context_transition_fn(self.vars, self.t, None,
                                             state=state_repr,
                                             next_state=next_state_repr)
        # Select a new goal every C steps, otherwise use context transition.
        low_level_context = [
            tf.cond(tf.equal(self.t % self.meta_action_every_n, 0),
                    lambda: tf.cast(action_fn(next_state, context=None), tf.float32),
                    lambda: values)]
        ops = [tf.assign(var, value)
               for var, value in zip(self.vars, low_level_context)]
        with tf.control_dependencies(ops):
          return [tf.assign_add(self.t, 1)]  # increment timer
        return ops

  def reset(self, mode, agent=None, action_fn=None, state=None):
    """Returns ops that reset the context.

    Args:
      mode: a string representing the mode=[train, explore, eval].
    Returns:
      a list of ops that reset the context.
    """
    if agent is None:
      values = self.sample_contexts(mode=mode, batch_size=1)[0]
      if values is None:
        return []
      values = [value[0] for value in values]
      values[0] = uvf_utils.tf_print(
          values[0],
          values,
          message='context:reset, mode=%s' % mode,
          first_n=10,
          name='context:reset:%s' % mode)
      all_ops = []
      for _, context_vars in sorted(self.context_vars.items()):
        ops = [tf.assign(var, value) for var, value in zip(context_vars, values)]
      all_ops += ops
      all_ops.append(self.set_env_context_op(values))
      all_ops.append(tf.assign(self.t, 0))  # reset timer
      return all_ops
    else:
      ops = agent.tf_context.reset(mode)
      # NOTE: The code is currently written in such a way that the higher level
      # policy does not provide a low-level context until the second
      # observation.  Insead, we just zero-out low-level contexts.
      for key, context_vars in sorted(self.context_vars.items()):
        ops += [tf.assign(var, tf.zeros_like(var)) for var, meta_var in
                zip(context_vars, agent.tf_context.context_vars[key])]

      ops.append(tf.assign(self.t, 0))  # reset timer
      return ops

  def create_vars(self, name, agent=None):
    """Create tf variables for contexts.

    Args:
      name: Name of the variables.
    Returns:
      A list of [num_context_dims] tensors.
    """
    if agent is not None:
      meta_vars = agent.create_vars(name)
    else:
      meta_vars = {}
    assert name not in self.context_vars, ('Conflict! %s is already '
                                           'initialized.') % name
    self.context_vars[name] = tuple([
        tf.Variable(
            tf.zeros(shape=spec.shape, dtype=spec.dtype),
            name='%s_context_%d' % (name, i))
        for i, spec in enumerate(self.context_specs)
    ])
    return self.context_vars[name], meta_vars

  @property
  def n(self):
    return len(self.context_specs)

  @property
  def vars(self):
    return self.context_vars[self.VAR_NAME]

  # pylint: disable=protected-access
  @property
  def gym_env(self):
    return self._tf_env.pyenv._gym_env

  @property
  def tf_env(self):
    return self._tf_env
  # pylint: enable=protected-access
