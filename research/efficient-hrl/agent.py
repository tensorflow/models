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

"""A UVF agent.
"""

import tensorflow as tf
import gin.tf
from agents import ddpg_agent
# pylint: disable=unused-import
import cond_fn
from utils import utils as uvf_utils
from context import gin_imports
# pylint: enable=unused-import
slim = tf.contrib.slim


@gin.configurable
class UvfAgentCore(object):
  """Defines basic functions for UVF agent. Must be inherited with an RL agent.

  Used as lower-level agent.
  """

  def __init__(self,
               observation_spec,
               action_spec,
               tf_env,
               tf_context,
               step_cond_fn=cond_fn.env_transition,
               reset_episode_cond_fn=cond_fn.env_restart,
               reset_env_cond_fn=cond_fn.false_fn,
               metrics=None,
               **base_agent_kwargs):
    """Constructs a UVF agent.

    Args:
      observation_spec: A TensorSpec defining the observations.
      action_spec: A BoundedTensorSpec defining the actions.
      tf_env: A Tensorflow environment object.
      tf_context: A Context class.
      step_cond_fn: A function indicating whether to increment the num of steps.
      reset_episode_cond_fn: A function indicating whether to restart the
      episode, resampling the context.
      reset_env_cond_fn: A function indicating whether to perform a manual reset
      of the environment.
      metrics: A list of functions that evaluate metrics of the agent.
      **base_agent_kwargs: A dictionary of parameters for base RL Agent.
    Raises:
      ValueError: If 'dqda_clipping' is < 0.
    """
    self._step_cond_fn = step_cond_fn
    self._reset_episode_cond_fn = reset_episode_cond_fn
    self._reset_env_cond_fn = reset_env_cond_fn
    self.metrics = metrics

    # expose tf_context methods
    self.tf_context = tf_context(tf_env=tf_env)
    self.set_replay = self.tf_context.set_replay
    self.sample_contexts = self.tf_context.sample_contexts
    self.compute_rewards = self.tf_context.compute_rewards
    self.gamma_index = self.tf_context.gamma_index
    self.context_specs = self.tf_context.context_specs
    self.context_as_action_specs = self.tf_context.context_as_action_specs
    self.init_context_vars = self.tf_context.create_vars

    self.env_observation_spec = observation_spec[0]
    merged_observation_spec = (uvf_utils.merge_specs(
        (self.env_observation_spec,) + self.context_specs),)
    self._context_vars = dict()
    self._action_vars = dict()

    self.BASE_AGENT_CLASS.__init__(
        self,
        observation_spec=merged_observation_spec,
        action_spec=action_spec,
        **base_agent_kwargs
    )

  def set_meta_agent(self, agent=None):
    self._meta_agent = agent

  @property
  def meta_agent(self):
    return self._meta_agent

  def actor_loss(self, states, actions, rewards, discounts,
                 next_states):
    """Returns the next action for the state.

    Args:
      state: A [num_state_dims] tensor representing a state.
      context: A list of [num_context_dims] tensor representing a context.
    Returns:
      A [num_action_dims] tensor representing the action.
    """
    return self.BASE_AGENT_CLASS.actor_loss(self, states)

  def action(self, state, context=None):
    """Returns the next action for the state.

    Args:
      state: A [num_state_dims] tensor representing a state.
      context: A list of [num_context_dims] tensor representing a context.
    Returns:
      A [num_action_dims] tensor representing the action.
    """
    merged_state = self.merged_state(state, context)
    return self.BASE_AGENT_CLASS.action(self, merged_state)

  def actions(self, state, context=None):
    """Returns the next action for the state.

    Args:
      state: A [-1, num_state_dims] tensor representing a state.
      context: A list of [-1, num_context_dims] tensor representing a context.
    Returns:
      A [-1, num_action_dims] tensor representing the action.
    """
    merged_states = self.merged_states(state, context)
    return self.BASE_AGENT_CLASS.actor_net(self, merged_states)

  def log_probs(self, states, actions, state_reprs, contexts=None):
    assert contexts is not None
    batch_dims = [tf.shape(states)[0], tf.shape(states)[1]]
    contexts = self.tf_context.context_multi_transition_fn(
        contexts, states=tf.to_float(state_reprs))

    flat_states = tf.reshape(states,
                             [batch_dims[0] * batch_dims[1], states.shape[-1]])
    flat_contexts = [tf.reshape(tf.cast(context, states.dtype),
                                [batch_dims[0] * batch_dims[1], context.shape[-1]])
                     for context in contexts]
    flat_pred_actions = self.actions(flat_states, flat_contexts)
    pred_actions = tf.reshape(flat_pred_actions,
                              batch_dims + [flat_pred_actions.shape[-1]])

    error = tf.square(actions - pred_actions)
    spec_range = (self._action_spec.maximum - self._action_spec.minimum) / 2
    normalized_error = error / tf.constant(spec_range) ** 2
    return -normalized_error

  @gin.configurable('uvf_add_noise_fn')
  def add_noise_fn(self, action_fn, stddev=1.0, debug=False,
                   clip=True, global_step=None):
    """Returns the action_fn with additive Gaussian noise.

    Args:
      action_fn: A callable(`state`, `context`) which returns a
        [num_action_dims] tensor representing a action.
      stddev: stddev for the Ornstein-Uhlenbeck noise.
      debug: Print debug messages.
    Returns:
      A [num_action_dims] action tensor.
    """
    if global_step is not None:
      stddev *= tf.maximum(  # Decay exploration during training.
          tf.train.exponential_decay(1.0, global_step, 1e6, 0.8), 0.5)
    def noisy_action_fn(state, context=None):
      """Noisy action fn."""
      action = action_fn(state, context)
      if debug:
        action = uvf_utils.tf_print(
            action, [action],
            message='[add_noise_fn] pre-noise action',
            first_n=100)
      noise_dist = tf.distributions.Normal(tf.zeros_like(action),
                                           tf.ones_like(action) * stddev)
      noise = noise_dist.sample()
      action += noise
      if debug:
        action = uvf_utils.tf_print(
            action, [action],
            message='[add_noise_fn] post-noise action',
            first_n=100)
      if clip:
        action = uvf_utils.clip_to_spec(action, self._action_spec)
      return action
    return noisy_action_fn

  def merged_state(self, state, context=None):
    """Returns the merged state from the environment state and contexts.

    Args:
      state: A [num_state_dims] tensor representing a state.
      context: A list of [num_context_dims] tensor representing a context.
        If None, use the internal context.
    Returns:
      A [num_merged_state_dims] tensor representing the merged state.
    """
    if context is None:
      context = list(self.context_vars)
    state = tf.concat([state,] + context, axis=-1)
    self._validate_states(self._batch_state(state))
    return state

  def merged_states(self, states, contexts=None):
    """Returns the batch merged state from the batch env state and contexts.

    Args:
      states: A [batch_size, num_state_dims] tensor representing a batch
        of states.
      contexts: A list of [batch_size, num_context_dims] tensor
        representing a batch of contexts. If None,
        use the internal context.
    Returns:
      A [batch_size, num_merged_state_dims] tensor representing the batch
        of merged states.
    """
    if contexts is None:
      contexts = [tf.tile(tf.expand_dims(context, axis=0),
                          (tf.shape(states)[0], 1)) for
                  context in self.context_vars]
    states = tf.concat([states,] + contexts, axis=-1)
    self._validate_states(states)
    return states

  def unmerged_states(self, merged_states):
    """Returns the batch state and contexts from the batch merged state.

    Args:
      merged_states: A [batch_size, num_merged_state_dims] tensor
        representing a batch of merged states.
    Returns:
      A [batch_size, num_state_dims] tensor and a list of
        [batch_size, num_context_dims] tensors representing the batch state
        and contexts respectively.
    """
    self._validate_states(merged_states)
    num_state_dims = self.env_observation_spec.shape.as_list()[0]
    num_context_dims_list = [c.shape.as_list()[0] for c in self.context_specs]
    states = merged_states[:, :num_state_dims]
    contexts = []
    i = num_state_dims
    for num_context_dims in num_context_dims_list:
      contexts.append(merged_states[:, i: i+num_context_dims])
      i += num_context_dims
    return states, contexts

  def sample_random_actions(self, batch_size=1):
    """Return random actions.

    Args:
      batch_size: Batch size.
    Returns:
      A [batch_size, num_action_dims] tensor representing the batch of actions.
    """
    actions = tf.concat(
        [
            tf.random_uniform(
                shape=(batch_size, 1),
                minval=self._action_spec.minimum[i],
                maxval=self._action_spec.maximum[i])
            for i in range(self._action_spec.shape[0].value)
        ],
        axis=1)
    return actions

  def clip_actions(self, actions):
    """Clip actions to spec.

    Args:
      actions: A [batch_size, num_action_dims] tensor representing
      the batch of actions.
    Returns:
      A [batch_size, num_action_dims] tensor representing the batch
      of clipped actions.
    """
    actions = tf.concat(
        [
            tf.clip_by_value(
                actions[:, i:i+1],
                self._action_spec.minimum[i],
                self._action_spec.maximum[i])
            for i in range(self._action_spec.shape[0].value)
        ],
        axis=1)
    return actions

  def mix_contexts(self, contexts, insert_contexts, indices):
    """Mix two contexts based on indices.

    Args:
      contexts: A list of [batch_size, num_context_dims] tensor representing
      the batch of contexts.
      insert_contexts: A list of [batch_size, num_context_dims] tensor
      representing the batch of contexts to be inserted.
      indices: A list of a list of integers denoting indices to replace.
    Returns:
      A list of resulting contexts.
    """
    if indices is None: indices = [[]] * len(contexts)
    assert len(contexts) == len(indices)
    assert all([spec.shape.ndims == 1 for spec in self.context_specs])
    mix_contexts = []
    for contexts_, insert_contexts_, indices_, spec in zip(
        contexts, insert_contexts, indices, self.context_specs):
      mix_contexts.append(
          tf.concat(
              [
                  insert_contexts_[:, i:i + 1] if i in indices_ else
                  contexts_[:, i:i + 1] for i in range(spec.shape.as_list()[0])
              ],
              axis=1))
    return mix_contexts

  def begin_episode_ops(self, mode, action_fn=None, state=None):
    """Returns ops that reset agent at beginning of episodes.

    Args:
      mode: a string representing the mode=[train, explore, eval].
    Returns:
      A list of ops.
    """
    all_ops = []
    for _, action_var in sorted(self._action_vars.items()):
      sample_action = self.sample_random_actions(1)[0]
      all_ops.append(tf.assign(action_var, sample_action))
    all_ops += self.tf_context.reset(mode=mode, agent=self._meta_agent,
                                     action_fn=action_fn, state=state)
    return all_ops

  def cond_begin_episode_op(self, cond, input_vars, mode, meta_action_fn):
    """Returns op that resets agent at beginning of episodes.

    A new episode is begun if the cond op evalues to `False`.

    Args:
      cond: a Boolean tensor variable.
      input_vars: A list of tensor variables.
      mode: a string representing the mode=[train, explore, eval].
    Returns:
      Conditional begin op.
    """
    (state, action, reward, next_state,
     state_repr, next_state_repr) = input_vars
    def continue_fn():
      """Continue op fn."""
      items = [state, action, reward, next_state,
               state_repr, next_state_repr] + list(self.context_vars)
      batch_items = [tf.expand_dims(item, 0) for item in items]
      (states, actions, rewards, next_states,
       state_reprs, next_state_reprs) = batch_items[:6]
      context_reward = self.compute_rewards(
          mode, state_reprs, actions, rewards, next_state_reprs,
          batch_items[6:])[0][0]
      context_reward = tf.cast(context_reward, dtype=reward.dtype)
      if self.meta_agent is not None:
        meta_action = tf.concat(self.context_vars, -1)
        items = [state, meta_action, reward, next_state,
                 state_repr, next_state_repr] + list(self.meta_agent.context_vars)
        batch_items = [tf.expand_dims(item, 0) for item in items]
        (states, meta_actions, rewards, next_states,
         state_reprs, next_state_reprs) = batch_items[:6]
        meta_reward = self.meta_agent.compute_rewards(
            mode, states, meta_actions, rewards,
            next_states, batch_items[6:])[0][0]
        meta_reward = tf.cast(meta_reward, dtype=reward.dtype)
      else:
        meta_reward = tf.constant(0, dtype=reward.dtype)

      with tf.control_dependencies([context_reward, meta_reward]):
        step_ops = self.tf_context.step(mode=mode, agent=self._meta_agent,
                                        state=state,
                                        next_state=next_state,
                                        state_repr=state_repr,
                                        next_state_repr=next_state_repr,
                                        action_fn=meta_action_fn)
      with tf.control_dependencies(step_ops):
        context_reward, meta_reward = map(tf.identity, [context_reward, meta_reward])
      return context_reward, meta_reward
    def begin_episode_fn():
      """Begin op fn."""
      begin_ops = self.begin_episode_ops(mode=mode, action_fn=meta_action_fn, state=state)
      with tf.control_dependencies(begin_ops):
        return tf.zeros_like(reward), tf.zeros_like(reward)
    with tf.control_dependencies(input_vars):
      cond_begin_episode_op = tf.cond(cond, continue_fn, begin_episode_fn)
    return cond_begin_episode_op

  def get_env_base_wrapper(self, env_base, **begin_kwargs):
    """Create a wrapper around env_base, with agent-specific begin/end_episode.

    Args:
      env_base: A python environment base.
      **begin_kwargs: Keyword args for begin_episode_ops.
    Returns:
      An object with begin_episode() and end_episode().
    """
    begin_ops = self.begin_episode_ops(**begin_kwargs)
    return uvf_utils.get_contextual_env_base(env_base, begin_ops)

  def init_action_vars(self, name, i=None):
    """Create and return a tensorflow Variable holding an action.

    Args:
      name: Name of the variables.
      i: Integer id.
    Returns:
      A [num_action_dims] tensor.
    """
    if i is not None:
      name += '_%d' % i
    assert name not in self._action_vars, ('Conflict! %s is already '
                                           'initialized.') % name
    self._action_vars[name] = tf.Variable(
        self.sample_random_actions(1)[0], name='%s_action' % (name))
    self._validate_actions(tf.expand_dims(self._action_vars[name], 0))
    return self._action_vars[name]

  @gin.configurable('uvf_critic_function')
  def critic_function(self, critic_vals, states, critic_fn=None):
    """Computes q values based on outputs from the critic net.

    Args:
      critic_vals: A tf.float32 [batch_size, ...] tensor representing outputs
        from the critic net.
      states: A [batch_size, num_state_dims] tensor representing a batch
        of states.
      critic_fn: A callable that process outputs from critic_net and
        outputs a [batch_size] tensor representing q values.
    Returns:
      A tf.float32 [batch_size] tensor representing q values.
    """
    if critic_fn is not None:
      env_states, contexts = self.unmerged_states(states)
      critic_vals = critic_fn(critic_vals, env_states, contexts)
    critic_vals.shape.assert_has_rank(1)
    return critic_vals

  def get_action_vars(self, key):
    return self._action_vars[key]

  def get_context_vars(self, key):
    return self.tf_context.context_vars[key]

  def step_cond_fn(self, *args):
    return self._step_cond_fn(self, *args)

  def reset_episode_cond_fn(self, *args):
    return self._reset_episode_cond_fn(self, *args)

  def reset_env_cond_fn(self, *args):
    return self._reset_env_cond_fn(self, *args)

  @property
  def context_vars(self):
    return self.tf_context.vars


@gin.configurable
class MetaAgentCore(UvfAgentCore):
  """Defines basic functions for UVF Meta-agent. Must be inherited with an RL agent.

  Used as higher-level agent.
  """

  def __init__(self,
               observation_spec,
               action_spec,
               tf_env,
               tf_context,
               sub_context,
               step_cond_fn=cond_fn.env_transition,
               reset_episode_cond_fn=cond_fn.env_restart,
               reset_env_cond_fn=cond_fn.false_fn,
               metrics=None,
               actions_reg=0.,
               k=2,
               **base_agent_kwargs):
    """Constructs a Meta agent.

    Args:
      observation_spec: A TensorSpec defining the observations.
      action_spec: A BoundedTensorSpec defining the actions.
      tf_env: A Tensorflow environment object.
      tf_context: A Context class.
      step_cond_fn: A function indicating whether to increment the num of steps.
      reset_episode_cond_fn: A function indicating whether to restart the
      episode, resampling the context.
      reset_env_cond_fn: A function indicating whether to perform a manual reset
      of the environment.
      metrics: A list of functions that evaluate metrics of the agent.
      **base_agent_kwargs: A dictionary of parameters for base RL Agent.
    Raises:
      ValueError: If 'dqda_clipping' is < 0.
    """
    self._step_cond_fn = step_cond_fn
    self._reset_episode_cond_fn = reset_episode_cond_fn
    self._reset_env_cond_fn = reset_env_cond_fn
    self.metrics = metrics
    self._actions_reg = actions_reg
    self._k = k

    # expose tf_context methods
    self.tf_context = tf_context(tf_env=tf_env)
    self.sub_context = sub_context(tf_env=tf_env)
    self.set_replay = self.tf_context.set_replay
    self.sample_contexts = self.tf_context.sample_contexts
    self.compute_rewards = self.tf_context.compute_rewards
    self.gamma_index = self.tf_context.gamma_index
    self.context_specs = self.tf_context.context_specs
    self.context_as_action_specs = self.tf_context.context_as_action_specs
    self.sub_context_as_action_specs = self.sub_context.context_as_action_specs
    self.init_context_vars = self.tf_context.create_vars

    self.env_observation_spec = observation_spec[0]
    merged_observation_spec = (uvf_utils.merge_specs(
        (self.env_observation_spec,) + self.context_specs),)
    self._context_vars = dict()
    self._action_vars = dict()

    assert len(self.context_as_action_specs) == 1
    self.BASE_AGENT_CLASS.__init__(
        self,
        observation_spec=merged_observation_spec,
        action_spec=self.sub_context_as_action_specs,
        **base_agent_kwargs
    )

  @gin.configurable('meta_add_noise_fn')
  def add_noise_fn(self, action_fn, stddev=1.0, debug=False,
                   global_step=None):
    noisy_action_fn = super(MetaAgentCore, self).add_noise_fn(
        action_fn, stddev,
        clip=True, global_step=global_step)
    return noisy_action_fn

  def actor_loss(self, states, actions, rewards, discounts,
                 next_states):
    """Returns the next action for the state.

    Args:
      state: A [num_state_dims] tensor representing a state.
      context: A list of [num_context_dims] tensor representing a context.
    Returns:
      A [num_action_dims] tensor representing the action.
    """
    actions = self.actor_net(states, stop_gradients=False)
    regularizer = self._actions_reg * tf.reduce_mean(
        tf.reduce_sum(tf.abs(actions[:, self._k:]), -1), 0)
    loss = self.BASE_AGENT_CLASS.actor_loss(self, states)
    return regularizer + loss


@gin.configurable
class UvfAgent(UvfAgentCore, ddpg_agent.TD3Agent):
  """A DDPG agent with UVF.
  """
  BASE_AGENT_CLASS = ddpg_agent.TD3Agent
  ACTION_TYPE = 'continuous'

  def __init__(self, *args, **kwargs):
    UvfAgentCore.__init__(self, *args, **kwargs)


@gin.configurable
class MetaAgent(MetaAgentCore, ddpg_agent.TD3Agent):
  """A DDPG meta-agent.
  """
  BASE_AGENT_CLASS = ddpg_agent.TD3Agent
  ACTION_TYPE = 'continuous'

  def __init__(self, *args, **kwargs):
    MetaAgentCore.__init__(self, *args, **kwargs)


@gin.configurable()
def state_preprocess_net(
    states,
    num_output_dims=2,
    states_hidden_layers=(100,),
    normalizer_fn=None,
    activation_fn=tf.nn.relu,
    zero_time=True,
    images=False):
  """Creates a simple feed forward net for embedding states.
  """
  with slim.arg_scope(
      [slim.fully_connected],
      activation_fn=activation_fn,
      normalizer_fn=normalizer_fn,
      weights_initializer=slim.variance_scaling_initializer(
          factor=1.0/3.0, mode='FAN_IN', uniform=True)):

    states_shape = tf.shape(states)
    states_dtype = states.dtype
    states = tf.to_float(states)
    if images:  # Zero-out x-y
      states *= tf.constant([0.] * 2 + [1.] * (states.shape[-1] - 2), dtype=states.dtype)
    if zero_time:
      states *= tf.constant([1.] * (states.shape[-1] - 1) + [0.], dtype=states.dtype)
    orig_states = states
    embed = states
    if states_hidden_layers:
      embed = slim.stack(embed, slim.fully_connected, states_hidden_layers,
                         scope='states')

    with slim.arg_scope([slim.fully_connected],
                        weights_regularizer=None,
                        weights_initializer=tf.random_uniform_initializer(
                            minval=-0.003, maxval=0.003)):
      embed = slim.fully_connected(embed, num_output_dims,
                                   activation_fn=None,
                                   normalizer_fn=None,
                                   scope='value')

    output = embed
    output = tf.cast(output, states_dtype)
    return output


@gin.configurable()
def action_embed_net(
    actions,
    states=None,
    num_output_dims=2,
    hidden_layers=(400, 300),
    normalizer_fn=None,
    activation_fn=tf.nn.relu,
    zero_time=True,
    images=False):
  """Creates a simple feed forward net for embedding actions.
  """
  with slim.arg_scope(
      [slim.fully_connected],
      activation_fn=activation_fn,
      normalizer_fn=normalizer_fn,
      weights_initializer=slim.variance_scaling_initializer(
          factor=1.0/3.0, mode='FAN_IN', uniform=True)):

    actions = tf.to_float(actions)
    if states is not None:
      if images:  # Zero-out x-y
        states *= tf.constant([0.] * 2 + [1.] * (states.shape[-1] - 2), dtype=states.dtype)
      if zero_time:
        states *= tf.constant([1.] * (states.shape[-1] - 1) + [0.], dtype=states.dtype)
      actions = tf.concat([actions, tf.to_float(states)], -1)

    embed = actions
    if hidden_layers:
      embed = slim.stack(embed, slim.fully_connected, hidden_layers,
                         scope='hidden')

    with slim.arg_scope([slim.fully_connected],
                        weights_regularizer=None,
                        weights_initializer=tf.random_uniform_initializer(
                            minval=-0.003, maxval=0.003)):
      embed = slim.fully_connected(embed, num_output_dims,
                                   activation_fn=None,
                                   normalizer_fn=None,
                                   scope='value')
      if num_output_dims == 1:
        return embed[:, 0, ...]
      else:
        return embed


def huber(x, kappa=0.1):
  return (0.5 * tf.square(x) * tf.to_float(tf.abs(x) <= kappa) +
          kappa * (tf.abs(x) - 0.5 * kappa) * tf.to_float(tf.abs(x) > kappa)
          ) / kappa


@gin.configurable()
class StatePreprocess(object):
  STATE_PREPROCESS_NET_SCOPE = 'state_process_net'
  ACTION_EMBED_NET_SCOPE = 'action_embed_net'

  def __init__(self, trainable=False,
               state_preprocess_net=lambda states: states,
               action_embed_net=lambda actions, *args, **kwargs: actions,
               ndims=None):
    self.trainable = trainable
    self._scope = tf.get_variable_scope().name
    self._ndims = ndims
    self._state_preprocess_net = tf.make_template(
        self.STATE_PREPROCESS_NET_SCOPE, state_preprocess_net,
        create_scope_now_=True)
    self._action_embed_net = tf.make_template(
        self.ACTION_EMBED_NET_SCOPE, action_embed_net,
        create_scope_now_=True)

  def __call__(self, states):
    batched = states.get_shape().ndims != 1
    if not batched:
      states = tf.expand_dims(states, 0)
    embedded = self._state_preprocess_net(states)
    if self._ndims is not None:
      embedded = embedded[..., :self._ndims]
    if not batched:
      return embedded[0]
    return embedded

  def loss(self, states, next_states, low_actions, low_states):
    batch_size = tf.shape(states)[0]
    d = int(low_states.shape[1])
    # Sample indices into meta-transition to train on.
    probs = 0.99 ** tf.range(d, dtype=tf.float32)
    probs *= tf.constant([1.0] * (d - 1) + [1.0 / (1 - 0.99)],
                         dtype=tf.float32)
    probs /= tf.reduce_sum(probs)
    index_dist = tf.distributions.Categorical(probs=probs, dtype=tf.int64)
    indices = index_dist.sample(batch_size)
    batch_size = tf.cast(batch_size, tf.int64)
    next_indices = tf.concat(
        [tf.range(batch_size, dtype=tf.int64)[:, None],
         (1 + indices[:, None]) % d], -1)
    new_next_states = tf.where(indices < d - 1,
                               tf.gather_nd(low_states, next_indices),
                               next_states)
    next_states = new_next_states

    embed1 = tf.to_float(self._state_preprocess_net(states))
    embed2 = tf.to_float(self._state_preprocess_net(next_states))
    action_embed = self._action_embed_net(
        tf.layers.flatten(low_actions), states=states)

    tau = 2.0
    fn = lambda z: tau * tf.reduce_sum(huber(z), -1)
    all_embed = tf.get_variable('all_embed', [1024, int(embed1.shape[-1])],
                                initializer=tf.zeros_initializer())
    upd = all_embed.assign(tf.concat([all_embed[batch_size:], embed2], 0))
    with tf.control_dependencies([upd]):
      close = 1 * tf.reduce_mean(fn(embed1 + action_embed - embed2))
      prior_log_probs = tf.reduce_logsumexp(
          -fn((embed1 + action_embed)[:, None, :] - all_embed[None, :, :]),
          axis=-1) - tf.log(tf.to_float(all_embed.shape[0]))
      far = tf.reduce_mean(tf.exp(-fn((embed1 + action_embed)[1:] - embed2[:-1])
                                  - tf.stop_gradient(prior_log_probs[1:])))
      repr_log_probs = tf.stop_gradient(
          -fn(embed1 + action_embed - embed2) - prior_log_probs) / tau
    return close + far, repr_log_probs, indices

  def get_trainable_vars(self):
    return (
        slim.get_trainable_variables(
            uvf_utils.join_scope(self._scope, self.STATE_PREPROCESS_NET_SCOPE)) +
        slim.get_trainable_variables(
            uvf_utils.join_scope(self._scope, self.ACTION_EMBED_NET_SCOPE)))


@gin.configurable()
class InverseDynamics(object):
  INVERSE_DYNAMICS_NET_SCOPE = 'inverse_dynamics'

  def __init__(self, spec):
    self._spec = spec

  def sample(self, states, next_states, num_samples, orig_goals, sc=0.5):
    goal_dim = orig_goals.shape[-1]
    spec_range = (self._spec.maximum - self._spec.minimum) / 2 * tf.ones([goal_dim])
    loc = tf.cast(next_states - states, tf.float32)[:, :goal_dim]
    scale = sc * tf.tile(tf.reshape(spec_range, [1, goal_dim]),
                         [tf.shape(states)[0], 1])
    dist = tf.distributions.Normal(loc, scale)
    if num_samples == 1:
      return dist.sample()
    samples = tf.concat([dist.sample(num_samples - 2),
                         tf.expand_dims(loc, 0),
                         tf.expand_dims(orig_goals, 0)], 0)
    return uvf_utils.clip_to_spec(samples, self._spec)
