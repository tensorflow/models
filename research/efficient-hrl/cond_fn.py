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

"""Defines many boolean functions indicating when to step and reset.
"""

import tensorflow as tf
import gin.tf


@gin.configurable
def env_transition(agent, state, action, transition_type, environment_steps,
                   num_episodes):
  """True if the transition_type is TRANSITION or FINAL_TRANSITION.

  Args:
    agent: RL agent.
    state: A [num_state_dims] tensor representing a state.
    action: Action performed.
    transition_type: Type of transition after action
    environment_steps: Number of steps performed by environment.
    num_episodes: Number of episodes.
  Returns:
    cond: Returns an op that evaluates to true if the transition type is
    not RESTARTING
  """
  del agent, state, action, num_episodes, environment_steps
  cond = tf.logical_not(transition_type)
  return cond


@gin.configurable
def env_restart(agent, state, action, transition_type, environment_steps,
                num_episodes):
  """True if the transition_type is RESTARTING.

  Args:
    agent: RL agent.
    state: A [num_state_dims] tensor representing a state.
    action: Action performed.
    transition_type: Type of transition after action
    environment_steps: Number of steps performed by environment.
    num_episodes: Number of episodes.
  Returns:
    cond: Returns an op that evaluates to true if the transition type equals
    RESTARTING.
  """
  del agent, state, action, num_episodes, environment_steps
  cond = tf.identity(transition_type)
  return cond


@gin.configurable
def every_n_steps(agent,
                  state,
                  action,
                  transition_type,
                  environment_steps,
                  num_episodes,
                  n=150):
  """True once every n steps.

  Args:
    agent: RL agent.
    state: A [num_state_dims] tensor representing a state.
    action: Action performed.
    transition_type: Type of transition after action
    environment_steps: Number of steps performed by environment.
    num_episodes: Number of episodes.
    n: Return true once every n steps.
  Returns:
    cond: Returns an op that evaluates to true if environment_steps
    equals 0 mod n. We increment the step before checking this condition, so
    we do not need to add one to environment_steps.
  """
  del agent, state, action, transition_type, num_episodes
  cond = tf.equal(tf.mod(environment_steps, n), 0)
  return cond


@gin.configurable
def every_n_episodes(agent,
                     state,
                     action,
                     transition_type,
                     environment_steps,
                     num_episodes,
                     n=2,
                     steps_per_episode=None):
  """True once every n episodes.

  Specifically, evaluates to True on the 0th step of every nth episode.
  Unlike environment_steps, num_episodes starts at 0, so we do want to add
  one to ensure it does not reset on the first call.

  Args:
    agent: RL agent.
    state: A [num_state_dims] tensor representing a state.
    action: Action performed.
    transition_type: Type of transition after action
    environment_steps: Number of steps performed by environment.
    num_episodes: Number of episodes.
    n: Return true once every n episodes.
    steps_per_episode: How many steps per episode. Needed to determine when a
    new episode starts.
  Returns:
    cond: Returns an op that evaluates to true on the last step of the episode
      (i.e. if num_episodes equals 0 mod n).
  """
  assert steps_per_episode is not None
  del agent, action, transition_type
  ant_fell = tf.logical_or(state[2] < 0.2, state[2] > 1.0)
  cond = tf.logical_and(
      tf.logical_or(
          ant_fell,
          tf.equal(tf.mod(num_episodes + 1, n), 0)),
      tf.equal(tf.mod(environment_steps, steps_per_episode), 0))
  return cond


@gin.configurable
def failed_reset_after_n_episodes(agent,
                                  state,
                                  action,
                                  transition_type,
                                  environment_steps,
                                  num_episodes,
                                  steps_per_episode=None,
                                  reset_state=None,
                                  max_dist=1.0,
                                  epsilon=1e-10):
  """Every n episodes, returns True if the reset agent fails to return.

  Specifically, evaluates to True if the distance between the state and the
  reset state is greater than max_dist at the end of the episode.

  Args:
    agent: RL agent.
    state: A [num_state_dims] tensor representing a state.
    action: Action performed.
    transition_type: Type of transition after action
    environment_steps: Number of steps performed by environment.
    num_episodes: Number of episodes.
    steps_per_episode: How many steps per episode. Needed to determine when a
    new episode starts.
    reset_state: State to which the reset controller should return.
    max_dist: Agent is considered to have successfully reset if its distance
    from the reset_state is less than max_dist.
    epsilon: small offset to ensure non-negative/zero distance.
  Returns:
    cond: Returns an op that evaluates to true if num_episodes+1 equals 0
    mod n. We add one to the num_episodes so the environment is not reset after
    the 0th step.
  """
  assert steps_per_episode is not None
  assert reset_state is not None
  del agent, state, action, transition_type, num_episodes
  dist = tf.sqrt(
      tf.reduce_sum(tf.squared_difference(state, reset_state)) + epsilon)
  cond = tf.logical_and(
      tf.greater(dist, tf.constant(max_dist)),
      tf.equal(tf.mod(environment_steps, steps_per_episode), 0))
  return cond


@gin.configurable
def q_too_small(agent,
                state,
                action,
                transition_type,
                environment_steps,
                num_episodes,
                q_min=0.5):
  """True of q is too small.

  Args:
    agent: RL agent.
    state: A [num_state_dims] tensor representing a state.
    action: Action performed.
    transition_type: Type of transition after action
    environment_steps: Number of steps performed by environment.
    num_episodes: Number of episodes.
    q_min: Returns true if the qval is less than q_min
  Returns:
    cond: Returns an op that evaluates to true if qval is less than q_min.
  """
  del transition_type, environment_steps, num_episodes
  state_for_reset_agent = tf.stack(state[:-1], tf.constant([0], dtype=tf.float))
  qval = agent.BASE_AGENT_CLASS.critic_net(
      tf.expand_dims(state_for_reset_agent, 0), tf.expand_dims(action, 0))[0, :]
  cond = tf.greater(tf.constant(q_min), qval)
  return cond


@gin.configurable
def true_fn(agent, state, action, transition_type, environment_steps,
            num_episodes):
  """Returns an op that evaluates to true.

  Args:
    agent: RL agent.
    state: A [num_state_dims] tensor representing a state.
    action: Action performed.
    transition_type: Type of transition after action
    environment_steps: Number of steps performed by environment.
    num_episodes: Number of episodes.
  Returns:
    cond: op that always evaluates to True.
  """
  del agent, state, action, transition_type, environment_steps, num_episodes
  cond = tf.constant(True, dtype=tf.bool)
  return cond


@gin.configurable
def false_fn(agent, state, action, transition_type, environment_steps,
             num_episodes):
  """Returns an op that evaluates to false.

  Args:
    agent: RL agent.
    state: A [num_state_dims] tensor representing a state.
    action: Action performed.
    transition_type: Type of transition after action
    environment_steps: Number of steps performed by environment.
    num_episodes: Number of episodes.
  Returns:
    cond: op that always evaluates to False.
  """
  del agent, state, action, transition_type, environment_steps, num_episodes
  cond = tf.constant(False, dtype=tf.bool)
  return cond
