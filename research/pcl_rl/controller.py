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

"""Controller coordinates sampling and training model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf
import numpy as np
import pickle
import random

flags = tf.flags
gfile = tf.gfile

FLAGS = flags.FLAGS


def find_best_eps_lambda(rewards, lengths):
  """Find the best lambda given a desired epsilon = FLAGS.max_divergence."""
  # perhaps not the best way to do this
  desired_div = FLAGS.max_divergence * np.mean(lengths)

  def calc_divergence(eps_lambda):
    max_reward = np.max(rewards)
    logz = (max_reward / eps_lambda +
            np.log(np.mean(np.exp((rewards - max_reward) / eps_lambda))))
    exprr = np.mean(np.exp(rewards / eps_lambda - logz) *
                    rewards / eps_lambda)
    return exprr - logz

  left = 0.0
  right = 1000.0

  if len(rewards) <= 8:
    return (left + right) / 2

  num_iter = max(4, 1 + int(np.log((right - left) / 0.1) / np.log(2.0)))
  for _ in xrange(num_iter):
    mid = (left + right) / 2
    cur_div = calc_divergence(mid)
    if cur_div > desired_div:
      left = mid
    else:
      right = mid

  return (left + right) / 2


class Controller(object):

  def __init__(self, env, env_spec, internal_dim,
               use_online_batch=True,
               batch_by_steps=False,
               unify_episodes=False,
               replay_batch_size=None,
               max_step=None,
               cutoff_agent=1,
               save_trajectories_file=None,
               use_trust_region=False,
               use_value_opt=False,
               update_eps_lambda=False,
               prioritize_by='rewards',
               get_model=None,
               get_replay_buffer=None,
               get_buffer_seeds=None):
    self.env = env
    self.env_spec = env_spec
    self.internal_dim = internal_dim
    self.use_online_batch = use_online_batch
    self.batch_by_steps = batch_by_steps
    self.unify_episodes = unify_episodes
    self.replay_batch_size = replay_batch_size
    self.max_step = max_step
    self.cutoff_agent = cutoff_agent
    self.save_trajectories_file = save_trajectories_file
    self.use_trust_region = use_trust_region
    self.use_value_opt = use_value_opt
    self.update_eps_lambda = update_eps_lambda
    self.prioritize_by = prioritize_by

    self.model = get_model()
    self.replay_buffer = get_replay_buffer()
    self.seed_replay_buffer(get_buffer_seeds())

    self.internal_state = np.array([self.initial_internal_state()] *
                                   len(self.env))
    self.last_obs = self.env_spec.initial_obs(len(self.env))
    self.last_act = self.env_spec.initial_act(len(self.env))
    self.last_pad = np.zeros(len(self.env))

    self.start_episode = np.array([True] * len(self.env))
    self.step_count = np.array([0] * len(self.env))
    self.episode_running_rewards = np.zeros(len(self.env))
    self.episode_running_lengths = np.zeros(len(self.env))
    self.episode_rewards = []
    self.greedy_episode_rewards = []
    self.episode_lengths = []
    self.total_rewards = []

    self.best_batch_rewards = None

  def setup(self, train=True):
    self.model.setup(train=train)

  def initial_internal_state(self):
    return np.zeros(self.model.policy.rnn_state_dim)

  def _sample_episodes(self, sess, greedy=False):
    """Sample episodes from environment using model."""
    # reset environments as necessary
    obs_after_reset = self.env.reset_if(self.start_episode)

    for i, obs in enumerate(obs_after_reset):
      if obs is not None:
        self.step_count[i] = 0
        self.internal_state[i] = self.initial_internal_state()
        for j in xrange(len(self.env_spec.obs_dims)):
          self.last_obs[j][i] = obs[j]
        for j in xrange(len(self.env_spec.act_dims)):
          self.last_act[j][i] = -1
        self.last_pad[i] = 0

    # maintain episode as a single unit if the last sampling
    # batch ended before the episode was terminated
    if self.unify_episodes:
      assert len(obs_after_reset) == 1
      new_ep = obs_after_reset[0] is not None
    else:
      new_ep = True

    self.start_id = 0 if new_ep else len(self.all_obs[:])

    initial_state = self.internal_state
    all_obs = [] if new_ep else self.all_obs[:]
    all_act = ([self.last_act] if new_ep else self.all_act[:])
    all_pad = [] if new_ep else self.all_pad[:]
    rewards = [] if new_ep else self.rewards[:]

    # start stepping in the environments
    step = 0
    while not self.env.all_done():
      self.step_count += 1 - np.array(self.env.dones)

      next_internal_state, sampled_actions = self.model.sample_step(
          sess, self.last_obs, self.internal_state, self.last_act,
          greedy=greedy)

      env_actions = self.env_spec.convert_actions_to_env(sampled_actions)
      next_obs, reward, next_dones, _ = self.env.step(env_actions)

      all_obs.append(self.last_obs)
      all_act.append(sampled_actions)
      all_pad.append(self.last_pad)
      rewards.append(reward)

      self.internal_state = next_internal_state
      self.last_obs = next_obs
      self.last_act = sampled_actions
      self.last_pad = np.array(next_dones).astype('float32')

      step += 1
      if self.max_step and step >= self.max_step:
        break

    self.all_obs = all_obs[:]
    self.all_act = all_act[:]
    self.all_pad = all_pad[:]
    self.rewards = rewards[:]

    # append final observation
    all_obs.append(self.last_obs)

    return initial_state, all_obs, all_act, rewards, all_pad

  def sample_episodes(self, sess, greedy=False):
    """Sample steps from the environment until we have enough for a batch."""

    # check if last batch ended with episode that was not terminated
    if self.unify_episodes:
      self.all_new_ep = self.start_episode[0]

    # sample episodes until we either have enough episodes or enough steps
    episodes = []
    total_steps = 0
    while total_steps < self.max_step * len(self.env):
      (initial_state,
       observations, actions, rewards,
       pads) = self._sample_episodes(sess, greedy=greedy)

      observations = list(zip(*observations))
      actions = list(zip(*actions))

      terminated = np.array(self.env.dones)

      self.total_rewards = np.sum(np.array(rewards[self.start_id:]) *
                                  (1 - np.array(pads[self.start_id:])), axis=0)
      self.episode_running_rewards *= 1 - self.start_episode
      self.episode_running_lengths *= 1 - self.start_episode
      self.episode_running_rewards += self.total_rewards
      self.episode_running_lengths += np.sum(1 - np.array(pads[self.start_id:]), axis=0)

      episodes.extend(self.convert_from_batched_episodes(
          initial_state, observations, actions, rewards,
          terminated, pads))
      total_steps += np.sum(1 - np.array(pads))

      # set next starting episodes
      self.start_episode = np.logical_or(terminated,
                                         self.step_count >= self.cutoff_agent)
      episode_rewards = self.episode_running_rewards[self.start_episode].tolist()
      self.episode_rewards.extend(episode_rewards)
      self.episode_lengths.extend(self.episode_running_lengths[self.start_episode].tolist())
      self.episode_rewards = self.episode_rewards[-100:]
      self.episode_lengths = self.episode_lengths[-100:]

      if (self.save_trajectories_file is not None and
          (self.best_batch_rewards is None or
           np.mean(self.total_rewards) > self.best_batch_rewards)):
        self.best_batch_rewards = np.mean(self.total_rewards)
        my_episodes = self.convert_from_batched_episodes(
          initial_state, observations, actions, rewards,
          terminated, pads)
        with gfile.GFile(self.save_trajectories_file, 'w') as f:
          pickle.dump(my_episodes, f)

      if not self.batch_by_steps:
        return (initial_state,
                observations, actions, rewards,
                terminated, pads)

    return self.convert_to_batched_episodes(episodes)

  def _train(self, sess,
             observations, initial_state, actions,
             rewards, terminated, pads):
    """Train model using batch."""
    avg_episode_reward = np.mean(self.episode_rewards)
    greedy_episode_reward = (np.mean(self.greedy_episode_rewards)
                             if self.greedy_episode_rewards else
                             avg_episode_reward)
    loss, summary = None, None
    if self.use_trust_region:
      # use trust region to optimize policy
      loss, _, summary = self.model.trust_region_step(
          sess,
          observations, initial_state, actions,
          rewards, terminated, pads,
          avg_episode_reward=avg_episode_reward,
          greedy_episode_reward=greedy_episode_reward)
    else:  # otherwise use simple gradient descent on policy
      loss, _, summary = self.model.train_step(
          sess,
          observations, initial_state, actions,
          rewards, terminated, pads,
          avg_episode_reward=avg_episode_reward,
          greedy_episode_reward=greedy_episode_reward)

    if self.use_value_opt:  # optionally perform specific value optimization
      self.model.fit_values(
          sess,
          observations, initial_state, actions,
          rewards, terminated, pads)

    return loss, summary

  def train(self, sess):
    """Sample some episodes and train on some episodes."""
    cur_step = sess.run(self.model.inc_global_step)
    self.cur_step = cur_step

    # on the first iteration, set target network close to online network
    if self.cur_step == 0:
      for _ in xrange(100):
        sess.run(self.model.copy_op)
    # on other iterations, just perform single target <-- online operation
    sess.run(self.model.copy_op)

    # sample from env
    (initial_state,
     observations, actions, rewards,
     terminated, pads) = self.sample_episodes(sess)

    # add to replay buffer
    self.add_to_replay_buffer(
        initial_state, observations, actions,
        rewards, terminated, pads)

    loss, summary = 0, None
    # train on online batch
    if self.use_online_batch:
      loss, summary = self._train(
          sess,
          observations, initial_state, actions,
          rewards, terminated, pads)

    # update relative entropy coefficient
    if self.update_eps_lambda:
      episode_rewards = np.array(self.episode_rewards)
      episode_lengths = np.array(self.episode_lengths)
      eps_lambda = find_best_eps_lambda(
          episode_rewards[-20:], episode_lengths[-20:])
      sess.run(self.model.objective.assign_eps_lambda,
               feed_dict={self.model.objective.new_eps_lambda: eps_lambda})

    # train on replay batch
    replay_batch, replay_probs = self.get_from_replay_buffer(
        self.replay_batch_size)
    if replay_batch:
      (initial_state,
       observations, actions, rewards,
       terminated, pads) = replay_batch

      loss, summary = self._train(
          sess,
          observations, initial_state, actions,
          rewards, terminated, pads)

    return loss, summary, self.total_rewards, self.episode_rewards

  def eval(self, sess):
    """Use greedy sampling."""
    (initial_state,
     observations, actions, rewards,
     pads, terminated) = self.sample_episodes(sess, greedy=True)

    total_rewards = np.sum(np.array(rewards) * (1 - np.array(pads)), axis=0)
    return total_rewards, self.episode_rewards

  def convert_from_batched_episodes(
      self, initial_state, observations, actions, rewards,
      terminated, pads):
    """Convert time-major batch of episodes to batch-major list of episodes."""

    rewards = np.array(rewards)
    pads = np.array(pads)
    observations = [np.array(obs) for obs in observations]
    actions = [np.array(act) for act in actions]

    total_rewards = np.sum(rewards * (1 - pads), axis=0)
    total_length = np.sum(1 - pads, axis=0).astype('int32')

    episodes = []
    num_episodes = rewards.shape[1]
    for i in xrange(num_episodes):
      length = total_length[i]
      ep_initial = initial_state[i]
      ep_obs = [obs[:length + 1, i, ...] for obs in observations]
      ep_act = [act[:length + 1, i, ...] for act in actions]
      ep_rewards = rewards[:length, i]

      episodes.append(
          [ep_initial, ep_obs, ep_act, ep_rewards, terminated[i]])

    return episodes

  def convert_to_batched_episodes(self, episodes, max_length=None):
    """Convert batch-major list of episodes to time-major batch of episodes."""
    lengths = [len(ep[-2]) for ep in episodes]
    max_length = max_length or max(lengths)

    new_episodes = []
    for ep, length in zip(episodes, lengths):
      initial, observations, actions, rewards, terminated = ep
      observations = [np.resize(obs, [max_length + 1] + list(obs.shape)[1:])
                      for obs in observations]
      actions = [np.resize(act, [max_length + 1] + list(act.shape)[1:])
                 for act in actions]
      pads = np.array([0] * length + [1] * (max_length - length))
      rewards = np.resize(rewards, [max_length]) * (1 - pads)
      new_episodes.append([initial, observations, actions, rewards,
                           terminated, pads])

    (initial, observations, actions, rewards,
     terminated, pads) = zip(*new_episodes)
    observations = [np.swapaxes(obs, 0, 1)
                    for obs in zip(*observations)]
    actions = [np.swapaxes(act, 0, 1)
               for act in zip(*actions)]
    rewards = np.transpose(rewards)
    pads = np.transpose(pads)

    return (initial, observations, actions, rewards, terminated, pads)

  def add_to_replay_buffer(self, initial_state,
                           observations, actions, rewards,
                           terminated, pads):
    """Add batch of episodes to replay buffer."""
    if self.replay_buffer is None:
      return

    rewards = np.array(rewards)
    pads = np.array(pads)
    total_rewards = np.sum(rewards * (1 - pads), axis=0)

    episodes = self.convert_from_batched_episodes(
      initial_state, observations, actions, rewards,
      terminated, pads)

    priorities = (total_rewards if self.prioritize_by == 'reward'
                  else self.cur_step)

    if not self.unify_episodes or self.all_new_ep:
      self.last_idxs = self.replay_buffer.add(
          episodes, priorities)
    else:
      # If we are unifying episodes, we attempt to
      # keep them unified in the replay buffer.
      # The first episode sampled in the current batch is a
      # continuation of the last episode from the previous batch
      self.replay_buffer.add(episodes[:1], priorities, self.last_idxs[-1:])
      if len(episodes) > 1:
        self.replay_buffer.add(episodes[1:], priorities)

  def get_from_replay_buffer(self, batch_size):
    """Sample a batch of episodes from the replay buffer."""
    if self.replay_buffer is None or len(self.replay_buffer) < 1 * batch_size:
      return None, None

    desired_count = batch_size * self.max_step
    # in the case of batch_by_steps, we sample larger and larger
    # amounts from the replay buffer until we have enough steps.
    while True:
      if batch_size > len(self.replay_buffer):
        batch_size = len(self.replay_buffer)
      episodes, probs = self.replay_buffer.get_batch(batch_size)
      count = sum(len(ep[-2]) for ep in episodes)
      if count >= desired_count or not self.batch_by_steps:
        break
      if batch_size == len(self.replay_buffer):
        return None, None
      batch_size *= 1.2

    return (self.convert_to_batched_episodes(episodes), probs)

  def seed_replay_buffer(self, episodes):
    """Seed the replay buffer with some episodes."""
    if self.replay_buffer is None:
      return

    # just need to add initial state
    for i in xrange(len(episodes)):
      episodes[i] = [self.initial_internal_state()] + episodes[i]

    self.replay_buffer.seed_buffer(episodes)
