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

"""Evaluation utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from collections import namedtuple
logging = tf.logging
import gin.tf


@gin.configurable
def evaluate_checkpoint_repeatedly(checkpoint_dir,
                                   evaluate_checkpoint_fn,
                                   eval_interval_secs=600,
                                   max_number_of_evaluations=None,
                                   checkpoint_timeout=None,
                                   timeout_fn=None):
  """Evaluates a checkpointed model at a set interval."""
  if max_number_of_evaluations is not None and max_number_of_evaluations <= 0:
    raise ValueError(
        '`max_number_of_evaluations` must be either None or a positive number.')

  number_of_evaluations = 0
  for checkpoint_path in tf.contrib.training.checkpoints_iterator(
      checkpoint_dir,
      min_interval_secs=eval_interval_secs,
      timeout=checkpoint_timeout,
      timeout_fn=timeout_fn):
    retries = 3
    for _ in range(retries):
      try:
        should_stop = evaluate_checkpoint_fn(checkpoint_path)
        break
      except tf.errors.DataLossError as e:
        logging.warn(
            'Encountered a DataLossError while evaluating a checkpoint. This '
            'can happen when reading a checkpoint before it is fully written. '
            'Retrying...'
        )
        time.sleep(2.0)


def compute_model_loss(sess, model_rollout_fn, states, actions):
  """Computes model loss."""
  preds, losses = [], []
  preds.append(states[0])
  losses.append(0)
  for state, action in zip(states[1:], actions[1:]):
    pred = model_rollout_fn(sess, preds[-1], action)
    loss = np.sqrt(np.sum((state - pred) ** 2))
    preds.append(pred)
    losses.append(loss)
  return preds, losses


def compute_average_reward(sess, env_base, step_fn, gamma, num_steps,
                           num_episodes):
  """Computes the discounted reward for a given number of steps.

  Args:
    sess: The tensorflow session.
    env_base: A python environment.
    step_fn: A function that takes in `sess` and returns a list of
      [state, action, reward, discount, transition_type] values.
    gamma: discounting factor to apply to the reward.
    num_steps: number of steps to compute the reward over.
    num_episodes: number of episodes to average the reward over.
  Returns:
    average_reward: a scalar of discounted reward.
    last_reward: last reward received.
  """
  average_reward = 0
  average_last_reward = 0
  average_meta_reward = 0
  average_last_meta_reward = 0
  average_success = 0.
  states, actions = None, None
  for i in range(num_episodes):
    env_base.end_episode()
    env_base.begin_episode()
    (reward, last_reward, meta_reward, last_meta_reward,
     states, actions) = compute_reward(
        sess, step_fn, gamma, num_steps)
    s_reward = last_meta_reward  # Navigation
    success = (s_reward > -5.0)  # When using diff=False
    logging.info('Episode = %d, reward = %s, meta_reward = %f, '
                 'last_reward = %s, last meta_reward = %f, success = %s',
                 i, reward, meta_reward, last_reward, last_meta_reward,
                 success)
    average_reward += reward
    average_last_reward += last_reward
    average_meta_reward += meta_reward
    average_last_meta_reward += last_meta_reward
    average_success += success
  average_reward /= num_episodes
  average_last_reward /= num_episodes
  average_meta_reward /= num_episodes
  average_last_meta_reward /= num_episodes
  average_success /= num_episodes
  return (average_reward, average_last_reward,
          average_meta_reward, average_last_meta_reward,
          average_success,
          states, actions)


def compute_reward(sess, step_fn, gamma, num_steps):
  """Computes the discounted reward for a given number of steps.

  Args:
    sess: The tensorflow session.
    step_fn: A function that takes in `sess` and returns a list of
      [state, action, reward, discount, transition_type] values.
    gamma: discounting factor to apply to the reward.
    num_steps: number of steps to compute the reward over.
  Returns:
    reward: cumulative discounted reward.
    last_reward: reward received at final step.
  """

  total_reward = 0
  total_meta_reward = 0
  gamma_step = 1
  states = []
  actions = []
  for _ in range(num_steps):
    state, action, transition_type, reward, meta_reward, discount, _, _ = step_fn(sess)
    total_reward += reward * gamma_step * discount
    total_meta_reward += meta_reward * gamma_step * discount
    gamma_step *= gamma
    states.append(state)
    actions.append(action)
  return (total_reward, reward, total_meta_reward, meta_reward,
          states, actions)
