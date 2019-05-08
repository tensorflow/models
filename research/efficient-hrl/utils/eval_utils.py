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
import os
import cv2
import yaml


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
  average_q_value = 0.
  average_meta_q_value = 0.
  average_context_norm = 0.
  states, actions = None, None
  for i in range(num_episodes):
    env_base.end_episode()
    env_base.begin_episode()
    (reward, last_reward, meta_reward, last_meta_reward,
     states, actions, q, meta_q, context_norm) = compute_reward(
        sess, step_fn, gamma, num_steps)
    s_reward = last_meta_reward  # Navigation
    success = (s_reward > -5.0)  # When using diff=False
    logging.info('Episode = %d, reward = %s, meta_reward = %f, '
                 'last_reward = %s, last meta_reward = %f, success = %s, average q value = %f, '
                 'average meta q value= %f, average context norm %f',
                 i, reward, meta_reward, last_reward, last_meta_reward,
                 success, q, meta_q, context_norm)
    average_reward += reward
    average_last_reward += last_reward
    average_meta_reward += meta_reward
    average_last_meta_reward += last_meta_reward
    average_success += success
    average_q_value += q
    average_meta_q_value += meta_q
    average_context_norm += context_norm
  average_reward /= num_episodes
  average_last_reward /= num_episodes
  average_meta_reward /= num_episodes
  average_last_meta_reward /= num_episodes
  average_success /= num_episodes
  average_q_value /= num_episodes
  average_meta_q_value /= num_episodes
  average_context_norm /= num_episodes
  return (average_reward, average_last_reward,
          average_meta_reward, average_last_meta_reward,
          average_success,
          states, actions,
          average_q_value,
          average_meta_q_value,
          average_context_norm)


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
  average_q_value = 0
  average_meta_q_value = 0
  states = []
  actions = []
  average_context = 0.
  for _ in range(num_steps):
    state, action, transition_type, reward, meta_reward, discount, context, _, q, meta_q = step_fn(sess)
    total_reward += reward * gamma_step * discount
    total_meta_reward += meta_reward * gamma_step * discount
    gamma_step *= gamma
    states.append(state)
    actions.append(action)
    average_context += np.linalg.norm(context)
    average_q_value += q
    average_meta_q_value += meta_q
  average_q_value = average_q_value/num_steps
  average_meta_q_value = average_meta_q_value/num_steps
  average_context = average_context/num_steps
  return (total_reward, reward, total_meta_reward, meta_reward,
          states, actions, average_q_value, average_meta_q_value, average_context)


def capture_video(sess, eval_step, env_base, total_steps, video_filename,
                  video_settings, reset_every):
  video_writer = None
  if not isinstance(video_settings, dict):
    try:
      video_settings = yaml.load(video_settings)
    except:
      tf.logging.info("Cannot convert video_settings to a dict via yaml!")
  if video_settings is not None and "fps" in video_settings:
    fps = video_settings["fps"]
  else:
    fps = 20

  video_dir = os.path.dirname(video_filename)
  os.makedirs(video_dir, exist_ok=True)

  arr = np.array([])
  arr_reward = np.array([])
  states = np.array([])

  def stack(a, b):
    if len(a) == 0:
      return np.array([b])

    else:
      return np.vstack((a, b))

  for t in range(total_steps):
    (next_state, action, time_step, post_reward, post_meta_reward,
     discount, contexts, state_repr, _, _) = eval_step(sess)
    states = stack(states, next_state[:2])
    if len(arr_reward) == 0:
      arr_reward = np.array([post_reward])
    else:
      arr_reward = np.vstack((arr_reward, np.array(post_reward)))

    if len(arr) == 0:
      arr = np.array([contexts[0]])
    else:
      arr = np.vstack((arr, np.array(contexts[0])))
    # id = env_base._gym_env.wrapped_env.sim.model.joint_name2id("movable_meta")
    qpos = env_base._gym_env.wrapped_env.sim.data.qpos
    # import ipdb; ipdb.set_trace()
    qpos[-2] = contexts[0][0] + next_state[0]
    qpos[-1] = contexts[0][1] + next_state[1]

    env_base.gym.render(mode='human')
    img = env_base.gym.render(mode='rgb_array')
    if video_writer is None:
      height, width = img.shape[:2]
      video_writer = cv2.VideoWriter(
        video_filename, apiPreference=cv2.CAP_ANY,
        fourcc=cv2.VideoWriter_fourcc(*'MPEG'),
        fps=fps, frameSize=(width, height))
    video_writer.write(img)

    if t % reset_every == 0:
      pass

  # import json
  # with open("meta_info.json", 'w') as f:
  #   json.dump(np.hstack((arr, arr_r, states)).tolist(),f, indent=4)

  if video_writer is not None:
    os.makedirs(os.path.dirname(video_filename), exist_ok=True)
    video_writer.release()
    tf.logging.info("Video written to %s" % video_filename)
