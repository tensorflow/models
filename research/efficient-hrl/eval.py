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

r"""Script for evaluating a UVF agent.

To run locally: See run_eval.py

To run on borg: See train_eval.borg
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
slim = tf.contrib.slim
import gin.tf
# pylint: disable=unused-import
import agent
import train
from utils import utils as uvf_utils
from utils import eval_utils
from environments import create_maze_env
# pylint: enable=unused-import

flags = tf.app.flags

flags.DEFINE_string('eval_dir', None,
                    'Directory for writing logs/summaries during eval.')
flags.DEFINE_string('checkpoint_dir', None,
                    'Directory containing checkpoints to eval.')
FLAGS = flags.FLAGS


def get_evaluate_checkpoint_fn(master, output_dir, eval_step_fns,
                               model_rollout_fn, gamma, max_steps_per_episode,
                               num_episodes_eval, num_episodes_videos,
                               tuner_hook, generate_videos,
                               generate_summaries, video_settings):
  """Returns a function that evaluates a given checkpoint.

  Args:
    master: BNS name of the TensorFlow master
    output_dir: The output directory to which the metric summaries are written.
    eval_step_fns: A dictionary of a functions that return a list of
      [state, action, reward, discount, transition_type] tensors,
      indexed by summary tag name.
    model_rollout_fn: Model rollout fn.
    gamma: Discount factor for the reward.
    max_steps_per_episode: Maximum steps to run each episode for.
    num_episodes_eval: Number of episodes to evaluate and average reward over.
    num_episodes_videos: Number of episodes to record for video.
    tuner_hook: A callable(average reward, global step) that updates a Vizier
      tuner trial.
    generate_videos: Whether to generate videos of the agent in action.
    generate_summaries: Whether to generate summaries.
    video_settings: Settings for generating videos of the agent.

  Returns:
    A function that evaluates a checkpoint.
  """
  sess = tf.Session(master, graph=tf.get_default_graph())
  sess.run(tf.global_variables_initializer())
  sess.run(tf.local_variables_initializer())
  summary_writer = tf.summary.FileWriter(output_dir)

  def evaluate_checkpoint(checkpoint_path):
    """Performs a one-time evaluation of the given checkpoint.

    Args:
      checkpoint_path: Checkpoint to evaluate.
    Returns:
      True if the evaluation process should stop
    """
    restore_fn = tf.contrib.framework.assign_from_checkpoint_fn(
        checkpoint_path,
        uvf_utils.get_all_vars(),
        ignore_missing_vars=True,
        reshape_variables=False)
    assert restore_fn is not None, 'cannot restore %s' % checkpoint_path
    restore_fn(sess)
    global_step = sess.run(slim.get_global_step())
    should_stop = False
    max_reward = -1e10
    max_meta_reward = -1e10

    for eval_tag, (eval_step, env_base,) in sorted(eval_step_fns.items()):
      if hasattr(env_base, 'set_sess'):
        env_base.set_sess(sess)  # set session

      if generate_summaries:
        tf.logging.info(
            '[%s] Computing average reward over %d episodes at global step %d.',
            eval_tag, num_episodes_eval, global_step)
        (average_reward, last_reward,
         average_meta_reward, last_meta_reward, average_success,
         states, actions) = eval_utils.compute_average_reward(
             sess, env_base, eval_step, gamma, max_steps_per_episode,
             num_episodes_eval)
        tf.logging.info('[%s] Average reward = %f', eval_tag, average_reward)
        tf.logging.info('[%s] Last reward = %f', eval_tag, last_reward)
        tf.logging.info('[%s] Average meta reward = %f', eval_tag, average_meta_reward)
        tf.logging.info('[%s] Last meta reward = %f', eval_tag, last_meta_reward)
        tf.logging.info('[%s] Average success = %f', eval_tag, average_success)
        if model_rollout_fn is not None:
          preds, model_losses = eval_utils.compute_model_loss(
              sess, model_rollout_fn, states, actions)
          for i, (pred, state, model_loss) in enumerate(
              zip(preds, states, model_losses)):
            tf.logging.info('[%s] Model rollout step %d: loss=%f', eval_tag, i,
                            model_loss)
            tf.logging.info('[%s] Model rollout step %d: pred=%s', eval_tag, i,
                            str(pred.tolist()))
            tf.logging.info('[%s] Model rollout step %d: state=%s', eval_tag, i,
                            str(state.tolist()))

        # Report the eval stats to the tuner.
        if average_reward > max_reward:
          max_reward = average_reward
        if average_meta_reward > max_meta_reward:
          max_meta_reward = average_meta_reward

        for (tag, value) in [('Reward/average_%s_reward', average_reward),
                             ('Reward/last_%s_reward', last_reward),
                             ('Reward/average_%s_meta_reward', average_meta_reward),
                             ('Reward/last_%s_meta_reward', last_meta_reward),
                             ('Reward/average_%s_success', average_success)]:
          summary_str = tf.Summary(value=[
              tf.Summary.Value(
                  tag=tag % eval_tag,
                  simple_value=value)
          ])
          summary_writer.add_summary(summary_str, global_step)
          summary_writer.flush()

      if generate_videos or should_stop:
        # Do a manual reset before generating the video to see the initial
        # pose of the robot, towards which the reset controller is moving.
        if hasattr(env_base, '_gym_env'):
          tf.logging.info('Resetting before recording video')
          if hasattr(env_base._gym_env, 'reset_model'):
            env_base._gym_env.reset_model()  # pylint: disable=protected-access
          else:
            env_base._gym_env.wrapped_env.reset_model()
        video_filename = os.path.join(output_dir, 'videos',
                                      '%s_step_%d.mp4' % (eval_tag,
                                                          global_step))
        eval_utils.capture_video(sess, eval_step, env_base,
                                max_steps_per_episode * num_episodes_videos,
                                video_filename, video_settings,
                                reset_every=max_steps_per_episode)

      should_stop = should_stop or (generate_summaries and tuner_hook and
                                    tuner_hook(max_reward, global_step))
    return bool(should_stop)

  return evaluate_checkpoint


def get_model_rollout(uvf_agent, tf_env):
  """Model rollout function."""
  state_spec = tf_env.observation_spec()[0]
  action_spec = tf_env.action_spec()[0]
  state_ph = tf.placeholder(dtype=state_spec.dtype, shape=state_spec.shape)
  action_ph = tf.placeholder(dtype=action_spec.dtype, shape=action_spec.shape)

  merged_state = uvf_agent.merged_state(state_ph)
  diff_value = uvf_agent.critic_net(tf.expand_dims(merged_state, 0),
                                    tf.expand_dims(action_ph, 0))[0]
  diff_value = tf.cast(diff_value, dtype=state_ph.dtype)
  state_ph.shape.assert_is_compatible_with(diff_value.shape)
  next_state = state_ph + diff_value

  def model_rollout_fn(sess, state, action):
    return sess.run(next_state, feed_dict={state_ph: state, action_ph: action})

  return model_rollout_fn


def get_eval_step(uvf_agent,
                  state_preprocess,
                  tf_env,
                  action_fn,
                  meta_action_fn,
                  environment_steps,
                  num_episodes,
                  mode='eval'):
  """Get one-step policy/env stepping ops.

  Args:
    uvf_agent: A UVF agent.
    tf_env: A TFEnvironment.
    action_fn: A function to produce actions given current state.
    meta_action_fn: A function to produce meta actions given current state.
    environment_steps: A variable to count the number of steps in the tf_env.
    num_episodes: A variable to count the number of episodes.
    mode: a string representing the mode=[train, explore, eval].

  Returns:
    A collect_experience_op that excute an action and store into the
    replay_buffer
  """

  tf_env.start_collect()
  state = tf_env.current_obs()
  action = action_fn(state, context=None)
  state_repr = state_preprocess(state)

  action_spec = tf_env.action_spec()
  action_ph = tf.placeholder(dtype=action_spec.dtype, shape=action_spec.shape)
  with tf.control_dependencies([state]):
    transition_type, reward, discount = tf_env.step(action_ph)

  def increment_step():
    return environment_steps.assign_add(1)

  def increment_episode():
    return num_episodes.assign_add(1)

  def no_op_int():
    return tf.constant(0, dtype=tf.int64)

  step_cond = uvf_agent.step_cond_fn(state, action,
                                     transition_type,
                                     environment_steps, num_episodes)
  reset_episode_cond = uvf_agent.reset_episode_cond_fn(
      state, action,
      transition_type, environment_steps, num_episodes)
  reset_env_cond = uvf_agent.reset_env_cond_fn(state, action,
                                               transition_type,
                                               environment_steps, num_episodes)

  increment_step_op = tf.cond(step_cond, increment_step, no_op_int)
  with tf.control_dependencies([increment_step_op]):
    increment_episode_op = tf.cond(reset_episode_cond, increment_episode,
                                   no_op_int)

  with tf.control_dependencies([reward, discount]):
    next_state = tf_env.current_obs()
    next_state_repr = state_preprocess(next_state)

  with tf.control_dependencies([increment_episode_op]):
    post_reward, post_meta_reward = uvf_agent.cond_begin_episode_op(
        tf.logical_not(reset_episode_cond),
        [state, action_ph, reward, next_state,
         state_repr, next_state_repr],
        mode=mode, meta_action_fn=meta_action_fn)

  # Important: do manual reset after getting the final reward from the
  # unreset environment.
  with tf.control_dependencies([post_reward, post_meta_reward]):
    cond_reset_op = tf.cond(reset_env_cond,
                            tf_env.reset,
                            tf_env.current_time_step)

  # Add a dummy control dependency to force the reset_op to run
  with tf.control_dependencies(cond_reset_op):
    post_reward, post_meta_reward = map(tf.identity, [post_reward, post_meta_reward])

  eval_step = [next_state, action_ph, transition_type, post_reward, post_meta_reward, discount, uvf_agent.context_vars, state_repr]

  if callable(action):
    def step_fn(sess):
      action_value = action(sess)
      return sess.run(eval_step, feed_dict={action_ph: action_value})
  else:
    action = uvf_utils.clip_to_spec(action, action_spec)
    def step_fn(sess):
      action_value = sess.run(action)
      return sess.run(eval_step, feed_dict={action_ph: action_value})

  return step_fn


@gin.configurable
def evaluate(checkpoint_dir,
             eval_dir,
             environment=None,
             num_bin_actions=3,
             agent_class=None,
             meta_agent_class=None,
             state_preprocess_class=None,
             gamma=1.0,
             num_episodes_eval=10,
             eval_interval_secs=60,
             max_number_of_evaluations=None,
             checkpoint_timeout=None,
             timeout_fn=None,
             tuner_hook=None,
             generate_videos=False,
             generate_summaries=True,
             num_episodes_videos=5,
             video_settings=None,
             eval_modes=('eval',),
             eval_model_rollout=False,
             policy_save_dir='policy',
             checkpoint_range=None,
             checkpoint_path=None,
             max_steps_per_episode=None,
             evaluate_nohrl=False):
  """Loads and repeatedly evaluates a checkpointed model at a set interval.

  Args:
    checkpoint_dir: The directory where the checkpoints reside.
    eval_dir: Directory to save the evaluation summary results.
    environment: A BaseEnvironment to evaluate.
    num_bin_actions: Number of bins for discretizing continuous actions.
    agent_class: An RL agent class.
    meta_agent_class: A Meta agent class.
    gamma: Discount factor for the reward.
    num_episodes_eval: Number of episodes to evaluate and average reward over.
    eval_interval_secs: The number of seconds between each evaluation run.
    max_number_of_evaluations: The max number of evaluations. If None the
      evaluation continues indefinitely.
    checkpoint_timeout: The maximum amount of time to wait between checkpoints.
      If left as `None`, then the process will wait indefinitely.
    timeout_fn: Optional function to call after a timeout.
    tuner_hook: A callable that takes the average reward and global step and
      updates a Vizier tuner trial.
    generate_videos: Whether to generate videos of the agent in action.
    generate_summaries: Whether to generate summaries.
    num_episodes_videos: Number of episodes to evaluate for generating videos.
    video_settings: Settings for generating videos of the agent.
      optimal action based on the critic.
    eval_modes: A tuple of eval modes.
    eval_model_rollout: Evaluate model rollout.
    policy_save_dir: Optional sub-directory where the policies are
      saved.
    checkpoint_range: Optional. If provided, evaluate all checkpoints in
      the range.
    checkpoint_path: Optional sub-directory specifying which checkpoint to
      evaluate. If None, will evaluate the most recent checkpoint.
  """
  tf_env = create_maze_env.TFPyEnvironment(environment)
  observation_spec = [tf_env.observation_spec()]
  action_spec = [tf_env.action_spec()]

  assert max_steps_per_episode, 'max_steps_per_episode need to be set'

  if agent_class.ACTION_TYPE == 'discrete':
    assert False
  else:
    assert agent_class.ACTION_TYPE == 'continuous'

  if meta_agent_class is not None:
    assert agent_class.ACTION_TYPE == meta_agent_class.ACTION_TYPE
    with tf.variable_scope('meta_agent'):
      meta_agent = meta_agent_class(
        observation_spec,
        action_spec,
        tf_env,
      )
  else:
    meta_agent = None

  with tf.variable_scope('uvf_agent'):
    uvf_agent = agent_class(
        observation_spec,
        action_spec,
        tf_env,
    )
    uvf_agent.set_meta_agent(agent=meta_agent)

  with tf.variable_scope('state_preprocess'):
    state_preprocess = state_preprocess_class()

  # run both actor and critic once to ensure networks are initialized
  # and gin configs will be saved
  # pylint: disable=protected-access
  temp_states = tf.expand_dims(
      tf.zeros(
          dtype=uvf_agent._observation_spec.dtype,
          shape=uvf_agent._observation_spec.shape), 0)
  # pylint: enable=protected-access
  temp_actions = uvf_agent.actor_net(temp_states)
  uvf_agent.critic_net(temp_states, temp_actions)

  # create eval_step_fns for each action function
  eval_step_fns = dict()
  meta_agent = uvf_agent.meta_agent
  for meta in [True] + [False] * evaluate_nohrl:
    meta_tag = 'hrl' if meta else 'nohrl'
    uvf_agent.set_meta_agent(meta_agent if meta else None)
    for mode in eval_modes:
      # wrap environment
      wrapped_environment = uvf_agent.get_env_base_wrapper(
          environment, mode=mode)
      action_wrapper = lambda agent_: agent_.action
      action_fn = action_wrapper(uvf_agent)
      meta_action_fn = action_wrapper(meta_agent)
      eval_step_fns['%s_%s' % (mode, meta_tag)] = (get_eval_step(
          uvf_agent=uvf_agent,
          state_preprocess=state_preprocess,
          tf_env=tf_env,
          action_fn=action_fn,
          meta_action_fn=meta_action_fn,
          environment_steps=tf.Variable(
              0, dtype=tf.int64, name='environment_steps'),
          num_episodes=tf.Variable(0, dtype=tf.int64, name='num_episodes'),
          mode=mode), wrapped_environment,)

  model_rollout_fn = None
  if eval_model_rollout:
    model_rollout_fn = get_model_rollout(uvf_agent, tf_env)

  tf.train.get_or_create_global_step()

  if policy_save_dir:
    checkpoint_dir = os.path.join(checkpoint_dir, policy_save_dir)

  tf.logging.info('Evaluating policies at %s', checkpoint_dir)
  tf.logging.info('Running episodes for max %d steps', max_steps_per_episode)

  evaluate_checkpoint_fn = get_evaluate_checkpoint_fn(
      '', eval_dir, eval_step_fns, model_rollout_fn, gamma,
      max_steps_per_episode, num_episodes_eval, num_episodes_videos, tuner_hook,
      generate_videos, generate_summaries, video_settings)

  if checkpoint_path is not None:
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
    evaluate_checkpoint_fn(checkpoint_path)
  elif checkpoint_range is not None:
    model_files = tf.gfile.Glob(
        os.path.join(checkpoint_dir, 'model.ckpt-*.index'))
    tf.logging.info('Found %s policies at %s', len(model_files), checkpoint_dir)
    model_files = {
        int(f.split('model.ckpt-', 1)[1].split('.', 1)[0]):
        os.path.splitext(f)[0]
        for f in model_files
    }
    model_files = {
        k: v
        for k, v in model_files.items()
        if k >= checkpoint_range[0] and k <= checkpoint_range[1]
    }
    tf.logging.info('Evaluating %d policies at %s',
                    len(model_files), checkpoint_dir)
    for _, checkpoint_path in sorted(model_files.items()):
      evaluate_checkpoint_fn(checkpoint_path)
  else:
    eval_utils.evaluate_checkpoint_repeatedly(
        checkpoint_dir,
        evaluate_checkpoint_fn,
        eval_interval_secs=eval_interval_secs,
        max_number_of_evaluations=max_number_of_evaluations,
        checkpoint_timeout=checkpoint_timeout,
        timeout_fn=timeout_fn)
