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

"""Trainer for coordinating single or multi-replica training.

Main point of entry for running models.  Specifies most of
the parameters used by different algorithms.
"""

import tensorflow as tf
import numpy as np
import random
import os
import pickle

from six.moves import xrange
import controller
import model
import policy
import baseline
import objective
import full_episode_objective
import trust_region
import optimizers
import replay_buffer
import expert_paths
import gym_wrapper
import env_spec

app = tf.app
flags = tf.flags
logging = tf.logging
gfile = tf.gfile

FLAGS = flags.FLAGS

flags.DEFINE_string('env', 'Copy-v0', 'environment name')
flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_integer('replay_batch_size', None, 'replay batch size; defaults to batch_size')
flags.DEFINE_integer('num_samples', 1,
                     'number of samples from each random seed initialization')
flags.DEFINE_integer('max_step', 200, 'max number of steps to train on')
flags.DEFINE_integer('cutoff_agent', 0,
                     'number of steps at which to cut-off agent. '
                     'Defaults to always cutoff')
flags.DEFINE_integer('num_steps', 100000, 'number of training steps')
flags.DEFINE_integer('validation_frequency', 100,
                     'every so many steps, output some stats')

flags.DEFINE_float('target_network_lag', 0.95,
                   'This exponential decay on online network yields target '
                   'network')
flags.DEFINE_string('sample_from', 'online',
                    'Sample actions from "online" network or "target" network')

flags.DEFINE_string('objective', 'pcl',
                    'pcl/upcl/a3c/trpo/reinforce/urex')
flags.DEFINE_bool('trust_region_p', False,
                  'use trust region for policy optimization')
flags.DEFINE_string('value_opt', None,
                    'leave as None to optimize it along with policy '
                    '(using critic_weight). Otherwise set to '
                    '"best_fit" (least squares regression), "lbfgs", or "grad"')
flags.DEFINE_float('max_divergence', 0.01,
                   'max divergence (i.e. KL) to allow during '
                   'trust region optimization')

flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
flags.DEFINE_float('clip_norm', 5.0, 'clip norm')
flags.DEFINE_float('clip_adv', 0.0, 'Clip advantages at this value.  '
                   'Leave as 0 to not clip at all.')
flags.DEFINE_float('critic_weight', 0.1, 'critic weight')
flags.DEFINE_float('tau', 0.1, 'entropy regularizer.'
                   'If using decaying tau, this is the final value.')
flags.DEFINE_float('tau_decay', None,
                   'decay tau by this much every 100 steps')
flags.DEFINE_float('tau_start', 0.1,
                   'start tau at this value')
flags.DEFINE_float('eps_lambda', 0.0, 'relative entropy regularizer.')
flags.DEFINE_bool('update_eps_lambda', False,
                  'Update lambda automatically based on last 100 episodes.')
flags.DEFINE_float('gamma', 1.0, 'discount')
flags.DEFINE_integer('rollout', 10, 'rollout')
flags.DEFINE_bool('use_target_values', False,
                  'use target network for value estimates')
flags.DEFINE_bool('fixed_std', True,
                  'fix the std in Gaussian distributions')
flags.DEFINE_bool('input_prev_actions', True,
                  'input previous actions to policy network')
flags.DEFINE_bool('recurrent', True,
                  'use recurrent connections')
flags.DEFINE_bool('input_time_step', False,
                  'input time step into value calucations')

flags.DEFINE_bool('use_online_batch', True, 'train on batches as they are sampled')
flags.DEFINE_bool('batch_by_steps', False,
                  'ensure each training batch has batch_size * max_step steps')
flags.DEFINE_bool('unify_episodes', False,
                  'Make sure replay buffer holds entire episodes, '
                  'even across distinct sampling steps')
flags.DEFINE_integer('replay_buffer_size', 5000, 'replay buffer size')
flags.DEFINE_float('replay_buffer_alpha', 0.5, 'replay buffer alpha param')
flags.DEFINE_integer('replay_buffer_freq', 0,
                     'replay buffer frequency (only supports -1/0/1)')
flags.DEFINE_string('eviction', 'rand',
                    'how to evict from replay buffer: rand/rank/fifo')
flags.DEFINE_string('prioritize_by', 'rewards',
                    'Prioritize replay buffer by "rewards" or "step"')
flags.DEFINE_integer('num_expert_paths', 0,
                     'number of expert paths to seed replay buffer with')

flags.DEFINE_integer('internal_dim', 256, 'RNN internal dim')
flags.DEFINE_integer('value_hidden_layers', 0,
                     'number of hidden layers in value estimate')
flags.DEFINE_integer('tf_seed', 42, 'random seed for tensorflow')

flags.DEFINE_string('save_trajectories_dir', None,
                    'directory to save trajectories to, if desired')
flags.DEFINE_string('load_trajectories_file', None,
                    'file to load expert trajectories from')

# supervisor flags
flags.DEFINE_bool('supervisor', False, 'use supervisor training')
flags.DEFINE_integer('task_id', 0, 'task id')
flags.DEFINE_integer('ps_tasks', 0, 'number of ps tasks')
flags.DEFINE_integer('num_replicas', 1, 'number of replicas used')
flags.DEFINE_string('master', 'local', 'name of master')
flags.DEFINE_string('save_dir', '', 'directory to save model to')
flags.DEFINE_string('load_path', '', 'path of saved model to load (if none in save_dir)')


class Trainer(object):
  """Coordinates single or multi-replica training."""

  def __init__(self):
    self.batch_size = FLAGS.batch_size
    self.replay_batch_size = FLAGS.replay_batch_size
    if self.replay_batch_size is None:
      self.replay_batch_size = self.batch_size
    self.num_samples = FLAGS.num_samples

    self.env_str = FLAGS.env
    self.env = gym_wrapper.GymWrapper(self.env_str,
                                      distinct=FLAGS.batch_size // self.num_samples,
                                      count=self.num_samples)
    self.eval_env = gym_wrapper.GymWrapper(
        self.env_str,
        distinct=FLAGS.batch_size // self.num_samples,
        count=self.num_samples)
    self.env_spec = env_spec.EnvSpec(self.env.get_one())

    self.max_step = FLAGS.max_step
    self.cutoff_agent = FLAGS.cutoff_agent
    self.num_steps = FLAGS.num_steps
    self.validation_frequency = FLAGS.validation_frequency

    self.target_network_lag = FLAGS.target_network_lag
    self.sample_from = FLAGS.sample_from
    assert self.sample_from in ['online', 'target']

    self.critic_weight = FLAGS.critic_weight
    self.objective = FLAGS.objective
    self.trust_region_p = FLAGS.trust_region_p
    self.value_opt = FLAGS.value_opt
    assert not self.trust_region_p or self.objective in ['pcl', 'trpo']
    assert self.objective != 'trpo' or self.trust_region_p
    assert self.value_opt is None or self.value_opt == 'None' or \
        self.critic_weight == 0.0
    self.max_divergence = FLAGS.max_divergence

    self.learning_rate = FLAGS.learning_rate
    self.clip_norm = FLAGS.clip_norm
    self.clip_adv = FLAGS.clip_adv
    self.tau = FLAGS.tau
    self.tau_decay = FLAGS.tau_decay
    self.tau_start = FLAGS.tau_start
    self.eps_lambda = FLAGS.eps_lambda
    self.update_eps_lambda = FLAGS.update_eps_lambda
    self.gamma = FLAGS.gamma
    self.rollout = FLAGS.rollout
    self.use_target_values = FLAGS.use_target_values
    self.fixed_std = FLAGS.fixed_std
    self.input_prev_actions = FLAGS.input_prev_actions
    self.recurrent = FLAGS.recurrent
    assert not self.trust_region_p or not self.recurrent
    self.input_time_step = FLAGS.input_time_step
    assert not self.input_time_step or (self.cutoff_agent <= self.max_step)

    self.use_online_batch = FLAGS.use_online_batch
    self.batch_by_steps = FLAGS.batch_by_steps
    self.unify_episodes = FLAGS.unify_episodes
    if self.unify_episodes:
      assert self.batch_size == 1

    self.replay_buffer_size = FLAGS.replay_buffer_size
    self.replay_buffer_alpha = FLAGS.replay_buffer_alpha
    self.replay_buffer_freq = FLAGS.replay_buffer_freq
    assert self.replay_buffer_freq in [-1, 0, 1]
    self.eviction = FLAGS.eviction
    self.prioritize_by = FLAGS.prioritize_by
    assert self.prioritize_by in ['rewards', 'step']
    self.num_expert_paths = FLAGS.num_expert_paths

    self.internal_dim = FLAGS.internal_dim
    self.value_hidden_layers = FLAGS.value_hidden_layers
    self.tf_seed = FLAGS.tf_seed

    self.save_trajectories_dir = FLAGS.save_trajectories_dir
    self.save_trajectories_file = (
        os.path.join(
            self.save_trajectories_dir, self.env_str.replace('-', '_'))
        if self.save_trajectories_dir else None)
    self.load_trajectories_file = FLAGS.load_trajectories_file

    self.hparams = dict((attr, getattr(self, attr))
                        for attr in dir(self)
                        if not attr.startswith('__') and
                        not callable(getattr(self, attr)))

  def hparams_string(self):
    return '\n'.join('%s: %s' % item for item in sorted(self.hparams.items()))

  def get_objective(self):
    tau = self.tau
    if self.tau_decay is not None:
      assert self.tau_start >= self.tau
      tau = tf.maximum(
          tf.train.exponential_decay(
              self.tau_start, self.global_step, 100, self.tau_decay),
          self.tau)

    if self.objective in ['pcl', 'a3c', 'trpo', 'upcl']:
      cls = (objective.PCL if self.objective in ['pcl', 'upcl'] else
             objective.TRPO if self.objective == 'trpo' else
             objective.ActorCritic)
      policy_weight = 1.0

      return cls(self.learning_rate,
                 clip_norm=self.clip_norm,
                 policy_weight=policy_weight,
                 critic_weight=self.critic_weight,
                 tau=tau, gamma=self.gamma, rollout=self.rollout,
                 eps_lambda=self.eps_lambda, clip_adv=self.clip_adv,
                 use_target_values=self.use_target_values)
    elif self.objective in ['reinforce', 'urex']:
      cls = (full_episode_objective.Reinforce
             if self.objective == 'reinforce' else
             full_episode_objective.UREX)
      return cls(self.learning_rate,
                 clip_norm=self.clip_norm,
                 num_samples=self.num_samples,
                 tau=tau, bonus_weight=1.0)  # TODO: bonus weight?
    else:
      assert False, 'Unknown objective %s' % self.objective

  def get_policy(self):
    if self.recurrent:
      cls = policy.Policy
    else:
      cls = policy.MLPPolicy
    return cls(self.env_spec, self.internal_dim,
               fixed_std=self.fixed_std,
               recurrent=self.recurrent,
               input_prev_actions=self.input_prev_actions)

  def get_baseline(self):
    cls = (baseline.UnifiedBaseline if self.objective == 'upcl' else
           baseline.Baseline)
    return cls(self.env_spec, self.internal_dim,
               input_prev_actions=self.input_prev_actions,
               input_time_step=self.input_time_step,
               input_policy_state=self.recurrent,  # may want to change this
               n_hidden_layers=self.value_hidden_layers,
               hidden_dim=self.internal_dim,
               tau=self.tau)

  def get_trust_region_p_opt(self):
    if self.trust_region_p:
      return trust_region.TrustRegionOptimization(
          max_divergence=self.max_divergence)
    else:
      return None

  def get_value_opt(self):
    if self.value_opt == 'grad':
      return optimizers.GradOptimization(
          learning_rate=self.learning_rate, max_iter=5, mix_frac=0.05)
    elif self.value_opt == 'lbfgs':
      return optimizers.LbfgsOptimization(max_iter=25, mix_frac=0.1)
    elif self.value_opt == 'best_fit':
      return optimizers.BestFitOptimization(mix_frac=1.0)
    else:
      return None

  def get_model(self):
    cls = model.Model
    return cls(self.env_spec, self.global_step,
               target_network_lag=self.target_network_lag,
               sample_from=self.sample_from,
               get_policy=self.get_policy,
               get_baseline=self.get_baseline,
               get_objective=self.get_objective,
               get_trust_region_p_opt=self.get_trust_region_p_opt,
               get_value_opt=self.get_value_opt)

  def get_replay_buffer(self):
    if self.replay_buffer_freq <= 0:
      return None
    else:
      assert self.objective in ['pcl', 'upcl'], 'Can\'t use replay buffer with %s' % (
          self.objective)
    cls = replay_buffer.PrioritizedReplayBuffer
    return cls(self.replay_buffer_size,
               alpha=self.replay_buffer_alpha,
               eviction_strategy=self.eviction)

  def get_buffer_seeds(self):
    return expert_paths.sample_expert_paths(
        self.num_expert_paths, self.env_str, self.env_spec,
        load_trajectories_file=self.load_trajectories_file)

  def get_controller(self, env):
    """Get controller."""
    cls = controller.Controller
    return cls(env, self.env_spec, self.internal_dim,
               use_online_batch=self.use_online_batch,
               batch_by_steps=self.batch_by_steps,
               unify_episodes=self.unify_episodes,
               replay_batch_size=self.replay_batch_size,
               max_step=self.max_step,
               cutoff_agent=self.cutoff_agent,
               save_trajectories_file=self.save_trajectories_file,
               use_trust_region=self.trust_region_p,
               use_value_opt=self.value_opt not in [None, 'None'],
               update_eps_lambda=self.update_eps_lambda,
               prioritize_by=self.prioritize_by,
               get_model=self.get_model,
               get_replay_buffer=self.get_replay_buffer,
               get_buffer_seeds=self.get_buffer_seeds)

  def do_before_step(self, step):
    pass

  def run(self):
    """Run training."""
    is_chief = FLAGS.task_id == 0 or not FLAGS.supervisor
    sv = None

    def init_fn(sess, saver):
      ckpt = None
      if FLAGS.save_dir and sv is None:
        load_dir = FLAGS.save_dir
        ckpt = tf.train.get_checkpoint_state(load_dir)
      if ckpt and ckpt.model_checkpoint_path:
        logging.info('restoring from %s', ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
      elif FLAGS.load_path:
        logging.info('restoring from %s', FLAGS.load_path)
        saver.restore(sess, FLAGS.load_path)

    if FLAGS.supervisor:
      with tf.device(tf.ReplicaDeviceSetter(FLAGS.ps_tasks, merge_devices=True)):
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        tf.set_random_seed(FLAGS.tf_seed)
        self.controller = self.get_controller(self.env)
        self.model = self.controller.model
        self.controller.setup()
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
          self.eval_controller = self.get_controller(self.eval_env)
          self.eval_controller.setup(train=False)

        saver = tf.train.Saver(max_to_keep=10)
        step = self.model.global_step
        sv = tf.Supervisor(logdir=FLAGS.save_dir,
                           is_chief=is_chief,
                           saver=saver,
                           save_model_secs=600,
                           summary_op=None,  # we define it ourselves
                           save_summaries_secs=60,
                           global_step=step,
                           init_fn=lambda sess: init_fn(sess, saver))
        sess = sv.PrepareSession(FLAGS.master)
    else:
      tf.set_random_seed(FLAGS.tf_seed)
      self.global_step = tf.contrib.framework.get_or_create_global_step()
      self.controller = self.get_controller(self.env)
      self.model = self.controller.model
      self.controller.setup()
      with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        self.eval_controller = self.get_controller(self.eval_env)
        self.eval_controller.setup(train=False)

      saver = tf.train.Saver(max_to_keep=10)
      sess = tf.Session()
      sess.run(tf.initialize_all_variables())
      init_fn(sess, saver)

    self.sv = sv
    self.sess = sess

    logging.info('hparams:\n%s', self.hparams_string())

    model_step = sess.run(self.model.global_step)
    if model_step >= self.num_steps:
      logging.info('training has reached final step')
      return

    losses = []
    rewards = []
    all_ep_rewards = []
    for step in xrange(1 + self.num_steps):

      if sv is not None and sv.ShouldStop():
        logging.info('stopping supervisor')
        break

      self.do_before_step(step)

      (loss, summary,
       total_rewards, episode_rewards) = self.controller.train(sess)
      _, greedy_episode_rewards = self.eval_controller.eval(sess)
      self.controller.greedy_episode_rewards = greedy_episode_rewards
      losses.append(loss)
      rewards.append(total_rewards)
      all_ep_rewards.extend(episode_rewards)

      if (random.random() < 0.1 and summary and episode_rewards and
          is_chief and sv and sv._summary_writer):
        sv.summary_computed(sess, summary)

      model_step = sess.run(self.model.global_step)
      if is_chief and step % self.validation_frequency == 0:
        logging.info('at training step %d, model step %d: '
                     'avg loss %f, avg reward %f, '
                     'episode rewards: %f, greedy rewards: %f',
                     step, model_step,
                     np.mean(losses), np.mean(rewards),
                     np.mean(all_ep_rewards),
                     np.mean(greedy_episode_rewards))

        losses = []
        rewards = []
        all_ep_rewards = []

      if model_step >= self.num_steps:
        logging.info('training has reached final step')
        break

    if is_chief and sv is not None:
      logging.info('saving final model to %s', sv.save_path)
      sv.saver.save(sess, sv.save_path, global_step=sv.global_step)


def main(unused_argv):
  logging.set_verbosity(logging.INFO)
  trainer = Trainer()
  trainer.run()


if __name__ == '__main__':
  app.run()
