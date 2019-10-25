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

r""""""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import os
import time

import tensorflow as tf

import gin.tf

flags = tf.app.flags


flags.DEFINE_multi_string('config_file', None,
                          'List of paths to the config files.')
flags.DEFINE_multi_string('params', None,
                          'Newline separated list of Gin parameter bindings.')

flags.DEFINE_string('train_dir', None,
                    'Directory for writing logs/summaries during training.')
flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use.')
flags.DEFINE_integer('task', 0, 'task id')
flags.DEFINE_integer('save_interval_secs', 300, 'The frequency at which '
                     'checkpoints are saved, in seconds.')
flags.DEFINE_integer('save_summaries_secs', 30, 'The frequency at which '
                     'summaries are saved, in seconds.')
flags.DEFINE_boolean('summarize_gradients', False,
                     'Whether to generate gradient summaries.')

FLAGS = flags.FLAGS

TrainOps = namedtuple('TrainOps',
                      ['train_op', 'meta_train_op', 'collect_experience_op'])


class TrainStep(object):
  """Handles training step."""

  def __init__(self,
               max_number_of_steps=0,
               num_updates_per_observation=1,
               num_collect_per_update=1,
               num_collect_per_meta_update=1,
               log_every_n_steps=1,
               policy_save_fn=None,
               save_policy_every_n_steps=0,
               should_stop_early=None):
    """Returns a function that is executed at each step of slim training.

    Args:
      max_number_of_steps: Optional maximum number of train steps to take.
      num_updates_per_observation: Number of updates per observation.
      log_every_n_steps: The frequency, in terms of global steps, that the loss
      and global step and logged.
      policy_save_fn: A tf.Saver().save function to save the policy.
      save_policy_every_n_steps: How frequently to save the policy.
      should_stop_early: Optional hook to report whether training should stop.
    Raises:
      ValueError: If policy_save_fn is not provided when
        save_policy_every_n_steps > 0.
    """
    if save_policy_every_n_steps and policy_save_fn is None:
      raise ValueError(
          'policy_save_fn is required when save_policy_every_n_steps > 0')
    self.max_number_of_steps = max_number_of_steps
    self.num_updates_per_observation = num_updates_per_observation
    self.num_collect_per_update = num_collect_per_update
    self.num_collect_per_meta_update = num_collect_per_meta_update
    self.log_every_n_steps = log_every_n_steps
    self.policy_save_fn = policy_save_fn
    self.save_policy_every_n_steps = save_policy_every_n_steps
    self.should_stop_early = should_stop_early
    self.last_global_step_val = 0
    self.train_op_fn = None
    self.collect_and_train_fn = None
    tf.logging.info('Training for %d max_number_of_steps',
                    self.max_number_of_steps)

  def train_step(self, sess, train_ops, global_step, _):
    """This function will be called at each step of training.

    This represents one step of the DDPG algorithm and can include:
    1. collect a <state, action, reward, next_state> transition
    2. update the target network
    3. train the actor
    4. train the critic

    Args:
      sess: A Tensorflow session.
      train_ops: A DdpgTrainOps tuple of train ops to run.
      global_step: The global step.

    Returns:
      A scalar total loss.
      A boolean should stop.
    """
    start_time = time.time()
    if self.train_op_fn is None:
      self.train_op_fn = sess.make_callable([train_ops.train_op, global_step])
      self.meta_train_op_fn = sess.make_callable([train_ops.meta_train_op, global_step])
      self.collect_fn = sess.make_callable([train_ops.collect_experience_op, global_step])
      self.collect_and_train_fn = sess.make_callable(
          [train_ops.train_op, global_step, train_ops.collect_experience_op])
      self.collect_and_meta_train_fn = sess.make_callable(
          [train_ops.meta_train_op, global_step, train_ops.collect_experience_op])
    for _ in range(self.num_collect_per_update - 1):
      self.collect_fn()
    for _ in range(self.num_updates_per_observation - 1):
      self.train_op_fn()

    total_loss, global_step_val, _ = self.collect_and_train_fn()
    if (global_step_val // self.num_collect_per_meta_update !=
        self.last_global_step_val // self.num_collect_per_meta_update):
      self.meta_train_op_fn()

    time_elapsed = time.time() - start_time
    should_stop = False
    if self.max_number_of_steps:
      should_stop = global_step_val >= self.max_number_of_steps
    if global_step_val != self.last_global_step_val:
      if (self.save_policy_every_n_steps and
          global_step_val // self.save_policy_every_n_steps !=
          self.last_global_step_val // self.save_policy_every_n_steps):
        self.policy_save_fn(sess)

      if (self.log_every_n_steps and
          global_step_val % self.log_every_n_steps == 0):
        tf.logging.info(
            'global step %d: loss = %.4f (%.3f sec/step) (%d steps/sec)',
            global_step_val, total_loss, time_elapsed, 1 / time_elapsed)

    self.last_global_step_val = global_step_val
    stop_early = bool(self.should_stop_early and self.should_stop_early())
    return total_loss, should_stop or stop_early


def create_counter_summaries(counters):
  """Add named summaries to counters, a list of tuples (name, counter)."""
  if counters:
    with tf.name_scope('Counters/'):
      for name, counter in counters:
        tf.summary.scalar(name, counter)


def gen_debug_batch_summaries(batch):
  """Generates summaries for the sampled replay batch."""
  states, actions, rewards, _, next_states = batch
  with tf.name_scope('batch'):
    for s in range(states.get_shape()[-1]):
      tf.summary.histogram('states_%d' % s, states[:, s])
    for s in range(states.get_shape()[-1]):
      tf.summary.histogram('next_states_%d' % s, next_states[:, s])
    for a in range(actions.get_shape()[-1]):
      tf.summary.histogram('actions_%d' % a, actions[:, a])
    tf.summary.histogram('rewards', rewards)
