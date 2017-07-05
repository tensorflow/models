# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Utilities for training adversarial text models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

# Dependency imports

import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'Master address.')
flags.DEFINE_integer('task', 0, 'Task id of the replica running the training.')
flags.DEFINE_integer('ps_tasks', 0, 'Number of parameter servers.')
flags.DEFINE_string('train_dir', '/tmp/text_train',
                    'Directory for logs and checkpoints.')
flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')
flags.DEFINE_boolean('log_device_placement', False,
                     'Whether to log device placement.')


def run_training(train_op,
                 loss,
                 global_step,
                 variables_to_restore=None,
                 pretrained_model_dir=None):
  """Sets up and runs training loop."""
  tf.gfile.MakeDirs(FLAGS.train_dir)

  # Create pretrain Saver
  if pretrained_model_dir:
    assert variables_to_restore
    tf.logging.info('Will attempt restore from %s: %s', pretrained_model_dir,
                    variables_to_restore)
    saver_for_restore = tf.train.Saver(variables_to_restore)

  # Init ops
  if FLAGS.sync_replicas:
    local_init_op = tf.get_collection('local_init_op')[0]
    ready_for_local_init_op = tf.get_collection('ready_for_local_init_op')[0]
  else:
    local_init_op = tf.train.Supervisor.USE_DEFAULT
    ready_for_local_init_op = tf.train.Supervisor.USE_DEFAULT

  is_chief = FLAGS.task == 0
  sv = tf.train.Supervisor(
      logdir=FLAGS.train_dir,
      is_chief=is_chief,
      save_summaries_secs=5 * 60,
      save_model_secs=5 * 60,
      local_init_op=local_init_op,
      ready_for_local_init_op=ready_for_local_init_op,
      global_step=global_step)

  # Delay starting standard services to allow possible pretrained model restore.
  with sv.managed_session(
      master=FLAGS.master,
      config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement),
      start_standard_services=False) as sess:
    # Initialization
    if is_chief:
      if pretrained_model_dir:
        maybe_restore_pretrained_model(sess, saver_for_restore,
                                       pretrained_model_dir)
      if FLAGS.sync_replicas:
        sess.run(tf.get_collection('chief_init_op')[0])
      sv.start_standard_services(sess)

    sv.start_queue_runners(sess)

    # Training loop
    global_step_val = 0
    while not sv.should_stop() and global_step_val < FLAGS.max_steps:
      global_step_val = train_step(sess, train_op, loss, global_step)
    sv.stop()

    # Final checkpoint
    if is_chief:
      sv.saver.save(sess, sv.save_path, global_step=global_step)


def maybe_restore_pretrained_model(sess, saver_for_restore, model_dir):
  """Restores pretrained model if there is no ckpt model."""
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  checkpoint_exists = ckpt and ckpt.model_checkpoint_path
  if checkpoint_exists:
    tf.logging.info('Checkpoint exists in FLAGS.train_dir; skipping '
                    'pretraining restore')
    return

  pretrain_ckpt = tf.train.get_checkpoint_state(model_dir)
  if not (pretrain_ckpt and pretrain_ckpt.model_checkpoint_path):
    raise ValueError(
        'Asked to restore model from %s but no checkpoint found.' % model_dir)
  saver_for_restore.restore(sess, pretrain_ckpt.model_checkpoint_path)


def train_step(sess, train_op, loss, global_step):
  """Runs a single training step."""
  start_time = time.time()
  _, loss_val, global_step_val = sess.run([train_op, loss, global_step])
  duration = time.time() - start_time

  # Logging
  if global_step_val % 10 == 0:
    examples_per_sec = FLAGS.batch_size / duration
    sec_per_batch = float(duration)

    format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f ' 'sec/batch)')
    tf.logging.info(format_str % (global_step_val, loss_val, examples_per_sec,
                                  sec_per_batch))

  if np.isnan(loss_val):
    raise OverflowError('Loss is nan')

  return global_step_val
