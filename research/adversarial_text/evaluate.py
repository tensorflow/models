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
"""Evaluates text classification model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time

# Dependency imports

import tensorflow as tf

import graphs

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('master', '',
                    'BNS name prefix of the Tensorflow eval master, '
                    'or "local".')
flags.DEFINE_string('eval_dir', '/tmp/text_eval',
                    'Directory where to write event logs.')
flags.DEFINE_string('eval_data', 'test', 'Specify which dataset is used. '
                    '("train", "valid", "test") ')

flags.DEFINE_string('checkpoint_dir', '/tmp/text_train',
                    'Directory where to read model checkpoints.')
flags.DEFINE_integer('eval_interval_secs', 60, 'How often to run the eval.')
flags.DEFINE_integer('num_examples', 32, 'Number of examples to run.')
flags.DEFINE_bool('run_once', False, 'Whether to run eval only once.')


def restore_from_checkpoint(sess, saver):
  """Restore model from checkpoint.

  Args:
    sess: Session.
    saver: Saver for restoring the checkpoint.

  Returns:
    bool: Whether the checkpoint was found and restored
  """
  ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
  if not ckpt or not ckpt.model_checkpoint_path:
    tf.logging.info('No checkpoint found at %s', FLAGS.checkpoint_dir)
    return False

  saver.restore(sess, ckpt.model_checkpoint_path)
  return True


def run_eval(eval_ops, summary_writer, saver):
  """Runs evaluation over FLAGS.num_examples examples.

  Args:
    eval_ops: dict<metric name, tuple(value, update_op)>
    summary_writer: Summary writer.
    saver: Saver.

  Returns:
    dict<metric name, value>, with value being the average over all examples.
  """
  sv = tf.train.Supervisor(logdir=FLAGS.eval_dir, saver=None, summary_op=None)
  with sv.managed_session(
      master=FLAGS.master, start_standard_services=False) as sess:
    if not restore_from_checkpoint(sess, saver):
      return
    sv.start_queue_runners(sess)

    metric_names, ops = zip(*eval_ops.items())
    value_ops, update_ops = zip(*ops)

    value_ops_dict = dict(zip(metric_names, value_ops))

    # Run update ops
    num_batches = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
    tf.logging.info('Running %d batches for evaluation.', num_batches)
    for i in range(num_batches):
      if (i + 1) % 10 == 0:
        tf.logging.info('Running batch %d/%d...', i + 1, num_batches)
      if (i + 1) % 50 == 0:
        _log_values(sess, value_ops_dict)
      sess.run(update_ops)

    _log_values(sess, value_ops_dict, summary_writer=summary_writer)


def _log_values(sess, value_ops, summary_writer=None):
  """Evaluate, log, and write summaries of the eval metrics in value_ops."""
  metric_names, value_ops = zip(*value_ops.items())
  values = sess.run(value_ops)

  tf.logging.info('Eval metric values:')
  summary = tf.summary.Summary()
  for name, val in zip(metric_names, values):
    summary.value.add(tag=name, simple_value=val)
    tf.logging.info('%s = %.3f', name, val)

  if summary_writer is not None:
    global_step_val = sess.run(tf.train.get_global_step())
    summary_writer.add_summary(summary, global_step_val)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.gfile.MakeDirs(FLAGS.eval_dir)
  tf.logging.info('Building eval graph...')
  output = graphs.get_model().eval_graph(FLAGS.eval_data)
  eval_ops, moving_averaged_variables = output

  saver = tf.train.Saver(moving_averaged_variables)
  summary_writer = tf.summary.FileWriter(
      FLAGS.eval_dir, graph=tf.get_default_graph())

  while True:
    run_eval(eval_ops, summary_writer, saver)
    if FLAGS.run_once:
      break
    time.sleep(FLAGS.eval_interval_secs)


if __name__ == '__main__':
  tf.app.run()
