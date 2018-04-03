# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for hooks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf  # pylint: disable=g-bad-import-order
from tensorflow.python.training import monitored_session  # pylint: disable=g-bad-import-order

from official.utils.logs import hooks


tf.logging.set_verbosity(tf.logging.ERROR)


class ExamplesPerSecondHookTest(tf.test.TestCase):
  """Tests for the ExamplesPerSecondHook."""

  def setUp(self):
    """Mock out logging calls to verify if correct info is being monitored."""
    self._actual_log = tf.logging.info
    self.logged_message = None

    def mock_log(*args, **kwargs):
      self.logged_message = args
      self._actual_log(*args, **kwargs)

    tf.logging.info = mock_log

    self.graph = tf.Graph()
    with self.graph.as_default():
      self.global_step = tf.train.get_or_create_global_step()
      self.train_op = tf.assign_add(self.global_step, 1)

  def tearDown(self):
    tf.logging.info = self._actual_log

  def test_raise_in_both_secs_and_steps(self):
    with self.assertRaises(ValueError):
      hooks.ExamplesPerSecondHook(
          batch_size=256,
          every_n_steps=10,
          every_n_secs=20)

  def test_raise_in_none_secs_and_steps(self):
    with self.assertRaises(ValueError):
      hooks.ExamplesPerSecondHook(
          batch_size=256,
          every_n_steps=None,
          every_n_secs=None)

  def _validate_log_every_n_steps(self, sess, every_n_steps, warm_steps):
    hook = hooks.ExamplesPerSecondHook(
        batch_size=256,
        every_n_steps=every_n_steps,
        warm_steps=warm_steps)
    hook.begin()
    mon_sess = monitored_session._HookedSession(sess, [hook])  # pylint: disable=protected-access
    sess.run(tf.global_variables_initializer())

    self.logged_message = ''
    for _ in range(every_n_steps):
      mon_sess.run(self.train_op)
      self.assertEqual(str(self.logged_message).find('exp/sec'), -1)

    mon_sess.run(self.train_op)
    global_step_val = sess.run(self.global_step)
    # assertNotRegexpMatches is not supported by python 3.1 and later
    if global_step_val > warm_steps:
      self.assertRegexpMatches(str(self.logged_message), 'exp/sec')
    else:
      self.assertEqual(str(self.logged_message).find('exp/sec'), -1)

    # Add additional run to verify proper reset when called multiple times.
    self.logged_message = ''
    mon_sess.run(self.train_op)
    global_step_val = sess.run(self.global_step)
    if every_n_steps == 1 and global_step_val > warm_steps:
      self.assertRegexpMatches(str(self.logged_message), 'exp/sec')
    else:
      self.assertEqual(str(self.logged_message).find('exp/sec'), -1)

    hook.end(sess)

  def test_examples_per_sec_every_1_steps(self):
    with self.graph.as_default(), tf.Session() as sess:
      self._validate_log_every_n_steps(sess, 1, 0)

  def test_examples_per_sec_every_5_steps(self):
    with self.graph.as_default(), tf.Session() as sess:
      self._validate_log_every_n_steps(sess, 5, 0)

  def test_examples_per_sec_every_1_steps_with_warm_steps(self):
    with self.graph.as_default(), tf.Session() as sess:
      self._validate_log_every_n_steps(sess, 1, 10)

  def test_examples_per_sec_every_5_steps_with_warm_steps(self):
    with self.graph.as_default(), tf.Session() as sess:
      self._validate_log_every_n_steps(sess, 5, 10)

  def _validate_log_every_n_secs(self, sess, every_n_secs):
    hook = hooks.ExamplesPerSecondHook(
        batch_size=256,
        every_n_steps=None,
        every_n_secs=every_n_secs)
    hook.begin()
    mon_sess = monitored_session._HookedSession(sess, [hook])  # pylint: disable=protected-access
    sess.run(tf.global_variables_initializer())

    self.logged_message = ''
    mon_sess.run(self.train_op)
    self.assertEqual(str(self.logged_message).find('exp/sec'), -1)
    time.sleep(every_n_secs)

    self.logged_message = ''
    mon_sess.run(self.train_op)
    self.assertRegexpMatches(str(self.logged_message), 'exp/sec')

    hook.end(sess)

  def test_examples_per_sec_every_1_secs(self):
    with self.graph.as_default(), tf.Session() as sess:
      self._validate_log_every_n_secs(sess, 1)

  def test_examples_per_sec_every_5_secs(self):
    with self.graph.as_default(), tf.Session() as sess:
      self._validate_log_every_n_secs(sess, 5)


if __name__ == '__main__':
  tf.test.main()
