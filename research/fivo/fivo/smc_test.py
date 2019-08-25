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

"""Tests for fivo.smc."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy
import tensorflow as tf

from fivo import smc

lse = scipy.special.logsumexp


def _simple_transition_fn(state, unused_t):
  if state is None:
    return tf.zeros([4], dtype=tf.float32)
  return tf.constant([5., 4., 1., 0.5]), tf.zeros([4], dtype=tf.float32)


def _resample_at_step_criterion(step):
  """A criterion that resamples once at a specific timestep."""
  def criterion(log_weights, t):
    batch_size = tf.shape(log_weights)[1]
    return tf.fill([batch_size], tf.equal(t, step))
  return criterion


class SMCTest(tf.test.TestCase):

  def test_never_resampling(self):
    """Test that never_resample_criterion makes smc not resample.

    Also test that the weights and log_z_hat are computed correctly when never
    resampling.
    """
    tf.set_random_seed(1234)
    with self.test_session() as sess:
      outs = smc.smc(
          _simple_transition_fn,
          num_steps=tf.convert_to_tensor([5, 3]),
          num_particles=2,
          resampling_criterion=smc.never_resample_criterion)
      log_z_hat, weights, resampled = sess.run(outs[0:3])
      gt_weights = np.array(
          [[[5, 1], [4, .5]],
           [[10, 2], [8, 1]],
           [[15, 3], [12, 1.5]],
           [[20, 4], [12, 1.5]],
           [[25, 5], [12, 1.5]]],
          dtype=np.float32)
      gt_log_z_hat = np.array(
          [lse([25, 5]) - np.log(2),
           lse([12, 1.5]) - np.log(2)],
          dtype=np.float32)
      self.assertAllClose(gt_log_z_hat, log_z_hat)
      self.assertAllClose(gt_weights, weights)
      self.assertAllEqual(np.zeros_like(resampled), resampled)

  def test_always_resampling(self):
    """Test always_resample_criterion makes smc always resample.

    Past a sequence end the filter should not resample, however.
    Also check that weights and log_z_hat estimate are correct.
    """
    tf.set_random_seed(1234)
    with self.test_session() as sess:
      outs = smc.smc(
          _simple_transition_fn,
          num_steps=tf.convert_to_tensor([5, 3]),
          num_particles=2,
          resampling_criterion=smc.always_resample_criterion)
      log_z_hat, weights, resampled = sess.run(outs[0:3])
      gt_weights = np.array(
          [[[5, 1], [4, .5]],
           [[5, 1], [4, .5]],
           [[5, 1], [4, .5]],
           [[5, 1], [0., 0.]],
           [[5, 1], [0., 0.]]],
          dtype=np.float32)
      gt_log_z_hat = np.array(
          [5*lse([5, 1]) - 5*np.log(2),
           3*lse([4, .5]) - 3*np.log(2)],
          dtype=np.float32)
      gt_resampled = np.array(
          [[1, 1], [1, 1], [1, 1], [1, 0], [1, 0]],
          dtype=np.float32)
      self.assertAllClose(gt_log_z_hat, log_z_hat)
      self.assertAllClose(gt_weights, weights)
      self.assertAllEqual(gt_resampled, resampled)

  def test_weights_reset_when_resampling_at_sequence_end(self):
    """Test that the weights are reset when resampling at the sequence end.

    When resampling happens on the last timestep of a sequence the weights
    should be set to zero on the next timestep and remain zero afterwards.
    """
    tf.set_random_seed(1234)
    with self.test_session() as sess:
      outs = smc.smc(
          _simple_transition_fn,
          num_steps=tf.convert_to_tensor([5, 3]),
          num_particles=2,
          resampling_criterion=_resample_at_step_criterion(2))
      log_z_hat, weights, resampled = sess.run(outs[0:3])
      gt_log_z = np.array(
          [lse([15, 3]) + lse([10, 2]) - 2*np.log(2),
           lse([12, 1.5]) - np.log(2)],
          dtype=np.float32)
      gt_resampled = np.array(
          [[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]],
          dtype=np.float32)
      gt_weights = np.array(
          [[[5, 1], [4, .5]],
           [[10, 2], [8, 1]],
           [[15, 3], [12, 1.5]],
           [[5, 1], [0, 0]],
           [[10, 2], [0, 0]]],
          dtype=np.float32)
      self.assertAllClose(gt_log_z, log_z_hat)
      self.assertAllEqual(gt_resampled, resampled)
      self.assertAllEqual(gt_weights, weights)

  def test_weights_not_updated_past_sequence_end(self):
    """Test that non-zero weights are not updated past the end of a sequence."""
    tf.set_random_seed(1234)
    with self.test_session() as sess:
      outs = smc.smc(
          _simple_transition_fn,
          num_steps=tf.convert_to_tensor([6, 4]),
          num_particles=2,
          resampling_criterion=_resample_at_step_criterion(1))
      log_z_hat, weights, resampled = sess.run(outs[0:3])
      gt_log_z_hat = np.array(
          [lse([10, 2]) + lse([20, 4]) - 2*np.log(2),
           lse([8, 1]) + lse([8, 1]) - 2*np.log(2)],
          dtype=np.float32)
      # Ensure that we only resample on the 2nd timestep.
      gt_resampled = np.array(
          [[0, 0], [1, 1], [0, 0], [0, 0], [0, 0], [0, 0]],
          dtype=np.float32)
      # Ensure that the weights after the end of the sequence don't change.
      # Ensure that the weights after resampling before the end of the sequence
      # do change.
      gt_weights = np.array(
          [[[5, 1], [4, .5]],
           [[10, 2], [8, 1]],
           [[5, 1], [4, .5]],
           [[10, 2], [8, 1]],
           [[15, 3], [8, 1]],
           [[20, 4], [8, 1]]],
          dtype=np.float32)
      self.assertAllClose(gt_log_z_hat, log_z_hat)
      self.assertAllEqual(gt_resampled, resampled)
      self.assertAllEqual(gt_weights, weights)

  def test_resampling_on_max_num_steps(self):
    """Test that everything is correct when resampling on step max_num_steps.

    When resampling on step max_num_steps (i.e. the last step of the longest
    sequence), ensure that there are no off-by-one errors preventing resampling
    and also that the weights are not updated.
    """
    tf.set_random_seed(1234)
    with self.test_session() as sess:
      outs = smc.smc(
          _simple_transition_fn,
          num_steps=tf.convert_to_tensor([4, 2]),
          num_particles=2,
          resampling_criterion=_resample_at_step_criterion(3))
      log_z_hat, weights, resampled = sess.run(outs[0:3])
      gt_log_z_hat = np.array(
          [lse([20, 4]) - np.log(2),
           lse([8, 1]) - np.log(2)],
          dtype=np.float32)
      # Ensure that we only resample on the 3rd timestep and that the second
      # filter doesn't resample at all because it is only run for 2 steps.
      gt_resampled = np.array(
          [[0, 0], [0, 0], [0, 0], [1, 0]],
          dtype=np.float32)
      gt_weights = np.array(
          [[[5, 1], [4, .5]],
           [[10, 2], [8, 1]],
           [[15, 3], [8, 1]],
           [[20, 4], [8, 1]]],
          dtype=np.float32)
      self.assertAllClose(gt_log_z_hat, log_z_hat)
      self.assertAllEqual(gt_resampled, resampled)
      self.assertAllEqual(gt_weights, weights)

  def test_multinomial_resampling(self):
    """Test that mulitnomial resampling selects the correct states."""
    tf.set_random_seed(1234)
    with self.test_session() as sess:
      # Setup input.
      inf = 1000.0  # Very large value in log space.
      num_samples = 2
      batch_size = 2
      log_weights = tf.convert_to_tensor([[inf, 0], [0, inf]])
      states = tf.convert_to_tensor([1, 2, 3, 4])
      # Run test.
      resampled_states = smc.multinomial_resampling(
          log_weights, states, num_samples, batch_size, random_seed=0)
      resampled_states_values = sess.run(resampled_states)
      self.assertAllEqual(resampled_states_values, [1, 4, 1, 4])

  def test_blend_tensor(self):
    """Test that relaxed resampling blends the correct states."""
    tf.set_random_seed(1234)
    with self.test_session() as sess:
      # Setup input.
      num_samples = 2
      batch_size = 2
      blending_weights = tf.convert_to_tensor(
          [[[0.5, 0.5], [0.25, 0.75]], [[0.75, 0.25], [0.5, 0.5]]])
      states = tf.convert_to_tensor([4., 8., 12., 16.])
      # Run test.
      blended_states = smc._blend_tensor(blending_weights, states,
                                         num_samples, batch_size)
      blended_states_values = sess.run(blended_states)
      self.assertAllClose(blended_states_values[:, 0], [8., 14., 6., 12.])


if __name__ == '__main__':
  tf.test.main()
