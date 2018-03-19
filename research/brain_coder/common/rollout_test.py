from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Tests for common.rollout."""

import numpy as np
import tensorflow as tf

from common import rollout as rollout_lib  # brain coder


class RolloutTest(tf.test.TestCase):

  def MakeRollout(self, states, actions, rewards, values=None, terminated=True):
    rollout = rollout_lib.Rollout()
    rollout.add_many(
        states=states, actions=actions, rewards=rewards, values=values,
        terminated=terminated)
    return rollout

  def testDiscount(self):
    discounted = np.array([1.0 / 2 ** n for n in range(4, -1, -1)])
    discounted[:2] += [1.0 / 2 ** n for n in range(1, -1, -1)]

    self.assertTrue(np.array_equal(
        rollout_lib.discount([0.0, 1.0, 0.0, 0.0, 1.0], 0.50),
        discounted))
    self.assertTrue(np.array_equal(
        rollout_lib.discount(np.array([0.0, 1.0, 0.0, 0.0, 1.0]), 0.50),
        discounted))

  def testDiscountedAdvantageAndRewards(self):
    # lambda=1, No bootstrapping.
    values = [0.1, 0.5, 0.5, 0.25]
    (empirical_values,
     generalized_advantage) = rollout_lib.discounted_advantage_and_rewards(
         [0.0, 0.0, 0.0, 1.0],
         values,
         gamma=0.75,
         lambda_=1.0)
    expected_discounted_r = (
        np.array([1.0 * 0.75 ** n for n in range(3, -1, -1)]))
    expected_adv = expected_discounted_r - values
    self.assertTrue(np.array_equal(empirical_values, expected_discounted_r))
    self.assertTrue(np.allclose(generalized_advantage, expected_adv))

    # lambda=1, With bootstrapping.
    values = [0.1, 0.5, 0.5, 0.25, 0.75]
    (empirical_values,
     generalized_advantage) = rollout_lib.discounted_advantage_and_rewards(
         [0.0, 0.0, 0.0, 1.0],
         values,
         gamma=0.75,
         lambda_=1.0)
    expected_discounted_r = (
        np.array([0.75 * 0.75 ** n for n in range(4, 0, -1)])
        + np.array([1.0 * 0.75 ** n for n in range(3, -1, -1)]))
    expected_adv = expected_discounted_r - values[:-1]
    self.assertTrue(np.array_equal(empirical_values, expected_discounted_r))
    self.assertTrue(np.allclose(generalized_advantage, expected_adv))

    # lambda=0.5, With bootstrapping.
    values = [0.1, 0.5, 0.5, 0.25, 0.75]
    rewards = [0.0, 0.0, 0.0, 1.0]
    l = 0.5  # lambda
    g = 0.75  # gamma
    (empirical_values,
     generalized_advantage) = rollout_lib.discounted_advantage_and_rewards(
         rewards,
         values,
         gamma=g,
         lambda_=l)
    expected_discounted_r = (
        np.array([0.75 * g ** n for n in range(4, 0, -1)])
        + np.array([1.0 * g ** n for n in range(3, -1, -1)]))
    expected_adv = [0.0] * len(values)
    for t in range(3, -1, -1):
      delta_t = rewards[t] + g * values[t + 1] - values[t]
      expected_adv[t] = delta_t + g * l * expected_adv[t + 1]
    expected_adv = expected_adv[:-1]
    self.assertTrue(np.array_equal(empirical_values, expected_discounted_r))
    self.assertTrue(np.allclose(generalized_advantage, expected_adv))

  def testProcessRollouts(self):
    g = 0.95
    rollouts = [
        self.MakeRollout(
            states=[3, 6, 9],
            actions=[1, 2, 3],
            rewards=[1.0, -1.0, 0.5],
            values=[0.5, 0.5, 0.1]),
        self.MakeRollout(
            states=[10],
            actions=[5],
            rewards=[1.0],
            values=[0.5])]
    batch = rollout_lib.process_rollouts(rollouts, gamma=g)

    self.assertEqual(2, batch.batch_size)
    self.assertEqual(3, batch.max_time)
    self.assertEqual([3, 1], batch.episode_lengths)
    self.assertEqual([0.5, 1.0], batch.total_rewards)
    self.assertEqual(
        [[3, 6, 9], [10, 0, 0]],
        batch.states.tolist())
    self.assertEqual(
        [[1, 2, 3], [5, 0, 0]],
        batch.actions.tolist())

    rew1, rew2 = rollouts[0].rewards, rollouts[1].rewards
    expected_discounted_rewards = [
        [rew1[0] + g * rew1[1] + g * g * rew1[2],
         rew1[1] + g * rew1[2],
         rew1[2]],
        [rew2[0], 0.0, 0.0]]
    expected_advantages = [
        [dr - v
         for dr, v
         in zip(expected_discounted_rewards[0], rollouts[0].values)],
        [expected_discounted_rewards[1][0] - rollouts[1].values[0], 0.0, 0.0]]
    self.assertTrue(
        np.allclose(expected_discounted_rewards, batch.discounted_r))
    self.assertTrue(
        np.allclose(expected_advantages, batch.discounted_adv))


if __name__ == '__main__':
  tf.test.main()
