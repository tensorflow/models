from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Tests for common.reward."""

from math import log
import numpy as np
import tensorflow as tf

from common import reward  # brain coder


class RewardTest(tf.test.TestCase):

  def testAbsDiff(self):
    self.assertEqual(5, reward.abs_diff(15, 20))
    self.assertEqual(5, reward.abs_diff(20, 15))

  def testModAbsDiff(self):
    self.assertEqual(5, reward.mod_abs_diff(15, 20, 25))
    self.assertEqual(5, reward.mod_abs_diff(20, 15, 25))
    self.assertEqual(2, reward.mod_abs_diff(1, 24, 25))
    self.assertEqual(2, reward.mod_abs_diff(24, 1, 25))

    self.assertEqual(0, reward.mod_abs_diff(0, 0, 5))
    self.assertEqual(1, reward.mod_abs_diff(0, 1, 5))
    self.assertEqual(2, reward.mod_abs_diff(0, 2, 5))
    self.assertEqual(2, reward.mod_abs_diff(0, 3, 5))
    self.assertEqual(1, reward.mod_abs_diff(0, 4, 5))

    self.assertEqual(0, reward.mod_abs_diff(-1, 4, 5))
    self.assertEqual(1, reward.mod_abs_diff(-5, 4, 5))
    self.assertEqual(1, reward.mod_abs_diff(-7, 4, 5))
    self.assertEqual(1, reward.mod_abs_diff(13, 4, 5))
    self.assertEqual(1, reward.mod_abs_diff(15, 4, 5))

  def testAbsoluteDistance_AbsDiffMethod(self):
    self.assertEqual(
        4,
        reward.absolute_distance([0], [4], 5, scalar_diff_fn=reward.abs_diff))
    self.assertEqual(
        0,
        reward.absolute_distance([4], [4], 5, scalar_diff_fn=reward.abs_diff))
    self.assertEqual(
        0,
        reward.absolute_distance([], [], 5, scalar_diff_fn=reward.abs_diff))
    self.assertEqual(
        5,
        reward.absolute_distance([1], [], 5, scalar_diff_fn=reward.abs_diff))
    self.assertEqual(
        5,
        reward.absolute_distance([], [1], 5, scalar_diff_fn=reward.abs_diff))
    self.assertEqual(
        0,
        reward.absolute_distance([1, 2, 3], [1, 2, 3], 5,
                                 scalar_diff_fn=reward.abs_diff))
    self.assertEqual(
        1,
        reward.absolute_distance([1, 2, 4], [1, 2, 3], 5,
                                 scalar_diff_fn=reward.abs_diff))
    self.assertEqual(
        1,
        reward.absolute_distance([1, 2, 2], [1, 2, 3], 5,
                                 scalar_diff_fn=reward.abs_diff))
    self.assertEqual(
        5,
        reward.absolute_distance([1, 2], [1, 2, 3], 5,
                                 scalar_diff_fn=reward.abs_diff))
    self.assertEqual(
        5,
        reward.absolute_distance([1, 2, 3, 4], [1, 2, 3], 5,
                                 scalar_diff_fn=reward.abs_diff))
    self.assertEqual(
        6,
        reward.absolute_distance([4, 4, 4], [1, 2, 3], 5,
                                 scalar_diff_fn=reward.abs_diff))

  def testAbsoluteDistance_ModDiffMethod(self):
    self.assertEqual(
        1,
        reward.absolute_distance([0], [4], 5,
                                 scalar_diff_fn=reward.mod_abs_diff))
    self.assertEqual(
        0,
        reward.absolute_distance([4], [4], 5,
                                 scalar_diff_fn=reward.mod_abs_diff))
    self.assertEqual(
        0,
        reward.absolute_distance([], [], 5,
                                 scalar_diff_fn=reward.mod_abs_diff))
    self.assertEqual(
        5,
        reward.absolute_distance([1], [], 5,
                                 scalar_diff_fn=reward.mod_abs_diff))
    self.assertEqual(
        5,
        reward.absolute_distance([], [1], 5,
                                 scalar_diff_fn=reward.mod_abs_diff))
    self.assertEqual(
        0,
        reward.absolute_distance([1, 2, 3], [1, 2, 3], 5,
                                 scalar_diff_fn=reward.mod_abs_diff))
    self.assertEqual(
        1,
        reward.absolute_distance([1, 2, 4], [1, 2, 3], 5,
                                 scalar_diff_fn=reward.mod_abs_diff))
    self.assertEqual(
        1,
        reward.absolute_distance([1, 2, 2], [1, 2, 3], 5,
                                 scalar_diff_fn=reward.mod_abs_diff))
    self.assertEqual(
        5,
        reward.absolute_distance([1, 2], [1, 2, 3], 5,
                                 scalar_diff_fn=reward.mod_abs_diff))
    self.assertEqual(
        5,
        reward.absolute_distance([1, 2, 3, 4], [1, 2, 3], 5,
                                 scalar_diff_fn=reward.mod_abs_diff))
    self.assertEqual(
        5,
        reward.absolute_distance([4, 4, 4], [1, 2, 3], 5,
                                 scalar_diff_fn=reward.mod_abs_diff))

  def testLogAbsoluteDistance(self):
    def log_diff(diff, base):
      return log(diff + 1) / log(base // 2 + 2)

    self.assertEqual(
        log_diff(1, 5),
        reward.log_absolute_distance([0], [4], 5))
    self.assertEqual(
        log_diff(2, 5),
        reward.log_absolute_distance([1], [4], 5))
    self.assertEqual(
        log_diff(2, 5),
        reward.log_absolute_distance([2], [4], 5))
    self.assertEqual(
        log_diff(1, 5),
        reward.log_absolute_distance([3], [4], 5))
    self.assertEqual(
        log_diff(3, 5),  # max_dist = base // 2 + 1 = 3
        reward.log_absolute_distance([], [4], 5))
    self.assertEqual(
        0 + log_diff(3, 5),  # max_dist = base // 2 + 1 = 3
        reward.log_absolute_distance([4, 4], [4], 5))
    self.assertEqual(
        0,
        reward.log_absolute_distance([4], [4], 5))
    self.assertEqual(
        0,
        reward.log_absolute_distance([], [], 5))
    self.assertEqual(
        1,
        reward.log_absolute_distance([1], [], 5))
    self.assertEqual(
        1,
        reward.log_absolute_distance([], [1], 5))

    self.assertEqual(
        0,
        reward.log_absolute_distance([1, 2, 3], [1, 2, 3], 5))
    self.assertEqual(
        log_diff(1, 5) / 3,  # divided by target length.
        reward.log_absolute_distance([1, 2, 4], [1, 2, 3], 5))
    self.assertEqual(
        log_diff(1, 5) / 3,
        reward.log_absolute_distance([1, 2, 2], [1, 2, 3], 5))
    self.assertEqual(
        log_diff(3, 5) / 3,  # max_dist
        reward.log_absolute_distance([1, 2], [1, 2, 3], 5))
    self.assertEqual(
        log_diff(3, 5) / 3,  # max_dist
        reward.log_absolute_distance([1, 2, 3, 4], [1, 2, 3], 5))
    # Add log differences for each position.
    self.assertEqual(
        (log_diff(2, 5) + log_diff(2, 5) + log_diff(1, 5)) / 3,
        reward.log_absolute_distance([4, 4, 4], [1, 2, 3], 5))

  def testAbsoluteDistanceReward(self):
    self.assertEqual(
        1,
        reward.absolute_distance_reward([1, 2, 3], [1, 2, 3], 5))
    self.assertEqual(
        1 - 1 / (5 * 3.),  # 1 - distance / (base * target_len)
        reward.absolute_distance_reward([1, 2, 4], [1, 2, 3], 5))
    self.assertEqual(
        1 - 1 / (5 * 3.),
        reward.absolute_distance_reward([1, 2, 2], [1, 2, 3], 5))
    self.assertTrue(np.isclose(
        1 - 5 / (5 * 3.),
        reward.absolute_distance_reward([1, 2], [1, 2, 3], 5)))
    self.assertTrue(np.isclose(
        1 - 5 / (5 * 3.),
        reward.absolute_distance_reward([1, 2, 3, 4], [1, 2, 3], 5)))
    # Add log differences for each position.
    self.assertEqual(
        1 - (3 + 2 + 1) / (5 * 3.),
        reward.absolute_distance_reward([4, 4, 4], [1, 2, 3], 5))
    self.assertEqual(
        1,
        reward.absolute_distance_reward([], [], 5))

  def testAbsoluteModDistanceReward(self):
    self.assertEqual(
        1,
        reward.absolute_mod_distance_reward([1, 2, 3], [1, 2, 3], 5))
    self.assertEqual(
        1 - 1 / (5 * 3.),  # 1 - distance / (base * target_len)
        reward.absolute_mod_distance_reward([1, 2, 4], [1, 2, 3], 5))
    self.assertEqual(
        1 - 1 / (5 * 3.),
        reward.absolute_mod_distance_reward([1, 2, 2], [1, 2, 3], 5))
    self.assertTrue(np.isclose(
        1 - 5 / (5 * 3.),
        reward.absolute_mod_distance_reward([1, 2], [1, 2, 3], 5)))
    self.assertTrue(np.isclose(
        1 - 5 / (5 * 3.),
        reward.absolute_mod_distance_reward([1, 2, 3, 4], [1, 2, 3], 5)))
    # Add log differences for each position.
    self.assertTrue(np.isclose(
        1 - (2 + 2 + 1) / (5 * 3.),
        reward.absolute_mod_distance_reward([4, 4, 4], [1, 2, 3], 5)))
    self.assertTrue(np.isclose(
        1 - (1 + 2 + 2) / (5 * 3.),
        reward.absolute_mod_distance_reward([0, 1, 2], [4, 4, 4], 5)))
    self.assertEqual(
        1,
        reward.absolute_mod_distance_reward([], [], 5))

  def testAbsoluteLogDistanceReward(self):
    def log_diff(diff, base):
      return log(diff + 1) / log(base // 2 + 2)

    self.assertEqual(
        1,
        reward.absolute_log_distance_reward([1, 2, 3], [1, 2, 3], 5))
    self.assertEqual(
        1 - log_diff(1, 5) / 3,  # divided by target length.
        reward.absolute_log_distance_reward([1, 2, 4], [1, 2, 3], 5))
    self.assertEqual(
        1 - log_diff(1, 5) / 3,
        reward.absolute_log_distance_reward([1, 2, 2], [1, 2, 3], 5))
    self.assertEqual(
        1 - log_diff(3, 5) / 3,  # max_dist
        reward.absolute_log_distance_reward([1, 2], [1, 2, 3], 5))
    self.assertEqual(
        1 - log_diff(3, 5) / 3,  # max_dist
        reward.absolute_log_distance_reward([1, 2, 3, 4], [1, 2, 3], 5))
    # Add log differences for each position.
    self.assertEqual(
        1 - (log_diff(2, 5) + log_diff(2, 5) + log_diff(1, 5)) / 3,
        reward.absolute_log_distance_reward([4, 4, 4], [1, 2, 3], 5))
    self.assertEqual(
        1 - (log_diff(1, 5) + log_diff(2, 5) + log_diff(2, 5)) / 3,
        reward.absolute_log_distance_reward([0, 1, 2], [4, 4, 4], 5))
    self.assertEqual(
        1,
        reward.absolute_log_distance_reward([], [], 5))

  def testDeltaRewardManager(self):
    reward_manager = reward.DeltaRewardManager(
        [1, 2, 3, 4], base=5, distance_fn=reward.absolute_distance)
    self.assertEqual(-3, reward_manager([1]))
    self.assertEqual(0, reward_manager([1]))
    self.assertEqual(4 / 5., reward_manager([1, 3]))
    self.assertEqual(-4 / 5, reward_manager([1]))
    self.assertEqual(3, reward_manager([1, 2, 3, 4]))
    self.assertEqual(-1, reward_manager([1, 2, 3]))
    self.assertEqual(0, reward_manager([1, 2, 3, 4, 3]))
    self.assertEqual(-1, reward_manager([1, 2, 3, 4, 3, 2]))
    self.assertEqual(2, reward_manager([1, 2, 3, 4]))
    self.assertEqual(0, reward_manager([1, 2, 3, 4]))
    self.assertEqual(0, reward_manager([1, 2, 3, 4]))

  def testFloorRewardMananger(self):
    reward_manager = reward.FloorRewardManager(
        [1, 2, 3, 4], base=5, distance_fn=reward.absolute_distance)
    self.assertEqual(1, reward_manager([1]))
    self.assertEqual(0, reward_manager([1]))
    self.assertEqual(4 / 5., reward_manager([1, 3]))
    self.assertEqual(0, reward_manager([1]))
    self.assertEqual(1 / 5., reward_manager([1, 2]))
    self.assertEqual(0, reward_manager([0, 1]))
    self.assertEqual(0, reward_manager([]))
    self.assertEqual(0, reward_manager([1, 2]))
    self.assertEqual(2, reward_manager([1, 2, 3, 4]))
    self.assertEqual(0, reward_manager([1, 2, 3]))
    self.assertEqual(-1, reward_manager([1, 2, 3, 4, 3]))
    self.assertEqual(0, reward_manager([1, 2, 3, 4, 3, 2]))
    self.assertEqual(1, reward_manager([1, 2, 3, 4]))
    self.assertEqual(0, reward_manager([1, 2, 3, 4]))
    self.assertEqual(0, reward_manager([1, 2, 3, 4]))

    reward_manager = reward.FloorRewardManager(
        [1, 2, 3, 4], base=5, distance_fn=reward.absolute_distance)
    self.assertEqual(1, reward_manager([1]))
    self.assertEqual(-1, reward_manager([1, 0, 0, 0, 0, 0]))
    self.assertEqual(0, reward_manager([1, 2, 3, 4, 0, 0]))
    self.assertEqual(0, reward_manager([1, 2, 3, 4, 0]))
    self.assertEqual(1, reward_manager([]))
    self.assertEqual(0, reward_manager([]))
    self.assertEqual(0, reward_manager([1]))
    self.assertEqual(1, reward_manager([1, 2]))
    self.assertEqual(-1, reward_manager([1, 2, 3, 4, 0, 0]))
    self.assertEqual(0, reward_manager([1, 1, 1, 1, 1]))
    self.assertEqual(1 + 2, reward_manager([1, 2, 3, 4]))


if __name__ == '__main__':
  tf.test.main()
