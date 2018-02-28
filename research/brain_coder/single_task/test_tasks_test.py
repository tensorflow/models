from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Tests for test_tasks."""

import numpy as np
import tensorflow as tf

from single_task import misc  # brain coder
from single_task import test_tasks  # brain coder


def get_reward(reward_fn, candidate):
  return sum(reward_fn(misc.bf_tokens_to_string(candidate)).episode_rewards)


class TestTasksTest(tf.test.TestCase):

  def testHillClimbingTask(self):
    task = test_tasks.BasicTaskManager(test_tasks.HillClimbingTask())
    reward_fns = task.rl_batch(1)
    reward_fn = reward_fns[0]
    self.assertTrue(np.isclose(get_reward(reward_fn, [1, 2, 0]), 8 / 12.))
    self.assertTrue(np.isclose(get_reward(reward_fn, [1, 2, 2, 0]), 11 / 12.))
    self.assertTrue(np.isclose(get_reward(reward_fn, [1, 2, 3, 0]), 1.0))
    self.assertTrue(
        np.isclose(get_reward(reward_fn, [1, 2, 3, 4, 5, 2, 0]), 1. + 8 / 12.))
    self.assertTrue(
        np.isclose(get_reward(reward_fn, [1, 2, 3, 4, 5, 6, 0]), 2.0))
    self.assertTrue(
        np.isclose(get_reward(reward_fn, [1, 2, 3, 4, 5, 6, 1, 8, 3, 0]), 3.0))
    self.assertTrue(
        np.isclose(get_reward(reward_fn, [1, 2, 3, 4, 5, 6, 7, 8, 7, 0]), 3.0))
    self.assertTrue(
        np.isclose(get_reward(reward_fn, [1, 2, 3, 4, 5, 6, 1, 8, 3, 1, 0]),
                   3.0 - 4 / 12.))
    self.assertTrue(
        np.isclose(
            get_reward(reward_fn, [1, 2, 3, 4, 5, 6, 1, 8, 3, 1, 1, 1, 1, 0]),
            2.0))
    self.assertTrue(
        np.isclose(get_reward(reward_fn, [1, 2, 3, 4, 5, 6, 7, 8, 7, 3, 0]),
                   3.0 + 1 / 12.))
    self.assertTrue(
        np.isclose(
            get_reward(reward_fn, [1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1,
                                   8, 5, 1, 6, 4, 2, 1, 8, 3, 0]),
            8.0))
    self.assertTrue(
        np.isclose(
            get_reward(reward_fn, [1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1,
                                   8, 5, 1, 6, 4, 2, 1, 8, 3, 1, 1, 0]),
            8.0 - 8 / 12.))
    self.assertTrue(
        np.isclose(get_reward(reward_fn, [1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3,
                                          2, 1, 8, 5, 1, 6, 4, 2, 1, 8, 3, 1, 1,
                                          1, 1, 1, 1, 1, 0]),
                   7.0))


if __name__ == '__main__':
  tf.test.main()
