from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Manage data for pretraining and RL tasks."""

import ast
from collections import namedtuple

from absl import logging

from single_task import code_tasks  # brain coder


RLBatch = namedtuple('RLBatch', ['reward_fns', 'batch_size', 'good_reward'])


class DataManager(object):
  """Interface between environment and model."""

  def __init__(self, global_config, run_number=None,
               do_code_simplification=False):
    """Constructs a DataManager.

    Args:
      global_config: A config_lib.Config instance containing all config. See
          config in defaults.py.
      run_number: Which run this is (of the same experiment). This should be set
          when a task cycle is defined in the config. A task cycle is a list of
          tasks to cycle through repeatedly, and the selected task is a function
          of the run number, i.e. 0-th run, 1-st run, 2-nd run, etc...
          This can be None if only a single task is set in the config.
      do_code_simplification: When global_config.env.config_for_iclr is True,
          use this option to create code simplification (code golf) tasks, vs
          fixed length coding tasks. If True, a task with code simplification
          reward will be constructed.

    Raises:
      ValueError: If global_config.env.task and global_config.env.task_cycle
          are both set, or both not set. Only one should be given.
      ValueError: If global_config.env.task_cycle is set but run_number is None.
    """
    env_config = global_config.env
    self.batch_size = global_config.batch_size

    if env_config.task_cycle:
      if env_config.task:
        raise ValueError('Do not set both `task` and `task_cycle`.')
      if run_number is None:
        raise ValueError('Do not use task_cycle for single-run experiment.')
      index = run_number % len(env_config.task_cycle)
      self.task_name = env_config.task_cycle[index]
      logging.info('run_number: %d,  task_cycle index: %d', run_number, index)
      logging.info('task_cycle: %s', env_config.task_cycle)
    elif env_config.task:
      self.task_name = env_config.task
    else:
      raise ValueError('Either `task` or `task_cycle` must be set.')
    logging.info('Task for this run: "%s"', self.task_name)

    logging.info('config_for_iclr=True; do_code_simplification=%s',
                 do_code_simplification)
    self.rl_task = code_tasks.make_task(
        task_name=self.task_name,
        override_kwargs=ast.literal_eval(env_config.task_kwargs),
        max_code_length=global_config.timestep_limit,
        require_correct_syntax=env_config.correct_syntax,
        do_code_simplification=do_code_simplification,
        correct_bonus=env_config.task_manager_config.correct_bonus,
        code_length_bonus=env_config.task_manager_config.code_length_bonus)

  def sample_rl_batch(self):
    """Create reward functions from the current task.

    Returns:
      RLBatch namedtuple instance, which holds functions and information for
      a minibatch of episodes.
      * reward_fns: A reward function for each episode. Maps code string to
          reward.
      * batch_size: Number of episodes in this minibatch.
      * good_reward: Estimated threshold of rewards which indicate the algorithm
          is starting to solve the task. This is a heuristic that tries to
          reduce the amount of stuff written to disk.
    """
    reward_fns = self.rl_task.rl_batch(self.batch_size)
    return RLBatch(
        reward_fns=reward_fns,
        batch_size=self.batch_size,
        good_reward=self.rl_task.good_reward)
