from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Utilities related to computing training batches from episode rollouts.

Implementations here are based on code from Open AI:
https://github.com/openai/universe-starter-agent/blob/master/a3c.py.
"""

from collections import namedtuple
import numpy as np
import scipy.signal

from common import utils  # brain coder


class Rollout(object):
  """Holds a rollout for an episode.

  A rollout is a record of the states observed in some environment and actions
  taken by the agent to arrive at those states. Other information includes
  rewards received after each action, values estimated for each state, whether
  the rollout concluded the episide, and total reward received. Everything
  should be given in time order.

  At each time t, the agent sees state s_t, takes action a_t, and then receives
  reward r_t. The agent may optionally estimate a state value V(s_t) for each
  state.

  For an episode of length T:
  states = [s_0, ..., s_(T-1)]
  actions = [a_0, ..., a_(T-1)]
  rewards = [r_0, ..., r_(T-1)]
  values = [V(s_0), ..., V(s_(T-1))]

  Note that there is an extra state s_T observed after taking action a_(T-1),
  but this is not included in the rollout.

  Rollouts have an `terminated` attribute which is True when the rollout is
  "finalized", i.e. it holds a full episode. terminated will be False when
  time steps are still being added to it.
  """

  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []
    self.values = []
    self.total_reward = 0.0
    self.terminated = False

  def add(self, state, action, reward, value=0.0, terminated=False):
    """Add the next timestep to this rollout.

    Args:
      state: The state observed at the start of this timestep.
      action: The action taken after observing the given state.
      reward: The reward received for taking the given action.
      value: The value estimated for the given state.
      terminated: Whether this timestep ends the episode.

    Raises:
      ValueError: If this.terminated is already True, meaning that the episode
          has already ended.
    """
    if self.terminated:
      raise ValueError(
          'Trying to add timestep to an already terminal rollout.')
    self.states += [state]
    self.actions += [action]
    self.rewards += [reward]
    self.values += [value]
    self.terminated = terminated
    self.total_reward += reward

  def add_many(self, states, actions, rewards, values=None, terminated=False):
    """Add many timesteps to this rollout.

    Arguments are the same as `add`, but are lists of equal size.

    Args:
      states: The states observed.
      actions: The actions taken.
      rewards: The rewards received.
      values: The values estimated for the given states.
      terminated: Whether this sequence ends the episode.

    Raises:
      ValueError: If the lengths of all the input lists are not equal.
      ValueError: If this.terminated is already True, meaning that the episode
          has already ended.
    """
    if len(states) != len(actions):
      raise ValueError(
          'Number of states and actions must be the same. Got %d states and '
          '%d actions' % (len(states), len(actions)))
    if len(states) != len(rewards):
      raise ValueError(
          'Number of states and rewards must be the same. Got %d states and '
          '%d rewards' % (len(states), len(rewards)))
    if values is not None and len(states) != len(values):
      raise ValueError(
          'Number of states and values must be the same. Got %d states and '
          '%d values' % (len(states), len(values)))
    if self.terminated:
      raise ValueError(
          'Trying to add timesteps to an already terminal rollout.')
    self.states += states
    self.actions += actions
    self.rewards += rewards
    self.values += values if values is not None else [0.0] * len(states)
    self.terminated = terminated
    self.total_reward += sum(rewards)

  def extend(self, other):
    """Append another rollout to this rollout."""
    assert not self.terminated
    self.states.extend(other.states)
    self.actions.extend(other.actions)
    self.rewards.extend(other.rewards)
    self.values.extend(other.values)
    self.terminated = other.terminated
    self.total_reward += other.total_reward


def discount(x, gamma):
  """Returns discounted sums for each value in x, with discount factor gamma.

  This can be used to compute the return (discounted sum of rewards) at each
  timestep given a sequence of rewards. See the definitions for return and
  REINFORCE in section 3 of https://arxiv.org/pdf/1602.01783.pdf.

  Let g^k mean gamma ** k.
  For list [x_0, ..., x_N], the following list of discounted sums is computed:
  [x_0 + g^1 * x_1 + g^2 * x_2 + ... g^N * x_N,
   x_1 + g^1 * x_2 + g^2 * x_3 + ... g^(N-1) * x_N,
   x_2 + g^1 * x_3 + g^2 * x_4 + ... g^(N-2) * x_N,
   ...,
   x_(N-1) + g^1 * x_N,
   x_N]

  Args:
    x: List of numbers [x_0, ..., x_N].
    gamma: Float between 0 and 1 (inclusive). This is the discount factor.

  Returns:
    List of discounted sums.
  """
  return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def discounted_advantage_and_rewards(rewards, values, gamma, lambda_=1.0):
  """Compute advantages and returns (discounted sum of rewards).

  For an episode of length T, rewards = [r_0, ..., r_(T-1)].
  Each reward r_t is observed after taking action a_t at state s_t. A final
  state s_T is observed but no reward is given at this state since no action
  a_T is taken (otherwise there would be a new state s_(T+1)).

  `rewards` and `values` are for a single episode. Return R_t is the discounted
  sum of future rewards starting at time t, where `gamma` is the discount
  factor.
  R_t = r_t + gamma * r_(t+1) + gamma**2 * r_(t+2) + ...
        + gamma**(T-1-t) * r_(T-1)

  Advantage A(a_t, s_t) is approximated by computing A(a_t, s_t) = R_t - V(s_t)
  where V(s_t) is an approximation of the value at that state, given in the
  `values` list. Returns R_t are needed for all REINFORCE algorithms. Advantage
  is used for the advantage actor critic variant of REINFORCE.
  See algorithm S3 in https://arxiv.org/pdf/1602.01783.pdf.

  Additionally another parameter `lambda_` controls the bias-variance tradeoff.
  See "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438.
  lambda_ = 1 reduces to regular advantage.
  0 <= lambda_ < 1 trades off variance for bias, with lambda_ = 0 being the
  most biased.

  Bootstrapping is also supported. If an episode does not end in a terminal
  state (either because the episode was ended early, or the environment does not
  have end states), the true return cannot be computed from the rewards alone.
  However, it can be estimated by computing the value (an approximation of
  return) of the last state s_T. Thus the `values` list will have an extra item:
  values = [V(s_0), ..., V(s_(T-1)), V(s_T)].

  Args:
    rewards: List of observed rewards [r_0, ..., r_(T-1)].
    values: List of estimated values [V(s_0), ..., V(s_(T-1))] with an optional
        extra V(s_T) item.
    gamma: Discount factor. Number between 0 and 1. 1 means no discount.
        If not 1, gamma is typically near 1, like 0.99.
    lambda_: Bias-variance tradeoff factor. Between 0 and 1.

  Returns:
    empirical_values: Returns at each timestep.
    generalized_advantage: Avantages at each timestep.

  Raises:
    ValueError: If shapes of `rewards` and `values` are not rank 1.
    ValueError: If len(values) not in (len(rewards), len(rewards) + 1).
  """
  rewards = np.asarray(rewards, dtype=np.float32)
  values = np.asarray(values, dtype=np.float32)
  if rewards.ndim != 1:
    raise ValueError('Single episode only. rewards must be rank 1.')
  if values.ndim != 1:
    raise ValueError('Single episode only. values must be rank 1.')
  if len(values) == len(rewards):
    # No bootstrapping.
    values = np.append(values, 0)
    empirical_values = discount(rewards, gamma)
  elif len(values) == len(rewards) + 1:
    # With bootstrapping.
    # Last value is for the terminal state (final state after last action was
    # taken).
    empirical_values = discount(np.append(rewards, values[-1]), gamma)[:-1]
  else:
    raise ValueError('values should contain the same number of items or one '
                     'more item than rewards')
  delta = rewards + gamma * values[1:] - values[:-1]
  generalized_advantage = discount(delta, gamma * lambda_)

  # empirical_values is the discounted sum of rewards into the future.
  # generalized_advantage is the target for each policy update.
  return empirical_values, generalized_advantage


"""Batch holds a minibatch of episodes.

Let bi = batch_index, i.e. the index of each episode in the minibatch.
Let t = time.

Attributes:
  states: States for each timestep in each episode. Indexed by states[bi, t].
  actions: Actions for each timestep in each episode. Indexed by actions[bi, t].
  discounted_adv: Advantages (computed by discounted_advantage_and_rewards)
      for each timestep in each episode. Indexed by discounted_adv[bi, t].
  discounted_r: Returns (discounted sum of rewards computed by
      discounted_advantage_and_rewards) for each timestep in each episode.
      Indexed by discounted_r[bi, t].
  total_rewards: Total reward for each episode, i.e. sum of rewards across all
      timesteps (not discounted). Indexed by total_rewards[bi].
  episode_lengths: Number of timesteps in each episode. If an episode has
      N actions, N rewards, and N states, then its length is N. Indexed by
      episode_lengths[bi].
  batch_size: Number of episodes in this minibatch. An integer.
  max_time: Maximum episode length in the batch. An integer.
"""  # pylint: disable=pointless-string-statement
Batch = namedtuple(
    'Batch',
    ['states', 'actions', 'discounted_adv', 'discounted_r', 'total_rewards',
     'episode_lengths', 'batch_size', 'max_time'])


def process_rollouts(rollouts, gamma, lambda_=1.0):
  """Convert a batch of rollouts into tensors ready to be fed into a model.

  Lists from each episode are stacked into 2D tensors and padded with 0s up to
  the maximum timestep in the batch.

  Args:
    rollouts: A list of Rollout instances.
    gamma: The discount factor. A number between 0 and 1 (inclusive). See gamma
        argument in discounted_advantage_and_rewards.
    lambda_: See lambda_ argument in discounted_advantage_and_rewards.

  Returns:
    Batch instance. states, actions, discounted_adv, and discounted_r are
    numpy arrays with shape (batch_size, max_episode_length). episode_lengths
    is a list of ints. total_rewards is a list of floats (total reward in each
    episode). batch_size and max_time are ints.

  Raises:
    ValueError: If any of the rollouts are not terminal.
  """
  for ro in rollouts:
    if not ro.terminated:
      raise ValueError('Can only process terminal rollouts.')

  episode_lengths = [len(ro.states) for ro in rollouts]
  batch_size = len(rollouts)
  max_time = max(episode_lengths)

  states = utils.stack_pad([ro.states for ro in rollouts], 0, max_time)
  actions = utils.stack_pad([ro.actions for ro in rollouts], 0, max_time)

  discounted_rewards = [None] * batch_size
  discounted_adv = [None] * batch_size
  for i, ro in enumerate(rollouts):
    disc_r, disc_adv = discounted_advantage_and_rewards(
        ro.rewards, ro.values, gamma, lambda_)
    discounted_rewards[i] = disc_r
    discounted_adv[i] = disc_adv
  discounted_rewards = utils.stack_pad(discounted_rewards, 0, max_time)
  discounted_adv = utils.stack_pad(discounted_adv, 0, max_time)

  total_rewards = [sum(ro.rewards) for ro in rollouts]

  return Batch(states=states,
               actions=actions,
               discounted_adv=discounted_adv,
               discounted_r=discounted_rewards,
               total_rewards=total_rewards,
               episode_lengths=episode_lengths,
               batch_size=batch_size,
               max_time=max_time)
