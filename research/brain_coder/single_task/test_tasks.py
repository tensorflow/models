from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Tasks that test correctness of algorithms."""

from common import reward as reward_lib  # brain coder
from single_task import misc  # brain coder


class BasicTaskManager(object):
  """Wraps a generic reward function."""

  def __init__(self, reward_fn):
    self.reward_fn = reward_fn
    self.good_reward = 1.0

  def _score_string(self, string):
    actions = misc.bf_string_to_tokens(string)
    reward, correct = self.reward_fn(actions)
    return misc.RewardInfo(
        episode_rewards=[0.0] * (len(string) - 1) + [reward],
        input_case=None,
        correct_output=None,
        code_output=actions,
        input_type=None,
        output_type=misc.IOType.integer,
        reason='correct' if correct else 'wrong')

  def rl_batch(self, batch_size):
    reward_fns = [self._score_string] * batch_size
    return reward_fns


class Trie(object):
  """Trie for sequences."""
  EOS = ()

  def __init__(self):
    self.trie = {}

  def insert(self, sequence):
    d = self.trie
    for e in sequence:
      if e not in d:
        d[e] = {}
      d = d[e]
    d[self.EOS] = True   # Terminate sequence.

  def prefix_match(self, sequence):
    """Return prefix of `sequence` which exists in the trie."""
    d = self.trie
    index = 0
    for i, e in enumerate(sequence + [self.EOS]):
      index = i
      if e in d:
        d = d[e]
        if e == self.EOS:
          return sequence, True
      else:
        break
    return sequence[:index], False

  def next_choices(self, sequence):
    d = self.trie
    for e in sequence:
      if e in d:
        d = d[e]
      else:
        raise ValueError('Sequence not a prefix: %s' % (sequence,))
    return d.keys()


class HillClimbingTask(object):
  """Simple task that tests reward hill climbing ability.

  There are a set of paths (sequences of tokens) which are rewarded. The total
  reward for a path is proportional to its length, so the longest path is the
  target. Shorter paths can be dead ends.
  """

  def __init__(self):
    # Paths are sequences of sub-sequences. Here we form unique sub-sequences
    # out of 3 arbitrary ints. We use sub-sequences instead of single entities
    # to make the task harder by making the episodes last longer, i.e. more
    # for the agent to remember.
    a = (1, 2, 3)
    b = (4, 5, 6)
    c = (7, 8, 7)
    d = (6, 5, 4)
    e = (3, 2, 1)
    f = (8, 5, 1)
    g = (6, 4, 2)
    h = (1, 8, 3)
    self.paths = Trie()
    self.paths.insert([a, b, h])
    self.paths.insert([a, b, c, d, e, f, g, h])
    self.paths.insert([a, b, c, d, e, b, a])
    self.paths.insert([a, b, g, h])
    self.paths.insert([a, e, f, g])
    self.correct_sequence = misc.flatten([a, b, c, d, e, f, g, h])

    def distance_fn(a, b):
      len_diff = abs(len(a) - len(b))
      return sum(reward_lib.mod_abs_diff(ai - 1, bi - 1, 8)
                 for ai, bi in zip(a, b)) + len_diff * 4  # 8 / 2 = 4
    self.distance_fn = distance_fn

  def __call__(self, actions):
    # Compute reward for action sequence.
    actions = [a for a in actions if a > 0]
    sequence = [tuple(actions[i: i + 3]) for i in xrange(0, len(actions), 3)]
    prefix, complete = self.paths.prefix_match(sequence)
    if complete:
      return float(len(prefix)), actions == self.correct_sequence
    if len(prefix) == len(sequence):
      return float(len(prefix)), False
    next_pred = sequence[len(prefix)]
    choices = self.paths.next_choices(prefix)
    if choices == [()]:
      return (len(prefix) - len(next_pred) / 3.0), False
    min_dist = min(self.distance_fn(c, next_pred) for c in choices)
    # +1 reward for each element in the sequence correct, plus fraction torwards
    # closest next element.
    # Maximum distance possible is num_actions * base / 2 = 3 * 8 / 2 = 12
    return (len(prefix) + (1 - min_dist / 12.0)), False


