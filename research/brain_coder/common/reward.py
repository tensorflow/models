from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Reward functions, distance functions, and reward managers."""

from abc import ABCMeta
from abc import abstractmethod
from math import log


# All sequences here are assumed to be lists of ints bounded
# between 0 and `base`-1 (inclusive).


#################################
### Scalar Distance Functions ###
#################################


def abs_diff(a, b, base=0):
  """Absolute value of difference between scalars.

  abs_diff is symmetric, i.e. `a` and `b` are interchangeable.

  Args:
    a: First argument. An int.
    b: Seconds argument. An int.
    base: Dummy argument so that the argument signature matches other scalar
        diff functions. abs_diff is the same in all bases.

  Returns:
    abs(a - b).
  """
  del base  # Unused.
  return abs(a - b)


def mod_abs_diff(a, b, base):
  """Shortest distance between `a` and `b` in the modular integers base `base`.

  The smallest distance between a and b is returned.
  Example: mod_abs_diff(1, 99, 100) ==> 2. It is not 98.

  mod_abs_diff is symmetric, i.e. `a` and `b` are interchangeable.

  Args:
    a: First argument. An int.
    b: Seconds argument. An int.
    base: The modulo base. A positive int.

  Returns:
    Shortest distance.
  """
  diff = abs(a - b)
  if diff >= base:
    diff %= base
  return min(diff, (-diff) + base)


###############################
### List Distance Functions ###
###############################


def absolute_distance(pred, target, base, scalar_diff_fn=abs_diff):
  """Asymmetric list distance function.

  List distance is the sum of element-wise distances, like Hamming distance, but
  where `pred` can be longer or shorter than `target`. For each position in both
  `pred` and `target`, distance between those elements is computed with
  `scalar_diff_fn`. For missing or extra elements in `pred`, the maximum
  distance is assigned, which is equal to `base`.

  Distance is 0 when `pred` and `target` are identical, and will be a positive
  integer when they are not.

  Args:
    pred: Prediction list. Distance from this list is computed.
    target: Target list. Distance to this list is computed.
    base: The integer base to use. For example, a list of chars would use base
        256.
    scalar_diff_fn: Element-wise distance function.

  Returns:
    List distance between `pred` and `target`.
  """
  d = 0
  for i, target_t in enumerate(target):
    if i >= len(pred):
      d += base  # A missing slot is worth the max distance.
    else:
      # Add element-wise distance for this slot.
      d += scalar_diff_fn(pred[i], target_t, base)
  if len(pred) > len(target):
    # Each extra slot is worth the max distance.
    d += (len(pred) - len(target)) * base
  return d


def log_absolute_distance(pred, target, base):
  """Asymmetric list distance function that uses log distance.

  A list distance which computes sum of element-wise distances, similar to
  `absolute_distance`. Unlike `absolute_distance`, this scales the resulting
  distance to be a float.

  Element-wise distance are log-scale. Distance between two list changes
  relatively less for elements that are far apart, but changes a lot (goes to 0
  faster) when values get close together.

  Args:
    pred: List of ints. Computes distance from this list to the target.
    target: List of ints. This is the "correct" list which the prediction list
        is trying to match.
    base: Integer base.

  Returns:
    Float distance normalized so that when `pred` is at most as long as `target`
    the distance is between 0.0 and 1.0. Distance grows unboundedly large
    as `pred` grows past `target` in length.
  """
  if not target:
    length_normalizer = 1.0
    if not pred:
      # Distance between [] and [] is 0.0 since they are equal.
      return 0.0
  else:
    length_normalizer = float(len(target))
  # max_dist is the maximum element-wise distance, before taking log and
  # scaling. Since we use `mod_abs_diff`, it would be (base // 2), but we add
  # 1 to it so that missing or extra positions get the maximum penalty.
  max_dist = base // 2 + 1

  # The log-distance will be scaled by a factor.
  # Note: +1 is added to the numerator and denominator to avoid log(0). This
  # only has a translational effect, i.e. log(dist + 1) / log(max_dist + 1).
  factor = log(max_dist + 1)

  d = 0.0  # Total distance to be computed.
  for i, target_t in enumerate(target):
    if i >= len(pred):
      # Assign the max element-wise distance for missing positions. This is 1.0
      # after scaling.
      d += 1.0
    else:
      # Add the log-dist divided by a scaling factor.
      d += log(mod_abs_diff(pred[i], target_t, base) + 1) / factor
  if len(pred) > len(target):
    # Add the max element-wise distance for each extra position.
    # Since max dist after scaling is 1, this is just the difference in list
    # lengths.
    d += (len(pred) - len(target))
  return d / length_normalizer  # Normalize again by the target length.


########################
### Reward Functions ###
########################

# Reward functions assign reward based on program output.
# Warning: only use these functions as the terminal rewards in episodes, i.e.
# for the "final" programs.


def absolute_distance_reward(pred, target, base, scalar_diff_fn=abs_diff):
  """Reward function based on absolute_distance function.

  Maximum reward, 1.0, is given when the lists are equal. Reward is scaled
  so that 0.0 reward is given when `pred` is the empty list (assuming `target`
  is not empty). Reward can go negative when `pred` is longer than `target`.

  This is an asymmetric reward function, so which list is the prediction and
  which is the target matters.

  Args:
    pred: Prediction sequence. This should be the sequence outputted by the
        generated code. List of ints n, where 0 <= n < base.
    target: Target sequence. The correct sequence that the generated code needs
        to output. List of ints n, where 0 <= n < base.
    base: Base of the computation.
    scalar_diff_fn: Element-wise distance function.

  Returns:
    Reward computed based on `pred` and `target`. A float.
  """
  unit_dist = float(base * len(target))
  if unit_dist == 0:
    unit_dist = base
  dist = absolute_distance(pred, target, base, scalar_diff_fn=scalar_diff_fn)
  return (unit_dist - dist) / unit_dist


def absolute_mod_distance_reward(pred, target, base):
  """Same as `absolute_distance_reward` but `mod_abs_diff` scalar diff is used.

  Args:
    pred: Prediction sequence. This should be the sequence outputted by the
        generated code. List of ints n, where 0 <= n < base.
    target: Target sequence. The correct sequence that the generated code needs
        to output. List of ints n, where 0 <= n < base.
    base: Base of the computation.

  Returns:
    Reward computed based on `pred` and `target`. A float.
  """
  return absolute_distance_reward(pred, target, base, mod_abs_diff)


def absolute_log_distance_reward(pred, target, base):
  """Compute reward using `log_absolute_distance`.

  Maximum reward, 1.0, is given when the lists are equal. Reward is scaled
  so that 0.0 reward is given when `pred` is the empty list (assuming `target`
  is not empty). Reward can go negative when `pred` is longer than `target`.

  This is an asymmetric reward function, so which list is the prediction and
  which is the target matters.

  This reward function has the nice property that much more reward is given
  for getting the correct value (at each position) than for there being any
  value at all. For example, in base 100, lets say pred = [1] * 1000
  and target = [10] * 1000. A lot of reward would be given for being 80%
  accurate (worst element-wise distance is 50, distances here are 9) using
  `absolute_distance`. `log_absolute_distance` on the other hand will give
  greater and greater reward increments the closer each predicted value gets to
  the target. That makes the reward given for accuracy somewhat independant of
  the base.

  Args:
    pred: Prediction sequence. This should be the sequence outputted by the
        generated code. List of ints n, where 0 <= n < base.
    target: Target sequence. The correct sequence that the generated code needs
        to output. List of ints n, where 0 <= n < base.
    base: Base of the computation.

  Returns:
    Reward computed based on `pred` and `target`. A float.
  """
  return 1.0 - log_absolute_distance(pred, target, base)


#######################
### Reward Managers ###
#######################

# Reward managers assign reward to many code attempts throughout an episode.


class RewardManager(object):
  """Reward managers administer reward across an episode.

  Reward managers are used for "editor" environments. These are environments
  where the agent has some way to edit its code over time, and run its code
  many time in the same episode, so that it can make incremental improvements.

  Reward managers are instantiated with a target sequence, which is the known
  correct program output. The manager is called on the output from a proposed
  code, and returns reward. If many proposal outputs are tried, reward may be
  some stateful function that takes previous tries into account. This is done,
  in part, so that an agent cannot accumulate unbounded reward just by trying
  junk programs as often as possible. So reward managers should not give the
  same reward twice if the next proposal is not better than the last.
  """
  __metaclass__ = ABCMeta

  def __init__(self, target, base, distance_fn=absolute_distance):
    self._target = list(target)
    self._base = base
    self._distance_fn = distance_fn

  @abstractmethod
  def __call__(self, sequence):
    """Call this reward manager like a function to get reward.

    Calls to reward manager are stateful, and will take previous sequences
    into account. Repeated calls with the same sequence may produce different
    rewards.

    Args:
      sequence: List of integers (each between 0 and base - 1). This is the
          proposal sequence. Reward will be computed based on the distance
          from this sequence to the target (distance function and target are
          given in the constructor), as well as previous sequences tried during
          the lifetime of this object.

    Returns:
      Float value. The reward received from this call.
    """
    return 0.0


class DeltaRewardManager(RewardManager):
  """Simple reward manager that assigns reward for the net change in distance.

  Given some (possibly asymmetric) list distance function, gives reward for
  relative changes in prediction distance to the target.

  For example, if on the first call the distance is 3.0, the change in distance
  is -3 (from starting distance of 0). That relative change will be scaled to
  produce a negative reward for this step. On the next call, the distance is 2.0
  which is a +1 change, and that will be scaled to give a positive reward.
  If the final call has distance 0 (the target is achieved), that is another
  positive change of +2. The total reward across all 3 calls is then 0, which is
  the highest posible episode total.

  Reward is scaled so that the maximum element-wise distance is worth 1.0.
  Maximum total episode reward attainable is 0.
  """

  def __init__(self, target, base, distance_fn=absolute_distance):
    super(DeltaRewardManager, self).__init__(target, base, distance_fn)
    self._last_diff = 0

  def _diff(self, seq):
    return self._distance_fn(seq, self._target, self._base)

  def _delta_reward(self, seq):
    # Reward is relative to previous sequence diff.
    # Reward is scaled so that maximum token difference is worth 1.0.
    # Reward = (last_diff - this_diff) / self.base.
    # Reward is positive if this sequence is closer to the target than the
    # previous sequence, and negative if this sequence is further away.
    diff = self._diff(seq)
    reward = (self._last_diff - diff) / float(self._base)
    self._last_diff = diff
    return reward

  def __call__(self, seq):
    return self._delta_reward(seq)


class FloorRewardManager(RewardManager):
  """Assigns positive reward for each step taken closer to the target.

  Given some (possibly asymmetric) list distance function, gives reward for
  whenever a new episode minimum distance is reached. No reward is given if
  the distance regresses to a higher value, so that the sum of rewards
  for the episode is positive.

  Reward is scaled so that the maximum element-wise distance is worth 1.0.
  Maximum total episode reward attainable is len(target).

  If the prediction sequence is longer than the target, a reward of -1 is given.
  Subsequence predictions which are also longer get 0 reward. The -1 penalty
  will be canceled out with a +1 reward when a prediction is given which is at
  most the length of the target.
  """

  def __init__(self, target, base, distance_fn=absolute_distance):
    super(FloorRewardManager, self).__init__(target, base, distance_fn)
    self._last_diff = 0
    self._min_diff = self._max_diff()
    self._too_long_penality_given = False

  def _max_diff(self):
    return self._distance_fn([], self._target, self._base)

  def _diff(self, seq):
    return self._distance_fn(seq, self._target, self._base)

  def _delta_reward(self, seq):
    # Reward is only given if this sequence is closer to the target than any
    # previous sequence.
    # Reward is scaled so that maximum token difference is worth 1.0
    # Reward = (min_diff - this_diff) / self.base
    # Reward is always positive.
    diff = self._diff(seq)
    if diff < self._min_diff:
      reward = (self._min_diff - diff) / float(self._base)
      self._min_diff = diff
    else:
      reward = 0.0
    return reward

  def __call__(self, seq):
    if len(seq) > len(self._target):  # Output is too long.
      if not self._too_long_penality_given:
        self._too_long_penality_given = True
        reward = -1.0
      else:
        reward = 0.0  # Don't give this penalty more than once.
      return reward

    reward = self._delta_reward(seq)
    if self._too_long_penality_given:
      reward += 1.0  # Return the subtracted reward.
      self._too_long_penality_given = False
    return reward

