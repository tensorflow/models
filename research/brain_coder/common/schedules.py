from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Schedule functions for controlling hparams over time."""

from abc import ABCMeta
from abc import abstractmethod
import math

from common import config_lib  # brain coder


class Schedule(object):
  """Schedule is a function which sets a hyperparameter's value over time.

  For example, a schedule can be used to decay an hparams, or oscillate it over
  time.

  This object is constructed with an instance of config_lib.Config (will be
  specific to each class implementation). For example if this is a decay
  schedule, the config may specify the rate of decay and decay start time. Then
  the object instance is called like a function, mapping global step (an integer
  counting how many calls to the train op have been made) to the hparam value.

  Properties of a schedule function f(t):
  0) Domain of t is the non-negative integers (t may be 0).
  1) Range of f is the reals.
  2) Schedule functions can assume that they will be called in time order. This
     allows schedules to be stateful.
  3) Schedule functions should be deterministic. Two schedule instances with the
     same config must always give the same value for each t, and regardless of
     what t's it was previously called on. Users may call f(t) on arbitrary
     (positive) time jumps. Essentially, multiple schedule instances used in
     replica training will behave the same.
  4) Duplicate successive calls on the same time are allowed.
  """
  __metaclass__ = ABCMeta

  @abstractmethod
  def __init__(self, config):
    """Construct this schedule with a config specific to each class impl.

    Args:
      config: An instance of config_lib.Config.
    """
    pass

  @abstractmethod
  def __call__(self, global_step):
    """Map `global_step` to a value.

    `global_step` is an integer counting how many calls to the train op have
    been made across all replicas (hence why it is global). Implementations
    may assume calls to be made in time order, i.e. `global_step` now >=
    previous `global_step` values.

    Args:
      global_step: Non-negative integer.

    Returns:
      Hparam value at this step. A number.
    """
    pass


class ConstSchedule(Schedule):
  """Constant function.

  config:
    const: Constant value at every step.

  f(t) = const.
  """

  def __init__(self, config):
    super(ConstSchedule, self).__init__(config)
    self.const = config.const

  def __call__(self, global_step):
    return self.const


class LinearDecaySchedule(Schedule):
  """Linear decay function.

  config:
    initial: Decay starts from this value.
    final: Decay ends at this value.
    start_time: Step when decay starts. Constant before it.
    end_time: When decay ends. Constant after it.

  f(t) is a linear function when start_time <= t <= end_time, with slope of
  (final - initial) / (end_time - start_time). f(t) = initial
  when t <= start_time. f(t) = final when t >= end_time.

  If start_time == end_time, this becomes a step function.
  """

  def __init__(self, config):
    super(LinearDecaySchedule, self).__init__(config)
    self.initial = config.initial
    self.final = config.final
    self.start_time = config.start_time
    self.end_time = config.end_time

    if self.end_time < self.start_time:
      raise ValueError('start_time must be before end_time.')

    # Linear interpolation.
    self._time_diff = float(self.end_time - self.start_time)
    self._diff = float(self.final - self.initial)
    self._slope = (
        self._diff / self._time_diff if self._time_diff > 0 else float('inf'))

  def __call__(self, global_step):
    if global_step <= self.start_time:
      return self.initial
    if global_step > self.end_time:
      return self.final
    return self.initial + (global_step - self.start_time) * self._slope


class ExponentialDecaySchedule(Schedule):
  """Exponential decay function.

  See https://en.wikipedia.org/wiki/Exponential_decay.

  Use this decay function to decay over orders of magnitude. For example, to
  decay learning rate from 1e-2 to 1e-6. Exponential decay will decay the
  exponent linearly.

  config:
    initial: Decay starts from this value.
    final: Decay ends at this value.
    start_time: Step when decay starts. Constant before it.
    end_time: When decay ends. Constant after it.

  f(t) is an exponential decay function when start_time <= t <= end_time. The
  decay rate and amplitude are chosen so that f(t) = initial when
  t = start_time, and f(t) = final when t = end_time. f(t) is constant for
  t < start_time or t > end_time. initial and final must be positive values.

  If start_time == end_time, this becomes a step function.
  """

  def __init__(self, config):
    super(ExponentialDecaySchedule, self).__init__(config)
    self.initial = config.initial
    self.final = config.final
    self.start_time = config.start_time
    self.end_time = config.end_time

    if self.initial <= 0 or self.final <= 0:
      raise ValueError('initial and final must be positive numbers.')

    # Linear interpolation in log space.
    self._linear_fn = LinearDecaySchedule(
        config_lib.Config(
            initial=math.log(self.initial),
            final=math.log(self.final),
            start_time=self.start_time,
            end_time=self.end_time))

  def __call__(self, global_step):
    return math.exp(self._linear_fn(global_step))


class SmootherstepDecaySchedule(Schedule):
  """Smootherstep decay function.

  A sigmoidal like transition from initial to final values. A smoother
  transition than linear and exponential decays, hence the name.
  See https://en.wikipedia.org/wiki/Smoothstep.

  config:
    initial: Decay starts from this value.
    final: Decay ends at this value.
    start_time: Step when decay starts. Constant before it.
    end_time: When decay ends. Constant after it.

  f(t) is fully defined here:
  https://en.wikipedia.org/wiki/Smoothstep#Variations.

  f(t) is smooth, as in its first-derivative exists everywhere.
  """

  def __init__(self, config):
    super(SmootherstepDecaySchedule, self).__init__(config)
    self.initial = config.initial
    self.final = config.final
    self.start_time = config.start_time
    self.end_time = config.end_time

    if self.end_time < self.start_time:
      raise ValueError('start_time must be before end_time.')

    self._time_diff = float(self.end_time - self.start_time)
    self._diff = float(self.final - self.initial)

  def __call__(self, global_step):
    if global_step <= self.start_time:
      return self.initial
    if global_step > self.end_time:
      return self.final
    x = (global_step - self.start_time) / self._time_diff

    # Smootherstep
    return self.initial + x * x * x * (x * (x * 6 - 15) + 10) * self._diff


class HardOscillatorSchedule(Schedule):
  """Hard oscillator function.

  config:
    high: Max value of the oscillator. Value at constant plateaus.
    low: Min value of the oscillator. Value at constant valleys.
    start_time: Global step when oscillation starts. Constant before this.
    period: Width of one oscillation, i.e. number of steps over which the
        oscillation takes place.
    transition_fraction: Fraction of the period spent transitioning between high
        and low values. 50% of this time is spent rising, and 50% of this time
        is spent falling. 50% of the remaining time is spent constant at the
        high value, and 50% of the remaining time is spent constant at the low
        value. transition_fraction = 1.0 means the entire period is spent
        rising and falling. transition_fraction = 0.0 means no time is spent
        rising and falling, i.e. the function jumps instantaneously between
        high and low.

  f(t) = high when t < start_time.
  f(t) is periodic when t >= start_time, with f(t + period) = f(t).
  f(t) is linear with positive slope when rising, and negative slope when
  falling. At the start of the period t0, f(t0) = high and begins to descend.
  At the middle of the period f is low and is constant until the ascension
  begins. f then rises from low to high and is constant again until the period
  repeats.

  Note: when transition_fraction is 0, f starts the period low and ends high.
  """

  def __init__(self, config):
    super(HardOscillatorSchedule, self).__init__(config)
    self.high = config.high
    self.low = config.low
    self.start_time = config.start_time
    self.period = float(config.period)
    self.transition_fraction = config.transition_fraction
    self.half_transition_fraction = config.transition_fraction / 2.0

    if self.transition_fraction < 0 or self.transition_fraction > 1.0:
      raise ValueError('transition_fraction must be between 0 and 1.0')
    if self.period <= 0:
      raise ValueError('period must be positive')

    self._slope = (
        float(self.high - self.low) / self.half_transition_fraction
        if self.half_transition_fraction > 0 else float('inf'))

  def __call__(self, global_step):
    if global_step < self.start_time:
      return self.high
    period_pos = ((global_step - self.start_time) / self.period) % 1.0
    if period_pos >= 0.5:
      # ascending
      period_pos -= 0.5
      if period_pos < self.half_transition_fraction:
        return self.low + period_pos * self._slope
      else:
        return self.high
    else:
      # descending
      if period_pos < self.half_transition_fraction:
        return self.high - period_pos * self._slope
      else:
        return self.low


_NAME_TO_CONFIG = {
    'const': ConstSchedule,
    'linear_decay': LinearDecaySchedule,
    'exp_decay': ExponentialDecaySchedule,
    'smooth_decay': SmootherstepDecaySchedule,
    'hard_osc': HardOscillatorSchedule,
}


def make_schedule(config):
  """Schedule factory.

  Given `config` containing a `fn` property, a Schedule implementation is
  instantiated with `config`. See `_NAME_TO_CONFIG` for `fn` options.

  Args:
    config: Config with a `fn` option that specifies which Schedule
        implementation to use. `config` is passed into the constructor.

  Returns:
    A Schedule impl instance.
  """
  schedule_class = _NAME_TO_CONFIG[config.fn]
  return schedule_class(config)
