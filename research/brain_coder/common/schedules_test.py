from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Tests for common.schedules."""

from math import exp
from math import sqrt
import numpy as np
from six.moves import xrange
import tensorflow as tf

from common import config_lib  # brain coder
from common import schedules  # brain coder


class SchedulesTest(tf.test.TestCase):

  def ScheduleTestHelper(self, config, schedule_subtype, io_values):
    """Run common checks for schedules.

    Args:
      config: Config object which is passed into schedules.make_schedule.
      schedule_subtype: The expected schedule type to be instantiated.
      io_values: List of (input, output) pairs. Must be in ascending input
          order. No duplicate inputs.
    """

    # Check that make_schedule makes the correct type.
    f = schedules.make_schedule(config)
    self.assertTrue(isinstance(f, schedule_subtype))

    # Check that multiple instances returned from make_schedule behave the same.
    fns = [schedules.make_schedule(config) for _ in xrange(3)]

    # Check that all the inputs map to the right outputs.
    for i, o in io_values:
      for f in fns:
        f_out = f(i)
        self.assertTrue(
            np.isclose(o, f_out),
            'Wrong value at input %d. Expected %s, got %s' % (i, o, f_out))

    # Check that a subset of the io_values are still correct.
    f = schedules.make_schedule(config)
    subseq = [io_values[i**2] for i in xrange(int(sqrt(len(io_values))))]
    if subseq[-1] != io_values[-1]:
      subseq.append(io_values[-1])
    for i, o in subseq:
      f_out = f(i)
      self.assertTrue(
          np.isclose(o, f_out),
          'Wrong value at input %d. Expected %s, got %s' % (i, o, f_out))

    # Check duplicate calls.
    f = schedules.make_schedule(config)
    for i, o in io_values:
      for _ in xrange(3):
        f_out = f(i)
        self.assertTrue(
            np.isclose(o, f_out),
            'Duplicate calls at input %d are not equal. Expected %s, got %s'
            % (i, o, f_out))

  def testConstSchedule(self):
    self.ScheduleTestHelper(
        config_lib.Config(fn='const', const=5),
        schedules.ConstSchedule,
        [(0, 5), (1, 5), (10, 5), (20, 5), (100, 5), (1000000, 5)])

  def testLinearDecaySchedule(self):
    self.ScheduleTestHelper(
        config_lib.Config(fn='linear_decay', initial=2, final=0, start_time=10,
                          end_time=20),
        schedules.LinearDecaySchedule,
        [(0, 2), (1, 2), (10, 2), (11, 1.8), (15, 1), (19, 0.2), (20, 0),
         (100000, 0)])

    # Test step function.
    self.ScheduleTestHelper(
        config_lib.Config(fn='linear_decay', initial=2, final=0, start_time=10,
                          end_time=10),
        schedules.LinearDecaySchedule,
        [(0, 2), (1, 2), (10, 2), (11, 0), (15, 0)])

  def testExponentialDecaySchedule(self):
    self.ScheduleTestHelper(
        config_lib.Config(fn='exp_decay', initial=exp(-1), final=exp(-6),
                          start_time=10, end_time=20),
        schedules.ExponentialDecaySchedule,
        [(0, exp(-1)), (1, exp(-1)), (10, exp(-1)), (11, exp(-1/2. - 1)),
         (15, exp(-5/2. - 1)), (19, exp(-9/2. - 1)), (20, exp(-6)),
         (100000, exp(-6))])

    # Test step function.
    self.ScheduleTestHelper(
        config_lib.Config(fn='exp_decay', initial=exp(-1), final=exp(-6),
                          start_time=10, end_time=10),
        schedules.ExponentialDecaySchedule,
        [(0, exp(-1)), (1, exp(-1)), (10, exp(-1)), (11, exp(-6)),
         (15, exp(-6))])

  def testSmootherstepDecaySchedule(self):
    self.ScheduleTestHelper(
        config_lib.Config(fn='smooth_decay', initial=2, final=0, start_time=10,
                          end_time=20),
        schedules.SmootherstepDecaySchedule,
        [(0, 2), (1, 2), (10, 2), (11, 1.98288), (15, 1), (19, 0.01712),
         (20, 0), (100000, 0)])

    # Test step function.
    self.ScheduleTestHelper(
        config_lib.Config(fn='smooth_decay', initial=2, final=0, start_time=10,
                          end_time=10),
        schedules.SmootherstepDecaySchedule,
        [(0, 2), (1, 2), (10, 2), (11, 0), (15, 0)])

  def testHardOscillatorSchedule(self):
    self.ScheduleTestHelper(
        config_lib.Config(fn='hard_osc', high=2, low=0, start_time=100,
                          period=10, transition_fraction=0.5),
        schedules.HardOscillatorSchedule,
        [(0, 2), (1, 2), (10, 2), (100, 2), (101, 1.2), (102, 0.4), (103, 0),
         (104, 0), (105, 0), (106, 0.8), (107, 1.6), (108, 2), (109, 2),
         (110, 2), (111, 1.2), (112, 0.4), (115, 0), (116, 0.8), (119, 2),
         (120, 2), (100001, 1.2), (100002, 0.4), (100005, 0), (100006, 0.8),
         (100010, 2)])

    # Test instantaneous step.
    self.ScheduleTestHelper(
        config_lib.Config(fn='hard_osc', high=2, low=0, start_time=100,
                          period=10, transition_fraction=0),
        schedules.HardOscillatorSchedule,
        [(0, 2), (1, 2), (10, 2), (99, 2), (100, 0), (104, 0), (105, 2),
         (106, 2), (109, 2), (110, 0)])


if __name__ == '__main__':
  tf.test.main()
