# Copyright 2018 The TensorFlow Authors.
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

"""Event class, which represents a periodic event in a light curve."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Event(object):
  """Represents a periodic event in a light curve."""

  def __init__(self, period, duration, t0):
    """Initializes the Event.

    Args:
      period: Period of the event, in days.
      duration: Duration of the event, in days.
      t0: Time of the first occurrence of the event, in days.
    """
    self._period = period
    self._duration = duration
    self._t0 = t0

  def __str__(self):
    return "<period={}, duration={}, t0={}>".format(self.period, self.duration,
                                                    self.t0)

  def __repr__(self):
    return "Event({})".format(str(self))

  @property
  def period(self):
    return self._period

  @property
  def duration(self):
    return self._duration

  @property
  def t0(self):
    return self._t0

  def equals(self, other_event, period_rtol=0.001, t0_durations=1):
    """Compares this Event to another Event, within the given tolerance.

    Args:
      other_event: An Event.
      period_rtol: Relative tolerance in matching the periods.
      t0_durations: Tolerance in matching the t0 values, in units of the other
        Event's duration.

    Returns:
      True if this Event is the same as other_event, within the given tolerance.
    """
    # First compare the periods.
    period_match = np.isclose(
        self.period, other_event.period, rtol=period_rtol, atol=1e-8)
    if not period_match:
      return False

    # To compare t0, we must consider that self.t0 and other_event.t0 may be at
    # different phases. Just comparing mod(self.t0, period) to
    # mod(other_event.t0, period) does not work because two similar values could
    # end up at different ends of [0, period).
    #
    # Define t0_diff to be the absolute difference, up to multiples of period.
    # This value is always in [0, period/2).
    t0_diff = np.mod(self.t0 - other_event.t0, other_event.period)
    if t0_diff > other_event.period / 2:
      t0_diff = other_event.period - t0_diff

    return t0_diff < t0_durations * other_event.duration
