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

"""Tests for periodic_event.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

from light_curve_util.periodic_event import Event


class EventTest(absltest.TestCase):

  def testEquals(self):
    event = Event(period=100, duration=5, t0=2)

    # Varying periods.
    self.assertFalse(event.equals(Event(period=0, duration=5, t0=2)))
    self.assertFalse(event.equals(Event(period=50, duration=5, t0=2)))
    self.assertFalse(event.equals(Event(period=99.89, duration=5, t0=2)))
    self.assertTrue(event.equals(Event(period=99.91, duration=5, t0=2)))
    self.assertTrue(event.equals(Event(period=100, duration=5, t0=2)))
    self.assertTrue(event.equals(Event(period=100.01, duration=5, t0=2)))
    self.assertFalse(event.equals(Event(period=101, duration=5, t0=2)))

    # Different period tolerance.
    self.assertTrue(
        event.equals(Event(period=99.1, duration=5, t0=2), period_rtol=0.01))
    self.assertTrue(
        event.equals(Event(period=100.9, duration=5, t0=2), period_rtol=0.01))
    self.assertFalse(
        event.equals(Event(period=98.9, duration=5, t0=2), period_rtol=0.01))
    self.assertFalse(
        event.equals(Event(period=101.1, duration=5, t0=2), period_rtol=0.01))

    # Varying t0.
    self.assertTrue(event.equals(Event(period=100, duration=5, t0=0)))
    self.assertTrue(event.equals(Event(period=100, duration=5, t0=2)))
    self.assertTrue(event.equals(Event(period=100, duration=5, t0=6.9)))
    self.assertFalse(event.equals(Event(period=100, duration=5, t0=7.1)))

    # t0 at the other end of [0, period).
    self.assertFalse(event.equals(Event(period=100, duration=5, t0=96.9)))
    self.assertTrue(event.equals(Event(period=100, duration=5, t0=97.1)))
    self.assertTrue(event.equals(Event(period=100, duration=5, t0=100)))
    self.assertTrue(event.equals(Event(period=100, duration=5, t0=102)))
    self.assertFalse(event.equals(Event(period=100, duration=5, t0=107.1)))

    # Varying duration.
    self.assertFalse(event.equals(Event(period=100, duration=5, t0=10)))
    self.assertFalse(event.equals(Event(period=100, duration=7, t0=10)))
    self.assertTrue(event.equals(Event(period=100, duration=9, t0=10)))

    # Different duration tolerance.
    self.assertFalse(
        event.equals(Event(period=100, duration=5, t0=10), t0_durations=1))
    self.assertTrue(
        event.equals(Event(period=100, duration=5, t0=10), t0_durations=2))


if __name__ == '__main__':
  absltest.main()
