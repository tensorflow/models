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

"""Tests for util.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np

from light_curve_util import periodic_event

from light_curve_util import util


class LightCurveUtilTest(absltest.TestCase):

  def testPhaseFoldTime(self):
    time = np.arange(0, 2, 0.1)

    # Simple.
    tfold = util.phase_fold_time(time, period=1, t0=0.45)
    expected = [
        -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, -0.45,
        -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45
    ]
    self.assertSequenceAlmostEqual(expected, tfold)

    # Large t0.
    tfold = util.phase_fold_time(time, period=1, t0=1.25)
    expected = [
        -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, -0.45, -0.35, -0.25,
        -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, -0.45, -0.35
    ]
    self.assertSequenceAlmostEqual(expected, tfold)

    # Negative t0.
    tfold = util.phase_fold_time(time, period=1, t0=-1.65)
    expected = [
        -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, -0.45, -0.35,
        -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45, -0.45
    ]
    self.assertSequenceAlmostEqual(expected, tfold)

    # Negative time.
    time = np.arange(-3, -1, 0.1)
    tfold = util.phase_fold_time(time, period=1, t0=0.55)
    expected = [
        0.45, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35, 0.45,
        -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35
    ]
    self.assertSequenceAlmostEqual(expected, tfold)

  def testSplit(self):
    all_time = [
        np.concatenate([
            np.arange(0, 1, 0.1),
            np.arange(1.5, 2, 0.1),
            np.arange(3, 4, 0.1)
        ])
    ]
    all_flux = [np.array([1] * 25)]

    # Gap width 0.5.
    split_time, split_flux = util.split(all_time, all_flux, gap_width=0.5)
    self.assertLen(split_time, 3)
    self.assertLen(split_flux, 3)
    self.assertSequenceAlmostEqual(
        [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], split_time[0])
    self.assertSequenceAlmostEqual([1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   split_flux[0])
    self.assertSequenceAlmostEqual([1.5, 1.6, 1.7, 1.8, 1.9], split_time[1])
    self.assertSequenceAlmostEqual([1, 1, 1, 1, 1], split_flux[1])
    self.assertSequenceAlmostEqual(
        [3., 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9], split_time[2])
    self.assertSequenceAlmostEqual([1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   split_flux[2])

    # Gap width 1.0.
    split_time, split_flux = util.split(all_time, all_flux, gap_width=1)
    self.assertLen(split_time, 2)
    self.assertLen(split_flux, 2)
    self.assertSequenceAlmostEqual([
        0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.5, 1.6, 1.7, 1.8, 1.9
    ], split_time[0])
    self.assertSequenceAlmostEqual(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], split_flux[0])
    self.assertSequenceAlmostEqual(
        [3., 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9], split_time[1])
    self.assertSequenceAlmostEqual([1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   split_flux[1])

  def testRemoveEvents(self):
    time = np.arange(20, dtype=np.float)
    flux = 10 * time

    # One event.
    events = [periodic_event.Event(period=4, duration=1.5, t0=3.5)]
    output_time, output_flux = util.remove_events(time, flux, events)
    self.assertSequenceAlmostEqual([1, 2, 5, 6, 9, 10, 13, 14, 17, 18],
                                   output_time)
    self.assertSequenceAlmostEqual(
        [10, 20, 50, 60, 90, 100, 130, 140, 170, 180], output_flux)

    # Two events.
    events.append(periodic_event.Event(period=7, duration=1.5, t0=6.5))
    output_time, output_flux = util.remove_events(time, flux, events)
    self.assertSequenceAlmostEqual([1, 2, 5, 9, 10, 17, 18], output_time)
    self.assertSequenceAlmostEqual([10, 20, 50, 90, 100, 170, 180], output_flux)

    # Multi segment light curve.
    time = [np.arange(10, dtype=np.float), np.arange(10, 20, dtype=np.float)]
    flux = [10 * t for t in time]
    output_time, output_flux = util.remove_events(time, flux, events)
    self.assertLen(output_time, 2)
    self.assertLen(output_flux, 2)
    self.assertSequenceAlmostEqual([1, 2, 5, 9], output_time[0])
    self.assertSequenceAlmostEqual([10, 20, 50, 90], output_flux[0])
    self.assertSequenceAlmostEqual([10, 17, 18], output_time[1])
    self.assertSequenceAlmostEqual([100, 170, 180], output_flux[1])

  def testInterpolateMaskedSpline(self):
    all_time = [
        np.arange(0, 10, dtype=np.float),
        np.arange(10, 20, dtype=np.float),
    ]
    all_masked_time = [
        np.array([0, 1, 2, 3, 8, 9], dtype=np.float),  # No 4, 5, 6, 7
        np.array([10, 11, 12, 13, 14, 15, 16], dtype=np.float),  # No 17, 18, 19
    ]
    all_masked_spline = [2 * t + 100 for t in all_masked_time]

    interp_spline = util.interpolate_masked_spline(all_time, all_masked_time,
                                                   all_masked_spline)
    self.assertLen(interp_spline, 2)
    self.assertSequenceAlmostEqual(
        [100, 102, 104, 106, 108, 110, 112, 114, 116, 118], interp_spline[0])
    self.assertSequenceAlmostEqual(
        [120, 122, 124, 126, 128, 130, 132, 132, 132, 132], interp_spline[1])

  def testCountTransitPoints(self):
    time = np.concatenate([
        np.arange(0, 10, 0.1, dtype=np.float),
        np.arange(15, 30, 0.1, dtype=np.float),
        np.arange(50, 100, 0.1, dtype=np.float)
    ])
    event = periodic_event.Event(period=10, duration=5, t0=9.95)

    points_in_transit = util.count_transit_points(time, event)
    np.testing.assert_array_equal([25, 50, 25, 0, 25, 50, 50, 50, 50],
                                  points_in_transit)


if __name__ == "__main__":
  absltest.main()
