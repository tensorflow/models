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
    # Single segment.
    all_time = np.concatenate([np.arange(0, 1, 0.1), np.arange(1.5, 2, 0.1)])
    all_flux = np.ones(15)

    # Gap width 0.5.
    split_time, split_flux = util.split(all_time, all_flux, gap_width=0.5)
    self.assertLen(split_time, 2)
    self.assertLen(split_flux, 2)
    self.assertSequenceAlmostEqual(np.arange(0, 1, 0.1), split_time[0])
    self.assertSequenceAlmostEqual(np.ones(10), split_flux[0])
    self.assertSequenceAlmostEqual(np.arange(1.5, 2, 0.1), split_time[1])
    self.assertSequenceAlmostEqual(np.ones(5), split_flux[1])

    # Multi segment.
    all_time = [
        np.concatenate([
            np.arange(0, 1, 0.1),
            np.arange(1.5, 2, 0.1),
            np.arange(3, 4, 0.1)
        ]),
        np.arange(4, 5, 0.1)
    ]
    all_flux = [np.ones(25), np.ones(10)]

    self.assertEqual(len(all_time), 2)
    self.assertEqual(len(all_time[0]), 25)
    self.assertEqual(len(all_time[1]), 10)

    self.assertEqual(len(all_flux), 2)
    self.assertEqual(len(all_flux[0]), 25)
    self.assertEqual(len(all_flux[1]), 10)

    # Gap width 0.5.
    split_time, split_flux = util.split(all_time, all_flux, gap_width=0.5)
    self.assertLen(split_time, 4)
    self.assertLen(split_flux, 4)
    self.assertSequenceAlmostEqual(np.arange(0, 1, 0.1), split_time[0])
    self.assertSequenceAlmostEqual(np.ones(10), split_flux[0])
    self.assertSequenceAlmostEqual(np.arange(1.5, 2, 0.1), split_time[1])
    self.assertSequenceAlmostEqual(np.ones(5), split_flux[1])
    self.assertSequenceAlmostEqual(np.arange(3, 4, 0.1), split_time[2])
    self.assertSequenceAlmostEqual(np.ones(10), split_flux[2])
    self.assertSequenceAlmostEqual(np.arange(4, 5, 0.1), split_time[3])
    self.assertSequenceAlmostEqual(np.ones(10), split_flux[3])

    # Gap width 1.0.
    split_time, split_flux = util.split(all_time, all_flux, gap_width=1)
    self.assertLen(split_time, 3)
    self.assertLen(split_flux, 3)
    self.assertSequenceAlmostEqual([
        0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.5, 1.6, 1.7, 1.8, 1.9
    ], split_time[0])
    self.assertSequenceAlmostEqual(np.ones(15), split_flux[0])
    self.assertSequenceAlmostEqual(np.arange(3, 4, 0.1), split_time[1])
    self.assertSequenceAlmostEqual(np.ones(10), split_flux[1])
    self.assertSequenceAlmostEqual(np.arange(4, 5, 0.1), split_time[2])
    self.assertSequenceAlmostEqual(np.ones(10), split_flux[2])

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

    # One segment totally removed with include_empty_segments = True.
    time = [np.arange(5, dtype=np.float), np.arange(10, 20, dtype=np.float)]
    flux = [10 * t for t in time]
    events = [periodic_event.Event(period=10, duration=2, t0=2.5)]
    output_time, output_flux = util.remove_events(
        time, flux, events, width_factor=3, include_empty_segments=True)
    self.assertLen(output_time, 2)
    self.assertLen(output_flux, 2)
    self.assertSequenceEqual([], output_time[0])
    self.assertSequenceEqual([], output_flux[0])
    self.assertSequenceAlmostEqual([16, 17, 18, 19], output_time[1])
    self.assertSequenceAlmostEqual([160, 170, 180, 190], output_flux[1])

    # One segment totally removed with include_empty_segments = False.
    time = [np.arange(5, dtype=np.float), np.arange(10, 20, dtype=np.float)]
    flux = [10 * t for t in time]
    events = [periodic_event.Event(period=10, duration=2, t0=2.5)]
    output_time, output_flux = util.remove_events(
        time, flux, events, width_factor=3, include_empty_segments=False)
    self.assertLen(output_time, 1)
    self.assertLen(output_flux, 1)
    self.assertSequenceAlmostEqual([16, 17, 18, 19], output_time[0])
    self.assertSequenceAlmostEqual([160, 170, 180, 190], output_flux[0])

  def testInterpolateMissingTime(self):
    # Fewer than 2 finite values.
    with self.assertRaises(ValueError):
      util.interpolate_missing_time(np.array([]))
    with self.assertRaises(ValueError):
      util.interpolate_missing_time(np.array([5.0]))
    with self.assertRaises(ValueError):
      util.interpolate_missing_time(np.array([5.0, np.nan]))
    with self.assertRaises(ValueError):
      util.interpolate_missing_time(np.array([np.nan, np.nan, np.nan]))

    # Small time arrays.
    self.assertSequenceAlmostEqual([0.5, 0.6],
                                   util.interpolate_missing_time(
                                       np.array([0.5, 0.6])))
    self.assertSequenceAlmostEqual([0.5, 0.6, 0.7],
                                   util.interpolate_missing_time(
                                       np.array([0.5, np.nan, 0.7])))

    # Time array of length 20 with some values NaN.
    time = np.array([
        np.nan, 0.5, 1.0, 1.5, 2.0, 2.5, np.nan, 3.5, 4.0, 4.5, 5.0, np.nan,
        np.nan, np.nan, np.nan, 7.5, 8.0, 8.5, np.nan, np.nan
    ])
    interp_time = util.interpolate_missing_time(time)
    self.assertSequenceAlmostEqual([
        0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
        7.0, 7.5, 8.0, 8.5, 9.0, 9.5
    ], interp_time)

    # Fill with 0.0 for missing values at the beginning and end.
    interp_time = util.interpolate_missing_time(time, fill_value=0.0)
    self.assertSequenceAlmostEqual([
        0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
        7.0, 7.5, 8.0, 8.5, 0.0, 0.0
    ], interp_time)

    # Interpolate with cadences.
    cadences = np.array([
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
        114, 115, 116, 117, 118, 119
    ])
    interp_time = util.interpolate_missing_time(time, cadences)
    self.assertSequenceAlmostEqual([
        0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5,
        7.0, 7.5, 8.0, 8.5, 9.0, 9.5
    ], interp_time)

    # Interpolate with missing cadences.
    time = np.array([0.6, 0.7, np.nan, np.nan, np.nan, 1.3, 1.4, 1.5])
    cadences = np.array([106, 107, 108, 109, 110, 113, 114, 115])
    interp_time = util.interpolate_missing_time(time, cadences)
    self.assertSequenceAlmostEqual([0.6, 0.7, 0.8, 0.9, 1.0, 1.3, 1.4, 1.5],
                                   interp_time)

  def testInterpolateMaskedSpline(self):
    all_time = [
        np.arange(0, 10, dtype=np.float),
        np.arange(10, 20, dtype=np.float),
        np.arange(20, 30, dtype=np.float),
    ]
    all_masked_time = [
        np.array([0, 1, 2, 3, 8, 9], dtype=np.float),  # No 4, 5, 6, 7
        np.array([10, 11, 12, 13, 14, 15, 16], dtype=np.float),  # No 17, 18, 19
        np.array([], dtype=np.float)
    ]
    all_masked_spline = [2 * t + 100 for t in all_masked_time]

    interp_spline = util.interpolate_masked_spline(all_time, all_masked_time,
                                                   all_masked_spline)
    self.assertLen(interp_spline, 3)
    self.assertSequenceAlmostEqual(
        [100, 102, 104, 106, 108, 110, 112, 114, 116, 118], interp_spline[0])
    self.assertSequenceAlmostEqual(
        [120, 122, 124, 126, 128, 130, 132, 132, 132, 132], interp_spline[1])
    self.assertTrue(np.all(np.isnan(interp_spline[2])))

  def testReshardArrays(self):
    xs = [
        np.array([1, 2, 3]),
        np.array([4]),
        np.array([5, 6, 7, 8, 9]),
        np.array([]),
    ]
    ys = [
        np.array([]),
        np.array([10, 20]),
        np.array([30, 40, 50, 60]),
        np.array([70]),
        np.array([80, 90]),
    ]
    reshard_xs = util.reshard_arrays(xs, ys)
    self.assertEqual(5, len(reshard_xs))
    np.testing.assert_array_equal([], reshard_xs[0])
    np.testing.assert_array_equal([1, 2], reshard_xs[1])
    np.testing.assert_array_equal([3, 4, 5, 6], reshard_xs[2])
    np.testing.assert_array_equal([7], reshard_xs[3])
    np.testing.assert_array_equal([8, 9], reshard_xs[4])

    with self.assertRaisesRegexp(ValueError,
                                 "xs and ys do not have the same total length"):
      util.reshard_arrays(xs, [np.array([10, 20, 30]), np.array([40, 50])])

  def testUniformCadenceLightCurve(self):
    input_cadence_no = np.array([13, 4, 5, 6, 8, 9, 11, 12])
    input_time = np.array([130, 40, 50, 60, 80, 90, 110, 120])
    input_flux = np.array([1300, 400, 500, 600, 800, np.nan, 1100, 1200])
    cadence_no, time, flux, mask = util.uniform_cadence_light_curve(
        input_cadence_no, input_time, input_flux)
    np.testing.assert_array_equal([4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                                  cadence_no)
    np.testing.assert_array_equal([40, 50, 60, 0, 80, 0, 0, 110, 120, 130],
                                  time)
    np.testing.assert_array_equal(
        [400, 500, 600, 0, 800, 0, 0, 1100, 1200, 1300], flux)
    np.testing.assert_array_equal([1, 1, 1, 0, 1, 0, 0, 1, 1, 1], mask)

    # Add duplicate cadence number.
    input_cadence_no = np.concatenate([input_cadence_no, np.array([13, 14])])
    input_time = np.concatenate([input_time, np.array([130, 140])])
    input_flux = np.concatenate([input_flux, np.array([1300, 1400])])
    with self.assertRaisesRegexp(ValueError, "Duplicate cadence number"):
      util.uniform_cadence_light_curve(input_cadence_no, input_time, input_flux)

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
