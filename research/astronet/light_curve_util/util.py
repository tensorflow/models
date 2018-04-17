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

"""Light curve utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
from six.moves import range  # pylint:disable=redefined-builtin


def phase_fold_time(time, period, t0):
  """Creates a phase-folded time vector.

  result[i] is the unique number in [-period / 2, period / 2)
  such that result[i] = time[i] - t0 + k_i * period, for some integer k_i.

  Args:
    time: 1D numpy array of time values.
    period: A positive real scalar; the period to fold over.
    t0: The center of the resulting folded vector; this value is mapped to 0.

  Returns:
    A 1D numpy array.
  """
  half_period = period / 2
  result = np.mod(time + (half_period - t0), period)
  result -= half_period
  return result


def split(all_time, all_flux, gap_width=0.75):
  """Splits a light curve on discontinuities (gaps).

  This function accepts a light curve that is either a single segment, or is
  piecewise defined (e.g. split by quarter breaks or gaps in the in the data).

  Args:
    all_time: Numpy array or list of numpy arrays; each is a sequence of time
        values.
    all_flux: Numpy array or list of numpy arrays; each is a sequence of flux
        values of the corresponding time array.
    gap_width: Minimum gap size (in time units) for a split.

  Returns:
    out_time: List of numpy arrays; the split time arrays.
    out_flux: List of numpy arrays; the split flux arrays.
  """
  # Handle single-segment inputs.
  # We must use an explicit length test on all_time because implicit conversion
  # to bool fails if all_time is a numpy array, and all_time.size is not defined
  # if all_time is a list of numpy arrays.
  if len(all_time) > 0 and not isinstance(all_time[0], collections.Iterable):  # pylint:disable=g-explicit-length-test
    all_time = [all_time]
    all_flux = [all_flux]

  out_time = []
  out_flux = []
  for time, flux in zip(all_time, all_flux):
    start = 0
    for end in range(1, len(time) + 1):
      # Choose the largest endpoint such that time[start:end] has no gaps.
      if end == len(time) or time[end] - time[end - 1] > gap_width:
        out_time.append(time[start:end])
        out_flux.append(flux[start:end])
        start = end

  return out_time, out_flux


def remove_events(all_time, all_flux, events, width_factor=1.0):
  """Removes events from a light curve.

  This function accepts either a single-segment or piecewise-defined light
  curve (e.g. one that is split by quarter breaks or gaps in the in the data).

  Args:
    all_time: Numpy array or list of numpy arrays; each is a sequence of time
        values.
    all_flux: Numpy array or list of numpy arrays; each is a sequence of flux
        values of the corresponding time array.
    events: List of Event objects to remove.
    width_factor: Fractional multiplier of the duration of each event to remove.

  Returns:
    output_time: Numpy array or list of numpy arrays; the time arrays with
        events removed.
    output_flux: Numpy array or list of numpy arrays; the flux arrays with
        events removed.
  """
  # Handle single-segment inputs.
  # We must use an explicit length test on all_time because implicit conversion
  # to bool fails if all_time is a numpy array and all_time.size is not defined
  # if all_time is a list of numpy arrays.
  if len(all_time) > 0 and not isinstance(all_time[0], collections.Iterable):  # pylint:disable=g-explicit-length-test
    all_time = [all_time]
    all_flux = [all_flux]
    single_segment = True
  else:
    single_segment = False

  output_time = []
  output_flux = []
  for time, flux in zip(all_time, all_flux):
    mask = np.ones_like(time, dtype=np.bool)
    for event in events:
      transit_dist = np.abs(phase_fold_time(time, event.period, event.t0))
      mask = np.logical_and(mask,
                            transit_dist > 0.5 * width_factor * event.duration)

    if single_segment:
      output_time = time[mask]
      output_flux = flux[mask]
    else:
      output_time.append(time[mask])
      output_flux.append(flux[mask])

  return output_time, output_flux


def interpolate_masked_spline(all_time, all_masked_time, all_masked_spline):
  """Linearly interpolates spline values across masked points.

  Args:
    all_time: List of numpy arrays; each is a sequence of time values.
    all_masked_time: List of numpy arrays; each is a sequence of time values
        with some values missing (masked).
    all_masked_spline: List of numpy arrays; the masked spline values
        corresponding to all_masked_time.

  Returns:
    interp_spline: List of numpy arrays; each is the masked spline with missing
        points linearly interpolated.
  """
  interp_spline = []
  for time, masked_time, masked_spline in zip(
      all_time, all_masked_time, all_masked_spline):
    if len(masked_time) > 0:  # pylint:disable=g-explicit-length-test
      interp_spline.append(np.interp(time, masked_time, masked_spline))
    else:
      interp_spline.append(np.full_like(time, np.nan))
  return interp_spline


def count_transit_points(time, event):
  """Computes the number of points in each transit of a given event.

  Args:
    time: Sorted numpy array of time values.
    event: An Event object.

  Returns:
    A numpy array containing the number of time points "in transit" for each
    transit occurring between the first and last time values.

  Raises:
    ValueError: If there are more than 10**6 transits.
  """
  t_min = np.min(time)
  t_max = np.max(time)

  # Tiny periods or erroneous time values could make this loop take forever.
  if (t_max - t_min) / event.period > 10**6:
    raise ValueError(
        "Too many transits! Time range is [%.2f, %.2f] and period is %.2e." %
        (t_min, t_max, event.period))

  # Make sure t0 is in [t_min, t_min + period).
  t0 = np.mod(event.t0 - t_min, event.period) + t_min

  # Prepare loop variables.
  points_in_transit = []
  i, j = 0, 0

  for transit_midpoint in np.arange(t0, t_max, event.period):
    transit_begin = transit_midpoint - event.duration / 2
    transit_end = transit_midpoint + event.duration / 2

    # Move time[i] to the first point >= transit_begin.
    while time[i] < transit_begin:
      # transit_begin is guaranteed to be < np.max(t) (provided duration >= 0).
      # Therefore, i cannot go out of range.
      i += 1

    # Move time[j] to the first point > transit_end.
    while time[j] <= transit_end:
      j += 1
      # j went out of range. We're finished.
      if j >= len(time):
        break

    # The points in the current transit duration are precisely time[i:j].
    # Since j is an exclusive index, there are exactly j-i points in transit.
    points_in_transit.append(j - i)

  return np.array(points_in_transit)
