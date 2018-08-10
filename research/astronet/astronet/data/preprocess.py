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

"""Functions for reading and preprocessing light curves."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from astronet.util import example_util
from light_curve_util import kepler_io
from light_curve_util import median_filter
from light_curve_util import util
from third_party.kepler_spline import kepler_spline


def read_light_curve(kepid, kepler_data_dir):
  """Reads a Kepler light curve.

  Args:
    kepid: Kepler id of the target star.
    kepler_data_dir: Base directory containing Kepler data. See
        kepler_io.kepler_filenames().

  Returns:
    all_time: A list of numpy arrays; the time values of the raw light curve.
    all_flux: A list of numpy arrays corresponding to the time arrays in
        all_time.

  Raises:
    IOError: If the light curve files for this Kepler ID cannot be found.
  """
  # Read the Kepler light curve.
  file_names = kepler_io.kepler_filenames(kepler_data_dir, kepid)
  if not file_names:
    raise IOError("Failed to find .fits files in %s for Kepler ID %s" %
                  (kepler_data_dir, kepid))

  return kepler_io.read_kepler_light_curve(file_names)


def process_light_curve(all_time, all_flux):
  """Removes low-frequency variability from a light curve.

  Args:
    all_time: A list of numpy arrays; the time values of the raw light curve.
    all_flux: A list of numpy arrays corresponding to the time arrays in
        all_time.

  Returns:
    time: 1D NumPy array; the time values of the light curve.
    flux: 1D NumPy array; the normalized flux values of the light curve.
  """
  # Split on gaps.
  all_time, all_flux = util.split(all_time, all_flux, gap_width=0.75)

  # Fit a piecewise-cubic spline with default arguments.
  spline = kepler_spline.fit_kepler_spline(all_time, all_flux, verbose=False)[0]

  # Concatenate the piecewise light curve and spline.
  time = np.concatenate(all_time)
  flux = np.concatenate(all_flux)
  spline = np.concatenate(spline)

  # In rare cases the piecewise spline contains NaNs in places the spline could
  # not be fit. We can't normalize those points if the spline isn't defined
  # there. Instead we just remove them.
  finite_i = np.isfinite(spline)
  if not np.all(finite_i):
    time = time[finite_i]
    flux = flux[finite_i]
    spline = spline[finite_i]

  # "Flatten" the light curve (remove low-frequency variability) by dividing by
  # the spline.
  flux /= spline

  return time, flux


def phase_fold_and_sort_light_curve(time, flux, period, t0):
  """Phase folds a light curve and sorts by ascending time.

  Args:
    time: 1D NumPy array of time values.
    flux: 1D NumPy array of flux values.
    period: A positive real scalar; the period to fold over.
    t0: The center of the resulting folded vector; this value is mapped to 0.

  Returns:
    folded_time: 1D NumPy array of phase folded time values in
        [-period / 2, period / 2), where 0 corresponds to t0 in the original
        time array. Values are sorted in ascending order.
    folded_flux: 1D NumPy array. Values are the same as the original input
        array, but sorted by folded_time.
  """
  # Phase fold time.
  time = util.phase_fold_time(time, period, t0)

  # Sort by ascending time.
  sorted_i = np.argsort(time)
  time = time[sorted_i]
  flux = flux[sorted_i]

  return time, flux


def generate_view(time, flux, num_bins, bin_width, t_min, t_max,
                  normalize=True):
  """Generates a view of a phase-folded light curve using a median filter.

  Args:
    time: 1D array of time values, sorted in ascending order.
    flux: 1D array of flux values.
    num_bins: The number of intervals to divide the time axis into.
    bin_width: The width of each bin on the time axis.
    t_min: The inclusive leftmost value to consider on the time axis.
    t_max: The exclusive rightmost value to consider on the time axis.
    normalize: Whether to center the median at 0 and minimum value at -1.

  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  view = median_filter.median_filter(time, flux, num_bins, bin_width, t_min,
                                     t_max)

  if normalize:
    view -= np.median(view)
    view /= np.abs(np.min(view))

  return view


def global_view(time, flux, period, num_bins=2001, bin_width_factor=1 / 2001):
  """Generates a 'global view' of a phase folded light curve.

  See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta

  Args:
    time: 1D array of time values, sorted in ascending order.
    flux: 1D array of flux values.
    period: The period of the event (in days).
    num_bins: The number of intervals to divide the time axis into.
    bin_width_factor: Width of the bins, as a fraction of period.

  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  return generate_view(
      time,
      flux,
      num_bins=num_bins,
      bin_width=period * bin_width_factor,
      t_min=-period / 2,
      t_max=period / 2)


def local_view(time,
               flux,
               period,
               duration,
               num_bins=201,
               bin_width_factor=0.16,
               num_durations=4):
  """Generates a 'local view' of a phase folded light curve.

  See Section 3.3 of Shallue & Vanderburg, 2018, The Astronomical Journal.
  http://iopscience.iop.org/article/10.3847/1538-3881/aa9e09/meta

  Args:
    time: 1D array of time values, sorted in ascending order.
    flux: 1D array of flux values.
    period: The period of the event (in days).
    duration: The duration of the event (in days).
    num_bins: The number of intervals to divide the time axis into.
    bin_width_factor: Width of the bins, as a fraction of duration.
    num_durations: The number of durations to consider on either side of 0 (the
        event is assumed to be centered at 0).

  Returns:
    1D NumPy array of size num_bins containing the median flux values of
    uniformly spaced bins on the phase-folded time axis.
  """
  return generate_view(
      time,
      flux,
      num_bins=num_bins,
      bin_width=duration * bin_width_factor,
      t_min=max(-period / 2, -duration * num_durations),
      t_max=min(period / 2, duration * num_durations))


def generate_example_for_tce(time, flux, tce):
  """Generates a tf.train.Example representing an input TCE.

  Args:
    time: 1D NumPy array; the time values of the light curve.
    flux: 1D NumPy array; the normalized flux values of the light curve.
    tce: Dict-like object containing at least 'tce_period', 'tce_duration', and
        'tce_time0bk'. Additional items are included as features in the output.

  Returns:
    A tf.train.Example containing features 'global_view', 'local_view', and all
    values present in `tce`.
  """
  period = tce["tce_period"]
  duration = tce["tce_duration"]
  t0 = tce["tce_time0bk"]

  time, flux = phase_fold_and_sort_light_curve(time, flux, period, t0)

  # Make output proto.
  ex = tf.train.Example()

  # Set time series features.
  example_util.set_float_feature(ex, "global_view",
                                 global_view(time, flux, period))
  example_util.set_float_feature(ex, "local_view",
                                 local_view(time, flux, period, duration))

  # Set other features in `tce`.
  for name, value in tce.items():
    example_util.set_feature(ex, name, [value])

  return ex
