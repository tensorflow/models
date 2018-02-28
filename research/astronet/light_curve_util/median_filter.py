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

"""Utility function for smoothing data using a median filter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def median_filter(x, y, num_bins, bin_width=None, x_min=None, x_max=None):
  """Computes the median y-value in uniform intervals (bins) along the x-axis.

  The interval [x_min, x_max) is divided into num_bins uniformly spaced
  intervals of width bin_width. The value computed for each bin is the median
  of all y-values whose corresponding x-value is in the interval.

  NOTE: x must be sorted in ascending order or the results will be incorrect.

  Args:
    x: 1D array of x-coordinates sorted in ascending order. Must have at least 2
        elements, and all elements cannot be the same value.
    y: 1D array of y-coordinates with the same size as x.
    num_bins: The number of intervals to divide the x-axis into. Must be at
        least 2.
    bin_width: The width of each bin on the x-axis. Must be positive, and less
        than x_max - x_min. Defaults to (x_max - x_min) / num_bins.
    x_min: The inclusive leftmost value to consider on the x-axis. Must be less
        than or equal to the largest value of x. Defaults to min(x).
    x_max: The exclusive rightmost value to consider on the x-axis. Must be
        greater than x_min. Defaults to max(x).

  Returns:
    1D NumPy array of size num_bins containing the median y-values of uniformly
    spaced bins on the x-axis.

  Raises:
    ValueError: If an argument has an inappropriate value.
  """
  if num_bins < 2:
    raise ValueError("num_bins must be at least 2. Got: %d" % num_bins)

  # Validate the lengths of x and y.
  x_len = len(x)
  if x_len < 2:
    raise ValueError("len(x) must be at least 2. Got: %s" % x_len)
  if x_len != len(y):
    raise ValueError("len(x) (got: %d) must equal len(y) (got: %d)" % (x_len,
                                                                       len(y)))

  # Validate x_min and x_max.
  x_min = x_min if x_min is not None else x[0]
  x_max = x_max if x_max is not None else x[-1]
  if x_min >= x_max:
    raise ValueError("x_min (got: %d) must be less than x_max (got: %d)" %
                     (x_min, x_max))
  if x_min > x[-1]:
    raise ValueError(
        "x_min (got: %d) must be less than or equal to the largest value of x "
        "(got: %d)" % (x_min, x[-1]))

  # Validate bin_width.
  bin_width = bin_width if bin_width is not None else (x_max - x_min) / num_bins
  if bin_width <= 0:
    raise ValueError("bin_width must be positive. Got: %d" % bin_width)
  if bin_width >= x_max - x_min:
    raise ValueError(
        "bin_width (got: %d) must be less than x_max - x_min (got: %d)" %
        (bin_width, x_max - x_min))

  bin_spacing = (x_max - x_min - bin_width) / (num_bins - 1)

  # Bins with no y-values will fall back to the global median.
  result = np.repeat(np.median(y), num_bins)

  # Find the first element of x >= x_min. This loop is guaranteed to produce
  # a valid index because we know that x_min <= x[-1].
  x_start = 0
  while x[x_start] < x_min:
    x_start += 1

  # The bin at index i is the median of all elements y[j] such that
  # bin_min <= x[j] < bin_max, where bin_min and bin_max are the endpoints of
  # bin i.
  bin_min = x_min  # Left endpoint of the current bin.
  bin_max = x_min + bin_width  # Right endpoint of the current bin.
  j_start = x_start  # Inclusive left index of the current bin.
  j_end = x_start  # Exclusive end index of the current bin.

  for i in range(num_bins):
    # Move j_start to the first index of x >= bin_min.
    while j_start < x_len and x[j_start] < bin_min:
      j_start += 1

    # Move j_end to the first index of x >= bin_max (exclusive end index).
    while j_end < x_len and x[j_end] < bin_max:
      j_end += 1

    if j_end > j_start:
      # Compute and insert the median bin value.
      result[i] = np.median(y[j_start:j_end])

    # Advance the bin.
    bin_min += bin_spacing
    bin_max += bin_spacing

  return result
