/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "light_curve_util/cc/median_filter.h"

#include "absl/strings/substitute.h"
#include "light_curve_util/cc/median.h"

using absl::Substitute;
using std::min;
using std::vector;

namespace astronet {

bool MedianFilter(const vector<double>& x, const vector<double>& y,
                  int num_bins, double bin_width, double x_min, double x_max,
                  vector<double>* result, std::string* error) {
  const std::size_t x_size = x.size();
  if (x_size < 2) {
    *error = Substitute("x.size() must be greater than 1. Got: $0", x_size);
    return false;
  }
  if (x_size != y.size()) {
    *error = Substitute("x.size() (got: $0) must equal y.size() (got: $1)",
                        x_size, y.size());
    return false;
  }
  const double x_first = x[0];
  const double x_last = x[x_size - 1];
  if (x_first >= x_last) {
    *error = Substitute(
        "The first element of x (got: $0) must be less than the last "
        "element (got: $1). Either x is not sorted or all elements are "
        "equal.",
        x_first, x_last);
    return false;
  }
  if (x_min >= x_max) {
    *error = Substitute("x_min (got: $0) must be less than x_max (got: $1)",
                        x_min, x_max);
    return false;
  }
  if (x_min > x_last) {
    *error = Substitute(
        "x_min (got: $0) must be less than or equal to the largest value of x "
        "(got: $1)",
        x_min, x_last);
    return false;
  }
  if (bin_width <= 0) {
    *error = Substitute("bin_width must be positive. Got: $0", bin_width);
    return false;
  }
  if (bin_width >= x_max - x_min) {
    *error = Substitute(
        "bin_width (got: $0) must be less than x_max - x_min (got: $1)",
        bin_width, x_max - x_min);
    return false;
  }
  if (num_bins < 2) {
    *error = Substitute("num_bins must be greater than 1. Got: $0", num_bins);
    return false;
  }

  result->resize(num_bins);

  // Compute the spacing between midpoints of adjacent bins.
  double bin_spacing = (x_max - x_min - bin_width) / (num_bins - 1);

  // Create a vector to hold the values of the current bin on each iteration.
  // Its initial size is twice the expected number of points per bin if x
  // values are uniformly spaced. It will be expanded as necessary.
  int points_per_bin =
      1 + static_cast<int>(x_size * min(1.0, bin_width / (x_last - x_first)));
  vector<double> bin_values(2 * points_per_bin);

  // Create a vector to hold the indices of any empty bins.
  vector<int> empty_bins;

  // Find the first element of x >= x_min. This loop is guaranteed to produce
  // a valid index because we know that x_min <= x_last.
  int x_start = 0;
  while (x[x_start] < x_min) ++x_start;

  // The bin at index i is the median of all elements y[j] such that
  // bin_min <= x[j] < bin_max, where bin_min and bin_max are the endpoints of
  // bin i.
  double bin_min = x_min;              // Left endpoint of the current bin.
  double bin_max = x_min + bin_width;  // Right endpoint of the current bin.
  int j_start = x_start;  // Index of the first element in the current bin.
  int j = x_start;        // Index of the current element in the current bin.

  for (int i = 0; i < num_bins; ++i) {
    // Move j_start to the first index of x >= bin_min.
    while (j_start < x_size && x[j_start] < bin_min) ++j_start;

    // Accumulate values y[j] such that bin_min <= x[j] < bin_max. After this
    // loop, j is the exclusive end index of the current bin.
    j = j_start;
    while (j < x_size && x[j] < bin_max) {
      if (j - j_start >= bin_values.size()) {
        bin_values.resize(2 * bin_values.size());  // Expand if necessary.
      }
      bin_values[j - j_start] = y[j];
      ++j;
    }

    int n = j - j_start;  // Number of points in the bin.
    if (n == 0) {
      empty_bins.push_back(i);  // Empty bin.
    } else {
      // Compute and insert the median bin value.
      (*result)[i] = InPlaceMedian(bin_values.begin(), bin_values.begin() + n);
    }

    // Advance the bin.
    bin_min += bin_spacing;
    bin_max += bin_spacing;
  }

  // For empty bins, fall back to the median y value between x_min and x_max.
  if (!empty_bins.empty()) {
    double median = Median(y.begin() + x_start, y.begin() + j);
    for (int i : empty_bins) {
      (*result)[i] = median;
    }
  }
  return true;
}

}  // namespace astronet
