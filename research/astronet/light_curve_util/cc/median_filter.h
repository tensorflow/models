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

#ifndef TENSORFLOW_MODELS_ASTRONET_LIGHT_CURVE_UTIL_CC_MEDIAN_FILTER_H_
#define TENSORFLOW_MODELS_ASTRONET_LIGHT_CURVE_UTIL_CC_MEDIAN_FILTER_H_

#include <iostream>

#include <string>
#include <vector>

namespace astronet {

// Computes the median y-value in uniform intervals (bins) along the x-axis.
//
// The interval [x_min, x_max) is divided into num_bins uniformly spaced
// intervals of width bin_width. The value computed for each bin is the median
// of all y-values whose corresponding x-value is in the interval.
//
// NOTE: x must be sorted in ascending order or the results will be incorrect.
//
// Input args:
//   x: Vector of x-coordinates sorted in ascending order. Must have at least 2
//       elements, and all elements cannot be the same value.
//   y: Vector of y-coordinates with the same size as x.
//   num_bins: The number of intervals to divide the x-axis into. Must be at
//       least 2.
//   bin_width: The width of each bin on the x-axis. Must be positive, and less
//       than x_max - x_min.
//   x_min: The inclusive leftmost value to consider on the x-axis. Must be less
//       than or equal to the largest value of x.
//   x_max: The exclusive rightmost value to consider on the x-axis. Must be
//       greater than x_min.
//
// Output args:
//   result: Vector of size num_bins containing the median y-values of uniformly
//       spaced bins on the x-axis.
//   error: String indicating an error (e.g. an invalid argument).
//
// Returns:
//   true if the algorithm succeeded. If false, see "error".
bool MedianFilter(const std::vector<double>& x, const std::vector<double>& y,
                  int num_bins, double bin_width, double x_min, double x_max,
                  std::vector<double>* result, std::string* error);

}  // namespace astronet

#endif  // TENSORFLOW_MODELS_ASTRONET_LIGHT_CURVE_UTIL_CC_MEDIAN_FILTER_H_
