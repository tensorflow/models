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

#include "light_curve_util/cc/normalize.h"

#include <algorithm>

#include "absl/strings/substitute.h"
#include "light_curve_util/cc/median.h"

using absl::Substitute;
using std::vector;

namespace astronet {

bool NormalizeMedianAndMinimum(const vector<double>& x, vector<double>* result,
                               std::string* error) {
  if (x.size() < 2) {
    *error = Substitute("x.size() must be greater than 1. Got: $0", x.size());
    return false;
  }

  // Find the median of x.
  vector<double> x_copy(x);
  const double median = InPlaceMedian(x_copy.begin(), x_copy.end());

  // Find the min element of x. As a post condition of InPlaceMedian, we only
  // need to search elements lower than the middle.
  const auto x_copy_middle = x_copy.begin() + x_copy.size() / 2;
  const auto minimum = std::min_element(x_copy.begin(), x_copy_middle);

  // Guaranteed to be positive, unless the median exactly equals the minimum.
  double normalizer = median - *minimum;
  if (normalizer <= 0) {
    *error = Substitute("Minimum and median have the same value: $0", median);
    return false;
  }

  result->resize(x.size());
  std::transform(
      x.begin(), x.end(), result->begin(),
      [median, normalizer](double v) { return (v - median) / normalizer; });
  return true;
}

}  // namespace astronet
