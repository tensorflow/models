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

#include "light_curve_util/cc/phase_fold.h"

#include <math.h>
#include <algorithm>
#include <numeric>

#include "absl/strings/substitute.h"

using absl::Substitute;
using std::vector;

namespace astronet {

void PhaseFoldTime(const vector<double>& time, double period, double t0,
                   vector<double>* result) {
  result->resize(time.size());
  double half_period = period / 2;

  // Compute a constant offset to subtract from each time value before taking
  // the remainder modulo the period. This offset ensures that t0 will be
  // centered at +/- period / 2 after the remainder operation.
  double offset = t0 - half_period;

  std::transform(time.begin(), time.end(), result->begin(),
                 [period, offset, half_period](double t) {
                   // If t > offset, then rem is in [0, period) with t0 at
                   // period / 2. Otherwise rem is in (-period, 0] with t0 at
                   // -period / 2. We shift appropriately to return a value in
                   // [-period / 2, period / 2) with t0 centered at 0.
                   double rem = fmod(t - offset, period);
                   return rem < 0 ? rem + half_period : rem - half_period;
                 });
}

// Accept time as a value, because we will phase fold in place.
bool PhaseFoldAndSortLightCurve(vector<double> time, const vector<double>& flux,
                                double period, double t0,
                                vector<double>* folded_time,
                                vector<double>* folded_flux,
                                std::string* error) {
  const std::size_t length = time.size();
  if (flux.size() != length) {
    *error =
        Substitute("time.size() (got: $0) must equal flux.size() (got: $1)",
                   length, flux.size());
    return false;
  }

  // Phase fold time in place.
  PhaseFoldTime(time, period, t0, &time);

  // Sort the indices of time by ascending value.
  vector<std::size_t> sorted_i(length);
  std::iota(sorted_i.begin(), sorted_i.end(), 0);
  std::sort(
      sorted_i.begin(), sorted_i.end(),
      [&time](std::size_t i, std::size_t j) { return time[i] < time[j]; });

  // Copy phase folded and sorted time and flux into the output.
  folded_time->resize(length);
  folded_flux->resize(length);
  for (int i = 0; i < length; ++i) {
    (*folded_time)[i] = time[sorted_i[i]];
    (*folded_flux)[i] = flux[sorted_i[i]];
  }
  return true;
}

}  // namespace astronet
