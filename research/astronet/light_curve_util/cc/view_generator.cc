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

#include "light_curve_util/cc/view_generator.h"

#include "absl/memory/memory.h"
#include "light_curve_util/cc/median_filter.h"
#include "light_curve_util/cc/normalize.h"
#include "light_curve_util/cc/phase_fold.h"

using std::vector;

namespace astronet {

// Accept time as a value, because we will phase fold in place.
std::unique_ptr<ViewGenerator> ViewGenerator::Create(const vector<double>& time,
                                                     const vector<double>& flux,
                                                     double period, double t0,
                                                     std::string* error) {
  vector<double> folded_time(time.size());
  vector<double> folded_flux(flux.size());
  if (!PhaseFoldAndSortLightCurve(time, flux, period, t0, &folded_time,
                                  &folded_flux, error)) {
    return nullptr;
  }
  return absl::WrapUnique(
      new ViewGenerator(std::move(folded_time), std::move(folded_flux)));
}

bool ViewGenerator::GenerateView(int num_bins, double bin_width, double t_min,
                                 double t_max, bool normalize,
                                 vector<double>* result, std::string* error) {
  result->resize(num_bins);
  if (!MedianFilter(time_, flux_, num_bins, bin_width, t_min, t_max, result,
                    error)) {
    return false;
  }
  if (normalize) {
    return NormalizeMedianAndMinimum(*result, result, error);
  }
  return true;
}

ViewGenerator::ViewGenerator(vector<double> time, vector<double> flux)
    : time_(std::move(time)), flux_(std::move(flux)) {}

}  // namespace astronet
