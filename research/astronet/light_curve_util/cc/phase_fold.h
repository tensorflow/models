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

#ifndef TENSORFLOW_MODELS_ASTRONET_LIGHT_CURVE_UTIL_CC_PHASE_FOLD_H_
#define TENSORFLOW_MODELS_ASTRONET_LIGHT_CURVE_UTIL_CC_PHASE_FOLD_H_

#include <iostream>

#include <string>
#include <vector>

namespace astronet {

// Creates a phase-folded time vector.
//
// Specifically, result[i] is the unique number in [-period / 2, period / 2)
// such that result[i] = time[i] - t0 + k_i * period, for some integer k_i.
//
// Input args:
//   time: Input vector of time values.
//   period: The period to fold over.
//   t0: The center of the resulting folded vector; this value is mapped to 0.
//
// Output args:
//   result: Output phase folded vector. Can be a pointer to the input vector to
//       perform the phase-folding in-place.
void PhaseFoldTime(const std::vector<double>& time, double period, double t0,
                   std::vector<double>* result);

// Phase folds a light curve and sorts by ascending phase-folded time.
//
// See the comment on PhaseFoldTime for a description of the phase folding
// technique for the time values. The flux values are not modified; they are
// simply permuted to correspond to the sorted phase folded time values.
//
// Input args:
//   time: Vector of time values.
//   flux: Vector of flux values with the same size as time.
//   period: The period to fold over.
//   t0: The center of the resulting folded vector; this value is mapped to 0.
//
// Output args:
//   folded_time: Output phase folded time values, sorted in ascending order.
//   folded_flux: Output flux values corresponding pointwise to folded_time.
//   error: String indicating an error (e.g. time and flux are different sizes).
//
// Returns:
//   true if the algorithm succeeded. If false, see "error".
bool PhaseFoldAndSortLightCurve(std::vector<double> time,
                                const std::vector<double>& flux, double period,
                                double t0, std::vector<double>* folded_time,
                                std::vector<double>* folded_flux,
                                std::string* error);

}  // namespace astronet

#endif  // TENSORFLOW_MODELS_ASTRONET_LIGHT_CURVE_UTIL_CC_PHASE_FOLD_H_
