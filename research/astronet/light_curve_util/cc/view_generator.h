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

#ifndef TENSORFLOW_MODELS_ASTRONET_LIGHT_CURVE_UTIL_CC_VIEW_GENERATOR_H_
#define TENSORFLOW_MODELS_ASTRONET_LIGHT_CURVE_UTIL_CC_VIEW_GENERATOR_H_

#include <memory>
#include <string>
#include <vector>

namespace astronet {

// Helper class for phase-folding a light curve and then generating "views" of
// the light curve using a median filter.
//
// This class wraps the functions in light_curve_util.h for intended use as a
// a Python extension. It keeps the phase-folded light curve in the class state
// to minimize expensive copies between the language barrier.
class ViewGenerator {
 public:
  // Factory function to create a new ViewGenerator.
  //
  // Input args:
  //   time: Vector of time values, not phase-folded.
  //   flux: Vector of flux values with the same size as time.
  //   period: The period to fold over.
  //   t0: The center of the resulting folded vector; this value is mapped to 0.
  //
  //  Output args:
  //   error: String indicating an error (e.g. time and flux are different
  //       sizes).
  //
  // Returns:
  //   A ViewGenerator. May be a nullptr in the case of an error; see the
  //       "error" string if so.
  static std::unique_ptr<ViewGenerator> Create(const std::vector<double>& time,
                                               const std::vector<double>& flux,
                                               double period, double t0,
                                               std::string* error);

  // Generates a "view" of the phase-folded light curve using a median filter.
  //
  // Note that the time values of the phase-folded light curve are in the range
  // [-period / 2, period / 2).
  //
  // This function applies astronet::MedianFilter() to the phase-folded and
  // sorted light curve, followed optionally by
  // astronet::NormalizeMedianAndMinimum(). See the comments on those
  // functions for more details.
  //
  // Input args:
  //   num_bins: The number of intervals to divide the time axis into. Must be
  //       at least 2.
  //   bin_width: The width of each bin on the time axis. Must be positive, and
  //       less than t_max - t_min.
  //   t_min: The inclusive leftmost value to consider on the time axis. This
  //       should probably be at least -period / 2, which is the minimum
  //       possible value of the phase-folded light curve. Must be less than the
  //       largest value of the phase-folded time axis.
  //   t_max: The exclusive rightmost value to consider on the time axis. This
  //       should probably be at most period / 2, which is the maximum possible
  //       value of the phase-folded light curve. Must be greater than t_min.
  //   normalize: Whether to normalize the output vector to have median 0 and
  //       minimum -1.
  //
  // Output args:
  //   result: Vector of size num_bins containing the median flux values of
  //       uniformly spaced bins on the phase-folded time axis.
  //   error: String indicating an error (e.g. an invalid argument).
  //
  // Returns:
  //   true if the algorithm succeeded. If false, see "error".
  bool GenerateView(int num_bins, double bin_width, double t_min, double t_max,
                    bool normalize, std::vector<double>* result,
                    std::string* error);

 protected:
  // This class can only be constructed by Create().
  ViewGenerator(std::vector<double> time, std::vector<double> flux);

  // phase-folded light curve, sorted by time in ascending order.
  std::vector<double> time_;
  std::vector<double> flux_;
};

}  // namespace astronet

#endif  // TENSORFLOW_MODELS_ASTRONET_LIGHT_CURVE_UTIL_CC_VIEW_GENERATOR_H_
