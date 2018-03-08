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

#ifndef TENSORFLOW_MODELS_ASTRONET_LIGHT_CURVE_UTIL_CC_TEST_UTIL_H_
#define TENSORFLOW_MODELS_ASTRONET_LIGHT_CURVE_UTIL_CC_TEST_UTIL_H_

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace astronet {

// Like testing::DoubleNear, but operates on pairs and can therefore be used in
// testing::Pointwise.
MATCHER(DoubleNear, "") {
  return testing::Value(std::get<0>(arg),
                        testing::DoubleNear(std::get<1>(arg), 1e-12));
}

// Returns the range {start, start + step, start + 2 * step, ...} up to the
// exclusive end value, stop.
inline std::vector<double> range(double start, double stop, double step) {
  std::vector<double> result;
  while (start < stop) {
    result.push_back(start);
    start += step;
  }
  return result;
}

}  // namespace astronet

#endif  // TENSORFLOW_MODELS_ASTRONET_LIGHT_CURVE_UTIL_CC_TEST_UTIL_H_
