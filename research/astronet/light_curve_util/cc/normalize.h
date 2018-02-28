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

#ifndef TENSORFLOW_MODELS_ASTRONET_LIGHT_CURVE_UTIL_CC_NORMALIZE_H_
#define TENSORFLOW_MODELS_ASTRONET_LIGHT_CURVE_UTIL_CC_NORMALIZE_H_

#include <iostream>

#include <string>
#include <vector>

namespace astronet {

// Normalizes a vector with an affine transformation such that its median is
// mapped to 0 and its minimum is mapped to -1.
//
// Input args:
//   x: Vector to normalize. Must have at least 2 elements and all elements
//       cannot be the same value.
//
// Output args:
//   result: Output normalized vector. Can be a pointer to the input vector to
//       perform the normalization in-place.
//   error: String indicating an error (e.g. an invalid argument).
//
// Returns:
//   true if the algorithm succeeded. If false, see "error".
bool NormalizeMedianAndMinimum(const std::vector<double>& x,
                               std::vector<double>* result, std::string* error);

}  // namespace astronet

#endif  // TENSORFLOW_MODELS_ASTRONET_LIGHT_CURVE_UTIL_CC_NORMALIZE_H_
