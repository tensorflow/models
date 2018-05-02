// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

// Declares 16-bit floating point types.

#ifndef DRAGNN_RUNTIME_MATH_FLOAT16_TYPES_H_
#define DRAGNN_RUNTIME_MATH_FLOAT16_TYPES_H_

#if defined(__F16C__)
#include <emmintrin.h>
#endif

#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/casts.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Represents a truncated 16-bit floating point value. This corresponds to
// `bfloat16` in TensorFlow. It just chops the last 16 least-significant bits
// off the significand of a 32-bit floating point value, leaving 7 significand
// bits, 8 exponent bits, and 1 sign bit.
struct TruncatedFloat16 {
  // Slow unpacking routine. Use avx_vector_array.h for normal operation.
  float DebugToFloat() const {
    uint32 upcast = bits;
    upcast <<= 16;
    return tensorflow::bit_cast<float>(upcast);
  }

  // Slow packing routine. Use avx_vector_array.h for normal operation.
  static TruncatedFloat16 DebugFromFloat(float value) {
    uint32 float_bits = tensorflow::bit_cast<uint32>(value);
    return TruncatedFloat16{static_cast<uint16>(float_bits >> 16)};
  }

  uint16 bits;
};

static_assert(sizeof(TruncatedFloat16) == sizeof(uint16), "Bad struct size");

// Currently, only CPUs with the F16C instruction set are supported. All use of
// this struct should be flag-guarded.
//
// If this becomes a problem, we can implement this method with Eigen's
// CUDA/Half.h.
#if defined(__F16C__)

// Represents an IEEE-754 16-bit floating point value. This has 10 significand
// bits, 5 exponent bits, and 1 sign bit.
//
// TODO(googleuser): Either add compatibility support, or delete this code if
// it turns out not to be helpful.
struct IeeeFloat16 {
  // Slow unpacking routine. Use avx_vector_array.h for normal operation.
  float DebugToFloat() const { return _cvtsh_ss(bits); }

  // Slow packing routine. Use avx_vector_array.h for normal operation.
  static IeeeFloat16 DebugFromFloat(float value) {
    return IeeeFloat16{_cvtss_sh(value, 0)};
  }

  uint16 bits;
};

static_assert(sizeof(IeeeFloat16) == sizeof(uint16), "Bad struct size");

#endif

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_MATH_FLOAT16_TYPES_H_
