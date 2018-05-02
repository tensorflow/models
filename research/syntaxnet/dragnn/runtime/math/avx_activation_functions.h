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

// Contains logic for activation functions and more-complex elementwise
// vectorized operations.
//
// Uses operator overloading to express computation that looks like regular
// code. Currently, overloaded operators are scoped away in an "internal"
// namespace so they won't be accidentally used.

#ifndef DRAGNN_RUNTIME_MATH_AVX_ACTIVATION_FUNCTIONS_H_
#define DRAGNN_RUNTIME_MATH_AVX_ACTIVATION_FUNCTIONS_H_

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "dragnn/runtime/math/avx_vector_array.h"


#define DRAGNN_AVXAF_ATTRIBUTE_ALWAYS_INLINE __attribute__((always_inline))
#ifdef __clang__
#define DRAGNN_AVXAF_GCC_UNROLL
#else
#define DRAGNN_AVXAF_GCC_UNROLL __attribute__((optimize("unroll-loops")))
#endif

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Public API
namespace activations {
// Calculates elementwise exp(x).
inline AvxFloatVec DRAGNN_AVXAF_ATTRIBUTE_ALWAYS_INLINE DRAGNN_AVXAF_GCC_UNROLL
Exponential(AvxFloatVec x);

// Calculates elementwise sigmoid(x) = 1/(1+exp(-x)).
inline AvxFloatVec DRAGNN_AVXAF_ATTRIBUTE_ALWAYS_INLINE Sigmoid(AvxFloatVec x);

// Calculates elementwise tanh(x).
inline AvxFloatVec DRAGNN_AVXAF_ATTRIBUTE_ALWAYS_INLINE Tanh(AvxFloatVec x);
}  // namespace activations

namespace activations {

// Calculates e^x by representing x = m * ln(2) + r. It does a polynomial
// expansion of e^r, and then multiplies in e^(m * ln(2)) = 2^m.
//
inline AvxFloatVec Exponential(AvxFloatVec x) {
  // EDSL-like helpers for writing vectorized code.
  auto Const = AvxFloatVec::Const;

  constexpr float explo = -88.3762626647949f;
  constexpr float exphi = 88.3762626647950f;

  const float cephes_exp_factors[] = {
      1.9875691500e-4f, 1.3981999507e-3f, 8.3334519073e-3f,
      4.1665795894e-2f, 1.6666665459e-1f, 5.0000001201e-1f,
  };

  // Clamp the input. i.e. assume exp(-88) is close to zero and exp(88) is
  // close to infinity.
  x.Clamp(explo, exphi);

  // Calculate `m = floor(x/ln(2) + 0.5)`.
  constexpr float inv_log2e = 1.44269504088896341f;
  AvxFloatVec m = Const(0.5f);
  m += Const(inv_log2e) * x;
  m.Floor();

  // Calculate `r = x - m*ln(2)` (see function-level comment).
  constexpr float neg_ln2 = -0.6931471805599453f;
  AvxFloatVec r = x;
  r += m * Const(neg_ln2);

  // Calculate a polynomial expansion of y = exp(r).
  AvxFloatVec r_squared(r * r);
  AvxFloatVec y = Const(cephes_exp_factors[0]);
  for (int i = 1; i < 6; ++i) {
    y = y * r + Const(cephes_exp_factors[i]);
  }
  y = y * r_squared + r;
  y += Const(1.0f);

  // Calculate `emm0 = 2^m`. This is done by converting emm0 into an integer,
  // and shifting it into the exponent bits of the desired floating-point
  // result. Recall that the exponent is unsigned with 127 representing 2^0.
  AvxFloatVec emm0 = m;
  emm0 += Const(127.0f);
  AvxIntVec emm0_i(emm0);
  emm0_i.LeftShift(23);

  // The final result is `2^m * exp(r)`.
  return AvxFloatVec(emm0_i.ReinterpretCastFloat() * y);
}

inline AvxFloatVec Tanh(AvxFloatVec x) {
  // EDSL-like helpers for writing vectorized code.
  auto Const = AvxFloatVec::Const;

  const float numerator_coefficients[] = {
      -2.76076847742355e-16f, 2.00018790482477e-13f, -8.60467152213735e-11f,
      5.12229709037114e-08f,  1.48572235717979e-05f, 6.37261928875436e-04f,
      4.89352455891786e-03f,
  };
  const float denominator_coefficients[] = {
      1.19825839466702e-06f,
      1.18534705686654e-04f,
      2.26843463243900e-03f,
      4.89352518554385e-03f,
  };

  // Clamp the inputs to the range [-9, 9] since anything outside this range
  // is +/-1.0 in single-precision.
  x.Clamp(-9.0f, 9.0f);

  // Compute x^2.
  AvxFloatVec x_squared(x * x);

  // Compute the numerator polynomial.
  AvxFloatVec p = Const(numerator_coefficients[0]);
  for (int i = 1; i < 7; ++i) {
    // p = p * x^2 + numerator_coefficients_i
    p = p * x_squared + Const(numerator_coefficients[i]);
  }

  // p = p * x
  p = AvxFloatVec(p * x);

  // Compute the denominator polynomial.
  AvxFloatVec q = Const(denominator_coefficients[0]);
  for (int i = 1; i < 4; ++i) {
    // q = q * x^2 + alqha_i
    q = q * x_squared + Const(denominator_coefficients[i]);
  }

  // Divide the numerator by the denominator.
  return p / q;
}

inline AvxFloatVec Sigmoid(AvxFloatVec x) {
  AvxFloatVec half = AvxFloatVec::Const(0.5);
  return half * Tanh(AvxFloatVec(half * x)) + half;
}

}  // namespace activations
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#undef DRAGNN_AVXAF_ATTRIBUTE_ALWAYS_INLINE
#undef DRAGNN_AVXAF_GCC_UNROLL

#endif  // DRAGNN_RUNTIME_MATH_AVX_ACTIVATION_FUNCTIONS_H_
