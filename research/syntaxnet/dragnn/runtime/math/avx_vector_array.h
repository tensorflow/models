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

// Wraps AVX vectors into convenient helper classes. This contains a class
// wrapping a single AVX register, AvxFloatVec, and a class to manipulate a
// batch of registers, AvxFloatVecArray. Use of the latter is recommended where
// applicable, since it will be unrolled into more vectorizable code.

#ifndef DRAGNN_RUNTIME_MATH_AVX_VECTOR_ARRAY_H_
#define DRAGNN_RUNTIME_MATH_AVX_VECTOR_ARRAY_H_

#include <cmath>
#if defined(__AVX__)
#include <immintrin.h>
#elif defined(__SSE4_2__)
#include <nmmintrin.h>
#endif

#include "dragnn/runtime/math/float16_types.h"

#define DRAGNN_AVXVA_ALWAYS_INLINE inline __attribute__((always_inline))
#ifdef __clang__

// Clang doesn't support __attribute__((optimize(...))).
#define DRAGNN_AVXVA_INLINED_UNROLLED inline __attribute__((always_inline))

#else

// Assume we're using GCC, which does.
#define DRAGNN_AVXVA_INLINED_UNROLLED   \
  inline __attribute__((always_inline)) \
      __attribute__((optimize("unroll-loops")))

#endif

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Number of single-precision floating point numbers that fit into a single SSE
// / AVX2 register (which are 128 and 256 bits respectively).
constexpr int kSseWidth = 128 / 32;  // = 4
constexpr int kAvxWidth = 256 / 32;  // = 8
constexpr int kSseWidthHalfPrecision = 128 / 16;  // = 8
constexpr int kAvxWidthHalfPrecision = 256 / 16;  // = 16

class AvxFloatVec;

namespace internal {
// This struct should always be eliminated by the compiler; it only exists so we
// can write `foo += bar * baz`, and have that compiled into a single FMA
// operation.
struct AvxMultiplyExpr {
  const AvxFloatVec &a;
  const AvxFloatVec &b;
};
}  // namespace internal

// Allows EDSL-like programming with AVX vectors.
inline internal::AvxMultiplyExpr operator*(const AvxFloatVec &a,
                                           const AvxFloatVec &b);
inline AvxFloatVec operator+(const internal::AvxMultiplyExpr &expr,
                             const AvxFloatVec &v);
inline AvxFloatVec operator+(const AvxFloatVec &a, const AvxFloatVec &b);
inline AvxFloatVec operator/(const AvxFloatVec &a, const AvxFloatVec &b);
inline AvxFloatVec operator-(const AvxFloatVec &a, const AvxFloatVec &b);

// API over a single AVX vector (register). The implementation will either use
// a real AVX vector, or a fixed array of floats for compatibility.
//
// Note that we include the "inline" directive in declarations, not just
// definitions, because it is necessary for the "always_inline" directive.
struct AvxFloatVec {
 public:
  AvxFloatVec() {}

  // Evaluates an AvxMultiplyExpr intermediary without adding anything. This is
  // not an implicit cast, because typically when we write `a * b` we want to
  // add it to something and use an FMA operation.
  explicit AvxFloatVec(const internal::AvxMultiplyExpr &expr);

  // Loads from an aligned region of memory.
  inline void Load(const float *source);

  // Loads a constant value.
  inline void LoadConstVector(const float val);

  // Stores to an aligned region of memory.
  inline void Store(float *dst) const;

  // Adds `a * b` to this value, using a fused multiply-add operation.
  inline void AddProductOf(const AvxFloatVec &a, const AvxFloatVec &b);

  // Element-wise floor.
  inline void Floor();

  // Element-wise clamps values between a min and max value.
  inline void Clamp(const float min_value, const float max_value);

  // Convenience method for more complex calculations.
  static DRAGNN_AVXVA_ALWAYS_INLINE AvxFloatVec Const(const float value) {
    AvxFloatVec result;
    result.LoadConstVector(value);
    return result;
  }

  // Syntactic sugar for computing an FMA operation.
  inline AvxFloatVec &operator+=(const internal::AvxMultiplyExpr &to_add);

  // Adds another vector element-wise.
  inline AvxFloatVec &operator+=(const AvxFloatVec &vec);

  // Subtracts another vector element-wise.
  inline AvxFloatVec &operator-=(const AvxFloatVec &vec);

  // Divides another vector element-wise.
  inline AvxFloatVec &operator/=(const AvxFloatVec &vec);

#if defined(__AVX__)
  __m256 ymm;
#elif defined(__SSE4_2__)
  __m128 xmm[2];
#else
  float ymm[8];
#endif
};

// Small wrapper around integer AVX vectors, exposing only methods we need for
// implementing the activation functions.
//
// As above, `inline` is specified here for the always_inline directive.
class AvxIntVec {
 public:
  // Constructs an AVX integer vector, by converting floating-point values.
  inline explicit AvxIntVec(const AvxFloatVec &v);

  // Left-shifts integer values.
  inline void LeftShift(int bits);

  // Reinterprets the register as a floating-point register, for bitwise tricks.
  inline AvxFloatVec ReinterpretCastFloat();

 private:
  // Underlying register.
#if defined(__AVX__)
  __m256i ymm_;
#elif defined(__SSE4_2__)
  __m128i xmm_[2];
#else
  int ymm_[8];
#endif
};

// Implements the index permutation that is effectively applied by the
// _mm256_unpack instructions. This permutation is equivalent to swapping the
// 3rd and 4th bits. See the PermutationFunctionIsEqualToTable test for the
// effective permutation that this encodes.
//
// We haven't done performance testing, but hopefully this is sufficiently fast
// for the compatibility routine. Hopefully in its use below, the compiler will
// determine it is being called with a constant (post-unrolling) and inline it.
DRAGNN_AVXVA_ALWAYS_INLINE int FastUnpackPermutation(int original_idx) {
  // Bit in the 4th index if the 3rd and 4th bits should be swapped.
  int should_swap = (original_idx + /* 0b0100 */ 4) & /* 0b1000 */ 8;

  // If should_swap is zero, leaves original_idx untouched. Otherwise, does an
  // xor with 0b1100, which will flip 10 to 01 and 01 to 10.
  return (should_swap | (should_swap >> 1)) ^ original_idx;
}

// API over an array of AVX vectors (registers). The methods on this class are
// annotated such that the compiler should unroll them.
template <int N>
struct AvxFloatVecArray {
 public:
  DRAGNN_AVXVA_INLINED_UNROLLED void Load(const float *source) {
    for (int i = 0; i < N; i++) {
      vectors[i].Load(source + 8 * i);
    }
  }

  DRAGNN_AVXVA_INLINED_UNROLLED void Load(const float *source, int max_idx) {
    for (int i = 0; i < N; i++) {
      if (i < max_idx) {
        vectors[i].Load(source + 8 * i);
      } else {
        // When testing with a memory sanitizer, we make sure not to read
        // uninitialized values. This is usually safe in normal operation
        // because such results are never stored (via corresponding
        // store-masking logic), but of course each algorithm must be tested to
        // ensure correct operation.
        //
        // It is also worth pointing out that exceptional values (NaN, etc.) can
        // slow down AVX/FMA floating point operations considerably. So we
        // should investigate whether this is worth enabling in all cases (and
        // forcing algorithms to provide a default).
#if defined(MEMORY_SANITIZER)
        vectors[i].LoadConstVector(0);
#endif
      }
    }
  }

  // Reads and unpacks truncated half-precision values.
  //
  // Currently, only matrix coefficients use compressed/half-precision values,
  // so it's not yet necessary to support max_idx masking (which will get a bit
  // more complicated).
  DRAGNN_AVXVA_INLINED_UNROLLED void Load(const TruncatedFloat16 *source);

#if defined(__F16C__)

  // Reads and unpacks IEEE-754 half-precision values.
  //
  // Currently, only matrix coefficients use compressed/half-precision values,
  // so it's not yet necessary to support max_idx masking (which will get a bit
  // more complicated).
  //
  // TODO(googleuser): Either add non-F16C compatibility support from Eigen,
  // or delete this code if it turns out not to be helpful.
  DRAGNN_AVXVA_INLINED_UNROLLED void Load(const IeeeFloat16 *source);
#endif

  DRAGNN_AVXVA_INLINED_UNROLLED void LoadConstVector(const float val) {
    for (int i = 0; i < N; i++) {
      vectors[i].LoadConstVector(val);
    }
  }

  DRAGNN_AVXVA_INLINED_UNROLLED void Store(float *dst) {
    for (int i = 0; i < N; i++) {
      vectors[i].Store(dst + 8 * i);
    }
  }

  DRAGNN_AVXVA_INLINED_UNROLLED void Store(float *dst, int max_idx) {
    for (int i = 0; i < N; i++) {
      // This is equivalent to writing `i < N && i < max_idx` above, but forces
      // the compiler to produce more efficient code (it's still creating jump
      // instructions, but the branching is probably more predictable, and the
      // loops are unrolled). In the future we could switch to VMASKMOV if
      // necessary.
      if (i < max_idx) {
        vectors[i].Store(dst + 8 * i);
      }
    }
  }

  template <class Function>
  DRAGNN_AVXVA_INLINED_UNROLLED void Apply(const Function &fcn) {
    for (int i = 0; i < N; i++) {
      vectors[i] = fcn(vectors[i]);
    }
  }

  AvxFloatVec vectors[N];
};

// Implementation details.
#if defined(__AVX__)
DRAGNN_AVXVA_ALWAYS_INLINE
AvxFloatVec::AvxFloatVec(const internal::AvxMultiplyExpr &expr) {
  ymm = _mm256_mul_ps(expr.a.ymm, expr.b.ymm);
}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxFloatVec::Load(const float *source) {
  ymm = _mm256_load_ps(source);
}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxFloatVec::LoadConstVector(const float val) {
  ymm = _mm256_set1_ps(val);
}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxFloatVec::Store(float *dst) const {
  _mm256_store_ps(dst, ymm);
}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxFloatVec::AddProductOf(
    const AvxFloatVec &a, const AvxFloatVec &b) {
#if defined(__AVX2__) && defined(__FMA__)
  ymm = _mm256_fmadd_ps(a.ymm, b.ymm, ymm);
#else
  *this += AvxFloatVec(a * b);
#endif
}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxFloatVec::Floor() {
  ymm = _mm256_floor_ps(ymm);
}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxFloatVec::Clamp(const float min_value,
                                                   const float max_value) {
  ymm = _mm256_min_ps(ymm, Const(max_value).ymm);
  ymm = _mm256_max_ps(ymm, Const(min_value).ymm);
}

DRAGNN_AVXVA_ALWAYS_INLINE AvxFloatVec &AvxFloatVec::operator+=(
    const AvxFloatVec &vec) {
  ymm = _mm256_add_ps(vec.ymm, ymm);
  return *this;
}

DRAGNN_AVXVA_ALWAYS_INLINE AvxFloatVec &AvxFloatVec::operator-=(
    const AvxFloatVec &vec) {
  ymm = _mm256_sub_ps(ymm, vec.ymm);
  return *this;
}

DRAGNN_AVXVA_ALWAYS_INLINE AvxFloatVec &AvxFloatVec::operator/=(
    const AvxFloatVec &vec) {
  ymm = _mm256_div_ps(ymm, vec.ymm);
  return *this;
}

DRAGNN_AVXVA_ALWAYS_INLINE AvxIntVec::AvxIntVec(const AvxFloatVec &v)
    : ymm_(_mm256_cvttps_epi32(v.ymm)) {}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxIntVec::LeftShift(int bits) {
#if defined(__AVX2__)
  ymm_ = _mm256_slli_epi32(ymm_, bits);
#else

  // Convert to SSE and back again. This is pretty slow, so don't use this code
  // except for compatibility purposes.
  __m256i upper_bits = _mm256_permute2f128_si256(ymm_, ymm_, 1);
  __m128i first = _mm256_castsi256_si128(ymm_);         // Lower bits as SSE
  __m128i second = _mm256_castsi256_si128(upper_bits);  // Upper bits as SSE
  first = _mm_slli_epi32(first, bits);
  second = _mm_slli_epi32(second, bits);
  ymm_ = _mm256_permute2f128_si256(_mm256_castsi128_si256(first),
                                   _mm256_castsi128_si256(second), (2 << 4));
#endif
}

AvxFloatVec DRAGNN_AVXVA_ALWAYS_INLINE AvxIntVec::ReinterpretCastFloat() {
  AvxFloatVec result;
  result.ymm = _mm256_castsi256_ps(ymm_);
  return result;
}

template <int N>
DRAGNN_AVXVA_INLINED_UNROLLED void AvxFloatVecArray<N>::Load(
    const TruncatedFloat16 *source) {
  static_assert(N % 2 == 0,
                "Load() from half floats requires even-sized vector arrays.");

  for (int i = 0; i < N / 2; i++) {
#if defined(__AVX2__)
    const __m256i input = _mm256_load_si256(
        reinterpret_cast<__m256i const *>(source + kAvxWidthHalfPrecision * i));
    vectors[2 * i].ymm = _mm256_castsi256_ps(
        _mm256_unpacklo_epi16(_mm256_setzero_si256(), input));
    vectors[2 * i + 1].ymm = _mm256_castsi256_ps(
        _mm256_unpackhi_epi16(_mm256_setzero_si256(), input));
#else

    // Compatibility AVX (not AVX2) implementation.
    __m128i input[2];
    input[0] = _mm_load_si128(
        reinterpret_cast<__m128i const *>(source + kAvxWidthHalfPrecision * i));
    input[1] = _mm_load_si128(reinterpret_cast<__m128i const *>(
        source + kAvxWidthHalfPrecision * i + kSseWidthHalfPrecision));

    // Unpack. This permutation is kinda cryptic and, to be honest, derived by
    // simply trying many combinations.
    vectors[2 * i].ymm = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_castsi128_ps(
            _mm_unpacklo_epi16(_mm_setzero_si128(), input[0]))),
        _mm_castsi128_ps(_mm_unpacklo_epi16(_mm_setzero_si128(), input[1])), 1);
    vectors[2 * i + 1].ymm = _mm256_insertf128_ps(
        _mm256_castps128_ps256(_mm_castsi128_ps(
            _mm_unpackhi_epi16(_mm_setzero_si128(), input[0]))),
        _mm_castsi128_ps(_mm_unpackhi_epi16(_mm_setzero_si128(), input[1])), 1);
#endif
  }
}

#if defined(__F16C__)
template <int N>
DRAGNN_AVXVA_INLINED_UNROLLED void AvxFloatVecArray<N>::Load(
    const IeeeFloat16 *source) {
  static_assert(N % 2 == 0,
                "Load() from half floats requires even-sized vector arrays.");

  for (int i = 0; i < N / 2; i++) {
    // TODO(googleuser): Experiment with doing a single AVX2 load and
    // dividing the result.
    __m128i first_half = _mm_load_si128(
        reinterpret_cast<__m128i const *>(source + kAvxWidthHalfPrecision * i));
    __m128i second_half = _mm_load_si128(reinterpret_cast<__m128i const *>(
        source + kAvxWidthHalfPrecision * i + kAvxWidth));
    vectors[2 * i].ymm = _mm256_cvtph_ps(first_half);
    vectors[2 * i + 1].ymm = _mm256_cvtph_ps(second_half);
  }
}
#endif

#elif defined(__SSE4_2__)
DRAGNN_AVXVA_ALWAYS_INLINE
AvxFloatVec::AvxFloatVec(const internal::AvxMultiplyExpr &expr) {
  for (int i = 0; i < 2; ++i) {
    xmm[i] = _mm_mul_ps(expr.a.xmm[i], expr.b.xmm[i]);
  }
}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxFloatVec::Load(const float *source) {
  for (int i = 0; i < 2; ++i) {
    xmm[i] = _mm_load_ps(&source[i * kSseWidth]);
  }
}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxFloatVec::LoadConstVector(const float val) {
  xmm[1] = xmm[0] = _mm_set1_ps(val);
}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxFloatVec::Store(float *dst) const {
  for (int i = 0; i < 2; ++i) {
    _mm_store_ps(&dst[i * kSseWidth], xmm[i]);
  }
}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxFloatVec::AddProductOf(
    const AvxFloatVec &a, const AvxFloatVec &b) {
  *this += AvxFloatVec(a * b);
}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxFloatVec::Floor() {
  for (int i = 0; i < 2; ++i) {
    xmm[i] = _mm_floor_ps(xmm[i]);
  }
}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxFloatVec::Clamp(const float min_value,
                                                   const float max_value) {
  for (int i = 0; i < 2; ++i) {
    xmm[i] = _mm_min_ps(xmm[i], Const(max_value).xmm[i]);
    xmm[i] = _mm_max_ps(xmm[i], Const(min_value).xmm[i]);
  }
}

DRAGNN_AVXVA_ALWAYS_INLINE AvxFloatVec &AvxFloatVec::operator+=(
    const AvxFloatVec &vec) {
  for (int i = 0; i < 2; ++i) {
    xmm[i] = _mm_add_ps(vec.xmm[i], xmm[i]);
  }
  return *this;
}

DRAGNN_AVXVA_ALWAYS_INLINE AvxFloatVec &AvxFloatVec::operator-=(
    const AvxFloatVec &vec) {
  for (int i = 0; i < 2; ++i) {
    xmm[i] = _mm_sub_ps(xmm[i], vec.xmm[i]);
  }
  return *this;
}

DRAGNN_AVXVA_ALWAYS_INLINE AvxFloatVec &AvxFloatVec::operator/=(
    const AvxFloatVec &vec) {
  for (int i = 0; i < 2; ++i) {
    xmm[i] = _mm_div_ps(xmm[i], vec.xmm[i]);
  }
  return *this;
}

DRAGNN_AVXVA_ALWAYS_INLINE AvxIntVec::AvxIntVec(const AvxFloatVec &v) {
  xmm_[0] = _mm_cvttps_epi32(v.xmm[0]);
  xmm_[1] = _mm_cvttps_epi32(v.xmm[1]);
}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxIntVec::LeftShift(int bits) {
  for (int i = 0; i < 2; ++i) {
    xmm_[i] = _mm_slli_epi32(xmm_[i], bits);
  }
}

AvxFloatVec DRAGNN_AVXVA_ALWAYS_INLINE AvxIntVec::ReinterpretCastFloat() {
  AvxFloatVec result;
  for (int i = 0; i < 2; ++i) {
    result.xmm[i] = _mm_castsi128_ps(xmm_[i]);
  }
  return result;
}

template <int N>
DRAGNN_AVXVA_INLINED_UNROLLED void AvxFloatVecArray<N>::Load(
    const TruncatedFloat16 *source) {
  static_assert(N % 2 == 0,
                "Load() from half floats requires even-sized vector arrays.");

  for (int i = 0; i < N / 2; i++) {
    __m128i input[2];
    input[0] = _mm_load_si128(
        reinterpret_cast<__m128i const *>(source + kAvxWidthHalfPrecision * i));
    input[1] = _mm_load_si128(reinterpret_cast<__m128i const *>(
        source + kAvxWidthHalfPrecision * i + kSseWidthHalfPrecision));

    vectors[2 * i].xmm[0] =
        _mm_castsi128_ps(_mm_unpacklo_epi16(_mm_setzero_si128(), input[0]));
    vectors[2 * i + 1].xmm[0] =
        _mm_castsi128_ps(_mm_unpackhi_epi16(_mm_setzero_si128(), input[0]));
    vectors[2 * i].xmm[1] =
        _mm_castsi128_ps(_mm_unpacklo_epi16(_mm_setzero_si128(), input[1]));
    vectors[2 * i + 1].xmm[1] =
        _mm_castsi128_ps(_mm_unpackhi_epi16(_mm_setzero_si128(), input[1]));
  }
}

#if defined(__F16C__)
template <int N>
DRAGNN_AVXVA_INLINED_UNROLLED void AvxFloatVecArray<N>::Load(
    const IeeeFloat16 *source) {
  static_assert(N % 2 == 0,
                "Load() from half floats requires even-sized vector arrays.");

  for (int i = 0; i < N / 2; i++) {
    __m128i first_half = _mm_load_si128(
        reinterpret_cast<__m128i const *>(source + kAvxWidthHalfPrecision * i));
    __m128i second_half = _mm_load_si128(reinterpret_cast<__m128i const *>(
        source + kAvxWidthHalfPrecision * i + kAvxWidth));
    vectors[2 * i].xmm[0] = _mm_cvtph_ps(first_half);
    vectors[2 * i + 1].xmm[0] = _mm_cvtph_ps(second_half);

    first_half = _mm_shuffle_epi32(first_half, _MM_SHUFFLE(0, 1, 3, 2));
    second_half = _mm_shuffle_epi32(second_half, _MM_SHUFFLE(0, 1, 3, 2));
    vectors[2 * i].xmm[1] = _mm_cvtph_ps(first_half);
    vectors[2 * i + 1].xmm[1] = _mm_cvtph_ps(second_half);
  }
}
#endif

#else

// Compatibility implementations. If you compile with -ftree-vectorize and
// -msse2 flags, you should still get decent performance (maybe 1/4 of the
// AVX/FMA version).
//
// See the class above for method documentation.
DRAGNN_AVXVA_ALWAYS_INLINE
AvxFloatVec::AvxFloatVec(const internal::AvxMultiplyExpr &expr) {
  for (int i = 0; i < 8; i++) {
    ymm[i] = expr.a.ymm[i] * expr.b.ymm[i];
  }
}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxFloatVec::Load(const float *source) {
  for (int i = 0; i < 8; i++) {
    ymm[i] = source[i];
  }
}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxFloatVec::LoadConstVector(const float val) {
  for (int i = 0; i < 8; i++) {
    ymm[i] = val;
  }
}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxFloatVec::Store(float *dst) const {
  for (int i = 0; i < 8; i++) {
    dst[i] = ymm[i];
  }
}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxFloatVec::AddProductOf(
    const AvxFloatVec &a, const AvxFloatVec &b) {
  for (int i = 0; i < 8; i++) {
    ymm[i] += a.ymm[i] * b.ymm[i];
  }
}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxFloatVec::Floor() {
  for (int i = 0; i < 8; i++) {
    ymm[i] = floor(ymm[i]);
  }
}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxFloatVec::Clamp(const float min_value,
                                                   const float max_value) {
  for (int i = 0; i < 8; i++) {
    ymm[i] = fmin(fmax(ymm[i], min_value), max_value);
  }
}

DRAGNN_AVXVA_ALWAYS_INLINE AvxFloatVec &AvxFloatVec::operator+=(
    const AvxFloatVec &vec) {
  for (int i = 0; i < 8; i++) {
    ymm[i] += vec.ymm[i];
  }
  return *this;
}

DRAGNN_AVXVA_ALWAYS_INLINE AvxFloatVec &AvxFloatVec::operator-=(
    const AvxFloatVec &vec) {
  for (int i = 0; i < 8; i++) {
    ymm[i] -= vec.ymm[i];
  }
  return *this;
}

DRAGNN_AVXVA_ALWAYS_INLINE AvxFloatVec &AvxFloatVec::operator/=(
    const AvxFloatVec &vec) {
  for (int i = 0; i < 8; i++) {
    ymm[i] /= vec.ymm[i];
  }
  return *this;
}

DRAGNN_AVXVA_ALWAYS_INLINE AvxIntVec::AvxIntVec(const AvxFloatVec &v) {
  for (int i = 0; i < 8; i++) {
    ymm_[i] = static_cast<int>(v.ymm[i]);
  }
}

DRAGNN_AVXVA_ALWAYS_INLINE void AvxIntVec::LeftShift(int bits) {
  for (int i = 0; i < 8; i++) {
    ymm_[i] = ymm_[i] << bits;
  }
}

DRAGNN_AVXVA_ALWAYS_INLINE AvxFloatVec AvxIntVec::ReinterpretCastFloat() {
  AvxFloatVec result;
  for (int i = 0; i < 8; i++) {
    result.ymm[i] = reinterpret_cast<float &>(ymm_[i]);
  }
  return result;
}

template <int N>
DRAGNN_AVXVA_INLINED_UNROLLED void AvxFloatVecArray<N>::Load(
    const TruncatedFloat16 *source) {
  static_assert(N % 2 == 0,
                "Load() from half floats requires even-sized vector arrays.");

  // Iterate through mock AVX vectors, each composed of 16 half-floats.
  for (int vec_idx = 0; vec_idx < N / 2; vec_idx++) {
    // Making this code a bit more verbose, by reading in-order to a temporary
    // array, results in faster performance. The compatibility version is still
    // pretty slow though.
    TruncatedFloat16 tmp[16];
    for (int i = 0; i < kAvxWidthHalfPrecision; ++i) {
      tmp[i] = source[i + kAvxWidthHalfPrecision * vec_idx];
    }
    float unpacked[16];
    for (int i = 0; i < kAvxWidthHalfPrecision; ++i) {
      unpacked[i] = tmp[i].DebugToFloat();
    }
    for (int i = 0; i < kAvxWidthHalfPrecision; ++i) {
      int permuted = FastUnpackPermutation(i);
      vectors[2 * vec_idx + (i / 8)].ymm[i % 8] = unpacked[permuted];
    }
  }
}

#if defined(__F16C__)
template <int N>
DRAGNN_AVXVA_INLINED_UNROLLED void AvxFloatVecArray<N>::Load(
    const IeeeFloat16 *source) {
  // Not actually required for the compatibility implementation, but it'd be
  // rather non-uniform if this API succeeded, and then compilation failed when
  // AVX2 was turned on.
  static_assert(N % 2 == 0,
                "Load() from half floats requires even-sized vector arrays.");

  // Iterate through mock AVX vectors, each composed of 16 half-floats.
  for (int i = 0; i < N * kAvxWidth; ++i) {
    vectors[i / 8].ymm[i % 8] = source[i].DebugToFloat();
  }
}
#endif
#endif

// The following operations are mostly syntax sugar, so they do not need
// architecture-specific implementations.

DRAGNN_AVXVA_ALWAYS_INLINE AvxFloatVec &AvxFloatVec::operator+=(
    const internal::AvxMultiplyExpr &to_add) {
  AddProductOf(to_add.a, to_add.b);
  return *this;
}

DRAGNN_AVXVA_ALWAYS_INLINE internal::AvxMultiplyExpr operator*(
    const AvxFloatVec &a, const AvxFloatVec &b) {
  return internal::AvxMultiplyExpr{a, b};
}

DRAGNN_AVXVA_ALWAYS_INLINE AvxFloatVec
operator+(const internal::AvxMultiplyExpr &expr, const AvxFloatVec &v) {
  AvxFloatVec result = v;
  result += expr;
  return result;
}

DRAGNN_AVXVA_ALWAYS_INLINE AvxFloatVec operator+(const AvxFloatVec &a,
                                                 const AvxFloatVec &b) {
  AvxFloatVec result = a;
  result += b;
  return result;
}

DRAGNN_AVXVA_ALWAYS_INLINE AvxFloatVec operator/(const AvxFloatVec &a,
                                                 const AvxFloatVec &b) {
  AvxFloatVec result = a;
  result /= b;
  return result;
}

DRAGNN_AVXVA_ALWAYS_INLINE AvxFloatVec operator-(const AvxFloatVec &a,
                                                 const AvxFloatVec &b) {
  AvxFloatVec result = a;
  result -= b;
  return result;
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#undef DRAGNN_AVXVA_ALWAYS_INLINE
#undef DRAGNN_AVXVA_INLINED_UNROLLED

#endif  // DRAGNN_RUNTIME_MATH_AVX_VECTOR_ARRAY_H_
