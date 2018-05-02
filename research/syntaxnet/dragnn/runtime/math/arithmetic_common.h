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

// Declarations of arithmetic operations and trivial generic implementations.
// Architecture-specific implementations should include this header and define
// template specializations that override the generic implementations.

#ifndef DRAGNN_RUNTIME_MATH_ARITHMETIC_COMMON_H_
#define DRAGNN_RUNTIME_MATH_ARITHMETIC_COMMON_H_

#include <stddef.h>
#include <algorithm>

#include "dragnn/runtime/math/types.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Performs output = scale * input.  Dimensions must match.
template <class T>
void ScaleElements(Vector<T> input, T scale, MutableVector<T> output);

// Performs output += scale * input.  Dimensions must match.
template <class T>
void AddScaledElements(Vector<T> input, T scale, MutableVector<T> output);

// Performs values = max(minimum, values) in place.
template <class T>
void MaxElements(T minimum, MutableVector<T> values);

// Performs output = matrix * input.  All vectors are interpreted as column
// vectors.  Dimensions must match.
template <class T>
void MultiplyMatrixAndVector(Matrix<T> matrix, Vector<T> input,
                             MutableVector<T> output);

// Performs output = bias + matrix * input.  All vectors are interpreted as
// column vectors.  Dimensions must match.
template <class T>
void MultiplyMatrixAndVectorWithBias(Matrix<T> matrix, Vector<T> bias,
                                     Vector<T> input, MutableVector<T> output);

// Implementation details below.

template <class T>
void ScaleElements(T scale, Vector<T> input, MutableVector<T> output) {
  DCHECK_EQ(input.size(), output.size());
  for (size_t i = 0; i < input.size(); ++i) output[i] = scale * input[i];
}

template <class T>
void AddScaledElements(T scale, Vector<T> input, MutableVector<T> output) {
  DCHECK_EQ(input.size(), output.size());
  for (size_t i = 0; i < input.size(); ++i) output[i] += scale * input[i];
}

template <class T>
void MaxElements(T minimum, MutableVector<T> values) {
  for (T &value : values) value = std::max(minimum, value);
}

namespace internal {

// Like MultiplyMatrixAndVectorWithBias(), but if |ignore_bias| is true, then
// the |bias| is treated as zero and its dimensions are not checked.
template <bool ignore_bias, class T>
void MultiplyMatrixAndVectorImpl(Matrix<T> matrix, Vector<T> bias,
                                 Vector<T> input, MutableVector<T> output) {
  DCHECK_EQ(matrix.num_columns(), input.size());
  if (!ignore_bias) DCHECK_EQ(matrix.num_rows(), bias.size());
  DCHECK_EQ(matrix.num_rows(), output.size());
  for (size_t i = 0; i < matrix.num_rows(); ++i) {
    const Vector<T> row = matrix.row(i);
    DCHECK_EQ(row.size(), input.size());
    T sum = ignore_bias ? T() : bias[i];
    for (size_t j = 0; j < row.size(); ++j) sum += row[j] * input[j];
    output[i] = sum;
  }
}

}  // namespace internal

template <class T>
void MultiplyMatrixAndVector(Matrix<T> matrix, Vector<T> input,
                             MutableVector<T> output) {
  internal::MultiplyMatrixAndVectorImpl<true>(matrix, {}, input, output);
}

template <class T>
void MultiplyMatrixAndVectorWithBias(Matrix<T> matrix, Vector<T> bias,
                                     Vector<T> input, MutableVector<T> output) {
  internal::MultiplyMatrixAndVectorImpl<false>(matrix, bias, input, output);
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_MATH_ARITHMETIC_COMMON_H_
