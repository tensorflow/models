// Copyright 2018 Google Inc. All Rights Reserved.
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

// Compatibility support for Eigen.

#ifndef DRAGNN_RUNTIME_MATH_EIGEN_H_
#define DRAGNN_RUNTIME_MATH_EIGEN_H_

#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/math/types.h"
#include "third_party/eigen3/Eigen/Core"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace internal {

// Returns a combination of bit-options for Eigen matrices.
constexpr int GetEigenMatrixOptions() {
  return Eigen::AutoAlign | Eigen::RowMajor;
}

// Returns a combination of bit-options for Eigen maps of runtime types.
constexpr int GetEigenMapOptions() {
  static_assert(kAlignmentBytes >= EIGEN_MAX_ALIGN_BYTES,
                "Runtime alignment is not compatible with Eigen alignment.");
  return Eigen::Aligned;
}

// Eigen matrix and (row) vector types.  Don't use these directly; instead use
// the public Map types and functions below to wrap runtime types.
template <class T>
using EigenVector =
    Eigen::Matrix<T, 1, Eigen::Dynamic, GetEigenMatrixOptions()>;
template <class T>
using EigenMatrix =
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, GetEigenMatrixOptions()>;

// Eigen stride for matrix types.
using EigenMatrixStride = Eigen::Stride<Eigen::Dynamic, 1>;

// Returns the Eigen stride associated with the |matrix|.
template <class T>
EigenMatrixStride GetEigenMatrixStride(MatrixImpl<T> matrix) {
  return EigenMatrixStride(matrix.row_stride(), 1);
}

}  // namespace internal

// Eigen wrappers around a runtime-allocated matrix or (row) vector.
template <class T>
using EigenVectorMap =
    Eigen::Map<const internal::EigenVector<T>, internal::GetEigenMapOptions()>;
template <class T>
using MutableEigenVectorMap =
    Eigen::Map<internal::EigenVector<T>, internal::GetEigenMapOptions()>;
template <class T>
using EigenMatrixMap =
    Eigen::Map<const internal::EigenMatrix<T>, internal::GetEigenMapOptions(),
               internal::EigenMatrixStride>;
template <class T>
using MutableEigenMatrixMap =
    Eigen::Map<internal::EigenMatrix<T>, internal::GetEigenMapOptions(),
               internal::EigenMatrixStride>;

// Returns an Eigen wrapper around the |vector| or |matrix|.
template <class T>
EigenVectorMap<T> AsEigenMap(Vector<T> vector) {
  return EigenVectorMap<T>(vector.data(), vector.size());
}
template <class T>
MutableEigenVectorMap<T> AsEigenMap(MutableVector<T> vector) {
  return MutableEigenVectorMap<T>(vector.data(), vector.size());
}
template <class T>
EigenMatrixMap<T> AsEigenMap(Matrix<T> matrix) {
  return EigenMatrixMap<T>(matrix.data(), matrix.num_rows(),
                           matrix.num_columns(),
                           internal::GetEigenMatrixStride(matrix));
}
template <class T>
MutableEigenMatrixMap<T> AsEigenMap(MutableMatrix<T> matrix) {
  return MutableEigenMatrixMap<T>(matrix.data(), matrix.num_rows(),
                                  matrix.num_columns(),
                                  internal::GetEigenMatrixStride(matrix));
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_MATH_EIGEN_H_
