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

// Utility functions that can transform different matrix types. This includes
// non-trivial transposes, and converting vectors/etc. to the matrix types. This
// library should NOT be used for any performance-critical work, and should NOT
// be included at all in the mobile runtime.

#ifndef DRAGNN_RUNTIME_MATH_TRANSFORMATIONS_H_
#define DRAGNN_RUNTIME_MATH_TRANSFORMATIONS_H_

#include "dragnn/runtime/math/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

namespace internal {

// Puts a format-agnostic API on matrix-like data types. This is convenient, but
// has the downside of potential confusing compiler errors (when a
// specialization does not exist), and isn't suitable for optimizations like
// blocked transformations.
template <class T>
T *GetMatrixElement(int row, int col, MatrixImpl<T> *matrix) {
  return &matrix->row(row)[col];
}

template <class T>
const T &GetMatrixElement(int row, int col, const MatrixImpl<T> &matrix) {
  return matrix.row(row)[col];
}

template <class T>
T *GetMatrixElement(
    int row, int col,
    BlockedMatrixImpl<T, BlockedMatrixFormat::kRowBlockedColumnMajor> *matrix) {
  int sub_matrix_idx = row / matrix->block_size();
  int vector_idx = sub_matrix_idx * matrix->num_columns() + col;
  int element_idx = row % matrix->block_size();
  return &matrix->vector(vector_idx)[element_idx];
}

template <class T>
const T &GetMatrixElement(
    int row, int col,
    const BlockedMatrixImpl<T, BlockedMatrixFormat::kRowBlockedColumnMajor>
        &matrix) {
  int sub_matrix_idx = row / matrix.block_size();
  int vector_idx = sub_matrix_idx * matrix.num_columns() + col;
  int element_idx = row % matrix.block_size();
  return matrix.vector(vector_idx)[element_idx];
}

template <class T>
T *GetMatrixElement(
    int row, int col,
    BlockedMatrixImpl<T, BlockedMatrixFormat::kColumnBlockedRowMajor> *matrix) {
  int sub_matrix_idx = col / matrix->block_size();
  int vector_idx = sub_matrix_idx * matrix->num_rows() + row;
  int element_idx = col % matrix->block_size();
  return &matrix->vector(vector_idx)[element_idx];
}

template <class T>
const T &GetMatrixElement(
    int row, int col,
    const BlockedMatrixImpl<T, BlockedMatrixFormat::kColumnBlockedRowMajor>
        &matrix) {
  int sub_matrix_idx = col / matrix.block_size();
  int vector_idx = sub_matrix_idx * matrix.num_rows() + row;
  int element_idx = col % matrix.block_size();
  return matrix.vector(vector_idx)[element_idx];
}

}  // namespace internal

// Generates values for a matrix, by calling a provided function on each
// row/column index. Thanks to the magic of templating, the function call should
// be inlined and not cause too much overhead being "called" on each index.
template <class Function, class OutputMatrix>
void GenerateMatrix(int num_rows, int num_columns, const Function &get_value,
                    OutputMatrix *output_matrix) {
  for (size_t row = 0; row < num_rows; ++row) {
    for (size_t column = 0; column < num_columns; ++column) {
      *(GetMatrixElement(row, column, output_matrix)) = get_value(row, column);
    }
  }
}

// Copies the first |num_rows| rows and |num_columns| columns of input_matrix to
// output_matrix.
template <class InputMatrix, class OutputMatrix>
void CopyMatrixPrefix(const InputMatrix &input_matrix, int num_rows,
                      int num_columns, OutputMatrix *output_matrix) {
  const auto &get_value = [input_matrix](int row, int column) {
    return GetMatrixElement(row, column, input_matrix);
  };
  GenerateMatrix(num_rows, num_columns, get_value, output_matrix);
}

// Copies matrices. The matrices can be of different types, but must have the
// same dimensions.
template <class InputMatrix, class OutputMatrix>
tensorflow::Status CopyMatrix(const InputMatrix &input_matrix,
                              OutputMatrix *output_matrix) {
  if (input_matrix.num_rows() != output_matrix->num_rows()) {
    return tensorflow::errors::InvalidArgument(
        "Input matrix num_rows (", input_matrix.num_rows(),
        ") != output matrix num_rows (", output_matrix->num_rows(), ")");
  }
  if (input_matrix.num_columns() != output_matrix->num_columns()) {
    return tensorflow::errors::InvalidArgument(
        "Input matrix num_columns (", input_matrix.num_columns(),
        ") != output matrix num_columns (", output_matrix->num_columns(), ")");
  }
  CopyMatrixPrefix(input_matrix, input_matrix.num_rows(),
                   input_matrix.num_columns(), output_matrix);
  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_MATH_TRANSFORMATIONS_H_
