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

#ifndef DRAGNN_RUNTIME_VARIABLE_STORE_H_
#define DRAGNN_RUNTIME_VARIABLE_STORE_H_

#include <string>
#include <vector>

#include "dragnn/protos/runtime.pb.h"
#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/math/types.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Interface for a store holding named, precomputed variables.  Implementations
// must be thread-compatible.
class VariableStore {
 public:
  VariableStore(const VariableStore &that) = delete;
  VariableStore &operator=(const VariableStore &that) = delete;
  virtual ~VariableStore() = default;

  // Looks for the variable with the |name|, formats its content according to
  // the requested |format| (see details below), and points the |area| at the
  // result.  The content of the variable before formatting is its content in
  // the Python codebase.  The |area| is valid while this lives, even after
  // Close().  On error, returns non-OK and modifies nothing.
  //
  // Upon success the output |dimensions| will be cleared and assigned to
  // the set of dimensions (num_elements,) in case of vectors, (num_rows,
  // num_columns) in case of regular matrices, and (num_rows, num_columns,
  // block_size) in case of blocked matrices.
  //
  // FORMAT_FLAT:
  //   Flattens the variable as if by tf.reshape(var, [-1]), and sets the |area|
  //   to a single sub-view that points at the flat array.
  //
  // FORMAT_ROW_MAJOR_MATRIX:
  //   Reshapes the variable into a matrix as if by tf.reshape(var, [-1, D]),
  //   where D is the variable's innermost dimension.  Points each sub-view of
  //   the |area| at the corresponding row of the formatted matrix.  Requires
  //   that the variable has rank at least 2.
  //
  // FORMAT_COLUMN_BLOCKED_ROW_MAJOR_MATRIX:
  //   The variable must have shape [num_sub_matrices, num_rows, block_size],
  //   and is imported as a column-blocked row-major matrix, as documented in
  //   BlockedMatrixFormat (in math/types.h).  The matrix may also be padded.
  virtual tensorflow::Status Lookup(const string &name,
                                    VariableSpec::Format format,
                                    std::vector<size_t> *dimensions,
                                    AlignedArea *area) = 0;

  // Looks up a FORMAT_FLAT variable as a Vector<T>.
  template <class T>
  tensorflow::Status Lookup(const string &name, Vector<T> *vector);

  // Looks up a FORMAT_ROW_MAJOR_MATRIX as a Matrix<T>.
  template <class T>
  tensorflow::Status Lookup(const string &name, Matrix<T> *matrix);

  // Looks up a FORMAT_COLUMN_BLOCKED_ROW_MAJOR_MATRIX as a BlockedMatrix<T>.
  template <class T>
  tensorflow::Status Lookup(const string &name, BlockedMatrix<T> *matrix);

  // Releases intermediate resources, if any.  Does not invalidate the contents
  // of variables returned by previous calls to Lookup*(), but future calls to
  // Lookup*() are unsupported.  On error, returns non-OK.
  virtual tensorflow::Status Close() = 0;

 protected:
  VariableStore() = default;
};

// Implementation details below.

template <class T>
tensorflow::Status VariableStore::Lookup(const string &name,
                                         Vector<T> *vector) {
  AlignedArea area;
  std::vector<size_t> dimensions;
  TF_RETURN_IF_ERROR(
      Lookup(name, VariableSpec::FORMAT_FLAT, &dimensions, &area));

  if (area.num_views() != 1) {
    return tensorflow::errors::FailedPrecondition(
        "Vector variable '", name, "' should have 1 sub-view but has ",
        area.num_views());
  }

  if (area.view_size() % sizeof(T) != 0) {
    return tensorflow::errors::FailedPrecondition(
        "Vector variable '", name, "' does not divide into elements of size ",
        sizeof(T));
  }

  *vector = Vector<T>(area.view(0));
  if (dimensions.size() != 1) {
    return tensorflow::errors::FailedPrecondition("Expected 1 dimensions, got ",
                                                  dimensions.size());
  }
  if (dimensions[0] != vector->size()) {
    return tensorflow::errors::FailedPrecondition(
        "Vector size (", vector->size(), ") disagrees with dimensions[0] (",
        dimensions[0], ")");
  }
  return tensorflow::Status::OK();
}

template <class T>
tensorflow::Status VariableStore::Lookup(const string &name,
                                         Matrix<T> *matrix) {
  AlignedArea area;
  std::vector<size_t> dimensions;
  TF_RETURN_IF_ERROR(
      Lookup(name, VariableSpec::FORMAT_ROW_MAJOR_MATRIX, &dimensions, &area));
  if (dimensions.size() != 2) {
    return tensorflow::errors::FailedPrecondition("Expected 2 dimensions, got ",
                                                  dimensions.size());
  }

  if (area.view_size() % sizeof(T) != 0) {
    return tensorflow::errors::FailedPrecondition(
        "Matrix variable '", name, "' does not divide into elements of size ",
        sizeof(T));
  }

  *matrix = Matrix<T>(area);
  if (dimensions[0] != matrix->num_rows()) {
    return tensorflow::errors::FailedPrecondition(
        "Matrix rows (", matrix->num_rows(), ") disagrees with dimensions[0] (",
        dimensions[0], ")");
  }
  if (dimensions[1] != matrix->num_columns()) {
    return tensorflow::errors::FailedPrecondition(
        "Matrix columns (", matrix->num_columns(),
        ") disagrees with dimensions[1] (", dimensions[1], ")");
  }
  return tensorflow::Status::OK();
}

template <class T>
tensorflow::Status VariableStore::Lookup(const string &name,
                                         BlockedMatrix<T> *matrix) {
  AlignedArea area;
  std::vector<size_t> dimensions;
  TF_RETURN_IF_ERROR(
      Lookup(name, VariableSpec::FORMAT_COLUMN_BLOCKED_ROW_MAJOR_MATRIX,
             &dimensions, &area));
  if (dimensions.size() != 3) {
    return tensorflow::errors::FailedPrecondition("Expected 3 dimensions, got ",
                                                  dimensions.size());
  }
  const size_t num_rows = dimensions[0];
  const size_t num_columns = dimensions[1];
  const size_t block_size = dimensions[2];
  if (area.view_size() != block_size * sizeof(T)) {
    return tensorflow::errors::FailedPrecondition(
        "Area view size (", area.view_size(),
        ") doesn't correspond to block size (", block_size,
        ") times data type size (", sizeof(T), ")");
  }
  if (num_rows * num_columns != area.num_views() * block_size) {
    return tensorflow::errors::FailedPrecondition(
        "Rows * cols (", num_rows * num_columns, ") != area view size (",
        area.num_views() * block_size, ")");
  }

  // Avoid modification on error.
  BlockedMatrix<T> local_matrix;
  TF_RETURN_IF_ERROR(local_matrix.Reset(area, num_rows, num_columns));

  *matrix = local_matrix;
  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_VARIABLE_STORE_H_
