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

#include "dragnn/runtime/trained_model_variable_store.h"

#include "dragnn/runtime/math/types.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

tensorflow::Status TrainedModelVariableStore::Reset(
    const string &saved_model_dir) {
  TF_RETURN_IF_ERROR(trained_model_.Reset(saved_model_dir));

  // Success; make modifications.
  variables_.clear();
  return tensorflow::Status::OK();
}

namespace {

// Copies flat data from the |tensor|, casted to T, into the |array| and points
// the |area| at it.  On error, returns non-OK.
template <class T>
tensorflow::Status ExtractFlat(const tensorflow::Tensor &tensor,
                               std::vector<size_t> *dimensions,
                               UniqueAlignedArray *array,
                               MutableAlignedArea *area) {
  const auto flat = tensor.flat<T>();
  const size_t bytes = flat.size() * sizeof(T);
  array->Reset(ComputeAlignedAreaSize(1, bytes));
  TF_RETURN_IF_ERROR(area->Reset(array->view(), 1, bytes));
  const MutableVector<T> row(area->view(0));
  for (size_t i = 0; i < flat.size(); ++i) row[i] = flat(i);
  dimensions->clear();
  dimensions->push_back(flat.size());
  return tensorflow::Status::OK();
}

// Copies the |tensor|, casted to T and reshaped as a matrix, into the |array|
// and points the |area| at it.  Requires that the |tensor| is rank 2 or more.
// On error, returns non-OK.
template <class T>
tensorflow::Status ExtractMatrix(const tensorflow::Tensor &tensor,
                                 std::vector<size_t> *dimensions,
                                 UniqueAlignedArray *array,
                                 MutableAlignedArea *area) {
  if (tensor.dims() < 2) {
    return tensorflow::errors::InvalidArgument(
        "Tensor must be rank >= 2 but is rank ", tensor.dims());
  }

  // Flatten all dims except the inner-most, creating a matrix.
  const auto reshaped = tensor.flat_inner_dims<T>();

  const size_t num_rows = reshaped.dimension(0);
  const size_t num_columns = reshaped.dimension(1);
  *dimensions = {num_rows, num_columns};

  const size_t view_size_bytes = num_columns * sizeof(T);
  array->Reset(ComputeAlignedAreaSize(num_rows, view_size_bytes));
  TF_RETURN_IF_ERROR(area->Reset(array->view(), num_rows, view_size_bytes));

  MutableMatrix<T> matrix(*area);
  for (size_t row = 0; row < num_rows; ++row) {
    for (size_t column = 0; column < num_columns; ++column) {
      matrix.row(row)[column] = reshaped(row, column);
    }
  }

  return tensorflow::Status::OK();
}

// Copies a blocked matrix from the |tensor|, casted to T, into the |array| and
// points the |area| at it.  Requires that the |tensor| is rank 3.  On error,
// returns non-OK.
template <class T>
tensorflow::Status ExtractBlockedMatrix(const tensorflow::Tensor &tensor,
                                        std::vector<size_t> *dimensions,
                                        UniqueAlignedArray *array,
                                        MutableAlignedArea *area) {
  if (tensor.dims() != 3) {
    return tensorflow::errors::InvalidArgument(
        "Tensor must be rank 3 but is rank ", tensor.dims());
  }

  const size_t num_sub_matrices = tensor.dim_size(0);
  const size_t num_rows = tensor.dim_size(1);
  const size_t block_size = tensor.dim_size(2);
  const size_t num_columns = num_sub_matrices * block_size;
  *dimensions = {num_rows, num_columns, block_size};

  // Given the order of dimensions in the |tensor|, flattening it into a matrix
  // via flat_inner_dims() and copying it to the |area| is equivalent to copying
  // the blocked matrix.
  std::vector<size_t> unused_dimensions;  // ignore non-blocked dimensions
  return ExtractMatrix<T>(tensor, &unused_dimensions, array, area);
}

}  // namespace

tensorflow::Status TrainedModelVariableStore::Lookup(
    const string &name, VariableSpec::Format format,
    std::vector<size_t> *dimensions, AlignedArea *area) {
  const Key key(name, format);
  const auto it = variables_.find(key);
  if (it != variables_.end()) {
    std::tie(std::ignore, *dimensions, *area) = it->second;
    return tensorflow::Status::OK();
  }

  Variable variable;
  TF_RETURN_IF_ERROR(GetVariableContents(name, format, &variable));

  // Success; make modifications.
  std::tie(std::ignore, *dimensions, *area) = variable;
  variables_[key] = std::move(variable);
  return tensorflow::Status::OK();
}

tensorflow::Status TrainedModelVariableStore::GetVariableContents(
    const string &name, VariableSpec::Format format, Variable *variable) {
  tensorflow::Tensor tensor;
  TF_RETURN_IF_ERROR(trained_model_.EvaluateTensor(name, &tensor));

  // Extract typed tensor data.
  UniqueAlignedArray *array = &std::get<0>(*variable);
  std::vector<size_t> *dimensions = &std::get<1>(*variable);
  MutableAlignedArea *area = &std::get<2>(*variable);

  if (tensor.dtype() == tensorflow::DT_FLOAT) {
    switch (format) {
      case VariableSpec::FORMAT_UNKNOWN:
        return tensorflow::errors::InvalidArgument("Unknown variable format");

      case VariableSpec::FORMAT_FLAT:
        return ExtractFlat<float>(tensor, dimensions, array, area);

      case VariableSpec::FORMAT_ROW_MAJOR_MATRIX:
        return ExtractMatrix<float>(tensor, dimensions, array, area);

      case VariableSpec::FORMAT_COLUMN_BLOCKED_ROW_MAJOR_MATRIX:
        return ExtractBlockedMatrix<float>(tensor, dimensions, array, area);
    }
  } else if (tensor.dtype() == tensorflow::DT_BFLOAT16) {
    switch (format) {
      case VariableSpec::FORMAT_UNKNOWN:
        return tensorflow::errors::InvalidArgument("Unknown variable format");

      case VariableSpec::FORMAT_FLAT:
        return ExtractFlat<tensorflow::bfloat16>(tensor, dimensions, array,
                                                 area);

      case VariableSpec::FORMAT_ROW_MAJOR_MATRIX:
        return ExtractMatrix<tensorflow::bfloat16>(tensor, dimensions, array,
                                                   area);

      case VariableSpec::FORMAT_COLUMN_BLOCKED_ROW_MAJOR_MATRIX:
        return ExtractBlockedMatrix<tensorflow::bfloat16>(tensor, dimensions,
                                                          array, area);
    }
  } else {
    // TODO(googleuser): Add clauses for additional types as needed.
    return tensorflow::errors::Unimplemented(
        "Data type not supported: ", tensorflow::DataType_Name(tensor.dtype()));
  }
}

tensorflow::Status TrainedModelVariableStore::Close() {
  return trained_model_.Close();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
