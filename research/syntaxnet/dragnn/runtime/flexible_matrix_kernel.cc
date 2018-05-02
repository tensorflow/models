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

#include "dragnn/runtime/flexible_matrix_kernel.h"

#include "dragnn/runtime/math/avx_vector_array.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Rounds a number, |rows|, up to a multiple of |multiple|. For example,
// PadRows(6, 4) will return 8, because 8 is the nearest number after 6 that is
// divisible by 4. This method requires that |multiple| be positive. It is used
// for pre-calculating the dimension of a blocked matrix, instead of having to
// read the entire matrix.
inline int PadRows(int rows, int multiple) {
  return multiple * ((rows + multiple - 1) / multiple);
}

}  // namespace

constexpr char FlexibleMatrixKernel::kSuffix[];

tensorflow::Status FlexibleMatrixKernel::Initialize(
    const string &debug_name, const string &weights_name, int output_dimension,
    VariableStore *variable_store) {
  padded_output_dimension_ = PadRows(output_dimension, kAvxWidth);

  // Try retrieving the flexible matrix variable using all matrix formats.  Only
  // one format will work (see FlexibleMatrixVariableStoreWrapper).
  const string variable_name =
      tensorflow::strings::StrCat(weights_name, kSuffix);

  // Handle the simpler non-blocked case first.
  tensorflow::Status status = variable_store->Lookup(variable_name, &weights_);
  if (status.ok()) {
    LOG(INFO) << "Matrix of size " << weights_.num_rows() << " x "
              << weights_.num_columns() << " for layer " << debug_name
              << " will be computed with non-blocked arithmetic";
    weights_type_ = WeightsType::kNormal;
    return status;
  }

  // Otherwise, we must have a blocked format.
  BlockedMatrix<float> blocked_transpose;
  TF_RETURN_IF_ERROR(variable_store->Lookup(variable_name, &blocked_transpose));
  const auto blocked = blocked_transpose.Transpose();

  // Blocked matrices must use a supported block size.
  switch (blocked.block_size()) {
    case 32:
      weights_type_ = WeightsType::kBlocked32;
      status = fast_weights_32_.Initialize(blocked);
      break;

    case 48:
      weights_type_ = WeightsType::kBlocked48;
      status = fast_weights_48_.Initialize(blocked);
      break;

    default:
      return tensorflow::errors::FailedPrecondition(
          "Unsupported block size: ", blocked.block_size(), " for weights ",
          weights_name, " of layer ", debug_name);
  }

  if (status.ok()) {
    LOG(INFO) << "Matrix of size " << blocked.num_rows() << " x "
              << blocked.num_columns() << " for layer " << debug_name
              << " will be computed with SGEMV<block_size="
              << blocked.block_size() << ">";
  } else {
    // This should (almost?) never happen, because SgevmMatrix::Initialize()
    // only fails on bad block sizes, and the switch above ensures that the
    // SgemvMatrix and variable agree on block size.
    LOG(ERROR) << "Error formatting SGEMV matrix: " << status
               << " - matrix size " << blocked.num_rows() << " x "
               << blocked.num_columns() << " for layer " << debug_name;
  }

  return status;
}

int FlexibleMatrixKernel::NumPaddedRows() const {
  switch (weights_type_) {
    case WeightsType::kNormal:
      return weights_.num_rows();
    case WeightsType::kBlocked32:
      return fast_weights_32_.matrix().num_rows();
    case WeightsType::kBlocked48:
      return fast_weights_48_.matrix().num_rows();
  }
}

int FlexibleMatrixKernel::NumColumns() const {
  switch (weights_type_) {
    case WeightsType::kNormal:
      return weights_.num_columns();
    case WeightsType::kBlocked32:
      return fast_weights_32_.matrix().num_columns();
    case WeightsType::kBlocked48:
      return fast_weights_48_.matrix().num_columns();
  }
}

bool FlexibleMatrixKernel::MatchesOutputDimension(int output_dimension) const {
  int max_padding = 0;
  if (weights_type_ == WeightsType::kBlocked32) {
    max_padding = 32;
  } else if (weights_type_ == WeightsType::kBlocked48) {
    max_padding = 48;
  }
  return (NumPaddedRows() >= output_dimension &&
          NumPaddedRows() <= output_dimension + max_padding);
}

string FlexibleMatrixKernel::TypeName(WeightsType value) {
  switch (value) {
    case WeightsType::kNormal:
      return "normal (non-blocked)";
    case WeightsType::kBlocked32:
      return "32-row blocked";
    case WeightsType::kBlocked48:
      return "48-row blocked";
  }
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
