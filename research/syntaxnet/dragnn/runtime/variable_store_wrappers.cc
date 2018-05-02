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

#include "dragnn/runtime/variable_store_wrappers.h"

#include <algorithm>
#include <tuple>
#include <utility>
#include <vector>

#include "dragnn/runtime/flexible_matrix_kernel.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Returns the name of the averaged version of the variable named |name|.
string GetAveragedName(const string &name) {
  return tensorflow::strings::StrCat(name, "/ExponentialMovingAverage");
}

// Rounds a number, |rows|, up to a multiple of |multiple|. For example,
// PadRows(6, 4) will return 8, because 8 is the nearest number after 6 that is
// divisible by 4. This method requires that |multiple| be positive. It is used
// for pre-calculating the dimension of a blocked matrix, instead of having to
// read the entire matrix.
int PadRows(int rows, int multiple) {
  DCHECK_GT(multiple, 0);
  return multiple * ((rows + multiple - 1) / multiple);
}

// Calculates effective speed of a blocked matrix kernel. Blocked kernels may do
// a bit more calculation than necessary (since each AVX/SSE register contains
// multiple values), so their effective speed is less in those cases.
float EffectiveGflops(int rows, int block_dim, float base_gflops) {
  float padded_rows = PadRows(rows, block_dim);
  return (rows / padded_rows) * base_gflops;
}

}  // namespace

TryAveragedVariableStoreWrapper::TryAveragedVariableStoreWrapper(
    std::unique_ptr<VariableStore> variable_store, bool allow_fallback)
    : wrapped_variable_store_(std::move(variable_store)),
      allow_fallback_(allow_fallback) {}

tensorflow::Status TryAveragedVariableStoreWrapper::Lookup(
    const string &name, VariableSpec::Format format,
    std::vector<size_t> *dimensions, AlignedArea *area) {
  tensorflow::Status status = wrapped_variable_store_->Lookup(
      GetAveragedName(name), format, dimensions, area);
  if (status.ok()) {
    LOG(INFO) << "Using averaged variable: " << GetAveragedName(name);
    return status;
  }

  if (allow_fallback_) {
    LOG(INFO) << "Falling back to non-averaged variable: " << name;
    return wrapped_variable_store_->Lookup(name, format, dimensions, area);
  }

  return tensorflow::errors::InvalidArgument(
      "Failed to retrieve averaged variable '", GetAveragedName(name),
      "' for variable '", name, "': ", status.error_message());
}

tensorflow::Status TryAveragedVariableStoreWrapper::Close() {
  return wrapped_variable_store_->Close();
}

CaptureUsedVariableStoreWrapper::CaptureUsedVariableStoreWrapper(
    std::unique_ptr<VariableStore> variable_store)
    : wrapped_variable_store_(std::move(variable_store)) {}

tensorflow::Status CaptureUsedVariableStoreWrapper::Lookup(
    const string &name, VariableSpec::Format format,
    std::vector<size_t> *dimensions, AlignedArea *area) {
  tensorflow::Status status =
      wrapped_variable_store_->Lookup(name, format, dimensions, area);
  if (status.ok()) {
    // Capture the variable if the wrapped store's Lookup() succeeds.
    VariableKey key(name, format);
    std::pair<VariableKey, VariableValue> value(
        key, VariableValue(*dimensions, *area));
    if (index_.find(key) != index_.end()) {
      variables_[index_[key]] = value;
    } else {
      index_[key] = variables_.size();
      variables_.push_back(value);
    }
  }
  return status;
}

tensorflow::Status CaptureUsedVariableStoreWrapper::Close() {
  return wrapped_variable_store_->Close();
}

FlexibleMatrixVariableStoreWrapper::FlexibleMatrixVariableStoreWrapper(
    std::unique_ptr<VariableStore> variable_store)
    : wrapped_variable_store_(std::move(variable_store)) {}

tensorflow::Status FlexibleMatrixVariableStoreWrapper::Lookup(
    const string &name, VariableSpec::Format format,
    std::vector<size_t> *dimensions, AlignedArea *area) {
  // Forward requests that don't match the relevant suffix.
  tensorflow::StringPiece name_piece = name;
  if (!tensorflow::str_util::ConsumeSuffix(&name_piece,
                                           FlexibleMatrixKernel::kSuffix)) {
    return wrapped_variable_store_->Lookup(name, format, dimensions, area);
  }
  const string basename = name_piece.ToString();

  // Fetch the non-blocked, non-transposed version of the matrix.  This wrapper
  // will be nested inside the capturing wrapper, so we can do multiple lookups
  // without capturing more variables than we need.
  Matrix<float> plain_matrix;
  TF_RETURN_IF_ERROR(wrapped_variable_store_->Lookup(basename, &plain_matrix));
  const int output_dimension = plain_matrix.num_columns();

  // Performance estimates for different methods. A mix of 32/48 blocked
  // matrices got 28 GFLOPS, whereas only unblocked got 2.8 GFLOPS.
  using Candidate = std::tuple<float, VariableSpec::Format, string>;
  const std::vector<Candidate> candidates = {
      Candidate(2.8f, VariableSpec::FORMAT_ROW_MAJOR_MATRIX,
                tensorflow::strings::StrCat(basename, "/transposed")),
      Candidate(EffectiveGflops(output_dimension, 32, 25.0f),
                VariableSpec::FORMAT_COLUMN_BLOCKED_ROW_MAJOR_MATRIX,
                tensorflow::strings::StrCat(basename, "/matrix/blocked32")),
      Candidate(EffectiveGflops(output_dimension, 48, 25.0f),
                VariableSpec::FORMAT_COLUMN_BLOCKED_ROW_MAJOR_MATRIX,
                tensorflow::strings::StrCat(basename, "/matrix/blocked48"))};
  const auto max_it = std::max_element(candidates.begin(), candidates.end());
  const VariableSpec::Format argmax_format = std::get<1>(*max_it);
  const string &argmax_name = std::get<2>(*max_it);

  // The requested |format| must match the best format.  If not, return error
  // and wait until the proper format is requested.
  if (format != argmax_format) {
    return tensorflow::errors::FailedPrecondition(
        "Sub-optimal matrix format: ", VariableSpec::Format_Name(format), " (",
        VariableSpec::Format_Name(argmax_format), " is best)");
  }

  return wrapped_variable_store_->Lookup(argmax_name, format, dimensions, area);
}

tensorflow::Status FlexibleMatrixVariableStoreWrapper::Close() {
  return wrapped_variable_store_->Close();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
