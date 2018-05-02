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

#include "dragnn/runtime/test/fake_variable_store.h"

#include <string.h>
#include <utility>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

void FakeVariableStore::AddOrDie(const string &name,
                                 const std::vector<std::vector<float>> &data,
                                 VariableSpec::Format format) {
  CHECK(variables_[name].empty()) << "Adding duplicate variable: " << name;
  FormatMap formats;

  // Add a flattened version.
  std::vector<std::vector<float>> flat(1);
  for (const auto &row : data) {
    for (const float value : row) flat[0].push_back(value);
  }
  formats[VariableSpec::FORMAT_FLAT] = Variable(flat);

  // Add the |data| in its natural row-major format.
  formats[VariableSpec::FORMAT_ROW_MAJOR_MATRIX] = Variable(data);

  // Add the |data| as a trivial blocked matrix with one block---i.e., block
  // size equal to the number of columns.  Conveniently, this matrix has the
  // same underlying data layout as a plain matrix.
  formats[VariableSpec::FORMAT_COLUMN_BLOCKED_ROW_MAJOR_MATRIX] =
      Variable(data);

  // If |format| is FORMAT_UNKNOWN, keep all formats.  Otherwise, only keep the
  // specified format.
  if (format == VariableSpec::FORMAT_UNKNOWN) {
    variables_[name] = std::move(formats);
  } else {
    variables_[name][format] = std::move(formats[format]);
  }
}

void FakeVariableStore::SetBlockedDimensionOverride(
    const string &name, const std::vector<size_t> &dimensions) {
  override_blocked_dimensions_[name] = dimensions;
}

tensorflow::Status FakeVariableStore::Lookup(const string &name,
                                             VariableSpec::Format format,
                                             std::vector<size_t> *dimensions,
                                             AlignedArea *area) {
  const auto it = variables_.find(name);
  if (it == variables_.end()) {
    return tensorflow::errors::InvalidArgument("Unknown variable: ", name);
  }
  FormatMap &formats = it->second;
  if (formats.find(format) == formats.end()) {
    return tensorflow::errors::InvalidArgument("Unknown variable: ", name);
  }
  Variable &variable = formats.at(format);

  dimensions->clear();
  switch (format) {
    case VariableSpec::FORMAT_UNKNOWN:
      // This case should not happen because the |formats| mapping never has
      // FORMAT_UNKNOWN as a key.
      LOG(FATAL) << "Tried to get a variable with FORMAT_UNKNOWN";

    case VariableSpec::FORMAT_FLAT:
      *dimensions = {variable->num_columns()};
      break;
    case VariableSpec::FORMAT_ROW_MAJOR_MATRIX:
      *dimensions = {variable->num_rows(), variable->num_columns()};
      break;
    case VariableSpec::FORMAT_COLUMN_BLOCKED_ROW_MAJOR_MATRIX:
      if (override_blocked_dimensions_.find(name) !=
          override_blocked_dimensions_.end()) {
        *dimensions = override_blocked_dimensions_[name];
      } else {
        *dimensions = {variable->num_rows(), variable->num_columns(),
                       variable->num_columns()};  // = block_size
      }
      break;
  }

  *area = variable.area();
  return tensorflow::Status::OK();
}

// Executes cleanup functions (see `cleanup_` comment).
SimpleFakeVariableStore::~SimpleFakeVariableStore() {
  for (const auto &fcn : cleanup_) {
    fcn();
  }
}

tensorflow::Status SimpleFakeVariableStore::Lookup(
    const string &name, VariableSpec::Format format,
    std::vector<size_t> *dimensions, AlignedArea *area) {
  // Test should call MockLookup() first.
  CHECK(dimensions_to_return_ != nullptr);
  CHECK(area_to_return_ != nullptr);
  *dimensions = *dimensions_to_return_;
  *area = *area_to_return_;
  dimensions_to_return_ = nullptr;
  area_to_return_ = nullptr;
  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
