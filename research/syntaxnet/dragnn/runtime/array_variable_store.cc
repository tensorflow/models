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

#include "dragnn/runtime/array_variable_store.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/cpu_info.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Increment this if the serialized format changes in an incompatible way that
// can't be detected through other means.  For example,
// * If kAlignmentBytes is changed, then kVersion need not change because there
//   is a separate field for detecting alignment mismatch.
// * If ArrayVariableStoreSpec.variable is no longer populated, perhaps replaced
//   by some other approach, then kVersion should be incremented.
const uint32 ArrayVariableStore::kVersion = 0;

tensorflow::Status ArrayVariableStore::Reset(const ArrayVariableStoreSpec &spec,
                                             AlignedView data) {
  if (!spec.has_version() || !spec.has_alignment_bytes() ||
      !spec.has_is_little_endian()) {
    return tensorflow::errors::InvalidArgument(
        "ArrayVariableStoreSpec is missing a required field: ",
        spec.ShortDebugString());
  }

  if (spec.version() != kVersion) {
    return tensorflow::errors::InvalidArgument(
        "ArrayVariableStoreSpec.version (", spec.version(),
        ") does not match the binary (", kVersion, ")");
  }

  if (spec.alignment_bytes() != internal::kAlignmentBytes) {
    return tensorflow::errors::InvalidArgument(
        "ArrayVariableStoreSpec.alignment_bytes (", spec.alignment_bytes(),
        ") does not match the binary (", internal::kAlignmentBytes, ")");
  }

  // TODO(googleuser): It should be possible to correct an endian-ness mismatch.
  // A rough outline is:
  // * VariableStore::Lookup() takes an additional argument set to sizeof(T).
  // * Capture sizeof(T) and write it into the VariableSpec.
  // * Detect endian mismatch and byte-swap variables with multi-byte types.
  if (spec.is_little_endian() != tensorflow::port::kLittleEndian) {
    return tensorflow::errors::InvalidArgument(
        "ArrayVariableStoreSpec.is_little_endian (", spec.is_little_endian(),
        ") does not match the binary (", tensorflow::port::kLittleEndian, ")");
  }

  for (const VariableSpec &variable_spec : spec.variable()) {
    // When the proto parser encounters an unknown enumerator on the wire, it
    // replaces it with the default value (i.e., FORMAT_UNKNOWN).  Therefore,
    // VariableSpec.format() will always return a valid enumerator.
    DCHECK(VariableSpec::Format_IsValid(variable_spec.format()));

    if (variable_spec.format() == VariableSpec::FORMAT_UNKNOWN) {
      return tensorflow::errors::InvalidArgument(
          "Unknown variable format: ", variable_spec.ShortDebugString());
    }

    if (variable_spec.format() == VariableSpec::FORMAT_FLAT &&
        variable_spec.num_views() != 1) {
      return tensorflow::errors::InvalidArgument(
          "Flat variables must have 1 view: ",
          variable_spec.ShortDebugString());
    }
  }

  // Build into a temp mapping to avoid modification on error.
  std::unique_ptr<std::map<Key, Value>> new_variables(
      new std::map<Key, Value>());

  // Slice sub-arrays off of the main byte array.
  const char *base = data.data();
  const char *const end = base + data.size();
  for (const VariableSpec &variable_spec : spec.variable()) {
    const size_t num_views = variable_spec.num_views();
    const size_t view_size = variable_spec.view_size();
    const size_t area_size = ComputeAlignedAreaSize(num_views, view_size);

    if (base + area_size > end) {
      return tensorflow::errors::InvalidArgument(
          "Variable would overrun main byte array: ",
          variable_spec.ShortDebugString());
    }

    AlignedView view;
    TF_RETURN_IF_ERROR(view.Reset(base, area_size));
    base += area_size;  // remove claimed slice

    // Set dimensions from the spec.
    std::vector<size_t> dimensions(variable_spec.dimension().begin(),
                                   variable_spec.dimension().end());

    Value value(std::move(dimensions), AlignedArea());
    AlignedArea &area = value.second;
    TF_RETURN_IF_ERROR(area.Reset(view, num_views, view_size));

    // Currently, blocked variables are meant for fast inference algorithms,
    // which do not tolerate padding. Raise errors if there is padding.
    if (variable_spec.format() ==
        VariableSpec::FORMAT_COLUMN_BLOCKED_ROW_MAJOR_MATRIX) {
      size_t padding = variable_spec.view_size() % internal::kAlignmentBytes;
      if (padding != 0) {
        return tensorflow::errors::Internal(
            "Currently, fast matrix-vector operations do not support padded "
            "blocked matrices, but variable '",
            variable_spec.name(), "' has padding ", padding);
      }
    }

    const Key key(variable_spec.name(), variable_spec.format());

    if (!new_variables->emplace(key, value).second) {
      return tensorflow::errors::InvalidArgument(
          "Duplicate variable: ", variable_spec.ShortDebugString());
    }
  }

  if (base != end) {
    return tensorflow::errors::InvalidArgument(
        "Variables do not completely cover main byte array: ", end - base,
        " bytes remaining");
  }

  // Success; make modifications.
  variables_ = std::move(new_variables);
  return tensorflow::Status::OK();
}

tensorflow::Status ArrayVariableStore::Lookup(const string &name,
                                              VariableSpec::Format format,
                                              std::vector<size_t> *dimensions,
                                              AlignedArea *area) {
  if (!variables_) {
    return tensorflow::errors::FailedPrecondition(
        "ArrayVariableStore not initialized");
  }

  const Key key(name, format);
  const auto it = variables_->find(key);
  if (it == variables_->end()) {
    return tensorflow::errors::NotFound(
        "ArrayVariableStore has no variable with name '", name, "' and format ",
        VariableSpec::Format_Name(format));
  }

  // Success; make modifications.
  const Value &value = it->second;
  *dimensions = value.first;
  *area = value.second;
  return tensorflow::Status::OK();
}

tensorflow::Status ArrayVariableStore::Close() {
  if (!variables_) {
    return tensorflow::errors::FailedPrecondition(
        "ArrayVariableStore not initialized");
  }
  variables_.reset();
  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
