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

#include "dragnn/runtime/attributes.h"

#include <set>

#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

tensorflow::Status Attributes::Reset(
    const tensorflow::protobuf::Map<string, string> &mapping) {
  // First pass: Parse each value in the |mapping|.
  for (const auto &name_value : mapping) {
    const string &name = name_value.first;
    const string &value = name_value.second;
    const auto it = attributes_.find(name);
    if (it == attributes_.end()) {
      return tensorflow::errors::InvalidArgument("Unknown attribute: ", name);
    }
    TF_RETURN_IF_ERROR(it->second->Parse(value));
  }

  // Second pass: Look for missing mandatory attributes.
  std::set<string> missing_mandatory_attributes;
  for (const auto &it : attributes_) {
    const string &name = it.first;
    Attribute *attribute = it.second;
    if (!attribute->IsMandatory()) continue;
    if (mapping.find(name) == mapping.end()) {
      missing_mandatory_attributes.insert(name);
    }
  }

  if (!missing_mandatory_attributes.empty()) {
    return tensorflow::errors::InvalidArgument(
        "Missing mandatory attributes: ",
        tensorflow::str_util::Join(missing_mandatory_attributes, " "));
  }

  return tensorflow::Status::OK();
}

void Attributes::Register(const string &name, Attribute *attribute) {
  const bool unique = attributes_.emplace(name, attribute).second;
  DCHECK(unique) << "Duplicate attribute '" << name << "'";
}

tensorflow::Status Attributes::ParseValue(const string &str, string *value) {
  *value = str;
  return tensorflow::Status::OK();
}

tensorflow::Status Attributes::ParseValue(const string &str, bool *value) {
  const string lowercased_str = tensorflow::str_util::Lowercase(str);
  if (lowercased_str != "true" && lowercased_str != "false") {
    return tensorflow::errors::InvalidArgument(
        "Attribute can't be parsed as bool: ", str);
  }
  *value = lowercased_str == "true";
  return tensorflow::Status::OK();
}

tensorflow::Status Attributes::ParseValue(const string &str, int32 *value) {
  if (!tensorflow::strings::safe_strto32(str, value)) {
    return tensorflow::errors::InvalidArgument(
        "Attribute can't be parsed as int32: ", str);
  }
  return tensorflow::Status::OK();
}

tensorflow::Status Attributes::ParseValue(const string &str, int64 *value) {
  if (!tensorflow::strings::safe_strto64(str, value)) {
    return tensorflow::errors::InvalidArgument(
        "Attribute can't be parsed as int64: ", str);
  }
  return tensorflow::Status::OK();
}

tensorflow::Status Attributes::ParseValue(const string &str, size_t *value) {
  int64 signed_value = 0;
  if (!tensorflow::strings::safe_strto64(str, &signed_value) ||
      signed_value < 0) {
    return tensorflow::errors::InvalidArgument(
        "Attribute can't be parsed as size_t: ", str);
  }
  *value = signed_value;
  return tensorflow::Status::OK();
}

tensorflow::Status Attributes::ParseValue(const string &str, float *value) {
  if (!tensorflow::strings::safe_strtof(str.c_str(), value)) {
    return tensorflow::errors::InvalidArgument(
        "Attribute can't be parsed as float: ", str);
  }
  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
