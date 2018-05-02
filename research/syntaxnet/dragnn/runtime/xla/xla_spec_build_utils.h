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

// Build-time utils for working with spec of XLA-based DRAGNN runtime models.

#ifndef DRAGNN_RUNTIME_XLA_XLA_SPEC_BUILD_UTILS_H_
#define DRAGNN_RUNTIME_XLA_XLA_SPEC_BUILD_UTILS_H_

#include <string>
#include <vector>

#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Appends to string |spec_names_def| a Bazel variable definition for a
// variable called |variable_name|. The variable is a list, each entry
// contains the relevant information used by the build rules (the model name,
// the component name, and graph path), for every component in any of the
// MasterSpec protos stored in |master_spec_paths| which have the model name
// set. The |base_path| is stripped off when generating the graph_paths.
// On error, returns non-OK.
tensorflow::Status MasterSpecsToBazelDef(
    const string &variable_name, const string &base_path,
    const std::vector<string> &master_spec_paths, string *spec_names_def);

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_XLA_XLA_SPEC_BUILD_UTILS_H_
