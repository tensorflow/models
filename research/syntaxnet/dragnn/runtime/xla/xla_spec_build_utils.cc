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

#include "dragnn/runtime/xla/xla_spec_build_utils.h"

#include <map>
#include <utility>
#include <vector>

#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/xla/xla_spec_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

tensorflow::Status MasterSpecsToBazelDef(
    const string &variable_name, const string &base_path,
    const std::vector<string> &master_spec_paths, string *spec_names_def) {
  std::map<std::pair<string, string>, string> component_graph_map;
  for (const string &path : master_spec_paths) {
    MasterSpec master_spec;
    TF_RETURN_IF_ERROR(tensorflow::ReadTextProto(tensorflow::Env::Default(),
                                                 path, &master_spec));

    // TODO(googleuser): Replace with non-fragile approach to get the GraphDef path.
    tensorflow::StringPiece path_prefix = tensorflow::StringPiece(path);
    if (tensorflow::str_util::ConsumePrefix(&path_prefix, base_path)) {
      tensorflow::str_util::ConsumePrefix(&path_prefix, "/");
    }
    path_prefix = path_prefix.substr(0, path_prefix.rfind('.'));

    // Adds an entry for each unique model/component, removing any
    // duplicates.  However, if duplicate model/components are found
    // with differing graph paths, this is flagged as an error (a
    // sanity check to ensure model name consistency).
    for (const ComponentSpec &component_spec : master_spec.component()) {
      const string &model_name = ModelNameForComponent(component_spec);
      if (model_name.empty()) continue;

      string &component_graph = component_graph_map[std::make_pair(
          model_name, component_spec.name())];

      const string &component_graph_path = tensorflow::strings::StrCat(
          path_prefix, ".xla-compiled-cells-", component_spec.name(),
          kFrozenGraphDefResourceFileSuffix);
      if (!component_graph.empty()) {
        return tensorflow::errors::InvalidArgument("Component '", model_name,
                                                   "::", component_spec.name(),
                                                   "is duplicated");
      }
      component_graph = component_graph_path;
    }
  }

  // Appends the Bazel expression which contains one string array for
  // each unique model/component.
  tensorflow::strings::StrAppend(spec_names_def, variable_name, " = [\n");
  for (const auto &component_data : component_graph_map) {
    tensorflow::strings::StrAppend(
        spec_names_def, "    [ '", component_data.first.first, "', '",
        component_data.first.second, "', '", component_data.second, "' ],\n");
  }
  tensorflow::strings::StrAppend(spec_names_def, "]\n");

  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
