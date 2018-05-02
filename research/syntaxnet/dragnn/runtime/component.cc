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

#include "dragnn/runtime/component.h"

#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

string GetNormalizedComponentBuilderName(const ComponentSpec &component_spec) {
  // The Python registration API is based on (relative) module paths, such as
  // "some.module.FooComponent".  Discard the module path prefix and use only
  // the final segment, which is the subclass name.
  const std::vector<string> segments = tensorflow::str_util::Split(
      component_spec.component_builder().registered_name(), ".");
  CHECK_GT(segments.size(), 0) << "No builder name for component spec: "
                               << component_spec.ShortDebugString();
  tensorflow::StringPiece subclass_name = segments.back();

  // In addition, remove a "Builder" suffix, if any.  In the Python codebase, a
  // ComponentBuilder builds a TF graph to perform some computation, whereas in
  // the runtime, a Component directly executes that computation.
  tensorflow::str_util::ConsumeSuffix(&subclass_name, "Builder");
  return subclass_name.ToString();
}

tensorflow::Status Component::Select(const ComponentSpec &spec,
                                     string *result) {
  const string normalized_builder_name =
      GetNormalizedComponentBuilderName(spec);

  // Iterate through all registered components, constructing them and querying
  // their Supports() methods.
  std::unique_ptr<Component> current_best;
  string current_best_name;

  for (const Registry::Registrar *component = registry()->components;
       component != nullptr; component = component->next()) {
    // component->object() is a function pointer to the subclass' constructor.
    std::unique_ptr<Component> next(component->object()());
    string next_name(component->name());

    if (!next->Supports(spec, normalized_builder_name)) {
      continue;
    }

    // First supported component.
    if (current_best == nullptr) {
      current_best = std::move(next);
      current_best_name = next_name;
      continue;
    }

    // The two must agree on which takes precedence.
    if (next->PreferredTo(*current_best)) {
      if (current_best->PreferredTo(*next)) {
        return tensorflow::errors::FailedPrecondition(
            "Classes '", current_best_name, "' and '", next_name,
            "' both think they should be preferred to each-other. Please "
            "add logic to their PreferredTo() methods to avoid this.");
      }
      current_best = std::move(next);
      current_best_name = next_name;
    } else if (!current_best->PreferredTo(*next)) {
      return tensorflow::errors::FailedPrecondition(
          "Classes '", current_best_name, "' and '", next_name,
          "' both think they should be dis-preferred to each-other. Please "
          "add logic to their PreferredTo() methods to avoid this.");
    }
  }

  if (current_best == nullptr) {
    return tensorflow::errors::NotFound(
        "Could not find a best spec for component '", spec.name(),
        "' with normalized builder name '", normalized_builder_name, "'");
  } else {
    *result = std::move(current_best_name);
    return tensorflow::Status::OK();
  }
}

}  // namespace runtime
}  // namespace dragnn

REGISTER_SYNTAXNET_CLASS_REGISTRY("DRAGNN Runtime Component",
                                  dragnn::runtime::Component);

}  // namespace syntaxnet
