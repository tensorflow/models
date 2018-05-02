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

#include "dragnn/runtime/component_transformation.h"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "dragnn/runtime/component.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

tensorflow::Status TransformComponents(const string &input_master_spec_path,
                                       const string &output_master_spec_path) {
  MasterSpec master_spec;
  TF_RETURN_IF_ERROR(tensorflow::ReadTextProto(
      tensorflow::Env::Default(), input_master_spec_path, &master_spec));

  for (ComponentSpec &component_spec : *master_spec.mutable_component()) {
    TF_RETURN_IF_ERROR(ComponentTransformer::ApplyAll(&component_spec));
  }

  return tensorflow::WriteTextProto(tensorflow::Env::Default(),
                                    output_master_spec_path, master_spec);
}

tensorflow::Status ComponentTransformer::ApplyAll(
    ComponentSpec *component_spec) {
  // Limit on the number of iterations, to prevent infinite loops.
  static constexpr int kMaxNumIterations = 1000;

  std::set<string> names;  // sorted for determinism
  for (const Registry::Registrar *registrar = registry()->components;
       registrar != nullptr; registrar = registrar->next()) {
    names.insert(registrar->name());
  }

  std::vector<std::unique_ptr<ComponentTransformer>> transformers;
  transformers.reserve(names.size());
  for (const string &name : names) transformers.emplace_back(Create(name));

  ComponentSpec local_spec = *component_spec;  // avoid modification on error
  for (int iteration = 0; iteration < kMaxNumIterations; ++iteration) {
    const ComponentSpec original_spec = local_spec;

    for (const auto &transformer : transformers) {
      const string component_type =
          GetNormalizedComponentBuilderName(local_spec);
      TF_RETURN_IF_ERROR(transformer->Transform(component_type, &local_spec));
    }

    if (tensorflow::protobuf::util::MessageDifferencer::Equals(local_spec,
                                                               original_spec)) {
      // Converged successfully; make modifications.
      *component_spec = local_spec;
      return tensorflow::Status::OK();
    }
  }

  return tensorflow::errors::Internal("Failed to converge within ",
                                      kMaxNumIterations,
                                      " ComponentTransformer iterations");
}

}  // namespace runtime
}  // namespace dragnn

REGISTER_SYNTAXNET_CLASS_REGISTRY("DRAGNN Runtime Component Transformer",
                                  dragnn::runtime::ComponentTransformer);

}  // namespace syntaxnet
