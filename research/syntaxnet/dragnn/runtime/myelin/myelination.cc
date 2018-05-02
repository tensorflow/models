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

#include "dragnn/runtime/myelin/myelination.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/myelin/myelin_cell_converter.h"
#include "dragnn/runtime/myelin/myelin_spec_utils.h"
#include "dragnn/runtime/trained_model.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Updates the Component subclass in the |component_spec| to a Myelin-based
// version.  On error, returns non-OK and modifies nothing.
tensorflow::Status MyelinateComponentSubclass(ComponentSpec *component_spec) {
  const string subclass = GetNormalizedComponentBuilderName(*component_spec);
  if (subclass != "DynamicComponent") {
    return tensorflow::errors::Unimplemented(
        "No Myelin-based version of Component subclass '", subclass, "'");
  }

  // By convention, the Myelin-based version of "FooComponent" should be named
  // "MyelinFooComponent".
  component_spec->mutable_component_builder()->set_registered_name(
      tensorflow::strings::StrCat("Myelin", subclass));
  return tensorflow::Status::OK();
}

// Appends the list of component specs in the |master_spec| whose names match
// |component_names| to |matching_components|.  On error, returns non-OK.
tensorflow::Status GetMatchingComponentSpecs(
    const std::set<string> &component_names, MasterSpec *master_spec,
    std::vector<ComponentSpec *> *matching_components) {
  // Index the components in the |master_spec| by name.
  std::map<string, ComponentSpec *> components;
  for (ComponentSpec &component_spec : *master_spec->mutable_component()) {
    if (!components.emplace(component_spec.name(), &component_spec).second) {
      return tensorflow::errors::InvalidArgument("Duplicate component name: ",
                                                 component_spec.name());
    }
  }

  // Append the components named in the |component_names|.
  for (const string &component_name : component_names) {
    if (components.find(component_name) == components.end()) {
        return tensorflow::errors::InvalidArgument("Unknown component name: ",
                                                   component_name);
    }
    matching_components->push_back(components[component_name]);
  }

  return tensorflow::Status::OK();
}

}  // namespace

tensorflow::Status MyelinateCells(const string &saved_model_dir,
                                  const string &master_spec_path,
                                  const std::set<string> &component_names,
                                  const string &output_dir) {
  MasterSpec master_spec;
  TF_RETURN_IF_ERROR(tensorflow::ReadTextProto(tensorflow::Env::Default(),
                                               master_spec_path, &master_spec));

  std::vector<ComponentSpec *> components;
  TF_RETURN_IF_ERROR(
      GetMatchingComponentSpecs(component_names, &master_spec, &components));

  // Returns the path to the output Flow file for the |component_spec|.
  const auto get_flow_path = [&](const ComponentSpec &component_spec) {
    return tensorflow::io::JoinPath(
        output_dir,
        tensorflow::strings::StrCat(component_spec.name(), ".flow"));
  };

  // Modify the MasterSpec first, to catch issues before loading the trained
  // model, which is slow.
  for (ComponentSpec *component_spec : components) {
    // Add a resource for the Flow file to each component.  The file will be
    // created in a second pass, after loading the trained model.
    TF_RETURN_IF_ERROR(
        AddMyelinFlowResource(get_flow_path(*component_spec), component_spec));

    // Replace the Component subclass with a Myelin-based version.
    TF_RETURN_IF_ERROR(MyelinateComponentSubclass(component_spec));

    // Set embedding_dim=-1 for all channels.
    for (auto &fixed_channel : *component_spec->mutable_fixed_feature()) {
      fixed_channel.set_embedding_dim(-1);
    }
    for (auto &linked_channel : *component_spec->mutable_linked_feature()) {
      linked_channel.set_embedding_dim(-1);
    }
  }

  // Write the updated MasterSpec.
  TF_RETURN_IF_ERROR(
      tensorflow::Env::Default()->RecursivelyCreateDir(output_dir));
  TF_RETURN_IF_ERROR(tensorflow::WriteTextProto(
      tensorflow::Env::Default(),
      tensorflow::io::JoinPath(output_dir, "master-spec"), master_spec));

  // Convert each component into a Flow and write it.
  TrainedModel trained_model;
  TF_RETURN_IF_ERROR(trained_model.Reset(saved_model_dir));
  for (const ComponentSpec *component_spec : components) {
    string flow_data;
    TF_RETURN_IF_ERROR(MyelinCellConverter::Convert(component_spec->name(),
                                                    trained_model, &flow_data));

    TF_RETURN_IF_ERROR(tensorflow::WriteStringToFile(
        tensorflow::Env::Default(), get_flow_path(*component_spec), flow_data));
  }

  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
