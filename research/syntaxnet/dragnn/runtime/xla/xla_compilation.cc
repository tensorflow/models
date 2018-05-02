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

#include "dragnn/runtime/xla/xla_compilation.h"

#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "dragnn/protos/export.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/trained_model.h"
#include "dragnn/runtime/xla/xla_cell_converter.h"
#include "dragnn/runtime/xla/xla_graph_utils.h"
#include "dragnn/runtime/xla/xla_spec_utils.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Updates the Component subclass in the |component_spec| to an XLA-based
// version.  On error, returns non-OK and modifies nothing.
tensorflow::Status XlaCompileComponentSubclass(ComponentSpec *component_spec) {
  const string subclass = GetNormalizedComponentBuilderName(*component_spec);
  if (subclass != "DynamicComponent") {
    return tensorflow::errors::Unimplemented(
        "No XLA-based version of Component subclass '", subclass, "'");
  }

  // By convention, the XLA-based version of "FooComponent" should be named
  // "XlaFooComponent".
  component_spec->mutable_component_builder()->set_registered_name(
      tensorflow::strings::StrCat("Xla", subclass));
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

tensorflow::Status XlaCompileCells(const string &saved_model_dir,
                                   const string &master_spec_path,
                                   const std::set<string> &component_names,
                                   const string &model_name,
                                   const string &output_dir) {
  MasterSpec master_spec;
  TF_RETURN_IF_ERROR(tensorflow::ReadTextProto(tensorflow::Env::Default(),
                                               master_spec_path, &master_spec));

  std::vector<ComponentSpec *> components;
  TF_RETURN_IF_ERROR(
      GetMatchingComponentSpecs(component_names, &master_spec, &components));

  // Returns the path to the output frozen GraphDef file for the
  // |component_spec|.
  const auto get_frozen_graph_def_path =
      [&](const ComponentSpec &component_spec) {
        return tensorflow::io::JoinPath(
            output_dir,
            tensorflow::strings::StrCat(component_spec.name(),
                                        kFrozenGraphDefResourceFileSuffix));
      };

  // Perform some changes to the MasterSpec first, to catch issues before
  // loading the trained models, which is slow.
  for (ComponentSpec *component_spec : components) {
    // Add a resource for the frozen GraphDef file to each component.  The file
    // will be created in a second pass, after loading the trained model.
    TF_RETURN_IF_ERROR(AddFrozenGraphDefResource(
        get_frozen_graph_def_path(*component_spec), component_spec));

    // Replace the Component subclass with an XLA-based version.
    TF_RETURN_IF_ERROR(XlaCompileComponentSubclass(component_spec));

    // Set embedding_dim=-1 for all channels.
    for (auto &fixed_channel : *component_spec->mutable_fixed_feature()) {
      fixed_channel.set_embedding_dim(-1);
    }
    for (auto &linked_channel : *component_spec->mutable_linked_feature()) {
      linked_channel.set_embedding_dim(-1);
    }
  }

  // Create output directory which contains the new master spec and
  // the frozen graphs.
  TF_RETURN_IF_ERROR(
      tensorflow::Env::Default()->RecursivelyCreateDir(output_dir));

  // Convert each component into a frozen GraphDef and write it. Also may
  // add a CompilationSpec.
  TrainedModel trained_model;
  TF_RETURN_IF_ERROR(trained_model.Reset(saved_model_dir));
  for (ComponentSpec *component_spec : components) {
    tensorflow::GraphDef frozen_graph_def;
    CellSubgraphSpec cell_subgraph_spec;
    TF_RETURN_IF_ERROR(
        XlaCellConverter::Convert(component_spec->name(), trained_model,
                                  &frozen_graph_def, &cell_subgraph_spec));
    TF_RETURN_IF_ERROR(SaveFrozenGraphDef(
        get_frozen_graph_def_path(*component_spec), frozen_graph_def));

    if (!model_name.empty()) {
      auto *compilation_spec = component_spec->MutableExtension(
          CompilationSpec::component_spec_extension);
      compilation_spec->set_model_name(model_name);
      *compilation_spec->mutable_cell_subgraph_spec() = cell_subgraph_spec;
    }
  }

  // Write the updated MasterSpec.
  TF_RETURN_IF_ERROR(tensorflow::WriteTextProto(
      tensorflow::Env::Default(),
      tensorflow::io::JoinPath(output_dir, "master-spec"), master_spec));

  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
