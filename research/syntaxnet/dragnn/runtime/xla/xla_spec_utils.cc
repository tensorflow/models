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

#include "dragnn/runtime/xla/xla_spec_utils.h"

#include <algorithm>

#include "dragnn/protos/export.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

const char *const kFrozenGraphDefResourceName = "frozen-graph";
const char *const kFrozenGraphDefResourceFileFormat = "proto";
const char *const kFrozenGraphDefResourceRecordFormat = "tensorflow.GraphDef";
const char *const kFrozenGraphDefResourceFileSuffix = "-frozen";

string ModelNameForComponent(const ComponentSpec &component_spec) {
  return component_spec.GetExtension(CompilationSpec::component_spec_extension)
      .model_name();
}

tensorflow::Status GetCellSubgraphSpecForComponent(
    const ComponentSpec &component_spec, CellSubgraphSpec *cell_subgraph_spec) {
  if (!component_spec.GetExtension(CompilationSpec::component_spec_extension)
           .has_cell_subgraph_spec()) {
    return tensorflow::errors::InvalidArgument(
        "Component ", component_spec.name(),
        " does not have a CellSubgraphSpec");
  }

  if (cell_subgraph_spec != nullptr) {
    *cell_subgraph_spec =
        component_spec.GetExtension(CompilationSpec::component_spec_extension)
            .cell_subgraph_spec();
  }
  return tensorflow::Status::OK();
}

tensorflow::Status LookupFrozenGraphDefResource(
    const ComponentSpec &component_spec,
    const Resource **frozen_graph_def_resource) {
  const Resource *found_resource = nullptr;
  for (const Resource &resource : component_spec.resource()) {
    if (resource.name() != kFrozenGraphDefResourceName) continue;

    if (found_resource != nullptr) {
      return tensorflow::errors::InvalidArgument(
          "Component '", component_spec.name(),
          "' contains duplicate frozen TF GraphDef resources");
    }

    if (resource.part_size() != 1) {
      return tensorflow::errors::InvalidArgument(
          "Component '", component_spec.name(),
          "' has malformed frozen TF GraphDef resource; expected 1 part");
    }

    const Part &part = resource.part(0);
    if (part.file_format() != kFrozenGraphDefResourceFileFormat) {
      return tensorflow::errors::InvalidArgument(
          "Component '", component_spec.name(),
          "' has malformed frozen TF GraphDef resource; wrong file format");
    }

    if (part.record_format() != kFrozenGraphDefResourceRecordFormat) {
      return tensorflow::errors::InvalidArgument(
          "Component '", component_spec.name(),
          "' has malformed frozen TF GraphDef resource; wrong record format");
    }

    found_resource = &resource;
  }

  if (found_resource == nullptr) {
    return tensorflow::errors::NotFound("Component '", component_spec.name(),
                                        "' has no frozen TF GraphDef resource");
  }

  // Success; make modifications.
  *frozen_graph_def_resource = found_resource;
  return tensorflow::Status::OK();
}

tensorflow::Status AddFrozenGraphDefResource(const string &path,
                                             ComponentSpec *component_spec) {
  if (std::any_of(component_spec->resource().begin(),
                  component_spec->resource().end(),
                  [](const Resource &resource) {
                    return resource.name() == kFrozenGraphDefResourceName;
                  })) {
    return tensorflow::errors::InvalidArgument(
        "Component '", component_spec->name(),
        "' already contains a frozen TF GraphDef resource");
  }

  // Success; make modifications.
  Resource *resource = component_spec->add_resource();
  resource->set_name(kFrozenGraphDefResourceName);
  Part *part = resource->add_part();
  part->set_file_pattern(path);
  part->set_file_format(kFrozenGraphDefResourceFileFormat);
  part->set_record_format(kFrozenGraphDefResourceRecordFormat);
  return tensorflow::Status::OK();
}

string MakeXlaInputFixedFeatureIdName(int channel_id, int index) {
  return MakeXlaInputLayerName(tensorflow::strings::StrCat(
      "fixed_channel_", channel_id, "_index_", index, "_ids"));
}

string MakeXlaInputLinkedActivationVectorName(int channel_id) {
  return MakeXlaInputLayerName(tensorflow::strings::StrCat(
      "linked_channel_", channel_id, "_activations"));
}

string MakeXlaInputLinkedOutOfBoundsIndicatorName(int channel_id) {
  return MakeXlaInputLayerName(tensorflow::strings::StrCat(
      "linked_channel_", channel_id, "_out_of_bounds"));
}

string MakeXlaInputRecurrentLayerName(const string &layer_name) {
  return MakeXlaInputLayerName(layer_name);
}

string MakeXlaInputLayerName(const string &layer_name) {
  return tensorflow::strings::StrCat("INPUT__", layer_name);
}

string MakeXlaOutputLayerName(const string &layer_name) {
  return tensorflow::strings::StrCat("OUTPUT__", layer_name);
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
