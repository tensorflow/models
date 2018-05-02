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

#include "dragnn/runtime/myelin/myelin_spec_utils.h"

#include <algorithm>

#include "dragnn/runtime/myelin/myelin_library.h"
#include "sling/base/status.h"
#include "sling/file/file.h"
#include "sling/myelin/kernel/tensorflow.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

const char *const kMyelinFlowResourceName = "myelin-flow";
const char *const kMyelinFlowResourceFileFormat = "model";
const char *const kMyelinFlowResourceRecordFormat = "sling.myelin.Flow";

tensorflow::Status LookupMyelinFlowResource(const ComponentSpec &component_spec,
                                            const Resource **flow_resource) {
  const Resource *found_resource = nullptr;
  for (const Resource &resource : component_spec.resource()) {
    if (resource.name() != kMyelinFlowResourceName) continue;

    if (found_resource != nullptr) {
      return tensorflow::errors::InvalidArgument(
          "Component '", component_spec.name(),
          "' contains duplicate Myelin Flow resources");
    }

    if (resource.part_size() != 1) {
      return tensorflow::errors::InvalidArgument(
          "Component '", component_spec.name(),
          "' has malformed Myelin Flow resource; expected 1 part");
    }

    const Part &part = resource.part(0);
    if (part.file_format() != kMyelinFlowResourceFileFormat) {
      return tensorflow::errors::InvalidArgument(
          "Component '", component_spec.name(),
          "' has malformed Myelin Flow resource; wrong file format");
    }

    if (part.record_format() != kMyelinFlowResourceRecordFormat) {
      return tensorflow::errors::InvalidArgument(
          "Component '", component_spec.name(),
          "' has malformed Myelin Flow resource; wrong record format");
    }

    found_resource = &resource;
  }

  if (found_resource == nullptr) {
    return tensorflow::errors::NotFound("Component '", component_spec.name(),
                                        "' has no Myelin Flow resource");
  }

  // Success; make modifications.
  *flow_resource = found_resource;
  return tensorflow::Status::OK();
}

tensorflow::Status AddMyelinFlowResource(const string &path,
                                         ComponentSpec *component_spec) {
  if (std::any_of(component_spec->resource().begin(),
                  component_spec->resource().end(),
                  [](const Resource &resource) {
                    return resource.name() == kMyelinFlowResourceName;
                  })) {
    return tensorflow::errors::InvalidArgument(
        "Component '", component_spec->name(),
        "' already contains a Myelin Flow resource");
  }

  // Success; make modifications.
  Resource *resource = component_spec->add_resource();
  resource->set_name(kMyelinFlowResourceName);
  Part *part = resource->add_part();
  part->set_file_pattern(path);
  part->set_file_format(kMyelinFlowResourceFileFormat);
  part->set_record_format(kMyelinFlowResourceRecordFormat);
  return tensorflow::Status::OK();
}

tensorflow::Status LoadMyelinFlow(const string &flow_path,
                                  sling::myelin::Flow *flow) {
  sling::File::Init();
  const sling::Status status = flow->Load(flow_path);
  if (!status.ok()) {
    return tensorflow::errors::Internal("Failed to load Myelin Flow from '",
                                        flow_path, ": ", status.ToString());
  }

  // Mark cell inputs and outputs.
  for (sling::myelin::Flow::Variable *variable : flow->vars()) {
    for (tensorflow::StringPiece alias : variable->aliases) {
      if (tensorflow::str_util::StartsWith(alias, "INPUT/")) {
        variable->in = true;
      }
      if (tensorflow::str_util::StartsWith(alias, "OUTPUT/")) {
        variable->out = true;
      }
    }
  }

  return tensorflow::Status::OK();
}

void RegisterMyelinLibraries(sling::myelin::Library *library) {
  // TODO(googleuser): Add more libraries?
  sling::myelin::RegisterTensorflowLibrary(library);
  library->RegisterTransformer(new PreMultipliedEmbeddings());
}

std::set<string> GetRecurrentLayerNames(const sling::myelin::Flow &flow) {
  std::set<string> names;
  for (const sling::myelin::Flow::Variable *variable : flow.vars()) {
    for (tensorflow::StringPiece alias : variable->aliases) {
      if (!tensorflow::str_util::ConsumePrefix(&alias, "INPUT/")) continue;
      if (tensorflow::str_util::ConsumePrefix(&alias, "fixed_channel_")) {
        continue;
      }
      if (tensorflow::str_util::ConsumePrefix(&alias, "linked_channel_")) {
        continue;
      }
      names.insert(alias.ToString());
    }
  }
  return names;
}

std::set<string> GetOutputLayerNames(const sling::myelin::Flow &flow) {
  std::set<string> names;
  for (const sling::myelin::Flow::Variable *variable : flow.vars()) {
    for (tensorflow::StringPiece alias : variable->aliases) {
      if (!tensorflow::str_util::ConsumePrefix(&alias, "OUTPUT/")) continue;
      names.insert(alias.ToString());
    }
  }
  return names;
}

string MakeMyelinInputFixedFeatureIdName(int channel_id, int index) {
  return tensorflow::strings::StrCat(
      "INPUT/fixed_channel_", channel_id, "_index_", index, "_ids");
}

string MakeMyelinInputLinkedActivationVectorName(int channel_id) {
  return tensorflow::strings::StrCat("INPUT/linked_channel_", channel_id,
                                     "_activations");
}

string MakeMyelinInputLinkedOutOfBoundsIndicatorName(int channel_id) {
  return tensorflow::strings::StrCat("INPUT/linked_channel_", channel_id,
                                     "_out_of_bounds");
}

string MakeMyelinInputRecurrentLayerName(const string &layer_name) {
  return tensorflow::strings::StrCat("INPUT/", layer_name);
}

string MakeMyelinOutputLayerName(const string &layer_name) {
  return tensorflow::strings::StrCat("OUTPUT/", layer_name);
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
