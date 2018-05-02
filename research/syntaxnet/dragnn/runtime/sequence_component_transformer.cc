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

#include <string>

#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/component_transformation.h"
#include "dragnn/runtime/sequence_extractor.h"
#include "dragnn/runtime/sequence_linker.h"
#include "dragnn/runtime/sequence_predictor.h"
#include "dragnn/runtime/transition_system_traits.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Returns true if the |component_spec| has recurrent links.
bool IsRecurrent(const ComponentSpec &component_spec) {
  for (const LinkedFeatureChannel &channel : component_spec.linked_feature()) {
    if (channel.source_component() == component_spec.name()) return true;
  }
  return false;
}

// Returns the sequence-based version of the |component_type| with specification
// |component_spec|, or an empty string if there is no sequence-based version.
string GetSequenceComponentType(const string &component_type,
                                const ComponentSpec &component_spec) {
  // TODO(googleuser): Implement a SequenceDynamicComponent that can handle
  // recurrent links.  This may require changes to the NetworkUnit API.
  static const char *kSupportedComponentTypes[] = {
      "BulkDynamicComponent",    //
      "BulkLstmComponent",       //
      "MyelinDynamicComponent",  //

  };
  for (const char *supported_type : kSupportedComponentTypes) {
    if (component_type == supported_type) {
      return tensorflow::strings::StrCat("Sequence", supported_type);
    }
  }

  // Also support non-recurrent DynamicComponents.  The BulkDynamicComponent
  // requires determinism, but the SequenceBulkDynamicComponent does not, so
  // it's not sufficient to only upgrade from BulkDynamicComponent.
  if (component_type == "DynamicComponent" && !IsRecurrent(component_spec)) {
    return "SequenceBulkDynamicComponent";
  }

  return string();
}

// Returns the |status| but coerces NOT_FOUND to OK.  Sets |found| to false iff
// the |status| was NOT_FOUND.
tensorflow::Status AllowNotFound(const tensorflow::Status &status,
                                 bool *found) {
  *found = status.code() != tensorflow::error::NOT_FOUND;
  return *found ? status : tensorflow::Status::OK();
}

// Transformer that checks whether a sequence-based component implementation
// could be used and, if compatible, modifies the ComponentSpec accordingly.
class SequenceComponentTransformer : public ComponentTransformer {
 public:
  // Implements ComponentTransformer.
  tensorflow::Status Transform(const string &component_type,
                               ComponentSpec *component_spec) override;
};

tensorflow::Status SequenceComponentTransformer::Transform(
    const string &component_type, ComponentSpec *component_spec) {
  const int num_features = component_spec->fixed_feature_size() +
                           component_spec->linked_feature_size();
  if (num_features == 0) return tensorflow::Status::OK();

  // Look for supporting SequenceExtractors.
  bool found = false;
  string extractor_types;
  for (const FixedFeatureChannel &channel : component_spec->fixed_feature()) {
    string type;
    TF_RETURN_IF_ERROR(AllowNotFound(
        SequenceExtractor::Select(channel, *component_spec, &type), &found));
    if (!found) return tensorflow::Status::OK();
    tensorflow::strings::StrAppend(&extractor_types, type, ",");
  }
  if (!extractor_types.empty()) extractor_types.pop_back();  // remove comma

  // Look for supporting SequenceLinkers.
  string linker_types;
  for (const LinkedFeatureChannel &channel : component_spec->linked_feature()) {
    string type;
    TF_RETURN_IF_ERROR(AllowNotFound(
        SequenceLinker::Select(channel, *component_spec, &type), &found));
    if (!found) return tensorflow::Status::OK();
    tensorflow::strings::StrAppend(&linker_types, type, ",");
  }
  if (!linker_types.empty()) linker_types.pop_back();  // remove comma

  // Look for a supporting SequencePredictor, if predictions are necessary.
  string predictor_type;
  if (!TransitionSystemTraits(*component_spec).is_deterministic) {
    TF_RETURN_IF_ERROR(AllowNotFound(
        SequencePredictor::Select(*component_spec, &predictor_type), &found));
    if (!found) return tensorflow::Status::OK();
  }

  // Look for a supporting sequence-based component type.
  const string sequence_component_type =
      GetSequenceComponentType(component_type, *component_spec);
  if (sequence_component_type.empty()) return tensorflow::Status::OK();

  // Success; make modifications.
  component_spec->mutable_backend()->set_registered_name("SequenceBackend");
  RegisteredModuleSpec *builder = component_spec->mutable_component_builder();
  builder->set_registered_name(sequence_component_type);
  (*builder->mutable_parameters())["sequence_extractors"] = extractor_types;
  (*builder->mutable_parameters())["sequence_linkers"] = linker_types;
  (*builder->mutable_parameters())["sequence_predictor"] = predictor_type;
  return tensorflow::Status::OK();
}

DRAGNN_RUNTIME_REGISTER_COMPONENT_TRANSFORMER(SequenceComponentTransformer);

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
