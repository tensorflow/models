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

#include "dragnn/runtime/sequence_extractor.h"

#include "tensorflow/core/lib/core/errors.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

tensorflow::Status SequenceExtractor::Select(
    const FixedFeatureChannel &channel, const ComponentSpec &component_spec,
    string *name) {
  string supporting_name;
  for (const Registry::Registrar *registrar = registry()->components;
       registrar != nullptr; registrar = registrar->next()) {
    Factory *factory_function = registrar->object();
    std::unique_ptr<SequenceExtractor> current_extractor(factory_function());
    if (!current_extractor->Supports(channel, component_spec)) continue;

    if (!supporting_name.empty()) {
      return tensorflow::errors::Internal(
          "Multiple SequenceExtractors support channel ",
          channel.ShortDebugString(), " of ComponentSpec (", supporting_name,
          " and ", registrar->name(), "): ", component_spec.ShortDebugString());
    }

    supporting_name = registrar->name();
  }

  if (supporting_name.empty()) {
    return tensorflow::errors::NotFound(
        "No SequenceExtractor supports channel ", channel.ShortDebugString(),
        " of ComponentSpec: ", component_spec.ShortDebugString());
  }

  // Success; make modifications.
  *name = supporting_name;
  return tensorflow::Status::OK();
}

tensorflow::Status SequenceExtractor::New(
    const string &name, const FixedFeatureChannel &channel,
    const ComponentSpec &component_spec,
    std::unique_ptr<SequenceExtractor> *extractor) {
  std::unique_ptr<SequenceExtractor> matching_extractor;
  TF_RETURN_IF_ERROR(
      SequenceExtractor::CreateOrError(name, &matching_extractor));
  TF_RETURN_IF_ERROR(matching_extractor->Initialize(channel, component_spec));

  // Success; make modifications.
  *extractor = std::move(matching_extractor);
  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn

REGISTER_SYNTAXNET_CLASS_REGISTRY("DRAGNN Runtime Sequence Extractor",
                                  dragnn::runtime::SequenceExtractor);

}  // namespace syntaxnet
