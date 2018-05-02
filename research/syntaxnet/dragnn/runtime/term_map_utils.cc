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

#include "dragnn/runtime/term_map_utils.h"

#include "dragnn/runtime/fml_parsing.h"
#include "syntaxnet/feature_extractor.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Attributes for extracting term map feature options
struct TermMapAttributes : public FeatureFunctionAttributes {
  // Minimum frequency for included terms.
  Optional<int32> min_frequency{"min-freq", 0, this};

  // Maximum number of terms to include.
  Optional<int32> max_num_terms{"max-num-terms", 0, this};
};

// Returns true if the |record_format| is compatible with a TermFrequencyMap.
bool CompatibleRecordFormat(const string &record_format) {
  return record_format.empty() || record_format == "TermFrequencyMap";
}

}  // namespace

const string *LookupTermMapResourcePath(const string &resource_name,
                                        const ComponentSpec &component_spec) {
  for (const Resource &resource : component_spec.resource()) {
    if (resource.name() != resource_name) continue;
    if (resource.part_size() != 1) continue;
    const Part &part = resource.part(0);
    if (part.file_format() != "text") continue;
    if (!CompatibleRecordFormat(part.record_format())) continue;
    return &part.file_pattern();
  }
  return nullptr;
}

tensorflow::Status ParseTermMapFml(const string &fml,
                                   const std::vector<string> &types,
                                   int *min_frequency, int *max_num_terms) {
  FeatureFunctionDescriptor function;
  TF_RETURN_IF_ERROR(ParseFeatureChainFml(fml, types, &function));
  if (function.argument() != 0) {
    return tensorflow::errors::InvalidArgument(
        "TermFrequencyMap-based feature should have no argument: ", fml);
  }

  TermMapAttributes attributes;
  TF_RETURN_IF_ERROR(attributes.Reset(function));

  // Success; make modifications.
  *min_frequency = attributes.min_frequency();
  *max_num_terms = attributes.max_num_terms();
  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
