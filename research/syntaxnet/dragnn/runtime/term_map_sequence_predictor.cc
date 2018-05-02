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

#include "dragnn/runtime/term_map_sequence_predictor.h"

#include "dragnn/runtime/term_map_utils.h"
#include "syntaxnet/shared_store.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

TermMapSequencePredictor::TermMapSequencePredictor(const string &resource_name)
    : resource_name_(resource_name) {}

TermMapSequencePredictor::~TermMapSequencePredictor() {
  if (!SharedStore::Release(term_map_)) {
    LOG(ERROR) << "Failed to release term map for resource " << resource_name_;
  }
}

bool TermMapSequencePredictor::SupportsTermMap(
    const ComponentSpec &component_spec) const {
  return LookupTermMapResourcePath(resource_name_, component_spec) != nullptr;
}

tensorflow::Status TermMapSequencePredictor::InitializeTermMap(
    const ComponentSpec &component_spec, int min_frequency, int max_num_terms) {
  const string *path =
      LookupTermMapResourcePath(resource_name_, component_spec);
  if (path == nullptr) {
    return tensorflow::errors::InvalidArgument(
        "No compatible resource named '", resource_name_,
        "' in ComponentSpec: ", component_spec.ShortDebugString());
  }

  term_map_ = SharedStoreUtils::GetWithDefaultName<TermFrequencyMap>(
      *path, min_frequency, max_num_terms);

  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
