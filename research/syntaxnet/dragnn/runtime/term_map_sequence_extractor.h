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

#ifndef DRAGNN_RUNTIME_TERM_MAP_SEQUENCE_EXTRACTOR_H_
#define DRAGNN_RUNTIME_TERM_MAP_SEQUENCE_EXTRACTOR_H_

#include <string>

#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/sequence_extractor.h"
#include "dragnn/runtime/term_map_utils.h"
#include "syntaxnet/base.h"
#include "syntaxnet/shared_store.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Base class for TermFrequencyMap-based sequence feature extractors.  Requires
// the component to have a single fixed feature and a TermFrequencyMap resource.
// Templated on a |TermMap| type, which should have a 3-arg constructor similar
// to TermFrequencyMap's.
template <class TermMap>
class TermMapSequenceExtractor : public SequenceExtractor {
 public:
  // Creates a sequence extractor that will load a term map from the resource
  // named |resource_name|.
  explicit TermMapSequenceExtractor(const string &resource_name);
  ~TermMapSequenceExtractor() override;

  // Returns true if the |channel| of the |component_spec| is compatible with
  // this.  Subclasses should call this from their Supports().
  bool SupportsTermMap(const FixedFeatureChannel &channel,
                       const ComponentSpec &component_spec) const;

  // Loads a term map from the |channel| of the |component_spec|, applying the
  // |min_frequency| and |max_num_terms| when loading the term map.  On error,
  // returns non-OK.  Subclasses should call this from their Initialize().
  tensorflow::Status InitializeTermMap(const FixedFeatureChannel &channel,
                                       const ComponentSpec &component_spec,
                                       int min_frequency, int max_num_terms);

 protected:
  // Returns the current term map.  Only valid after InitializeTermMap().
  const TermMap &term_map() const { return *term_map_; }

 private:
  // Name of the resouce from which to load a term map.
  const string resource_name_;

  // Mapping from terms to feature IDs.  Owned by SharedStore.
  const TermMap *term_map_ = nullptr;
};

// Implementation details below.

template <class TermMap>
TermMapSequenceExtractor<TermMap>::TermMapSequenceExtractor(
    const string &resource_name)
    : resource_name_(resource_name) {}

template <class TermMap>
TermMapSequenceExtractor<TermMap>::~TermMapSequenceExtractor() {
  if (!SharedStore::Release(term_map_)) {
    LOG(ERROR) << "Failed to release term map for resource " << resource_name_;
  }
}

template <class TermMap>
bool TermMapSequenceExtractor<TermMap>::SupportsTermMap(
    const FixedFeatureChannel &channel,
    const ComponentSpec &component_spec) const {
  return LookupTermMapResourcePath(resource_name_, component_spec) != nullptr &&
         channel.size() == 1;
}

template <class TermMap>
tensorflow::Status TermMapSequenceExtractor<TermMap>::InitializeTermMap(
    const FixedFeatureChannel &channel, const ComponentSpec &component_spec,
    int min_frequency, int max_num_terms) {
  const string *path =
      LookupTermMapResourcePath(resource_name_, component_spec);
  if (path == nullptr) {
    return tensorflow::errors::InvalidArgument(
        "No compatible resource named '", resource_name_,
        "' in ComponentSpec: ", component_spec.ShortDebugString());
  }

  term_map_ = SharedStoreUtils::GetWithDefaultName<TermMap>(
      *path, min_frequency, max_num_terms);

  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_TERM_MAP_SEQUENCE_EXTRACTOR_H_
