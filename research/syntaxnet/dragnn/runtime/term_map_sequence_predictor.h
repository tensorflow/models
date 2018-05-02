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

#ifndef DRAGNN_RUNTIME_TERM_MAP_SEQUENCE_PREDICTOR_H_
#define DRAGNN_RUNTIME_TERM_MAP_SEQUENCE_PREDICTOR_H_

#include <string>

#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/sequence_predictor.h"
#include "syntaxnet/base.h"
#include "syntaxnet/term_frequency_map.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Base class for predictors whose output label set is defined by a term map.
// Requires the component to have a TermFrequencyMap resource.
class TermMapSequencePredictor : public SequencePredictor {
 public:
  // Creates a sequence predictor that will load a term map from the resource
  // named |resource_name|.
  explicit TermMapSequencePredictor(const string &resource_name);
  ~TermMapSequencePredictor() override;

  // Returns true if the |component_spec| is compatible with this.  Subclasses
  // should call this from their Supports().
  bool SupportsTermMap(const ComponentSpec &component_spec) const;

  // Loads a term map from the |component_spec|, applying the |min_frequency|
  // and |max_num_terms| when loading the term map.  On error, returns non-OK.
  // Subclasses should call this from their Initialize().
  tensorflow::Status InitializeTermMap(const ComponentSpec &component_spec,
                                       int min_frequency, int max_num_terms);

 protected:
  // Returns the current term map.  Only valid after InitializeTermMap().
  const TermFrequencyMap &term_map() const { return *term_map_; }

 private:
  // Name of the resouce from which to load a term map.
  const string resource_name_;

  // Mapping from strings to feature IDs.  Owned by SharedStore.
  const TermFrequencyMap *term_map_ = nullptr;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_TERM_MAP_SEQUENCE_PREDICTOR_H_
