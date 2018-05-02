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

#ifndef DRAGNN_RUNTIME_SEQUENCE_EXTRACTOR_H_
#define DRAGNN_RUNTIME_SEQUENCE_EXTRACTOR_H_

#include <memory>
#include <string>
#include <vector>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/protos/spec.pb.h"
#include "syntaxnet/base.h"
#include "syntaxnet/registry.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Interface for feature extraction for sequence inputs.

//
// This extractor can be used to avoid ComputeSession overhead in simple cases;
// for example, extracting a sequence of character or word IDs for an LSTM.
class SequenceExtractor : public RegisterableClass<SequenceExtractor> {
 public:
  // Sets |extractor| to an instance of the subclass named |name| initialized
  // from the |channel| of the |component_spec|.  On error, returns non-OK and
  // modifies nothing.
  static tensorflow::Status New(const string &name,
                                const FixedFeatureChannel &channel,
                                const ComponentSpec &component_spec,
                                std::unique_ptr<SequenceExtractor> *extractor);

  SequenceExtractor(const SequenceExtractor &) = delete;
  SequenceExtractor &operator=(const SequenceExtractor &) = delete;
  virtual ~SequenceExtractor() = default;

  // Sets |name| to the registered name of the SequenceExtractor that supports
  // the |channel| of the |component_spec|.  On error, returns non-OK and
  // modifies nothing.  The returned statuses include:
  // * OK: If a supporting SequenceExtractor was found.
  // * INTERNAL: If an error occurred while searching for a compatible match.
  // * NOT_FOUND: If the search was error-free, but no compatible match was
  //              found.
  static tensorflow::Status Select(const FixedFeatureChannel &channel,
                                   const ComponentSpec &component_spec,
                                   string *name);

  // Overwrites |ids| with the sequence of features extracted from the |input|.
  // On error, returns non-OK.
  virtual tensorflow::Status GetIds(InputBatchCache *input,
                                    std::vector<int32> *ids) const = 0;

 protected:
  SequenceExtractor() = default;

 private:
  // Helps prevent use of the Create() method; use New() instead.
  using RegisterableClass<SequenceExtractor>::Create;

  // Returns true if this supports the |channel| of the |component_spec|.
  // Implementations must coordinate to ensure that at most one supports any
  // given |component_spec|.
  virtual bool Supports(const FixedFeatureChannel &channel,
                        const ComponentSpec &component_spec) const = 0;

  // Initializes this from the |channel| of the |component_spec|.  On error,
  // returns non-OK.
  virtual tensorflow::Status Initialize(
      const FixedFeatureChannel &channel,
      const ComponentSpec &component_spec) = 0;
};

}  // namespace runtime
}  // namespace dragnn

DECLARE_SYNTAXNET_CLASS_REGISTRY("DRAGNN Runtime Sequence Extractor",
                                 dragnn::runtime::SequenceExtractor);

}  // namespace syntaxnet

#define DRAGNN_RUNTIME_REGISTER_SEQUENCE_EXTRACTOR(subclass) \
  REGISTER_SYNTAXNET_CLASS_COMPONENT(                        \
      ::syntaxnet::dragnn::runtime::SequenceExtractor, #subclass, subclass)

#endif  // DRAGNN_RUNTIME_SEQUENCE_EXTRACTOR_H_
