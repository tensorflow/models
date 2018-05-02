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

#ifndef DRAGNN_RUNTIME_SEQUENCE_LINKER_H_
#define DRAGNN_RUNTIME_SEQUENCE_LINKER_H_

#include <stddef.h>
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

// Interface for link extraction for sequence inputs.

//
// This can be used to avoid ComputeSession overhead in simple cases; for
// example, extracting a sequence of identity or reverse-identity links.
class SequenceLinker : public RegisterableClass<SequenceLinker> {
 public:
  // Sets |linker| to an instance of the subclass named |name| initialized from
  // the |channel| of the |component_spec|.  On error, returns non-OK and
  // modifies nothing.
  static tensorflow::Status New(const string &name,
                                const LinkedFeatureChannel &channel,
                                const ComponentSpec &component_spec,
                                std::unique_ptr<SequenceLinker> *linker);

  SequenceLinker(const SequenceLinker &) = delete;
  SequenceLinker &operator=(const SequenceLinker &) = delete;
  virtual ~SequenceLinker() = default;

  // Sets |name| to the registered name of the SequenceLinker that supports the
  // |channel| of the |component_spec|.  On error, returns non-OK and modifies
  // nothing.  The returned statuses include:
  // * OK: If a supporting SequenceLinker was found.
  // * INTERNAL: If an error occurred while searching for a compatible match.
  // * NOT_FOUND: If the search was error-free, but no compatible match was
  //              found.
  static tensorflow::Status Select(const LinkedFeatureChannel &channel,
                                   const ComponentSpec &component_spec,
                                   string *name);

  // Overwrites |links| with the sequence of translated link step indices for
  // the |input|.  Specifically, sets links[i] to the (possibly out-of-bounds)
  // step index to fetch from the source component for the i'th element of the
  // target sequence.  Assumes that |source_num_steps| is the number of steps
  // taken by the source component.  On error, returns non-OK.
  virtual tensorflow::Status GetLinks(size_t source_num_steps,
                                      InputBatchCache *input,
                                      std::vector<int32> *links) const = 0;

 protected:
  SequenceLinker() = default;

 private:
  // Helps prevent use of the Create() method; use New() instead.
  using RegisterableClass<SequenceLinker>::Create;

  // Returns true if this supports the |channel| of the |component_spec|.
  // Implementations must coordinate to ensure that at most one supports any
  // given |component_spec|.
  virtual bool Supports(const LinkedFeatureChannel &channel,
                        const ComponentSpec &component_spec) const = 0;

  // Initializes this from the |channel| of the |component_spec|.  On error,
  // returns non-OK.
  virtual tensorflow::Status Initialize(
      const LinkedFeatureChannel &channel,
      const ComponentSpec &component_spec) = 0;
};

}  // namespace runtime
}  // namespace dragnn

DECLARE_SYNTAXNET_CLASS_REGISTRY("DRAGNN Runtime Sequence Linker",
                                 dragnn::runtime::SequenceLinker);

}  // namespace syntaxnet

#define DRAGNN_RUNTIME_REGISTER_SEQUENCE_LINKER(subclass) \
  REGISTER_SYNTAXNET_CLASS_COMPONENT(                     \
      ::syntaxnet::dragnn::runtime::SequenceLinker, #subclass, subclass)

#endif  // DRAGNN_RUNTIME_SEQUENCE_LINKER_H_
