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

#include <stddef.h>
#include <vector>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/sequence_linker.h"
#include "dragnn/runtime/transition_system_traits.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Applies an identity function.
class IdentitySequenceLinker : public SequenceLinker {
 public:
  // Implements SequenceLinker.
  bool Supports(const LinkedFeatureChannel &channel,
                const ComponentSpec &component_spec) const override;
  tensorflow::Status Initialize(const LinkedFeatureChannel &channel,
                                const ComponentSpec &component_spec) override;
  tensorflow::Status GetLinks(size_t source_num_steps, InputBatchCache *input,
                              std::vector<int32> *links) const override;
};

bool IdentitySequenceLinker::Supports(
    const LinkedFeatureChannel &channel,
    const ComponentSpec &component_spec) const {
  TransitionSystemTraits traits(component_spec);

  // Note: Add more "||" clauses as needed.
  return (channel.fml() == "input.focus" ||
          channel.fml() == "char-input.focus") &&
         channel.source_translator() == "identity" && traits.is_sequential;
}

tensorflow::Status IdentitySequenceLinker::Initialize(
    const LinkedFeatureChannel &channel, const ComponentSpec &component_spec) {
  return tensorflow::Status::OK();
}

tensorflow::Status IdentitySequenceLinker::GetLinks(
    size_t source_num_steps, InputBatchCache *input,
    std::vector<int32> *links) const {
  links->resize(source_num_steps);
  int32 index = 0;
  for (int32 &link : *links) link = index++;
  return tensorflow::Status::OK();
}

DRAGNN_RUNTIME_REGISTER_SEQUENCE_LINKER(IdentitySequenceLinker);

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
