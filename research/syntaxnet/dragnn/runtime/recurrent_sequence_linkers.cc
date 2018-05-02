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

// Links to the previous step in the same component.  Templated on a bool that
// indicates the direction that the transition system runs in.
template <bool left_to_right>
class RecurrentSequenceLinker : public SequenceLinker {
 public:
  // Implements SequenceLinker.
  bool Supports(const LinkedFeatureChannel &channel,
                const ComponentSpec &component_spec) const override;
  tensorflow::Status Initialize(const LinkedFeatureChannel &channel,
                                const ComponentSpec &component_spec) override;
  tensorflow::Status GetLinks(size_t source_num_steps, InputBatchCache *input,
                              std::vector<int32> *links) const override;
};

template <bool left_to_right>
bool RecurrentSequenceLinker<left_to_right>::Supports(
    const LinkedFeatureChannel &channel,
    const ComponentSpec &component_spec) const {
  TransitionSystemTraits traits(component_spec);

  // Here, fml="bias" and source_translator="history" are a DRAGNN recipe for
  // linking to the previous transition step.  More concretely,
  //   * "bias" always extracts index 0.
  //   * "history" subtracts the index it is given from (#steps - 1).
  // Putting the two together, we link to (#steps - 1 - 0); i.e., the previous
  // transition step.
  return (channel.fml() == "bias" || channel.fml() == "bias(0)") &&
         channel.source_component() == component_spec.name() &&
         channel.source_translator() == "history" &&
         traits.is_left_to_right == left_to_right && traits.is_sequential;
}

template <bool left_to_right>
tensorflow::Status RecurrentSequenceLinker<left_to_right>::Initialize(
    const LinkedFeatureChannel &channel, const ComponentSpec &component_spec) {
  return tensorflow::Status::OK();
}

template <bool left_to_right>
tensorflow::Status RecurrentSequenceLinker<left_to_right>::GetLinks(
    size_t source_num_steps, InputBatchCache *input,
    std::vector<int32> *links) const {
  links->resize(source_num_steps);

  if (left_to_right) {
    int32 index = -1;
    for (int32 &link : *links) link = index++;
  } else {
    int32 index = static_cast<int32>(source_num_steps) - 1;
    for (int32 &link : *links) link = --index;
  }

  return tensorflow::Status::OK();
}

using LeftToRightRecurrentSequenceLinker = RecurrentSequenceLinker<true>;
using RightToLeftRecurrentSequenceLinker = RecurrentSequenceLinker<false>;

DRAGNN_RUNTIME_REGISTER_SEQUENCE_LINKER(LeftToRightRecurrentSequenceLinker);
DRAGNN_RUNTIME_REGISTER_SEQUENCE_LINKER(RightToLeftRecurrentSequenceLinker);

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
