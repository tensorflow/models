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

#ifndef DRAGNN_RUNTIME_TRANSITION_SYSTEM_TRAITS_H_
#define DRAGNN_RUNTIME_TRANSITION_SYSTEM_TRAITS_H_

#include "dragnn/protos/spec.pb.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Traits describing the transition system used by some component.
struct TransitionSystemTraits {
  // Creates a set of traits describing the |component_spec|.
  explicit TransitionSystemTraits(const ComponentSpec &component_spec);

  // Whether the transition system is deterministic---i.e., it can be advanced
  // without computing logits and making predictions.
  const bool is_deterministic;

  // Whether the transition system is sequential---i.e., compatible with
  // SequenceBackend, SequenceExtractor, and so on.
  const bool is_sequential;

  // Whether the transition system advances from left to right in the underlying
  // input sequence.  This only makes sense if |sequential| is true.
  const bool is_left_to_right;

  // Whether the transition steps correspond to characters or tokens.  This only
  // makes sense if |sequential| is true.
  //
  // TODO(googleuser): Distinguish between full-text character transition systems
  // and per-word ones?
  const bool is_character_scale;
  const bool is_token_scale;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_TRANSITION_SYSTEM_TRAITS_H_
