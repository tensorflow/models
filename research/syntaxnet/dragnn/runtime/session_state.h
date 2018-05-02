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

#ifndef DRAGNN_RUNTIME_SESSION_STATE_H_
#define DRAGNN_RUNTIME_SESSION_STATE_H_

#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/network_states.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// State associated with a ComputeSession being evaluated by a DRAGNN network,
// reusable across multiple evaluations.  Unlike the ComputeSession, which is
// both the input and output of the network, this state is strictly internal to
// the network.  Production code should allocate these via a SessionStatePool.
struct SessionState {
  // The network states that connect the pipeline of components.
  NetworkStates network_states;

  // Generic set of typed extensions.
  Extensions extensions;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_SESSION_STATE_H_
