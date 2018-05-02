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
#include <string>

#include "dragnn/core/compute_session.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/feed_forward_network_kernel.h"
#include "dragnn/runtime/feed_forward_network_layer.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/network_unit.h"
#include "dragnn/runtime/network_unit_base.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// A network unit that evaluates a feed-forward multi-layer perceptron.
class FeedForwardNetwork : public NetworkUnitBase {
 public:
  // Implements NetworkUnit.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override;
  string GetLogitsName() const override { return kernel_.logits_name(); }
  tensorflow::Status Evaluate(size_t step_index, SessionState *session_state,
                              ComputeSession *compute_session) const override;

 private:
  // Kernel that implements the feed-forward network.
  FeedForwardNetworkKernel kernel_;
};

tensorflow::Status FeedForwardNetwork::Initialize(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    NetworkStateManager *network_state_manager,
    ExtensionManager *extension_manager) {
  TF_RETURN_IF_ERROR(kernel_.Initialize(component_spec, variable_store,
                                        network_state_manager));

  const bool use_concatenated_input = true;
  TF_RETURN_IF_ERROR(InitializeBase(use_concatenated_input, component_spec,
                                    variable_store, network_state_manager,
                                    extension_manager));

  // Check dimensions across layers.  This must be done after InitializeBase(),
  // when concatenated_input_dim() is known.
  return kernel_.ValidateInputDimension(concatenated_input_dim());
}

tensorflow::Status FeedForwardNetwork::Evaluate(
    size_t step_index, SessionState *session_state,
    ComputeSession *compute_session) const {
  Vector<float> input;
  TF_RETURN_IF_ERROR(EvaluateBase(session_state, compute_session, &input));
  for (const FeedForwardNetworkLayer &layer : kernel_.layers()) {
    input = layer.Apply(input, session_state->network_states, step_index);
  }
  return tensorflow::Status::OK();
}

DRAGNN_RUNTIME_REGISTER_NETWORK_UNIT(FeedForwardNetwork);

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
