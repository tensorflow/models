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

// The DynamicComponent is the runtime analogue of the DynamicComponentBuilder
// in the Python codebase.  The role of the DynamicComponent is to manage the
// loop over transition steps, including:
//   * Allocating stepwise memory for network states and operands.
//   * Performing some computation at each step.
//   * Advancing the transition state until terminal.
//
// Note that the number of transition taken on any given evaluation of the
// DynamicComponent cannot be determined in advance.
//
// The core computational work is delegated to a NetworkUnit, which is evaluated
// at each transition step.  This makes the DynamicComponent flexible, since it
// can be applied to any NetworkUnit implementation, but it can be significantly
// more efficient to use a task-specific component implementation.  For example,
// the "shift-only" transition system merely scans the input tokens, and in that
// case we could replace the incremental loop with a "bulk" computation.

#include <stddef.h>
#include <string>
#include <vector>

#include "dragnn/core/compute_session.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/network_unit.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/transition_system_traits.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Performs an incremental computation, one transition at a time.
class DynamicComponent : public Component {
 protected:
  // Implements Component.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override;
  tensorflow::Status Evaluate(SessionState *session_state,
                              ComputeSession *compute_session,
                              ComponentTrace *component_trace) const override;

  // This class is intended to support all DynamicComponent layers. We currently
  // prefer to return `true` here and throw errors in Initialize() if a
  // particular feature is not supported.
  bool Supports(const ComponentSpec &spec,
                const string &normalized_builder_name) const override {
    return normalized_builder_name == "DynamicComponent";
  }

  // This class is not optimized, so any other supported subclasses of Component
  // should be preferred.
  bool PreferredTo(const Component &other) const override { return false; }

 private:
  // Name of this component.
  string name_;

  // Network unit that produces logits.
  std::unique_ptr<NetworkUnit> network_unit_;

  // Whether the transition system is deterministic.
  bool deterministic_ = false;

  // Handle to the network unit logits.  Valid iff |deterministic_| is false.
  LayerHandle<float> logits_handle_;
};

tensorflow::Status DynamicComponent::Initialize(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    NetworkStateManager *network_state_manager,
    ExtensionManager *extension_manager) {
  name_ = component_spec.name();
  if (!component_spec.attention_component().empty()) {
    return tensorflow::errors::Unimplemented("Attention is not supported");
  }

  TF_RETURN_IF_ERROR(NetworkUnit::CreateOrError(
      NetworkUnit::GetClassName(component_spec), &network_unit_));
  TF_RETURN_IF_ERROR(network_unit_->Initialize(component_spec, variable_store,
                                               network_state_manager,
                                               extension_manager));

  // Logits are unnecesssary when the component is deterministic.
  deterministic_ = TransitionSystemTraits(component_spec).is_deterministic;
  if (!deterministic_) {
    const string logits_name = network_unit_->GetLogitsName();
    if (logits_name.empty()) {
      return tensorflow::errors::InvalidArgument(
          "Network unit does not produce logits: ",
          component_spec.network_unit().ShortDebugString());
    }

    size_t dimension = 0;
    TF_RETURN_IF_ERROR(network_state_manager->LookupLayer(
        name_, logits_name, &dimension, &logits_handle_));

    if (dimension != component_spec.num_actions()) {
      return tensorflow::errors::InvalidArgument(
          "Dimension mismatch between network unit logits (", dimension,
          ") and ComponentSpec.num_actions (", component_spec.num_actions(),
          ") in component '", name_, "'");
    }
  }

  return tensorflow::Status::OK();
}

// No batches or beams.
constexpr int kNumItems = 1;

tensorflow::Status DynamicComponent::Evaluate(
    SessionState *session_state, ComputeSession *compute_session,
    ComponentTrace *component_trace) const {
  NetworkStates &network_states = session_state->network_states;
  for (size_t step_index = 0; !compute_session->IsTerminal(name_);
       ++step_index) {
    network_states.AddStep();
    TF_RETURN_IF_ERROR(
        network_unit_->Evaluate(step_index, session_state, compute_session));

    // If the component is deterministic, take the oracle transition instead of
    // predicting the next transition using the logits.
    if (deterministic_) {
      compute_session->AdvanceFromOracle(name_);
    } else {
      // AddStep() may invalidate the logits (due to reallocation), so the layer
      // lookup cannot be hoisted out of this loop.
      const Vector<float> logits(
          network_states.GetLayer(logits_handle_).row(step_index));
      if (!compute_session->AdvanceFromPrediction(name_, logits.data(),
                                                  kNumItems, logits.size())) {
        return tensorflow::errors::Internal(
            "Error in ComputeSession::AdvanceFromPrediction()");
      }
    }
  }

  return tensorflow::Status::OK();
}

DRAGNN_RUNTIME_REGISTER_COMPONENT(DynamicComponent);

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
