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
#include "dragnn/protos/trace.pb.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/fixed_embeddings.h"
#include "dragnn/runtime/linked_embeddings.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/myelin/myelin_dynamic_component_base.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/session_state.h"
#include "syntaxnet/base.h"
#include "sling/myelin/compute.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// A Myelin-based version of DynamicComponent.

//
// This implementation of MyelinDynamicComponentBase has the most generality
// w.r.t. input features and links, but suffers from ComputeSession overhead.
class MyelinDynamicComponent : public MyelinDynamicComponentBase {
 public:
  // Implements Component.
  tensorflow::Status Evaluate(SessionState *session_state,
                              ComputeSession *compute_session,
                              ComponentTrace *component_trace) const override;

 protected:
  // Unlike other specializations, this component will only be active if the
  // spec is explicitly modified to support Myelin (and flow resources are
  // generated).
  bool Supports(const ComponentSpec &spec,
                const string &normalized_builder_name) const override {
    return normalized_builder_name == "MyelinDynamicComponent";
  }
  bool PreferredTo(const Component &other) const override { return false; }

 private:
  // Forbid batches and beams.
  static constexpr int kEvaluateNumItems = 1;
};

tensorflow::Status MyelinDynamicComponent::Evaluate(
    SessionState *session_state, ComputeSession *compute_session,
    ComponentTrace *component_trace) const {
  NetworkStates &network_states = session_state->network_states;
  FixedEmbeddings &fixed_embeddings = GetFixedEmbeddings(session_state);
  LinkedEmbeddings &linked_embeddings = GetLinkedEmbeddings(session_state);

  sling::myelin::Instance &instance = GetInstance(session_state);
  for (size_t step_index = 0; !compute_session->IsTerminal(name());
       ++step_index) {
    network_states.AddStep();
    TF_RETURN_IF_ERROR(fixed_embeddings.Reset(&fixed_embedding_manager(),
                                              network_states, compute_session));
    TF_RETURN_IF_ERROR(linked_embeddings.Reset(
        &linked_embedding_manager(), network_states, compute_session));

    // Bind inputs and outputs into the |instance|.
    BindInputIds(fixed_embeddings, &instance);
    BindInputLinks(linked_embeddings, &instance);
    BindInputRecurrences(step_index, network_states, &instance);
    BindOutputLayers(step_index, network_states, &instance);

    // Invoke the cell in the |instance|.
    instance.Compute();
    MaybeTrace(step_index, &instance, component_trace);

    // If the component is deterministic, take the oracle transition instead of
    // predicting the next transition using the logits.
    if (deterministic()) {
      compute_session->AdvanceFromOracle(name());
    } else {
      // AddStep() may invalidate the logits (due to reallocation), so the layer
      // lookup cannot be hoisted out of this loop.
      const Vector<float> logits(
          network_states.GetLayer(logits_handle()).row(step_index));
      if (!compute_session->AdvanceFromPrediction(
              name(), logits.data(), kEvaluateNumItems, logits.size())) {
        return tensorflow::errors::Internal(
            "Error in ComputeSession::AdvanceFromPrediction()");
      }
    }
  }

  return tensorflow::Status::OK();
}

DRAGNN_RUNTIME_REGISTER_COMPONENT(MyelinDynamicComponent);

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
