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
#include <vector>

#include "dragnn/core/compute_session.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/myelin/myelin_dynamic_component_base.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/sequence_features.h"
#include "dragnn/runtime/sequence_links.h"
#include "dragnn/runtime/sequence_model.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "sling/myelin/compute.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// A Myelin-based version of DynamicComponent for sequence-based models.

class SequenceMyelinDynamicComponent : public MyelinDynamicComponentBase {
 public:
  // Implements Component.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override;
  tensorflow::Status Evaluate(SessionState *session_state,
                              ComputeSession *compute_session,
                              ComponentTrace *component_trace) const override;

 protected:
  // Implements Component.
  bool Supports(const ComponentSpec &component_spec,
                const string &normalized_builder_name) const override;
  bool PreferredTo(const Component &) const override { return false; }

 private:
  // Binds the fixed feature IDs for the |target_index|'th element of the
  // |features| to the |instance|.  Uses locals in the |network_states|.
  void BindInputIds(const SequenceFeatures &features, int target_index,
                    const NetworkStates &network_states,
                    sling::myelin::Instance *instance) const;

  // Binds the linked embeddings for the |target_index|'th element in the
  // |links| to the |instance|.
  void BindInputLinks(const SequenceLinks &links, int target_index,
                      sling::myelin::Instance *instance) const;

  // Sequence-based model evaluator.
  SequenceModel sequence_model_;

  // Intermediate values used by sequence models.
  SharedExtensionHandle<SequenceModel::EvaluateState> evaluate_state_handle_;
};

bool SequenceMyelinDynamicComponent::Supports(
    const ComponentSpec &component_spec,
    const string &normalized_builder_name) const {
  return normalized_builder_name == "SequenceMyelinDynamicComponent" &&
         SequenceModel::Supports(component_spec);
}

tensorflow::Status SequenceMyelinDynamicComponent::Initialize(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    NetworkStateManager *network_state_manager,
    ExtensionManager *extension_manager) {
  // Initialize the base class first, so its FixedEmbeddingManager and
  // LinkedEmbeddingManager can be wrapped in sequence-based versions.
  TF_RETURN_IF_ERROR(MyelinDynamicComponentBase::Initialize(
      component_spec, variable_store, network_state_manager,
      extension_manager));

  TF_RETURN_IF_ERROR(sequence_model_.Initialize(
      component_spec, kLogitsName, &fixed_embedding_manager(),
      &linked_embedding_manager(), network_state_manager));

  extension_manager->GetShared(&evaluate_state_handle_);
  return tensorflow::Status::OK();
}

void SequenceMyelinDynamicComponent::BindInputIds(
    const SequenceFeatures &features, int target_index,
    const NetworkStates &network_states,
    sling::myelin::Instance *instance) const {
  for (size_t channel_id = 0; channel_id < features.num_channels();
       ++channel_id) {
    const MutableVector<int32> id_vector = network_states.GetLocal(
        fixed_embedding_manager().id_handle(channel_id, 0));
    id_vector[0] = features.GetId(channel_id, target_index);
    BindInput(Vector<int32>(id_vector), input_ids()[channel_id].id, instance);
  }
}

void SequenceMyelinDynamicComponent::BindInputLinks(
    const SequenceLinks &links, int target_index,
    sling::myelin::Instance *instance) const {
  Vector<float> embedding;
  bool is_out_of_bounds = false;
  for (size_t channel_id = 0; channel_id < links.num_channels(); ++channel_id) {
    links.Get(channel_id, target_index, &embedding, &is_out_of_bounds);
    BindInputLink(embedding, is_out_of_bounds, input_links()[channel_id],
                  instance);
  }
}

tensorflow::Status SequenceMyelinDynamicComponent::Evaluate(
    SessionState *session_state, ComputeSession *compute_session,
    ComponentTrace *component_trace) const {
  NetworkStates &network_states = session_state->network_states;
  SequenceModel::EvaluateState &state =
      session_state->extensions.Get(evaluate_state_handle_);
  TF_RETURN_IF_ERROR(
      sequence_model_.Preprocess(session_state, compute_session, &state));

  // Avoid ComputeSession overhead by directly iterating over the feature IDs.
  // Handle forward and reverse iteration via an index and increment.
  int target_index = sequence_model_.left_to_right() ? 0 : state.num_steps - 1;
  const int target_increment = sequence_model_.left_to_right() ? 1 : -1;
  sling::myelin::Instance &instance = GetInstance(session_state);
  for (size_t step_index = 0; step_index < state.num_steps;
       ++step_index, target_index += target_increment) {
    // Bind inputs and outputs into the |instance|.
    BindInputIds(state.features, target_index, network_states, &instance);
    BindInputLinks(state.links, target_index, &instance);
    BindInputRecurrences(step_index, network_states, &instance);
    BindOutputLayers(step_index, network_states, &instance);

    // Invoke the cell in the |instance|.
    instance.Compute();
    MaybeTrace(step_index, &instance, component_trace);
  }

  return sequence_model_.Predict(network_states, &state);
}

DRAGNN_RUNTIME_REGISTER_COMPONENT(SequenceMyelinDynamicComponent);

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
