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
#include <string.h>

#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "dragnn/runtime/bulk_network_unit.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/fixed_embeddings.h"
#include "dragnn/runtime/linked_embeddings.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/sequence_model.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/variable_store.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Sequence-based bulk version of DynamicComponent.
class SequenceBulkDynamicComponent : public Component {
 public:
  // Implements Component.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override;
  tensorflow::Status Evaluate(SessionState *session_state,
                              ComputeSession *compute_session,
                              ComponentTrace *component_trace) const override;
  bool Supports(const ComponentSpec &component_spec,
                const string &normalized_builder_name) const override;
  bool PreferredTo(const Component &other) const override { return false; }

 private:
  // Evaluates all input features in the |state|, concatenates them into a
  // matrix of inputs in the |network_states|, and returns the matrix.
  Matrix<float> EvaluateInputs(const SequenceModel::EvaluateState &state,
                               const NetworkStates &network_states) const;

  // Managers for input embeddings.
  FixedEmbeddingManager fixed_embedding_manager_;
  LinkedEmbeddingManager linked_embedding_manager_;

  // Sequence-based model evaluator.
  SequenceModel sequence_model_;

  // Network unit for bulk inference.
  std::unique_ptr<BulkNetworkUnit> bulk_network_unit_;

  // Concatenated input matrix.
  LocalMatrixHandle<float> inputs_handle_;

  // Intermediate values used by sequence models.
  SharedExtensionHandle<SequenceModel::EvaluateState> evaluate_state_handle_;
};

bool SequenceBulkDynamicComponent::Supports(
    const ComponentSpec &component_spec,
    const string &normalized_builder_name) const {
  // Require embedded fixed features.
  for (const FixedFeatureChannel &channel : component_spec.fixed_feature()) {
    if (channel.embedding_dim() < 0) return false;
  }

  // Require non-transformed and non-recurrent linked features.
  // TODO(googleuser): Make SequenceLinks support transformed linked features?
  for (const LinkedFeatureChannel &channel : component_spec.linked_feature()) {
    if (channel.embedding_dim() >= 0) return false;
    if (channel.source_component() == component_spec.name()) return false;
  }

  return normalized_builder_name == "SequenceBulkDynamicComponent" &&
         SequenceModel::Supports(component_spec);
}

// Returns the sum of the dimensions of all channels in the |manager|.
template <class EmbeddingManager>
size_t SumEmbeddingDimensions(const EmbeddingManager &manager) {
  size_t sum = 0;
  for (size_t i = 0; i < manager.num_channels(); ++i) {
    sum += manager.embedding_dim(i);
  }
  return sum;
}

tensorflow::Status SequenceBulkDynamicComponent::Initialize(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    NetworkStateManager *network_state_manager,
    ExtensionManager *extension_manager) {
  TF_RETURN_IF_ERROR(BulkNetworkUnit::CreateOrError(
      BulkNetworkUnit::GetClassName(component_spec), &bulk_network_unit_));
  TF_RETURN_IF_ERROR(
      bulk_network_unit_->Initialize(component_spec, variable_store,
                                     network_state_manager, extension_manager));

  TF_RETURN_IF_ERROR(fixed_embedding_manager_.Reset(
      component_spec, variable_store, network_state_manager));
  TF_RETURN_IF_ERROR(linked_embedding_manager_.Reset(
      component_spec, variable_store, network_state_manager));

  const size_t concatenated_input_dim =
      SumEmbeddingDimensions(fixed_embedding_manager_) +
      SumEmbeddingDimensions(linked_embedding_manager_);
  TF_RETURN_IF_ERROR(
      bulk_network_unit_->ValidateInputDimension(concatenated_input_dim));
  TF_RETURN_IF_ERROR(
      network_state_manager->AddLocal(concatenated_input_dim, &inputs_handle_));

  TF_RETURN_IF_ERROR(sequence_model_.Initialize(
      component_spec, bulk_network_unit_->GetLogitsName(),
      &fixed_embedding_manager_, &linked_embedding_manager_,
      network_state_manager));

  extension_manager->GetShared(&evaluate_state_handle_);
  return tensorflow::Status::OK();
}

tensorflow::Status SequenceBulkDynamicComponent::Evaluate(
    SessionState *session_state, ComputeSession *compute_session,
    ComponentTrace *component_trace) const {
  const NetworkStates &network_states = session_state->network_states;
  SequenceModel::EvaluateState &state =
      session_state->extensions.Get(evaluate_state_handle_);
  TF_RETURN_IF_ERROR(
      sequence_model_.Preprocess(session_state, compute_session, &state));

  const Matrix<float> inputs = EvaluateInputs(state, network_states);
  TF_RETURN_IF_ERROR(bulk_network_unit_->Evaluate(inputs, session_state));

  return sequence_model_.Predict(network_states, &state);
}

Matrix<float> SequenceBulkDynamicComponent::EvaluateInputs(
    const SequenceModel::EvaluateState &state,
    const NetworkStates &network_states) const {
  const MutableMatrix<float> inputs = network_states.GetLocal(inputs_handle_);

  // Declared here for reuse in the loop below.
  bool is_out_of_bounds = false;
  Vector<float> embedding;

  // Handle forward and reverse iteration via a start index and increment.
  int target_index = sequence_model_.left_to_right() ? 0 : state.num_steps - 1;
  const int target_increment = sequence_model_.left_to_right() ? 1 : -1;
  for (size_t step_index = 0; step_index < state.num_steps;
       ++step_index, target_index += target_increment) {
    const MutableVector<float> row = inputs.row(step_index);
    float *output = row.data();

    for (size_t channel_id = 0; channel_id < state.features.num_channels();
         ++channel_id) {
      embedding = state.features.GetEmbedding(channel_id, target_index);
      memcpy(output, embedding.data(), embedding.size() * sizeof(float));
      output += embedding.size();
    }

    for (size_t channel_id = 0; channel_id < state.links.num_channels();
         ++channel_id) {
      state.links.Get(channel_id, target_index, &embedding, &is_out_of_bounds);
      memcpy(output, embedding.data(), embedding.size() * sizeof(float));
      output += embedding.size();
    }

    DCHECK_EQ(output, row.end());
  }

  return Matrix<float>(inputs);
}

DRAGNN_RUNTIME_REGISTER_COMPONENT(SequenceBulkDynamicComponent);

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
