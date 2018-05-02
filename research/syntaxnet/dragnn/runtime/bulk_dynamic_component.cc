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

#include <memory>
#include <string>

#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "dragnn/runtime/bulk_network_unit.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/network_unit_base.h"
#include "dragnn/runtime/transition_system_traits.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Network unit that allows us to make calls to NetworkUnitBase and extract
// features. We may want to provide more optimized versions of this class.
class BulkFeatureExtractorNetwork : public NetworkUnitBase {
 public:
  // Returns true if this supports the |component_spec|.  Requires:
  // * A deterministic transition system, which can be advanced from the oracle.
  // * No recurrent linked features (i.e. from this system).
  static bool Supports(const ComponentSpec &component_spec);

  // Implements NetworkUnit.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override;

  // Advances the |compute_session| through all oracle transitions and extracts
  // fixed and linked embeddings, concatenates them into an input matrix stored
  // in the NetworkStates in the |session_state|, and points the |inputs| at it.
  // Also adds steps to the NetworkStates.  On error, returns non-OK.
  tensorflow::Status EvaluateInputs(SessionState *session_state,
                                    ComputeSession *compute_session,
                                    Matrix<float> *inputs) const;

 private:
  // Implements NetworkUnit.  Evaluate() is "final" to encourage inlining.
  string GetLogitsName() const override { return ""; }
  tensorflow::Status Evaluate(size_t step_index, SessionState *session_state,
                              ComputeSession *compute_session) const final;

  // Name of the containing component.
  string name_;

  // Concatenated input matrix.
  LocalMatrixHandle<float> inputs_handle_;
};

bool BulkFeatureExtractorNetwork::Supports(
    const ComponentSpec &component_spec) {
  if (!TransitionSystemTraits(component_spec).is_deterministic) return false;

  // Forbid recurrent linked features.
  for (const LinkedFeatureChannel &channel : component_spec.linked_feature()) {
    if (channel.source_component() == component_spec.name()) return false;
  }

  return true;
}

tensorflow::Status BulkFeatureExtractorNetwork::Initialize(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    NetworkStateManager *network_state_manager,
    ExtensionManager *extension_manager) {
  name_ = component_spec.name();

  if (!Supports(component_spec)) {
    return tensorflow::errors::InvalidArgument(
        "BulkFeatureExtractorNetwork does not support component '", name_, "'");
  }

  const bool use_concatenated_input = true;
  TF_RETURN_IF_ERROR(InitializeBase(use_concatenated_input, component_spec,
                                    variable_store, network_state_manager,
                                    extension_manager));

  return network_state_manager->AddLocal(concatenated_input_dim(),
                                         &inputs_handle_);
}

tensorflow::Status BulkFeatureExtractorNetwork::EvaluateInputs(
    SessionState *session_state, ComputeSession *compute_session,
    Matrix<float> *inputs) const {
  // TODO(googleuser): Try the ComputeSession's bulk feature extraction API?
  for (size_t step_idx = 0; !compute_session->IsTerminal(name_); ++step_idx) {
    session_state->network_states.AddStep();
    TF_RETURN_IF_ERROR(Evaluate(step_idx, session_state, compute_session));
    compute_session->AdvanceFromOracle(name_);
  }

  *inputs = session_state->network_states.GetLocal(inputs_handle_);
  return tensorflow::Status::OK();
}

tensorflow::Status BulkFeatureExtractorNetwork::Evaluate(
    size_t step_index, SessionState *session_state,
    ComputeSession *compute_session) const {
  Vector<float> input;
  TF_RETURN_IF_ERROR(EvaluateBase(session_state, compute_session, &input));

  MutableMatrix<float> all_inputs =
      session_state->network_states.GetLocal(inputs_handle_);

  // TODO(googleuser): Punch a hole in EvaluateBase so it writes directly to
  // all_inputs.row(step_index).
  //
  // In the future, we could entirely eliminate copying, by providing a variant
  // of LstmCellFunction::RunInputComputation that adds a partial vector of
  // inputs, e.g. instead of RunInputComputation(x), we compute
  //
  // RunInputComputation(x[0:32]) + RunInputComputation(x[32:64])
  //
  // where perhaps x[0:32] points directly at a fixed word feature vector, and
  // x[32:64] points directly at the previous layer's outputs (as a linked
  // feature).
  MutableVector<float> output = all_inputs.row(step_index);
  DCHECK_EQ(input.size(), output.size());

  // TODO(googleuser): Try memcpy() or a custom vectorized copy.
  for (int i = 0; i < input.size(); ++i) {
    output[i] = input[i];
  }

  return tensorflow::Status::OK();
}

// Bulk version of a DynamicComponent---i.e., a component that was originally
// dynamic but can be automatically upgraded to a bulk version.
class BulkDynamicComponent : public Component {
 protected:
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
  bool PreferredTo(const Component &other) const override { return true; }

 private:
  // Feature extractor that builds the input activation matrix.
  BulkFeatureExtractorNetwork bulk_feature_extractor_;

  // Network unit for bulk computation.
  std::unique_ptr<BulkNetworkUnit> bulk_network_unit_;
};

// In addition to the BulkFeatureExtractorNetwork requirements, the bulk LSTM
// requires no attention (the runtime doesn't support attention yet).
bool BulkDynamicComponent::Supports(
    const ComponentSpec &component_spec,
    const string &normalized_builder_name) const {
  return BulkFeatureExtractorNetwork::Supports(component_spec) &&
         (normalized_builder_name == "DynamicComponent" ||
          normalized_builder_name == "BulkDynamicComponent") &&
         component_spec.attention_component().empty();
}

tensorflow::Status BulkDynamicComponent::Initialize(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    NetworkStateManager *network_state_manager,
    ExtensionManager *extension_manager) {
  TF_RETURN_IF_ERROR(bulk_feature_extractor_.Initialize(
      component_spec, variable_store, network_state_manager,
      extension_manager));

  TF_RETURN_IF_ERROR(BulkNetworkUnit::CreateOrError(
      BulkNetworkUnit::GetClassName(component_spec), &bulk_network_unit_));
  TF_RETURN_IF_ERROR(
      bulk_network_unit_->Initialize(component_spec, variable_store,
                                     network_state_manager, extension_manager));
  return bulk_network_unit_->ValidateInputDimension(
      bulk_feature_extractor_.concatenated_input_dim());
}

tensorflow::Status BulkDynamicComponent::Evaluate(
    SessionState *session_state, ComputeSession *compute_session,
    ComponentTrace *component_trace) const {
  Matrix<float> inputs;
  TF_RETURN_IF_ERROR(bulk_feature_extractor_.EvaluateInputs(
      session_state, compute_session, &inputs));
  return bulk_network_unit_->Evaluate(inputs, session_state);
}

DRAGNN_RUNTIME_REGISTER_COMPONENT(BulkDynamicComponent);

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
