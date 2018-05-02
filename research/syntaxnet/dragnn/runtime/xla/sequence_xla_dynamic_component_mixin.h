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

#ifndef DRAGNN_RUNTIME_XLA_SEQUENCE_XLA_DYNAMIC_COMPONENT_MIXIN_H_
#define DRAGNN_RUNTIME_XLA_SEQUENCE_XLA_DYNAMIC_COMPONENT_MIXIN_H_

#include <stddef.h>
#include <string>
#include <type_traits>
#include <vector>

#include "dragnn/core/compute_session.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/sequence_features.h"
#include "dragnn/runtime/sequence_links.h"
#include "dragnn/runtime/sequence_model.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/variable_store.h"
#include "dragnn/runtime/xla/xla_dynamic_component_base.h"
#include "syntaxnet/base.h"
#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// A mixin that converts an XlaDynamicComponent variant into a sequence-based
// version.  The |Base| must be a subclass of XlaDynamicComponentBase.
template <class Base>
class SequenceXlaDynamicComponentMixin : public Base {
 public:
  static_assert(std::is_base_of<XlaDynamicComponentBase, Base>::value,
                "SequenceXlaDynamicComponentMixin must template on a subclass "
                "of XlaDynamicComponentBase");

  // Implements Component.
  bool Supports(const ComponentSpec &component_spec,
                const string &normalized_builder_name) const override;
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override;
  tensorflow::Status Evaluate(SessionState *session_state,
                              ComputeSession *compute_session,
                              ComponentTrace *component_trace) const override;

 private:
  // Binds the fixed feature IDs for the |target_index|'th element of the
  // |features| to the |instance|.  Uses locals in the |network_states|.
  void BindInputIds(const SequenceFeatures &features, int target_index,
                    const NetworkStates &network_states,
                    tensorflow::XlaCompiledCpuFunction *instance) const;

  // Binds the linked embeddings for the |target_index|'th element in the
  // |links| to the |instance|.
  void BindInputLinks(const SequenceLinks &links, int target_index,
                      tensorflow::XlaCompiledCpuFunction *instance) const;

  // Sequence-based model evaluator.
  SequenceModel sequence_model_;

  // Intermediate values used by sequence models.
  SharedExtensionHandle<SequenceModel::EvaluateState> evaluate_state_handle_;
};

template <class Base>
bool SequenceXlaDynamicComponentMixin<Base>::Supports(
    const ComponentSpec &component_spec,
    const string &normalized_builder_name) const {
  tensorflow::StringPiece name = normalized_builder_name;
  return tensorflow::str_util::ConsumePrefix(&name, "Sequence") &&
         Base::Supports(component_spec, name.ToString()) &&
         SequenceModel::Supports(component_spec);
}

template <class Base>
tensorflow::Status SequenceXlaDynamicComponentMixin<Base>::Initialize(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    NetworkStateManager *network_state_manager,
    ExtensionManager *extension_manager) {
  // Initialize the base class first, so its FixedEmbeddingManager and
  // LinkedEmbeddingManager can be wrapped in sequence-based versions.
  TF_RETURN_IF_ERROR(Base::Initialize(component_spec, variable_store,
                                      network_state_manager,
                                      extension_manager));

  TF_RETURN_IF_ERROR(sequence_model_.Initialize(
      component_spec, Base::kLogitsName, &Base::fixed_embedding_manager(),
      &Base::linked_embedding_manager(), network_state_manager));

  extension_manager->GetShared(&evaluate_state_handle_);
  return tensorflow::Status::OK();
}

template <class Base>
void SequenceXlaDynamicComponentMixin<Base>::BindInputIds(
    const SequenceFeatures &features, int target_index,
    const NetworkStates &network_states,
    tensorflow::XlaCompiledCpuFunction *instance) const {
  for (size_t channel_id = 0; channel_id < features.num_channels();
       ++channel_id) {
    const MutableVector<int32> id_vector = network_states.GetLocal(
        Base::fixed_embedding_manager().id_handle(channel_id, 0));
    id_vector[0] = features.GetId(channel_id, target_index);
    Base::BindInput(Vector<int32>(id_vector), Base::input_ids()[channel_id].id,
                    instance);
  }
}

template <class Base>
void SequenceXlaDynamicComponentMixin<Base>::BindInputLinks(
    const SequenceLinks &links, int target_index,
    tensorflow::XlaCompiledCpuFunction *instance) const {
  Vector<float> embedding;
  bool is_out_of_bounds = false;
  for (size_t channel_id = 0; channel_id < links.num_channels(); ++channel_id) {
    links.Get(channel_id, target_index, &embedding, &is_out_of_bounds);
    Base::BindInputLink(embedding, is_out_of_bounds,
                        Base::input_links()[channel_id], instance);
  }
}

template <class Base>
tensorflow::Status SequenceXlaDynamicComponentMixin<Base>::Evaluate(
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
  tensorflow::XlaCompiledCpuFunction &instance =
      Base::GetInstance(session_state);
  for (size_t step_index = 0; step_index < state.num_steps;
       ++step_index, target_index += target_increment) {
    // Bind inputs and outputs into the |instance|.
    BindInputIds(state.features, target_index, network_states, &instance);
    BindInputLinks(state.links, target_index, &instance);
    Base::BindInputRecurrences(step_index, network_states, &instance);

    // Invoke the cell in the |instance|.
    if (!instance.Run()) {
      return tensorflow::errors::Internal("Error executing cell for ",
                                          Base::name(), ": ",
                                          instance.error_msg());
    }

    // Realizes the binding: copy outputs out of the |instance|.
    Base::BindOutputLayers(step_index, network_states, &instance);

    Base::MaybeTrace(step_index, &instance, component_trace);
  }

  return sequence_model_.Predict(network_states, &state);
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_XLA_SEQUENCE_XLA_DYNAMIC_COMPONENT_MIXIN_H_
