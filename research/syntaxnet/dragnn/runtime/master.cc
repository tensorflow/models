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

#include "dragnn/runtime/master.h"

#include <utility>
#include <vector>

#include "dragnn/protos/runtime.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

constexpr int kMaxBeamSize = 1;

// Combines, using MergeFrom(), each step trace in the |source| with the
// corresponding step of the |target|.  If |source| has more steps, then
// |target| is extended to match.
void MergeTraces(const ComponentTrace &source, ComponentTrace *target) {
  while (target->step_trace_size() < source.step_trace_size()) {
    target->add_step_trace();
  }
  for (int i = 0; i < source.step_trace_size(); ++i) {
    target->mutable_step_trace(i)->MergeFrom(source.step_trace(i));
  }
}

// Combines, using MergeTraces(), each component trace in the |source| with the
// corresponding component of the |target|.  If |source| has more components,
// then |target| is extended to match.
void MergeTraces(const MasterTrace &source, MasterTrace *target) {
  while (target->component_trace_size() < source.component_trace_size()) {
    target->add_component_trace();
  }
  for (int i = 0; i < source.component_trace_size(); ++i) {
    MergeTraces(source.component_trace(i), target->mutable_component_trace(i));
  }
}

}  // namespace

tensorflow::Status Master::Initialize(
    const MasterSpec &master_spec,
    std::unique_ptr<VariableStore> variable_store) {
  if (variable_store_ != nullptr) {
    return tensorflow::errors::FailedPrecondition("Can't initialize twice");
  }

  if (variable_store == nullptr) {
    return tensorflow::errors::InvalidArgument("No VariableStore");
  }
  variable_store_ = std::move(variable_store);

  const auto &master_performance_settings = master_spec.GetExtension(
      MasterPerformanceSettings::master_spec_extension);
  session_state_pool_.reset(new SessionStatePool(
      master_performance_settings.session_state_pool_max_free_states()));

  components_.reserve(master_spec.component_size());
  for (const ComponentSpec &component_spec : master_spec.component()) {
    const auto &component_performance_settings = component_spec.GetExtension(
        ComponentPerformanceSettings::component_spec_extension);
    components_.emplace_back();
    ComponentConfig &component = components_.back();
    component.name = component_spec.name();
    component.pre_allocate_num_steps =
        component_performance_settings.pre_allocate_num_steps();

    TF_RETURN_IF_ERROR(
        network_state_manager_.AddComponent(component_spec.name()));
    const string component_type =
        GetNormalizedComponentBuilderName(component_spec);
    TF_RETURN_IF_ERROR(
        Component::CreateOrError(component_type, &component.instance));
    TF_RETURN_IF_ERROR(component.instance->Initialize(
        component_spec, variable_store_.get(), &network_state_manager_,
        &extension_manager_));
  }

  return variable_store_->Close();
}

tensorflow::Status Master::Evaluate(ComputeSession *compute_session,
                                    MasterTrace *master_trace) const {
  if (variable_store_ == nullptr) {
    return tensorflow::errors::FailedPrecondition("Not initialized");
  }

  if (compute_session == nullptr) {
    return tensorflow::errors::InvalidArgument("No ComputeSession");
  }

  if (master_trace != nullptr) {
    master_trace->Clear();
    compute_session->SetTracing(true);
  }
  const auto ensure_tracing_disabled = tensorflow::gtl::MakeCleanup([=] {
    if (master_trace != nullptr) compute_session->SetTracing(false);
  });

  const ScopedSessionState session_state(session_state_pool_.get());
  session_state->network_states.Reset(&network_state_manager_);
  session_state->extensions.Reset(&extension_manager_);

  for (const ComponentConfig &component : components_) {
    // TODO(googleuser): Generically trace all layers?
    ComponentTrace *component_trace = nullptr;
    if (master_trace != nullptr) {
      component_trace = master_trace->add_component_trace();
      component_trace->set_name(component.name);
    }

    compute_session->InitializeComponentData(component.name, kMaxBeamSize);
    TF_RETURN_IF_ERROR(session_state->network_states.StartNextComponent(
        component.pre_allocate_num_steps));
    TF_RETURN_IF_ERROR(component.instance->Evaluate(
        session_state.get(), compute_session, component_trace));
    compute_session->FinalizeData(component.name);
  }

  if (master_trace != nullptr) {
    // Use only the first trace from the compute session.
    const std::vector<MasterTrace> traces = compute_session->GetTraceProtos();
    if (!traces.empty()) MergeTraces(traces[0], master_trace);
  }

  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
