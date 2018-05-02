// Copyright 2018 Google Inc. All Rights Reserved.
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

#include "dragnn/runtime/mst_solver_component_base.h"

#include <stddef.h>

#include "dragnn/runtime/attributes.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_unit.h"
#include "tensorflow/core/lib/core/errors.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Attributes used by the MST solver.
struct MstSolverAttributes : public Attributes {
  // Whether to solve for a spanning forest instead of a spanning tree.
  Optional<bool> forest{"forest", false, this};

  // Training-only attributes, ignored in the runtime.
  Ignored loss{"loss", this};
};

}  // namespace

MstSolverComponentBase::MstSolverComponentBase(const string &builder_name,
                                               const string &backend_name)
    : builder_name_(builder_name), backend_name_(backend_name) {}

bool MstSolverComponentBase::Supports(
    const ComponentSpec &component_spec,
    const string &normalized_builder_name) const {
  const string network_unit = NetworkUnit::GetClassName(component_spec);
  return (normalized_builder_name == "BulkAnnotatorComponent" ||
          normalized_builder_name == builder_name_) &&
         (component_spec.backend().registered_name() == "StatelessComponent" ||
          component_spec.backend().registered_name() == backend_name_) &&
         component_spec.transition_system().registered_name() == "heads" &&
         network_unit == "MstSolverNetwork" &&
         component_spec.fixed_feature_size() == 0 &&
         component_spec.linked_feature_size() == 1;
}

tensorflow::Status MstSolverComponentBase::Initialize(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    NetworkStateManager *network_state_manager,
    ExtensionManager *extension_manager) {
  MstSolverAttributes attributes;
  TF_RETURN_IF_ERROR(
      attributes.Reset(component_spec.network_unit().parameters()));
  forest_ = attributes.forest();

  const LinkedFeatureChannel &link = component_spec.linked_feature(0);
  size_t dimension = 0;
  TF_RETURN_IF_ERROR(network_state_manager->LookupLayer(
      link.source_component(), link.source_layer(), &dimension,
      &adjacency_handle_));

  if (dimension != 1) {
    return tensorflow::errors::InvalidArgument(
        "Adjacency matrix has dimension ", dimension, " but expected 1");
  }

  extension_manager->GetShared(&heads_handle_);
  extension_manager->GetShared(&solver_handle_);
  return tensorflow::Status::OK();
}

tensorflow::Status MstSolverComponentBase::ComputeHeads(
    SessionState *session_state,
    tensorflow::gtl::ArraySlice<Index> *heads) const {
  Matrix<float> adjacency(
      session_state->network_states.GetLayer(adjacency_handle_));
  const size_t num_nodes = adjacency.num_rows();

  Solver &solver = session_state->extensions.Get(solver_handle_);
  TF_RETURN_IF_ERROR(solver.Init(forest_, num_nodes));

  for (size_t target = 0; target < num_nodes; ++target) {
    Vector<float> source_scores = adjacency.row(target);
    for (size_t source = 0; source < num_nodes; ++source) {
      if (source == target) {
        solver.AddRoot(source, source_scores[source]);
      } else {
        solver.AddArc(source, target, source_scores[source]);
      }
    }
  }

  std::vector<Index> &argmax = session_state->extensions.Get(heads_handle_);
  argmax.resize(num_nodes);
  TF_RETURN_IF_ERROR(solver.Solve(&argmax));

  *heads = argmax;
  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
