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

#include "dragnn/runtime/head_selection_component_base.h"

#include <stddef.h>
#include <algorithm>

#include "tensorflow/core/lib/core/errors.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

HeadSelectionComponentBase::HeadSelectionComponentBase(
    const string &builder_name, const string &backend_name)
    : builder_name_(builder_name), backend_name_(backend_name) {}

bool HeadSelectionComponentBase::Supports(
    const ComponentSpec &component_spec,
    const string &normalized_builder_name) const {
  return (normalized_builder_name == "BulkAnnotatorComponent" ||
          normalized_builder_name == builder_name_) &&
         (component_spec.backend().registered_name() == "StatelessComponent" ||
          component_spec.backend().registered_name() == backend_name_) &&
         component_spec.transition_system().registered_name() == "heads" &&
         component_spec.network_unit().registered_name() == "IdentityNetwork" &&
         component_spec.fixed_feature_size() == 0 &&
         component_spec.linked_feature_size() == 1;
}

tensorflow::Status HeadSelectionComponentBase::Initialize(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    NetworkStateManager *network_state_manager,
    ExtensionManager *extension_manager) {
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
  return tensorflow::Status::OK();
}

const std::vector<int> &HeadSelectionComponentBase::ComputeHeads(
    SessionState *session_state) const {
  Matrix<float> adjacency(
      session_state->network_states.GetLayer(adjacency_handle_));
  std::vector<int> &heads = session_state->extensions.Get(heads_handle_);
  heads.resize(adjacency.num_rows());
  for (size_t i = 0; i < adjacency.num_rows(); ++i) {
    Vector<float> row = adjacency.row(i);
    const int head = std::max_element(row.begin(), row.end()) - row.begin();
    heads[i] = head != i ? head : -1;  // self-loops are roots
  }
  return heads;
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
