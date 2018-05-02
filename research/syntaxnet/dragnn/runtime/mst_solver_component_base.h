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

#ifndef DRAGNN_RUNTIME_MST_SOLVER_COMPONENT_BASE_H_
#define DRAGNN_RUNTIME_MST_SOLVER_COMPONENT_BASE_H_

#include <vector>

#include "dragnn/core/compute_session.h"
#include "dragnn/mst/mst_solver.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Base class for MST parsing components, which select heads jointly by finding
// the maximum spanning tree of the input tokens.
//
// This base class only computes the selected heads, while subclasses apply the
// heads to the annotations in the ComputeSession.
class MstSolverComponentBase : public Component {
 public:
  // NB: This definition of Index should match the MstSolver TF op wrappers.
  using Index = uint16;

  // Partially implements Component.
  bool Supports(const ComponentSpec &component_spec,
                const string &normalized_builder_name) const override;
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override;
  bool PreferredTo(const Component &other) const override { return false; }

 protected:
  // Creates a component that supports the |builder_name| and |backend_name|.
  MstSolverComponentBase(const string &builder_name,
                         const string &backend_name);

  // Points |heads| at the list of heads computed from the |session_state|,
  // where a self-loop indicates a root.  Returns non-OK on error.
  tensorflow::Status ComputeHeads(
      SessionState *session_state,
      tensorflow::gtl::ArraySlice<Index> *heads) const;

 private:
  using Solver = MstSolver<Index, float>;

  // Names of the supported component builder and backend.
  const string builder_name_;
  const string backend_name_;

  // Whether to solve for a spanning forest instead of a spanning tree.
  bool forest_ = false;

  // Directed adjacency matrix input.
  PairwiseLayerHandle<float> adjacency_handle_;

  // List of selected head indices.
  SharedExtensionHandle<std::vector<Index>> heads_handle_;

  // Reusable MST solver.
  SharedExtensionHandle<Solver> solver_handle_;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_MST_SOLVER_COMPONENT_BASE_H_
