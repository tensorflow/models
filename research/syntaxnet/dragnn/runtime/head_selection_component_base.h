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

#ifndef DRAGNN_RUNTIME_HEAD_SELECTION_COMPONENT_BASE_H_
#define DRAGNN_RUNTIME_HEAD_SELECTION_COMPONENT_BASE_H_

#include <vector>

#include "dragnn/core/compute_session.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Base class for head-selection components, which select heads independently
// per token.  Although this process is not guaranteed to produce a tree, for
// accurate parsers it often produces a tree.
//
// This base class only computes the selected heads, while subclasses apply
// those heads to the annotations in the ComputeSession.
class HeadSelectionComponentBase : public Component {
 public:
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
  HeadSelectionComponentBase(const string &builder_name,
                             const string &backend_name);

  // Returns the list of heads computed from the |session_state|, where -1
  // indicates a root.
  const std::vector<int> &ComputeHeads(SessionState *session_state) const;

 private:
  // Names of the supported component builder and backend.
  const string builder_name_;
  const string backend_name_;

  // Directed adjacency matrix input.
  PairwiseLayerHandle<float> adjacency_handle_;

  // List of selected head indices.
  SharedExtensionHandle<std::vector<int>> heads_handle_;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_HEAD_SELECTION_COMPONENT_BASE_H_
