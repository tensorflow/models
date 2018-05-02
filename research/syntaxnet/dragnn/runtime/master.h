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

#ifndef DRAGNN_RUNTIME_MASTER_H_
#define DRAGNN_RUNTIME_MASTER_H_

#include <memory>
#include <string>
#include <vector>

#include "dragnn/core/compute_session.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/session_state_pool.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// A DRAGNN master, which evaluates a series of components.

class Master {
 public:
  // Creates an uninitialized master.  Call Initialize() before use.
  Master() = default;

  // Initializes the components in this based on the |master_spec|, which may
  // have performance tuning settings attached (see runtime.proto).  Retrieves
  // pre-trained variables from the |variable_store|, which must not be closed.
  // On error, returns non-OK.
  tensorflow::Status Initialize(const MasterSpec &master_spec,
                                std::unique_ptr<VariableStore> variable_store);

  // Evaluates the pipeline of components on the |compute_session|, which must
  // be based on the same MasterSpec as this and populated with input data.  If
  // |master_trace| is non-null, overwrites it with extracted traces.  On error,
  // returns non-OK.
  tensorflow::Status Evaluate(ComputeSession *compute_session,
                              MasterTrace *master_trace) const;

 private:
  // A Component with some associated configuration.
  struct ComponentConfig {
    // Name of the component.
    string name;

    // Number of steps to pre-allocate operands for the component.
    size_t pre_allocate_num_steps = 0;

    // Component instance to initialize and evaluate.
    std::unique_ptr<Component> instance;
  };

  // Store of pre-trained variables used by the |components_|.  Must be declared
  // before the |components_| to ensure it outlives them.
  std::unique_ptr<VariableStore> variable_store_;

  // Manager for the network states in the |components_|.
  NetworkStateManager network_state_manager_;

  // Manager for SessionState extensions.
  ExtensionManager extension_manager_;

  // Ordered list of components to evaluate.
  std::vector<ComponentConfig> components_;

  // Pool of session states used when evaluating the |components_|.  This must
  // be destroyed before the |components_|, in case there are state extensions
  // that depend on the |components_|.  Declaring this after the |components_|
  // ensures the proper destructor ordering.
  std::unique_ptr<SessionStatePool> session_state_pool_;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_MASTER_H_
