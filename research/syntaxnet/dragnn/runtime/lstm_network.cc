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

#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/lstm_cell/cell_function.h"
#include "dragnn/runtime/lstm_network_kernel.h"
#include "dragnn/runtime/math/avx_activation_functions.h"
#include "dragnn/runtime/network_unit.h"
#include "dragnn/runtime/network_unit_base.h"
#include "dragnn/runtime/transition_system_traits.h"
#include "dragnn/runtime/variable_store.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// A network unit that evaluates a LSTM.
//
// NOTE: For efficiency, unlike the Python API, lstm_h and lstm_c are not
// exposed; any subsequent components should reference 'layer_0'. This seems to
// be the case for all current DRAGNN models.
class LSTMNetwork : public NetworkUnitBase {
 public:
  // Implements NetworkUnit.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override;
  string GetLogitsName() const override { return kernel_.GetLogitsName(); }
  tensorflow::Status Evaluate(size_t step_index, SessionState *session_state,
                              ComputeSession *compute_session) const override;

 private:
  // Kernel that implements the LSTM.
  LSTMNetworkKernel kernel_{/*bulk=*/false};
};

tensorflow::Status LSTMNetwork::Initialize(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    NetworkStateManager *network_state_manager,
    ExtensionManager *extension_manager) {
  TF_RETURN_IF_ERROR(kernel_.Initialize(component_spec, variable_store,
                                        network_state_manager,
                                        extension_manager));

  const bool use_concatenated_input = true;
  return InitializeBase(use_concatenated_input, component_spec, variable_store,
                        network_state_manager, extension_manager);
}

tensorflow::Status LSTMNetwork::Evaluate(
    size_t step_index, SessionState *session_state,
    ComputeSession *compute_session) const {
  Vector<float> input;
  TF_RETURN_IF_ERROR(EvaluateBase(session_state, compute_session, &input));
  return kernel_.Apply(step_index, input, session_state);
}

DRAGNN_RUNTIME_REGISTER_NETWORK_UNIT(LSTMNetwork);

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
