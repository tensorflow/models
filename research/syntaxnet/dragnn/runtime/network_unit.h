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

#ifndef DRAGNN_RUNTIME_NETWORK_UNIT_H_
#define DRAGNN_RUNTIME_NETWORK_UNIT_H_

#include <stddef.h>
#include <string>

#include "dragnn/core/compute_session.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "syntaxnet/registry.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Interface for network units for sequential inference.

class NetworkUnit : public RegisterableClass<NetworkUnit> {
 public:
  NetworkUnit(const NetworkUnit &that) = delete;
  NetworkUnit &operator=(const NetworkUnit &that) = delete;
  virtual ~NetworkUnit() = default;

  // Returns the network unit class name specified in the |component_spec|.
  static string GetClassName(const ComponentSpec &component_spec);

  // Initializes this to the configuration in the |component_spec|.  Retrieves
  // pre-trained variables from the |variable_store|, which must outlive this.
  // Adds layers and local operands to the |network_state_manager|, which must
  // be positioned at the current component.  Requests SessionState extensions
  // from the |extension_manager|.  On error, returns non-OK.
  virtual tensorflow::Status Initialize(
      const ComponentSpec &component_spec, VariableStore *variable_store,
      NetworkStateManager *network_state_manager,
      ExtensionManager *extension_manager) = 0;

  // Returns the name of the layer that contains classification logits, or an
  // empty string if this does not produce logits.  Requires that Initialize()
  // was called.
  virtual string GetLogitsName() const = 0;

  // Evaluates this network unit on the |session_state| and |compute_session|.
  // Requires that:
  // * The network states in the |session_state| is positioned at the current
  //   component, which must have at least |step_index|+1 steps.
  // * The same component in the |compute_session| must have traversed
  //   |step_index| transitions.
  // * Initialize() was called.
  // On error, returns non-OK.
  virtual tensorflow::Status Evaluate(
      size_t step_index, SessionState *session_state,
      ComputeSession *compute_session) const = 0;

 protected:
  NetworkUnit() = default;

 private:
  // Helps prevent use of the Create() method; use CreateOrError() instead.
  using RegisterableClass<NetworkUnit>::Create;
};

}  // namespace runtime
}  // namespace dragnn

DECLARE_SYNTAXNET_CLASS_REGISTRY("DRAGNN Runtime Network Unit",
                                 dragnn::runtime::NetworkUnit);

}  // namespace syntaxnet

// Registers a subclass using its class name as a string.
#define DRAGNN_RUNTIME_REGISTER_NETWORK_UNIT(subclass) \
  REGISTER_SYNTAXNET_CLASS_COMPONENT(                  \
      ::syntaxnet::dragnn::runtime::NetworkUnit, #subclass, subclass)

#endif  // DRAGNN_RUNTIME_NETWORK_UNIT_H_
