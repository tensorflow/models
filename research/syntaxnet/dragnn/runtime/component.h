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

#ifndef DRAGNN_RUNTIME_COMPONENT_H_
#define DRAGNN_RUNTIME_COMPONENT_H_

#include <string>

#include "dragnn/core/compute_session.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
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

// Helper method, currently only used by myelination.cc.
string GetNormalizedComponentBuilderName(const ComponentSpec &component_spec);

// Interface for components.

class Component : public RegisterableClass<Component> {
 public:
  Component(const Component &that) = delete;
  Component &operator=(const Component &that) = delete;
  virtual ~Component() = default;

  // Initializes this to the configuration in the |component_spec|.  Retrieves
  // pre-trained variables from the |variable_store|, which must outlive this.
  // Adds layers and local operands to the |network_state_manager|, which must
  // be positioned at the current component.  Requests SessionState extensions
  // from the |extension_manager|.  On error, returns non-OK.
  virtual tensorflow::Status Initialize(
      const ComponentSpec &component_spec, VariableStore *variable_store,
      NetworkStateManager *network_state_manager,
      ExtensionManager *extension_manager) = 0;

  // Evaluates this on the |session_state| and |compute_session|, which must
  // both be positioned at the current component.  If |component_trace| is
  // non-null, overwrites it with extracted traces.  On error, returns non-OK.
  virtual tensorflow::Status Evaluate(
      SessionState *session_state, ComputeSession *compute_session,
      ComponentTrace *component_trace) const = 0;

  // Returns the best component for a spec, searching through all registered
  // subclasses. This allows specialized implementations to be used.
  //
  // Sets |result| on success, otherwise returns an error message if a single
  // best matching component could not be found.  Returned statuses include:
  // * OK: If a single best matching component was found.
  // * FAILED_PRECONDITION: If an error occurred during the search.
  // * NOT_FOUND: If the search was error-free, but no matches were found.
  static tensorflow::Status Select(const ComponentSpec &spec, string *result);

 protected:
  Component() = default;

  // Whether this component supports a given spec. |spec| is the full component
  // spec and |normalized_builder_name| is the component builder name, with
  // Python modules and the suffix "Builder" stripped.
  virtual bool Supports(const ComponentSpec &spec,
                        const string &normalized_builder_name) const = 0;

  // Whether to prefer this component to another. (Both components must say that
  // they support the spec.)
  //
  // Components must agree on whether they are more or less specialized than
  // another component. Feel free to expose methods for subclasses to identify
  // themselves; currently, we only have unoptimized implementations (which say
  // they are never preferred) and optimized implementations (which say they are
  // always preferred).
  virtual bool PreferredTo(const Component &other) const = 0;

 private:
  // Helps prevent use of the Create() method; use CreateOrError() instead.
  using RegisterableClass<Component>::Create;
};

}  // namespace runtime
}  // namespace dragnn

DECLARE_SYNTAXNET_CLASS_REGISTRY("DRAGNN Runtime Component",
                                 dragnn::runtime::Component);

}  // namespace syntaxnet

// Registers a subclass using its class name as a string.
#define DRAGNN_RUNTIME_REGISTER_COMPONENT(subclass)                           \
  REGISTER_SYNTAXNET_CLASS_COMPONENT(::syntaxnet::dragnn::runtime::Component, \
                                     #subclass, subclass)

#endif  // DRAGNN_RUNTIME_COMPONENT_H_
