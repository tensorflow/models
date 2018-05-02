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

#ifndef DRAGNN_RUNTIME_BULK_NETWORK_UNIT_H_
#define DRAGNN_RUNTIME_BULK_NETWORK_UNIT_H_

#include <stddef.h>
#include <functional>
#include <memory>
#include <string>

#include "dragnn/core/compute_session.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "syntaxnet/registry.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Interface for network units for bulk inference.

//
// TODO(googleuser): The current approach assumes that fixed and
// linked embeddings are computed and concatenated outside the network unit,
// which is simple and composable.  However, it could be more efficient to,
// e.g., pass the fixed and linked embeddings individually or compute them
// internally.  That would elide the concatenation and could increase cache
// coherency.
class BulkNetworkUnit : public RegisterableClass<BulkNetworkUnit> {
 public:
  BulkNetworkUnit(const BulkNetworkUnit &that) = delete;
  BulkNetworkUnit &operator=(const BulkNetworkUnit &that) = delete;
  virtual ~BulkNetworkUnit() = default;

  // Returns the bulk network unit class name specified in the |component_spec|.
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

  // Returns OK iff this is compatible with the input |dimension|.
  virtual tensorflow::Status ValidateInputDimension(size_t dimension) const = 0;

  // Returns the name of the layer that contains classification logits, or an
  // empty string if this does not produce logits.  Requires that Initialize()
  // was called.
  virtual string GetLogitsName() const = 0;

  // Evaluates this network on the bulk |inputs|, using intermediate operands
  // and output layers in the |session_state|.  On error, returns non-OK.
  virtual tensorflow::Status Evaluate(Matrix<float> inputs,
                                      SessionState *session_state) const = 0;

 protected:
  BulkNetworkUnit() = default;

 private:
  // Helps prevent use of the Create() method; use CreateOrError() instead.
  using RegisterableClass<BulkNetworkUnit>::Create;
};

}  // namespace runtime
}  // namespace dragnn

DECLARE_SYNTAXNET_CLASS_REGISTRY("DRAGNN Runtime Bulk Network Unit",
                                 dragnn::runtime::BulkNetworkUnit);

}  // namespace syntaxnet

// Registers a subclass using its class name as a string.
#define DRAGNN_RUNTIME_REGISTER_BULK_NETWORK_UNIT(subclass) \
  REGISTER_SYNTAXNET_CLASS_COMPONENT(                       \
      ::syntaxnet::dragnn::runtime::BulkNetworkUnit, #subclass, subclass)

#endif  // DRAGNN_RUNTIME_BULK_NETWORK_UNIT_H_
