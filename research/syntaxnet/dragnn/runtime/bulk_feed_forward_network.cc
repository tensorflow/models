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

#include <stddef.h>
#include <string>

#include "dragnn/core/compute_session.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/bulk_network_unit.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/feed_forward_network_kernel.h"
#include "dragnn/runtime/feed_forward_network_layer.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// A network unit that evaluates a feed-forward multi-layer perceptron.
class BulkFeedForwardNetwork : public BulkNetworkUnit {
 public:
  // Implements BulkNetworkUnit.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override;
  tensorflow::Status ValidateInputDimension(size_t dimension) const override;
  string GetLogitsName() const override { return kernel_.logits_name(); }
  tensorflow::Status Evaluate(Matrix<float> inputs,
                              SessionState *session_state) const override;

 private:
  // Kernel that implements the feed-forward network.
  FeedForwardNetworkKernel kernel_;
};

tensorflow::Status BulkFeedForwardNetwork::Initialize(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    NetworkStateManager *network_state_manager,
    ExtensionManager *extension_manager) {
  for (const LinkedFeatureChannel &channel : component_spec.linked_feature()) {
    if (channel.source_component() == component_spec.name()) {
      return tensorflow::errors::InvalidArgument(
          "BulkFeedForwardNetwork forbids recurrent links");
    }
  }

  return kernel_.Initialize(component_spec, variable_store,
                            network_state_manager);
}

tensorflow::Status BulkFeedForwardNetwork::ValidateInputDimension(
    size_t dimension) const {
  return kernel_.ValidateInputDimension(dimension);
}

tensorflow::Status BulkFeedForwardNetwork::Evaluate(
    Matrix<float> inputs, SessionState *session_state) const {
  for (const FeedForwardNetworkLayer &layer : kernel_.layers()) {
    inputs = layer.Apply(inputs, session_state->network_states);
  }
  return tensorflow::Status::OK();
}

DRAGNN_RUNTIME_REGISTER_BULK_NETWORK_UNIT(BulkFeedForwardNetwork);

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
