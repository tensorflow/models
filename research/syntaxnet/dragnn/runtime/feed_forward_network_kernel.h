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

#ifndef DRAGNN_RUNTIME_FEED_FORWARD_NETWORK_KERNEL_H_
#define DRAGNN_RUNTIME_FEED_FORWARD_NETWORK_KERNEL_H_

#include <stddef.h>
#include <string>
#include <vector>

#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/feed_forward_network_layer.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// A kernel that evaluates a multi-layer perceptron.
class FeedForwardNetworkKernel {
 public:
  // Initializes this to the configuration in the |component_spec|.  Retrieves
  // pre-trained variables from the |variable_store|, which must outlive this.
  // Adds layers and local operands to the |network_state_manager|, which must
  // be positioned at the current component.  On error, returns non-OK.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager);

  // Returns OK iff this is compatible with the input |dimension|.
  tensorflow::Status ValidateInputDimension(size_t dimension) const;

  // Accessors.
  const std::vector<FeedForwardNetworkLayer> &layers() const { return layers_; }
  const string &logits_name() const { return logits_name_; }

 private:
  // List of layers, including hidden layers and the logits, if any.
  std::vector<FeedForwardNetworkLayer> layers_;

  // Name of the logits layer.
  string logits_name_;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_FEED_FORWARD_NETWORK_KERNEL_H_
