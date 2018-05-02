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

#ifndef DRAGNN_RUNTIME_LSTM_NETWORK_KERNEL_H_
#define DRAGNN_RUNTIME_LSTM_NETWORK_KERNEL_H_

#include <memory>
#include <string>

#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/feed_forward_network_layer.h"
#include "dragnn/runtime/lstm_cell/cell_function.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Kernel that evaluates an LSTM network.
class LSTMNetworkKernel {
 public:
  // Creates a kernel for bulk or non-bulk computations.
  explicit LSTMNetworkKernel(bool bulk) : bulk_(bulk) {}

  // Initializes this to the configuration in the |component_spec|.  Retrieves
  // pre-trained variables from the |variable_store|, which must outlive this.
  // Adds layers and local operands to the |network_state_manager|, which must
  // be positioned at the current component.  Requests SessionState extensions
  // from the |extension_manager|.  On error, returns non-OK.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager);

  // Returns the name of the logits layer, or an empty string if none.
  string GetLogitsName() const;

  // Applies this to the |input| activations for the |step_index|'th step using
  // the |session_state|.  Requires that this was created in non-bulk mode.  On
  // error, returns non-OK.
  tensorflow::Status Apply(size_t step_index, Vector<float> input,
                           SessionState *session_state) const;

  // As above, but for matrices.  Requires that this was created in bulk mode.
  tensorflow::Status Apply(Matrix<float> all_inputs,
                           SessionState *session_state) const;

 private:
  // Whether this is a bulk or non-bulk kernel.
  const bool bulk_;

  // Whether this has a logits layer.
  bool has_logits_ = false;

  // Main cell function, which is an instance of either LstmCellFunction<float>
  // or LstmCellFunctionBase<TruncatedFloat16>.
  std::unique_ptr<LstmCellFunctionBase> cell_function_;

  // LSTM cell state and output.
  LocalVectorHandle<float> cell_state_;
  LocalVectorHandle<float> cell_output_;

  // LSTM cell input.  Only used if |bulk_| is false.
  LocalVectorHandle<float> cell_input_vector_;

  // LSTM cell input.  Only used if |bulk_| is true.
  LocalMatrixHandle<float> cell_input_matrix_;

  // Hidden outputs.
  LayerHandle<float> hidden_;

  // The softmax is an affine transformation of the hidden state.
  FeedForwardNetworkLayer softmax_layer_;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_LSTM_NETWORK_KERNEL_H_
