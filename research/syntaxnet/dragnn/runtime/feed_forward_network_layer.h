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

#ifndef DRAGNN_RUNTIME_FEED_FORWARD_NETWORK_LAYER_H_
#define DRAGNN_RUNTIME_FEED_FORWARD_NETWORK_LAYER_H_

#include <stddef.h>
#include <string>

#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/activation_functions.h"
#include "dragnn/runtime/flexible_matrix_kernel.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Configuration and parameters of some layer of a multi-layer perceptron.
class FeedForwardNetworkLayer {
 public:
  // Name of the logits layer produced by a softmax.
  static constexpr char kLogitsName[] = "logits";

  // Creates an uninitialized layer.  Call Initialize() before use.
  FeedForwardNetworkLayer() = default;

  // Initializes this as a layer named |layer_name| of the component named
  // |component_name| that produces activations of size |output_dimension|,
  // and applies the |activation_function| to the output.  Adds this layer to
  // the |network_state_manager| and retrieves trained parameters from the
  // |variable_store| using the |variable_suffix|.  On error, returns non-OK.
  tensorflow::Status Initialize(const string &component_name,
                                const string &layer_name,
                                size_t output_dimension,
                                ActivationFunction activation_function,
                                const string &variable_suffix,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager);

  // For convenience, initializes this as a softmax that produces a layer named
  // |kLogitsName|.
  tensorflow::Status InitializeSoftmax(
      const ComponentSpec &component_spec, VariableStore *variable_store,
      NetworkStateManager *network_state_manager);

  // Returns OK iff this is compatible with input activation vectors of size
  // |input_dim| and sets |output_dim| to the output dimension of this layer.
  tensorflow::Status CheckInputDimAndGetOutputDim(size_t input_dim,
                                                  size_t *output_dim) const;

  // Applies the weights and biases of this layer to the |input| activations,
  // writes the resulting output activations into the |step_index|'th row of
  // the relevant output layer in the |network_states|, and returns the row.
  MutableVector<float> Apply(Vector<float> input,
                             const NetworkStates &network_states,
                             size_t step_index) const;

  // As above, but applies to a step-wise matrix of |inputs|.
  MutableMatrix<float> Apply(Matrix<float> inputs,
                             const NetworkStates &network_states) const;

 private:
  // Name of the layer, for debug purposes.
  string debug_name_;

  // Handle of the layer in the network states.
  LayerHandle<float> handle_;

  // Weight matrix and bias vector for computing the layer activations.
  FlexibleMatrixKernel matrix_kernel_;
  Vector<float> biases_;

  // The activation function to apply to the output.
  ActivationFunction activation_function_ = ActivationFunction::kIdentity;
};

// Implementation details below.

inline MutableVector<float> FeedForwardNetworkLayer::Apply(
    Vector<float> input, const NetworkStates &network_states,
    size_t step_index) const {
  const MutableVector<float> output =
      network_states.GetLayer(handle_).row(step_index);

  matrix_kernel_.MatrixVectorProduct(input, biases_, output);

  ApplyActivationFunction(activation_function_, output);
  return output;
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_FEED_FORWARD_NETWORK_LAYER_H_
