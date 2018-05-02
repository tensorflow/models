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

#include "dragnn/runtime/feed_forward_network_layer.h"

#include "tensorflow/core/lib/core/errors.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

constexpr char FeedForwardNetworkLayer::kLogitsName[];

tensorflow::Status FeedForwardNetworkLayer::Initialize(
    const string &component_name, const string &layer_name,
    size_t output_dimension, ActivationFunction activation_function,
    const string &variable_suffix, VariableStore *variable_store,
    NetworkStateManager *network_state_manager) {
  debug_name_ = tensorflow::strings::StrCat(component_name, "/", layer_name);
  activation_function_ = activation_function;

  const string weights_name =
      tensorflow::strings::StrCat(component_name, "/weights_", variable_suffix);
  const string biases_name =
      tensorflow::strings::StrCat(component_name, "/bias_", variable_suffix);

  TF_RETURN_IF_ERROR(variable_store->Lookup(biases_name, &biases_));
  TF_RETURN_IF_ERROR(matrix_kernel_.Initialize(
      debug_name_, weights_name, output_dimension, variable_store));

  TF_RETURN_IF_ERROR(
      network_state_manager->AddLayer(layer_name, output_dimension, &handle_));
  if (!matrix_kernel_.MatchesOutputDimension(output_dimension)) {
    return tensorflow::errors::InvalidArgument(
        "Weight matrix shape should be output dimension plus padding. ",
        debug_name_, ": weights=[", matrix_kernel_.NumPaddedRows(), ", ",
        matrix_kernel_.NumColumns(), "] vs output=", output_dimension);
  }

  // NOTE(gatoatigrado): Do we need to pad the bias vector?
  if (biases_.size() != output_dimension) {
    return tensorflow::errors::InvalidArgument(
        "Bias vector shape does not match output dimension in ", debug_name_,
        ": biases=[", biases_.size(), "] vs output=", output_dimension);
  }

  return tensorflow::Status::OK();
}

tensorflow::Status FeedForwardNetworkLayer::InitializeSoftmax(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    NetworkStateManager *network_state_manager) {
  return Initialize(component_spec.name(), kLogitsName,
                    component_spec.num_actions(), ActivationFunction::kIdentity,
                    "softmax", variable_store, network_state_manager);
}

tensorflow::Status FeedForwardNetworkLayer::CheckInputDimAndGetOutputDim(
    size_t input_dim, size_t *output_dim) const {
  if (matrix_kernel_.NumColumns() != input_dim) {
    return tensorflow::errors::InvalidArgument(
        "Weight matrix shape does not match input dimension in ", debug_name_,
        ": weights=[", matrix_kernel_.NumPaddedRows(), ", ",
        matrix_kernel_.NumColumns(), "] vs input=", input_dim);
  }

  *output_dim = matrix_kernel_.NumPaddedRows();
  return tensorflow::Status::OK();
}

MutableMatrix<float> FeedForwardNetworkLayer::Apply(
    Matrix<float> inputs, const NetworkStates &network_states) const {
  const MutableMatrix<float> outputs = network_states.GetLayer(handle_);

  size_t row = 0;
  for (; row + 1 < inputs.num_rows(); row += 2) {
    matrix_kernel_.MatrixVectorVectorProduct(
        inputs.row(row), inputs.row(row + 1), biases_, biases_,
        outputs.row(row), outputs.row(row + 1));
    ApplyActivationFunction(activation_function_, outputs.row(row));
    ApplyActivationFunction(activation_function_, outputs.row(row + 1));
  }

  if (row < inputs.num_rows()) {
    Vector<float> input_row = inputs.row(row);
    MutableVector<float> output_row = outputs.row(row);
    matrix_kernel_.MatrixVectorProduct(input_row, biases_, output_row);
    ApplyActivationFunction(activation_function_, output_row);
  }

  return outputs;
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
