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

#include "dragnn/runtime/feed_forward_network_kernel.h"

#include "dragnn/runtime/activation_functions.h"
#include "dragnn/runtime/attributes.h"
#include "dragnn/runtime/transition_system_traits.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Attributes used by the feed-forward network.
struct FeedForwardNetworkAttributes : public Attributes {
  // Hidden layer sizes; e.g., "64,64,32".
  Optional<std::vector<size_t>> hidden_layer_sizes{
      "hidden_layer_sizes", {}, this};

  // Whether to omit the "logits" layer.
  Optional<bool> omit_logits{"omit_logits", false, this};

  // Only the default settings are supported for these attributes.
  Optional<bool> layer_norm_input{"layer_norm_input", false, this};
  Optional<bool> layer_norm_hidden{"layer_norm_hidden", false, this};
  Optional<string> nonlinearity{"nonlinearity", "relu", this};

  // Training-only attributes, ignored in the runtime.
  Ignored dropout_keep_prob{"dropout_keep_prob", this};
  Ignored dropout_per_sequence{"dropout_per_sequence", this};
  Ignored dropout_all_layers{"dropout_all_layers", this};
  Ignored initialize_bias_zero{"initialize_bias_zero", this};
  Ignored initialize_softmax_zero{"initialize_softmax_zero", this};
  Ignored initialize_hidden_orthogonal{"initialize_hidden_orthogonal", this};
};

}  // namespace

tensorflow::Status FeedForwardNetworkKernel::Initialize(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    NetworkStateManager *network_state_manager) {
  FeedForwardNetworkAttributes attributes;
  TF_RETURN_IF_ERROR(
      attributes.Reset(component_spec.network_unit().parameters()));

  // Check for unsupported attribute values.
  if (attributes.layer_norm_input() || attributes.layer_norm_hidden()) {
    return tensorflow::errors::Unimplemented("Layer norm is not supported");
  }
  if (attributes.nonlinearity() != "relu") {
    return tensorflow::errors::Unimplemented("Non-linearity is not supported: ",
                                             attributes.nonlinearity());
  }

  // Add all hidden layers.
  for (const size_t hidden_layer_size : attributes.hidden_layer_sizes()) {
    const size_t height = layers_.size();
    layers_.emplace_back();
    TF_RETURN_IF_ERROR(layers_.back().Initialize(
        component_spec.name(), tensorflow::strings::StrCat("layer_", height),
        hidden_layer_size, ActivationFunction::kRelu,
        tensorflow::strings::StrCat(height), variable_store,
        network_state_manager));
  }

  // Add "last_layer" as an alias for the last hidden layer, if any.
  if (!layers_.empty()) {
    TF_RETURN_IF_ERROR(network_state_manager->AddLayerAlias(
        "last_layer",
        tensorflow::strings::StrCat("layer_", layers_.size() - 1)));
  }

  // Add a linear "logits" layer, if necessary.
  const bool has_logits =
      !TransitionSystemTraits(component_spec).is_deterministic &&
      !attributes.omit_logits();
  if (has_logits) {
    logits_name_ = FeedForwardNetworkLayer::kLogitsName;
    layers_.emplace_back();
    TF_RETURN_IF_ERROR(layers_.back().InitializeSoftmax(
        component_spec, variable_store, network_state_manager));
  }

  return tensorflow::Status::OK();
}

tensorflow::Status FeedForwardNetworkKernel::ValidateInputDimension(
    size_t dimension) const {
  for (const FeedForwardNetworkLayer &layer : layers_) {
    TF_RETURN_IF_ERROR(
        layer.CheckInputDimAndGetOutputDim(dimension, &dimension));
  }

  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
