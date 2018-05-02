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

#include "dragnn/runtime/sequence_model.h"

#include <vector>

#include "dragnn/runtime/attributes.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/sequence_backend.h"
#include "dragnn/runtime/transition_system_traits.h"
#include "tensorflow/core/lib/core/errors.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Proper backend for sequence-based models.
constexpr char kSupportedBackend[] = "SequenceBackend";

// Attributes for sequence-based comopnents, attached to the component builder.
// See SequenceComponentTransformer.
struct ComponentBuilderAttributes : public Attributes {
  // Registered names of the sequence extractors to use.
  Mandatory<std::vector<string>> sequence_extractors{"sequence_extractors",
                                                     this};

  // Registered names of the sequence linkers to use per channel, in order.
  Mandatory<std::vector<string>> sequence_linkers{"sequence_linkers", this};

  // Registered name of the sequence predictor to use.
  Mandatory<string> sequence_predictor{"sequence_predictor", this};
};

}  // namespace

bool SequenceModel::Supports(const ComponentSpec &component_spec) {
  // Require single-embedding fixed and linked features.
  for (const FixedFeatureChannel &channel : component_spec.fixed_feature()) {
    if (channel.size() != 1) return false;
  }
  for (const LinkedFeatureChannel &channel : component_spec.linked_feature()) {
    if (channel.size() != 1) return false;
  }

  const bool has_fixed_feature = component_spec.fixed_feature_size() > 0;
  bool has_recurrent_link = false;
  bool has_non_recurrent_link = false;
  for (const LinkedFeatureChannel &channel : component_spec.linked_feature()) {
    if (channel.source_component() == component_spec.name()) {
      has_recurrent_link = true;
    } else {
      has_non_recurrent_link = true;
    }
  }

  // Recurrent links must be accompanied by fixed features or non-recurrent
  // links, so the number of recurrent steps can be pre-computed.
  if (has_recurrent_link && !has_fixed_feature && !has_non_recurrent_link) {
    return false;
  }

  const int num_features = component_spec.fixed_feature_size() +
                           component_spec.linked_feature_size();
  return component_spec.backend().registered_name() == kSupportedBackend &&
         num_features > 0;
}

tensorflow::Status SequenceModel::Initialize(
    const ComponentSpec &component_spec, const string &logits_name,
    const FixedEmbeddingManager *fixed_embedding_manager,
    const LinkedEmbeddingManager *linked_embedding_manager,
    NetworkStateManager *network_state_manager) {
  component_name_ = component_spec.name();

  if (component_spec.backend().registered_name() != kSupportedBackend) {
    return tensorflow::errors::InvalidArgument(
        "Invalid component backend: ",
        component_spec.backend().registered_name());
  }

  TransitionSystemTraits traits(component_spec);
  deterministic_ = traits.is_deterministic;
  left_to_right_ = traits.is_left_to_right;

  ComponentBuilderAttributes component_builder_attributes;
  TF_RETURN_IF_ERROR(component_builder_attributes.Reset(
      component_spec.component_builder().parameters()));

  TF_RETURN_IF_ERROR(sequence_feature_manager_.Reset(
      fixed_embedding_manager, component_spec,
      component_builder_attributes.sequence_extractors()));
  TF_RETURN_IF_ERROR(sequence_link_manager_.Reset(
      linked_embedding_manager, component_spec,
      component_builder_attributes.sequence_linkers()));

  have_fixed_features_ = sequence_feature_manager_.num_channels() > 0;
  have_linked_features_ = sequence_link_manager_.num_channels() > 0;
  if (!have_fixed_features_ && !have_linked_features_) {
    return tensorflow::errors::InvalidArgument("No fixed or linked features");
  }

  if (!deterministic_) {
    size_t dimension = 0;
    TF_RETURN_IF_ERROR(network_state_manager->LookupLayer(
        component_name_, logits_name, &dimension, &logits_handle_));
    if (dimension != component_spec.num_actions()) {
      return tensorflow::errors::InvalidArgument(
          "Logits dimension mismatch between NetworkStates (", dimension,
          ") and ComponentSpec (", component_spec.num_actions(), ")");
    }

    TF_RETURN_IF_ERROR(SequencePredictor::New(
        component_builder_attributes.sequence_predictor(), component_spec,
        &sequence_predictor_));
  }

  return tensorflow::Status::OK();
}

tensorflow::Status SequenceModel::Preprocess(
    SessionState *session_state, ComputeSession *compute_session,
    EvaluateState *evaluate_state) const {
  InputBatchCache *input_batch_cache = compute_session->GetInputBatchCache();
  if (input_batch_cache == nullptr) {
    return tensorflow::errors::InvalidArgument("Null input batch");
  }

  // The feature handling below is complicated by the need to support recurrent
  // links.  See the comment on SequenceLinks::Reset().
  NetworkStates &network_states = session_state->network_states;
  TF_RETURN_IF_ERROR(evaluate_state->features.Reset(&sequence_feature_manager_,
                                                    input_batch_cache));
  if (have_fixed_features_) {
    network_states.AddSteps(evaluate_state->features.num_steps());
  }
  TF_RETURN_IF_ERROR(evaluate_state->links.Reset(
      /*add_steps=*/!have_fixed_features_, &sequence_link_manager_,
      &network_states, input_batch_cache));

  // Initialize() ensures that there is at least one fixed or linked feature;
  // use it to determine the number of steps.
  size_t num_steps = 0;
  if (have_fixed_features_ && have_linked_features_) {
    num_steps = evaluate_state->features.num_steps();
    if (num_steps != evaluate_state->links.num_steps()) {
      return tensorflow::errors::FailedPrecondition(
          "Sequence length mismatch between fixed features (", num_steps,
          ") and linked features (", evaluate_state->links.num_steps(), ")");
    }
  } else if (have_fixed_features_) {
    num_steps = evaluate_state->features.num_steps();
  } else {
    num_steps = evaluate_state->links.num_steps();
  }

  // Tell the backend the current input size, so it can handle requests for
  // linked features from downstream components.
  static_cast<SequenceBackend *>(
      compute_session->GetReadiedComponent(component_name_))
      ->SetSequenceSize(num_steps);

  evaluate_state->num_steps = num_steps;
  evaluate_state->input = input_batch_cache;
  return tensorflow::Status::OK();
}

tensorflow::Status SequenceModel::Predict(const NetworkStates &network_states,
                                          EvaluateState *evaluate_state) const {
  if (!deterministic_) {
    const Matrix<float> logits(network_states.GetLayer(logits_handle_));
    TF_RETURN_IF_ERROR(
        sequence_predictor_->Predict(logits, evaluate_state->input));
  }
  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
