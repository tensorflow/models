// Copyright 2018 Google Inc. All Rights Reserved.
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

#include "dragnn/core/compute_session.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/math/eigen.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/network_unit.h"
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

// Produces pairwise activations via a biaffine product between source and
// target token activations, as in the Dozat parser.  This is the runtime
// version of the BiaffineDigraphNetwork, but is implemented as a Component
// instead of a NetworkUnit so it can control operand allocation.
class BiaffineDigraphComponent : public Component {
 public:
  // Implements Component.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override;
  tensorflow::Status Evaluate(SessionState *session_state,
                              ComputeSession *compute_session,
                              ComponentTrace *component_trace) const override;
  bool Supports(const ComponentSpec &component_spec,
                const string &normalized_builder_name) const override;
  bool PreferredTo(const Component &other) const override { return false; }

 private:
  // Weights for computing source-target arc potentials.
  Matrix<float> arc_weights_;

  // Weights for computing source-selection potentials.
  Vector<float> source_weights_;

  // Weights and bias for root-target arc potentials.
  Vector<float> root_weights_;
  float root_bias_ = 0.0;

  // Source and target token activation inputs.
  LayerHandle<float> sources_handle_;
  LayerHandle<float> targets_handle_;

  // Directed adjacency matrix output.
  PairwiseLayerHandle<float> adjacency_handle_;

  // Handles for intermediate computations.
  LocalMatrixHandle<float> target_product_handle_;
};

bool BiaffineDigraphComponent::Supports(
    const ComponentSpec &component_spec,
    const string &normalized_builder_name) const {
  const string network_unit = NetworkUnit::GetClassName(component_spec);
  return (normalized_builder_name == "BulkFeatureExtractorComponent" ||
          normalized_builder_name == "BiaffineDigraphComponent") &&
         network_unit == "BiaffineDigraphNetwork";
}

// Finds the link named |name| in the |component_spec| and points the |handle|
// at the corresponding layer in the |network_state_manager|.  The layer must
// also match the |required_dimension|.  Returns non-OK on error.
tensorflow::Status FindAndValidateLink(
    const ComponentSpec &component_spec,
    const NetworkStateManager &network_state_manager, const string &name,
    size_t required_dimension, LayerHandle<float> *handle) {
  const LinkedFeatureChannel *link = nullptr;
  for (const LinkedFeatureChannel &channel : component_spec.linked_feature()) {
    if (channel.name() == name) {
      link = &channel;
      break;
    }
  }

  if (link == nullptr) {
    return tensorflow::errors::InvalidArgument(
        component_spec.name(), ": link '", name, "' does not exist");
  }

  const string error_suffix = tensorflow::strings::StrCat(
      " in link { ", link->ShortDebugString(), " }");

  if (link->embedding_dim() != -1) {
    return tensorflow::errors::InvalidArgument(
        component_spec.name(), ": transformed links are forbidden",
        error_suffix);
  }

  if (link->size() != 1) {
    return tensorflow::errors::InvalidArgument(
        component_spec.name(), ": multi-embedding links are forbidden",
        error_suffix);
  }

  if (link->source_component() == component_spec.name()) {
    return tensorflow::errors::InvalidArgument(
        component_spec.name(), ": recurrent links are forbidden", error_suffix);
  }

  if (link->fml() != "input.focus" || link->source_translator() != "identity") {
    return tensorflow::errors::InvalidArgument(
        component_spec.name(), ": non-trivial link translation is forbidden",
        error_suffix);
  }

  size_t dimension = 0;
  TF_RETURN_IF_ERROR(network_state_manager.LookupLayer(
      link->source_component(), link->source_layer(), &dimension, handle));

  if (dimension != required_dimension) {
    return tensorflow::errors::InvalidArgument(
        component_spec.name(), ": link '", name, "' has dimension ", dimension,
        " instead of ", required_dimension, error_suffix);
  }

  return tensorflow::Status::OK();
}

tensorflow::Status BiaffineDigraphComponent::Initialize(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    NetworkStateManager *network_state_manager,
    ExtensionManager *extension_manager) {
  TF_RETURN_IF_ERROR(variable_store->Lookup(
      tensorflow::strings::StrCat(component_spec.name(), "/weights_arc"),
      &arc_weights_));
  const size_t source_dimension = arc_weights_.num_rows();
  const size_t target_dimension = arc_weights_.num_columns();

  TF_RETURN_IF_ERROR(variable_store->Lookup(
      tensorflow::strings::StrCat(component_spec.name(), "/weights_source"),
      &source_weights_));
  if (source_weights_.size() != source_dimension) {
    return tensorflow::errors::InvalidArgument(
        component_spec.name(), ": dimension mismatch between weights_arc [",
        source_dimension, ",", target_dimension, "] and weights_source [",
        source_weights_.size(), "]");
  }

  TF_RETURN_IF_ERROR(variable_store->Lookup(
      tensorflow::strings::StrCat(component_spec.name(), "/root_weights"),
      &root_weights_));
  if (root_weights_.size() != target_dimension) {
    return tensorflow::errors::InvalidArgument(
        component_spec.name(), ": dimension mismatch between weights_arc [",
        source_dimension, ",", target_dimension, "] and root_weights [",
        root_weights_.size(), "]");
  }

  Vector<float> root_bias;
  TF_RETURN_IF_ERROR(variable_store->Lookup(
      tensorflow::strings::StrCat(component_spec.name(), "/root_bias"),
      &root_bias));
  if (root_bias.size() != 1) {
    return tensorflow::errors::InvalidArgument(
        component_spec.name(), ": root_bias must be a singleton");
  }
  root_bias_ = root_bias[0];

  if (component_spec.fixed_feature_size() != 0) {
    return tensorflow::errors::InvalidArgument(
        component_spec.name(), ": fixed features are forbidden");
  }
  if (component_spec.linked_feature_size() != 2) {
    return tensorflow::errors::InvalidArgument(
        component_spec.name(), ": two linked features are required");
  }

  TF_RETURN_IF_ERROR(FindAndValidateLink(component_spec, *network_state_manager,
                                         "sources", source_dimension,
                                         &sources_handle_));
  TF_RETURN_IF_ERROR(FindAndValidateLink(component_spec, *network_state_manager,
                                         "targets", target_dimension,
                                         &targets_handle_));
  TF_RETURN_IF_ERROR(
      network_state_manager->AddLayer("adjacency", 1, &adjacency_handle_));
  TF_RETURN_IF_ERROR(network_state_manager->AddLocal(source_dimension,
                                                     &target_product_handle_));

  return tensorflow::Status::OK();
}

tensorflow::Status BiaffineDigraphComponent::Evaluate(
    SessionState *session_state, ComputeSession *compute_session,
    ComponentTrace *component_trace) const {
  NetworkStates &network_states = session_state->network_states;

  // Infer the number of steps from the source and target activations.
  EigenMatrixMap<float> sources =
      AsEigenMap(Matrix<float>(network_states.GetLayer(sources_handle_)));
  EigenMatrixMap<float> targets =
      AsEigenMap(Matrix<float>(network_states.GetLayer(targets_handle_)));
  const size_t num_steps = sources.rows();
  if (targets.rows() != num_steps) {
    return tensorflow::errors::InvalidArgument(
        "step count mismatch between sources (", num_steps, ") and targets (",
        targets.rows(), ")");
  }

  // Since this component has a pairwise layer, allocate steps in one shot.
  network_states.AddSteps(num_steps);
  MutableEigenMatrixMap<float> adjacency =
      AsEigenMap(network_states.GetLayer(adjacency_handle_));
  MutableEigenMatrixMap<float> target_product =
      AsEigenMap(network_states.GetLocal(target_product_handle_));

  // First compute the adjacency matrix of combined arc and source scores.
  // Note: .noalias() ensures that the RHS is assigned directly to the LHS;
  // otherwise, Eigen may allocate a temp matrix to hold the result of the
  // matmul on the RHS and then copy that to the LHS.  See
  // http://eigen.tuxfamily.org/dox/TopicLazyEvaluation.html
  target_product.noalias() = targets * AsEigenMap(arc_weights_).transpose();
  target_product.rowwise() += AsEigenMap(source_weights_);
  adjacency.noalias() = target_product * sources.transpose();

  // Now overwrite the diagonal with root-selection scores.
  // Note: .array() allows the scalar addition of |root_bias_| to broadcast
  // across the diagonal.  See
  // https://eigen.tuxfamily.org/dox/group__TutorialArrayClass.html
  adjacency.diagonal().noalias() =
      AsEigenMap(root_weights_) * targets.transpose();
  adjacency.diagonal().array() += root_bias_;

  return tensorflow::Status::OK();
}

DRAGNN_RUNTIME_REGISTER_COMPONENT(BiaffineDigraphComponent);

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
