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

#include "dragnn/runtime/myelin/myelin_dynamic_component_base.h"

#include <string.h>
#include <algorithm>
#include <set>

#include "dragnn/runtime/myelin/myelin_spec_utils.h"
#include "dragnn/runtime/transition_system_traits.h"
#include "tensorflow/core/lib/core/errors.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

constexpr char MyelinDynamicComponentBase::kLogitsName[];

tensorflow::Status MyelinDynamicComponentBase::Validate(
    const ComponentSpec &component_spec) {
  if (!component_spec.attention_component().empty()) {
    return tensorflow::errors::Unimplemented("Attention is not supported");
  }

  for (const auto &fixed_feature : component_spec.fixed_feature()) {
    if (fixed_feature.embedding_dim() != -1) {
      return tensorflow::errors::InvalidArgument(
          "Myelin requires non-embedded fixed features");
    }
  }

  for (const auto &linked_feature : component_spec.linked_feature()) {
    if (linked_feature.embedding_dim() != -1) {
      return tensorflow::errors::InvalidArgument(
          "Myelin requires non-multiplied linked features");
    }
  }

  return tensorflow::Status::OK();
}

tensorflow::Status MyelinDynamicComponentBase::LookupVector(
    const string &name, sling::myelin::Type type, int dimension,
    sling::myelin::Tensor **vector) const {
  *vector = nullptr;  // so it is null if we error out
  sling::myelin::Tensor *tensor = network_.GetParameter(name);
  if (tensor == nullptr) {
    return tensorflow::errors::NotFound("No Myelin tensor named '", name, "'");
  }

  if (tensor->type() != type) {
    return tensorflow::errors::InvalidArgument(
        "Myelin tensor has wrong type '", name, "' ", tensor->TypeString(),
        " (expected ", sling::myelin::TypeTraits::of(type).name(), ")");
  }

  int num_nontrivial_dims = 0;
  for (int i = 0; i < tensor->rank(); ++i) {
    if (tensor->dim(i) > 1) ++num_nontrivial_dims;
  }
  if (num_nontrivial_dims > 1) {
    return tensorflow::errors::InvalidArgument(
        "Myelin tensor has non-vector-like shape: '", name, "' ",
        tensor->TypeString());
  }

  // Since the |tensor| is vector-like, elements() is equivalent to the vector
  // dimension and smooths over edges like rank=0.
  if (dimension >= 0 && tensor->elements() != dimension) {
    return tensorflow::errors::InvalidArgument(
        "Myelin vector has the wrong dimension '", name, "' ",
        tensor->TypeString(), " (expected ", dimension, ")");
  }

  if (internal::kAlignmentBytes % tensor->byte_alignment() != 0) {
    return tensorflow::errors::FailedPrecondition(
        "Myelin vector '", name, "' has incompatible byte alignment ",
        tensor->byte_alignment(), " (vs ", internal::kAlignmentBytes, ")");
  }

  for (int i = 0; i < tensor->rank(); ++i) {
    if (internal::kAlignmentBytes % tensor->minalign(i) != 0) {
      return tensorflow::errors::FailedPrecondition(
          "Myelin vector '", name, "' has incompatible minimum alignment ",
          tensor->minalign(i), " for dimension ", i, " (vs ",
          internal::kAlignmentBytes, ")");
    }
  }

  // Success; update |vector| to non-null.
  *vector = tensor;
  return tensorflow::Status::OK();
}

tensorflow::Status MyelinDynamicComponentBase::InitializeInputIds() {
  const int num_channels = fixed_embedding_manager_.num_channels();
  input_ids_.resize(fixed_embedding_manager_.num_embeddings());
  for (int channel_id = 0; channel_id < num_channels; ++channel_id) {
    DCHECK(!fixed_embedding_manager_.is_embedded(channel_id));
    const int channel_base = fixed_embedding_manager_.channel_base(channel_id);
    const int channel_size = fixed_embedding_manager_.channel_size(channel_id);
    for (int index = 0; index < channel_size; ++index) {
      InputId &input = input_ids_[channel_base + index];
      const string name = MakeMyelinInputFixedFeatureIdName(channel_id, index);
      TF_RETURN_IF_ERROR(
          LookupVector(name, sling::myelin::DT_INT32, 1, &input.id));
      VLOG(1) << "Component '" << name_ << "' fixed channel " << channel_id
              << " index " << index << ": Added feature ID";
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status MyelinDynamicComponentBase::InitializeInputLinks() {
  const int num_channels = linked_embedding_manager_.num_channels();
  input_links_.resize(num_channels);
  for (int channel_id = 0; channel_id < num_channels; ++channel_id) {
    InputLink &input = input_links_[channel_id];
    const int dimension = linked_embedding_manager_.embedding_dim(channel_id);
    const string activations_name =
        MakeMyelinInputLinkedActivationVectorName(channel_id);
    const string out_of_bounds_name =
        MakeMyelinInputLinkedOutOfBoundsIndicatorName(channel_id);
    TF_RETURN_IF_ERROR(LookupVector(activations_name, sling::myelin::DT_FLOAT,
                                    dimension, &input.activations));
    VLOG(1) << "Component '" << name_ << "' linked channel " << channel_id
            << ": Added activations";

    // Allow NOT_FOUND, for linked embedding channels that don't multiply the
    // input activations with an embedding matrix.
    const tensorflow::Status status = LookupVector(
        out_of_bounds_name, sling::myelin::DT_FLOAT, 1, &input.out_of_bounds);
    if (status.ok()) {
      VLOG(1) << "Component '" << name_ << "' linked channel " << channel_id
              << ": Added out-of-bounds indicator for multiplication";
    } else if (status.code() == tensorflow::error::NOT_FOUND) {
      VLOG(1) << "Component '" << name_ << "' linked channel " << channel_id
              << ": No out-of-bounds indicator; not multiplied";
    } else {
      return status;
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status MyelinDynamicComponentBase::InitializeInputRecurrences(
    const sling::myelin::Flow &flow, const NetworkStateManager &manager) {
  for (const string &layer_name : GetRecurrentLayerNames(flow)) {
    input_recurrences_.emplace_back();
    InputRecurrence &input = input_recurrences_.back();
    const string name = MakeMyelinInputRecurrentLayerName(layer_name);
    size_t dimension = 1;
    TF_RETURN_IF_ERROR(
        manager.LookupLayer(name_, layer_name, &dimension, &input.handle));
    TF_RETURN_IF_ERROR(LookupVector(name, sling::myelin::DT_FLOAT, dimension,
                                    &input.previous_output));
    VLOG(1) << "Component '" << name_ << "' recurrence '" << layer_name
            << "': Added link to previous output";
  }
  return tensorflow::Status::OK();
}

tensorflow::Status MyelinDynamicComponentBase::InitializeOutputLayers(
    const sling::myelin::Flow &flow, NetworkStateManager *manager) {
  // Mapping from cell output tensor to layer name, for detecting layer aliases.
  std::map<const sling::myelin::Tensor *, string> tensor_to_layer;
  for (const string &layer_name : GetOutputLayerNames(flow)) {
    output_layers_.emplace_back();
    OutputLayer &output = output_layers_.back();
    const string name = MakeMyelinOutputLayerName(layer_name);
    TF_RETURN_IF_ERROR(
        LookupVector(name, sling::myelin::DT_FLOAT, -1, &output.layer));

    // Add a new output layer or create an alias to an existing one.
    if (tensor_to_layer.find(output.layer) == tensor_to_layer.end()) {
      tensor_to_layer[output.layer] = layer_name;
      const size_t dimension = output.layer->elements();
      TF_RETURN_IF_ERROR(
          manager->AddLayer(layer_name, dimension, &output.handle));
      VLOG(1) << "Component '" << name_ << "' output '" << layer_name
              << "': Added new layer";
    } else {
      const string &original_name = tensor_to_layer[output.layer];
      output_layers_.pop_back();  // not a "real" output
      TF_RETURN_IF_ERROR(manager->AddLayerAlias(layer_name, original_name));
      VLOG(1) << "Component '" << name_ << "' output '" << layer_name
              << "': Alias of '" << original_name << "'";
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status MyelinDynamicComponentBase::InitializeConstantVectors() {
  // Find the maximum recurrent layer dimension; the |zeros_| must be this big.
  int max_dimension = 1;  // ensure at least one element, for |zero_|
  for (const InputRecurrence &input : input_recurrences_) {
    max_dimension = std::max(max_dimension, input.previous_output->elements());
  }

  // Allocate the backing array and parcel it out into sub-views.
  const std::vector<size_t> sizes = {sizeof(float),
                                     max_dimension * sizeof(float)};
  array_.Reset(ComputeTotalBytesWithAlignmentPadding(sizes));
  memset(array_.view().data(), 0, array_.view().size());  // = 0.0 for float
  std::vector<MutableAlignedView> views;
  TF_RETURN_IF_ERROR(array_.view().Split(sizes, &views));
  DCHECK_EQ(views.size(), 2);

  // Promote to typed vectors.
  one_ = Vector<float>(views[0]);
  zero_ = Vector<float>(views[1], 1);
  zeros_ = Vector<float>(views[1]);
  DCHECK_EQ(zero_.size(), 1);
  DCHECK_EQ(one_.size(), 1);
  DCHECK_EQ(zeros_.size(), max_dimension);

  // All memory was already zeroed, so only |one_| needs to be initialized.
  MutableVector<float> mutable_one(views[0]);
  mutable_one[0] = 1.0;
  return tensorflow::Status::OK();
}

tensorflow::Status MyelinDynamicComponentBase::MaybeInitializeLogits(
    const ComponentSpec &component_spec, const NetworkStateManager &manager) {
  // Logits are unnecessary when the component is deterministic.
  deterministic_ = TransitionSystemTraits(component_spec).is_deterministic;
  if (deterministic_) return tensorflow::Status::OK();

  size_t dimension = 0;
  TF_RETURN_IF_ERROR(
      manager.LookupLayer(name_, kLogitsName, &dimension, &logits_handle_));

  if (dimension != component_spec.num_actions()) {
    return tensorflow::errors::InvalidArgument(
        "Dimension mismatch between classification logits (", dimension,
        ") and ComponentSpec.num_actions (", component_spec.num_actions(),
        ") in component '", name_, "'");
  }

  return tensorflow::Status::OK();
}

tensorflow::Status MyelinDynamicComponentBase::Initialize(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    NetworkStateManager *network_state_manager,
    ExtensionManager *extension_manager) {
  name_ = component_spec.name();
  TF_RETURN_IF_ERROR(Validate(component_spec));

  const Resource *resource = nullptr;
  TF_RETURN_IF_ERROR(LookupMyelinFlowResource(component_spec, &resource));
  const string &flow_path = resource->part(0).file_pattern();

  sling::myelin::Flow flow;
  TF_RETURN_IF_ERROR(LoadMyelinFlow(flow_path, &flow));
  VLOG(1) << "Original Flow for '" << name_ << "':\n" << flow.ToString();

  // TODO(googleuser): Add support for optional profiling, via something like:
  // if (...) network_.set_profiling(true)
  // network_.Compile(flow, library);
  // ...
  // instance->Compute();
  // sling::myelin::Profile profile(instance);
  // VLOG(1) << profile.ASCIIReport();
  RegisterMyelinLibraries(&library_);
  flow.Analyze(library_);
  VLOG(1) << "Analyzed Flow for '" << name_ << "':\n" << flow.ToString();
  if (!network_.Compile(flow, library_)) {
    return tensorflow::errors::Internal(
        "Failed to compile Myelin network for component '", name_, "'");
  }

  cell_ = network_.GetCell(name_);
  if (cell_ == nullptr) {
    return tensorflow::errors::FailedPrecondition(
        "No function named '", name_, "' in Myelin network for component '",
        name_, "'");
  }
  VLOG(1) << name_ << ": " << cell_->code().size() << " bytes of Myelin code";

  // Configure the inputs and outputs of the Myelin cell.  As with NetworkUnit
  // and NetworkUnitBase, output layers and input features must be initialized
  // in a particular order to enable recurrent inputs.  Specifically, we must
  // populate output layers first, so they are available for recurrent access,
  // both by the |input_recurrences_| and the |linked_embedding_manager_|.
  TF_RETURN_IF_ERROR(InitializeOutputLayers(flow, network_state_manager));

  TF_RETURN_IF_ERROR(fixed_embedding_manager_.Reset(
      component_spec, variable_store, network_state_manager));
  TF_RETURN_IF_ERROR(linked_embedding_manager_.Reset(
      component_spec, variable_store, network_state_manager));

  TF_RETURN_IF_ERROR(InitializeInputIds());
  TF_RETURN_IF_ERROR(InitializeInputLinks());
  TF_RETURN_IF_ERROR(InitializeInputRecurrences(flow, *network_state_manager));

  TF_RETURN_IF_ERROR(InitializeConstantVectors());
  TF_RETURN_IF_ERROR(
      MaybeInitializeLogits(component_spec, *network_state_manager));

  extension_manager->GetShared(&fixed_embeddings_handle_);
  extension_manager->GetShared(&linked_embeddings_handle_);
  extension_manager->AddLocal(&instance_handle_);
  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
