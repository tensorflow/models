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

#include "dragnn/runtime/xla/xla_dynamic_component_base.h"

#include <string.h>
#include <algorithm>

#include "dragnn/protos/export.pb.h"
#include "dragnn/runtime/transition_system_traits.h"
#include "dragnn/runtime/xla/xla_spec_utils.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

constexpr char XlaDynamicComponentBase::kLogitsName[];

tensorflow::Status XlaDynamicComponentBase::Validate(
    const ComponentSpec &component_spec) {
  if (!component_spec.attention_component().empty()) {
    return tensorflow::errors::Unimplemented("Attention is not supported");
  }

  for (const auto &fixed_feature : component_spec.fixed_feature()) {
    if (fixed_feature.embedding_dim() != -1) {
      return tensorflow::errors::InvalidArgument(
          "XLA requires non-embedded fixed features");
    }
  }

  for (const auto &linked_feature : component_spec.linked_feature()) {
    if (linked_feature.embedding_dim() != -1) {
      return tensorflow::errors::InvalidArgument(
          "XLA requires non-multiplied linked features");
    }
  }

  return tensorflow::Status::OK();
}

tensorflow::Status XlaDynamicComponentBase::ValidateTensor(
    const string &name, const xla::PrimitiveType type, int dimension,
    const xla::Shape &shape, int *elements_out) {
  if (shape.element_type() != type) {
    return tensorflow::errors::InvalidArgument(
        "XLA tensor '", name, "' has wrong type ",
        xla::PrimitiveType_Name(shape.element_type()), " (expected ",
        xla::PrimitiveType_Name(type), ")");
  }

  int num_nontrivial_dims = 0;
  int64 elements = 1;
  for (int64 dim : shape.dimensions()) {
    if (dim > 1) {
      ++num_nontrivial_dims;
      elements *= dim;
    }
  }
  if (num_nontrivial_dims > 1) {
    return tensorflow::errors::InvalidArgument(
        "XLA tensor has non-vector-like shape: '", name, "' ",
        xla::ShapeUtil::HumanString(shape));
  }
  if (dimension >= 0 && elements != dimension) {
    return tensorflow::errors::InvalidArgument(
        "XLA input shape has the wrong dimension '", name, "' ",
        xla::ShapeUtil::HumanString(shape), " (expected ", dimension, ")");
  }
  *elements_out = static_cast<int>(elements);

  return tensorflow::Status::OK();
}

tensorflow::Status XlaDynamicComponentBase::LookupInputVector(
    const string &name, const xla::PrimitiveType type, int dimension,
    const tensorflow::XlaCompiledCpuFunction &instance,
    InputHandle *input_handle) const {
  input_handle->index = -1;  // set to invalid if we error out

  const int index = instance.LookupArgIndex(name);
  if (index == -1 || index >= program_shape_->parameters_size()) {
    return tensorflow::errors::NotFound("No XLA tensor named '", name, "'");
  }

  const xla::Shape &shape = program_shape_->parameters(index);
  TF_RETURN_IF_ERROR(
      ValidateTensor(name, type, dimension, shape, &input_handle->elements));
  input_handle->index = index;

  return tensorflow::Status::OK();
}

tensorflow::Status XlaDynamicComponentBase::LookupOutputVector(
    const string &name, const xla::PrimitiveType type, int dimension,
    const tensorflow::XlaCompiledCpuFunction &instance,
    OutputHandle *output_handle) const {
  output_handle->index = -1;  // set to invalid if we error out

  const int index = instance.LookupResultIndex(name);
  if (index == -1) {
    return tensorflow::errors::NotFound("No XLA tensor named '", name, "'");
  }
  const xla::Shape &result_shape = program_shape_->result();
  if (result_shape.element_type() != xla::TUPLE) {
    return tensorflow::errors::InvalidArgument("XLA output is not a tuple");
  }
  if (index >= result_shape.tuple_shapes_size()) {
    return tensorflow::errors::InvalidArgument("Invalid XLA output index: ",
                                               index);
  }
  const xla::Shape &shape = result_shape.tuple_shapes(index);

  TF_RETURN_IF_ERROR(
      ValidateTensor(name, type, dimension, shape, &output_handle->elements));
  output_handle->index = index;
  output_handle->bytes = xla::ShapeUtil::ByteSizeOf(shape);

  return tensorflow::Status::OK();
}

tensorflow::Status XlaDynamicComponentBase::InitializeInputIds(
    const tensorflow::XlaCompiledCpuFunction &instance) {
  const int num_channels = fixed_embedding_manager_.num_channels();
  input_ids_.resize(fixed_embedding_manager_.num_embeddings());
  for (int channel_id = 0; channel_id < num_channels; ++channel_id) {
    DCHECK(!fixed_embedding_manager_.is_embedded(channel_id));
    const int channel_base = fixed_embedding_manager_.channel_base(channel_id);
    const int channel_size = fixed_embedding_manager_.channel_size(channel_id);
    for (int index = 0; index < channel_size; ++index) {
      InputId &input = input_ids_[channel_base + index];
      const string name = MakeXlaInputFixedFeatureIdName(channel_id, index);
      TF_RETURN_IF_ERROR(
          LookupInputVector(name, xla::S32, 1, instance, &input.id));
      VLOG(1) << "Component '" << name_ << "' fixed channel " << channel_id
              << " index " << index << ": Added feature ID";
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status XlaDynamicComponentBase::InitializeInputLinks(
    const tensorflow::XlaCompiledCpuFunction &instance) {
  const int num_channels = linked_embedding_manager_.num_channels();
  input_links_.resize(num_channels);
  for (int channel_id = 0; channel_id < num_channels; ++channel_id) {
    InputLink &input = input_links_[channel_id];
    const int dimension = linked_embedding_manager_.embedding_dim(channel_id);
    const string activations_name =
        MakeXlaInputLinkedActivationVectorName(channel_id);
    const string out_of_bounds_name =
        MakeXlaInputLinkedOutOfBoundsIndicatorName(channel_id);
    TF_RETURN_IF_ERROR(LookupInputVector(activations_name, xla::F32, dimension,
                                         instance, &input.activations));
    VLOG(1) << "Component '" << name_ << "' linked channel " << channel_id
            << ": Added activations";

    // Allow NOT_FOUND, for linked embedding channels that don't multiply the
    // input activations with an embedding matrix.
    const tensorflow::Status status = LookupInputVector(
        out_of_bounds_name, xla::F32, 1, instance, &input.out_of_bounds);
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

tensorflow::Status XlaDynamicComponentBase::InitializeInputRecurrences(
    const CellSubgraphSpec &cell_subgraph_spec,
    const NetworkStateManager &manager,
    const tensorflow::XlaCompiledCpuFunction &instance) {
  for (const auto &cell_input : cell_subgraph_spec.input()) {
    if (cell_input.type() != CellSubgraphSpec::Input::TYPE_RECURRENT) continue;

    const string &layer_name = cell_input.name();
    input_recurrences_.emplace_back();
    InputRecurrence &input = input_recurrences_.back();
    const string name = MakeXlaInputRecurrentLayerName(layer_name);
    size_t dimension = 1;
    TF_RETURN_IF_ERROR(
        manager.LookupLayer(name_, layer_name, &dimension, &input.handle));
    TF_RETURN_IF_ERROR(LookupInputVector(name, xla::F32, dimension, instance,
                                         &input.previous_output));
    VLOG(1) << "Component '" << name_ << "' recurrence '" << layer_name
            << "': Added link to previous output";
  }
  return tensorflow::Status::OK();
}

tensorflow::Status XlaDynamicComponentBase::InitializeOutputLayers(
    const CellSubgraphSpec &cell_subgraph_spec, NetworkStateManager *manager,
    const tensorflow::XlaCompiledCpuFunction &instance) {
  // Mapping from output tensor name to layer name, for detecting layer aliases.
  std::map<string, string> tensor_to_layer;
  for (const auto &cell_output : cell_subgraph_spec.output()) {
    const string &layer_name = cell_output.name();
    output_layers_.emplace_back();
    OutputLayer &output = output_layers_.back();
    const string name = MakeXlaOutputLayerName(layer_name);

    // Add a new output layer or create an alias to an existing one.
    if (tensor_to_layer.find(cell_output.tensor()) == tensor_to_layer.end()) {
      TF_RETURN_IF_ERROR(
          LookupOutputVector(name, xla::F32, -1, instance, &output.layer));

      tensor_to_layer[cell_output.tensor()] = layer_name;
      const size_t dimension = output.layer.elements;
      TF_RETURN_IF_ERROR(
          manager->AddLayer(layer_name, dimension, &output.handle));
      VLOG(1) << "Component '" << name_ << "' output '" << layer_name
              << "': Added new layer";
    } else {
      const string &original_name = tensor_to_layer[cell_output.tensor()];
      output_layers_.pop_back();  // not a "real" output
      TF_RETURN_IF_ERROR(manager->AddLayerAlias(layer_name, original_name));
      VLOG(1) << "Component '" << name_ << "' output '" << layer_name
              << "': Alias of '" << original_name << "'";
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status XlaDynamicComponentBase::InitializeConstantVectors() {
  // Find the maximum recurrent layer dimension; the |zeros_| must be this big.
  int max_dimension = 1;  // ensure at least one element, for |zero_|
  for (const InputRecurrence &input : input_recurrences_) {
    max_dimension = std::max(max_dimension, input.previous_output.elements);
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

tensorflow::Status XlaDynamicComponentBase::MaybeInitializeLogits(
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

tensorflow::Status XlaDynamicComponentBase::Initialize(
    const ComponentSpec &component_spec, VariableStore *variable_store,
    NetworkStateManager *network_state_manager,
    ExtensionManager *extension_manager) {
  name_ = component_spec.name();
  TF_RETURN_IF_ERROR(Validate(component_spec));

  CellSubgraphSpec cell_subgraph_spec;
  TF_RETURN_IF_ERROR(
      InitializeFromComponentSpec(component_spec, &cell_subgraph_spec));

  // Cache the XLA StaticData after InitializeFromComponentSpec().
  static_data_ = &XlaStaticData();

  // Make a temporary instance to determine shape and input/output indices.
  tensorflow::XlaCompiledCpuFunction instance(
      *static_data_, tensorflow::XlaCompiledCpuFunction::AllocMode::
                         RESULTS_PROFILES_AND_TEMPS_ONLY);

  program_shape_ = instance.ProgramShape();
  if (program_shape_ == nullptr) {
    // Note: this fails when the proto dependency is missing.
    return tensorflow::errors::InvalidArgument("XLA program shape missing");
  }
  VLOG(1) << "XLA program shape = " << program_shape_->DebugString();

  // Configure the inputs and outputs of the XLA cell.  As with NetworkUnit
  // and NetworkUnitBase, output layers and input features must be initialized
  // in a particular order to enable recurrent inputs.  Specifically, we must
  // populate output layers first, so they are available for recurrent access,
  // both by the |input_recurrences_| and the |linked_embedding_manager_|.
  TF_RETURN_IF_ERROR(InitializeOutputLayers(cell_subgraph_spec,
                                            network_state_manager, instance));

  TF_RETURN_IF_ERROR(fixed_embedding_manager_.Reset(
      component_spec, variable_store, network_state_manager));
  TF_RETURN_IF_ERROR(linked_embedding_manager_.Reset(
      component_spec, variable_store, network_state_manager));

  TF_RETURN_IF_ERROR(InitializeInputIds(instance));
  TF_RETURN_IF_ERROR(InitializeInputLinks(instance));
  TF_RETURN_IF_ERROR(InitializeInputRecurrences(
      cell_subgraph_spec, *network_state_manager, instance));

  TF_RETURN_IF_ERROR(InitializeConstantVectors());
  TF_RETURN_IF_ERROR(
      MaybeInitializeLogits(component_spec, *network_state_manager));

  extension_manager->GetShared(&fixed_embeddings_handle_);
  extension_manager->GetShared(&linked_embeddings_handle_);
  extension_manager->AddLocal(&instance_handle_);
  return tensorflow::Status::OK();
}

tensorflow::Status XlaDynamicComponentBase::Evaluate(
    SessionState *session_state, ComputeSession *compute_session,
    ComponentTrace *component_trace) const {
  NetworkStates &network_states = session_state->network_states;
  FixedEmbeddings &fixed_embeddings =
      session_state->extensions.Get(fixed_embeddings_handle_);
  LinkedEmbeddings &linked_embeddings =
      session_state->extensions.Get(linked_embeddings_handle_);

  tensorflow::XlaCompiledCpuFunction &instance = GetInstance(session_state);

  for (size_t step_index = 0; !compute_session->IsTerminal(name());
       ++step_index) {
    network_states.AddStep();
    TF_RETURN_IF_ERROR(fixed_embeddings.Reset(&fixed_embedding_manager(),
                                              network_states, compute_session));
    TF_RETURN_IF_ERROR(linked_embeddings.Reset(
        &linked_embedding_manager(), network_states, compute_session));

    // Bind inputs into the |instance|.
    BindInputIds(fixed_embeddings, &instance);
    BindInputLinks(linked_embeddings, &instance);
    BindInputRecurrences(step_index, network_states, &instance);

    // Invoke the cell in the |instance|.
    if (!instance.Run()) {
      return tensorflow::errors::Internal("Error executing cell for ", name(),
                                          ": ", instance.error_msg());
    }

    // Realizes the binding: copy outputs out of the |instance|.
    BindOutputLayers(step_index, network_states, &instance);

    MaybeTrace(step_index, &instance, component_trace);

    // If the component is deterministic, take the oracle transition instead of
    // predicting the next transition using the logits.
    if (deterministic()) {
      compute_session->AdvanceFromOracle(name());
    } else {
      // AddStep() may invalidate the logits (due to reallocation), so the layer
      // lookup cannot be hoisted out of this loop.
      const Vector<float> logits(
          network_states.GetLayer(logits_handle()).row(step_index));
      if (!compute_session->AdvanceFromPrediction(
              name(), logits.data(), kEvaluateNumItems, logits.size())) {
        return tensorflow::errors::Internal(
            "Error in ComputeSession::AdvanceFromPrediction()");
      }
    }
  }

  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
