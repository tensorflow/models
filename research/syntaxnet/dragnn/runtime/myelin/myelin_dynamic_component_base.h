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

#ifndef DRAGNN_RUNTIME_MYELIN_MYELIN_DYNAMIC_COMPONENT_BASE_H_
#define DRAGNN_RUNTIME_MYELIN_MYELIN_DYNAMIC_COMPONENT_BASE_H_

#include <stddef.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "dragnn/protos/cell_trace.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/fixed_embeddings.h"
#include "dragnn/runtime/linked_embeddings.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/myelin/myelin_tracing.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "sling/myelin/compute.h"
#include "sling/myelin/flow.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/dynamic_annotations.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Base class for Myelin-based versions of DynamicComponent.

//
// Roughly, this is a base class for a version of DynamicComponent where the
// per-transition-step computation is performed by a Myelin cell instead of a
// NetworkUnit.  This class implements Initialize() and provides methods that
// can be useful for inference, but does not implement Evaluate().
//
// At initialization time, this class creates lists of configuration structs
// that associate each input or output of the Myelin cell with an operand that
// the DRAGNN runtime manages.  See, e.g., InputId and InitializeInputIds().
//
// At inference time, subclasses can bind the relevant DRAGNN runtime operands
// to the inputs and outputs of the Myelin instance (see, e.g., BindInputIds())
// and evaluate the Myelin cell.  Like DynamicComponent, the cell should be
// evaluated once per transition and the results used to advance the transition
// system state.
//
// Except as noted below, this is a drop-in replacement for DynamicComponent:
// * The name of the logits layer is hard-coded (see kLogitsName).
// * The fixed and linked channels must have embedding_dim=-1, because the fixed
//   lookups and linked multiplications are handled within Myelin.
//
// The MyelinDynamicComponent subclass provides a general-purpose implementation
// of Evaluate().  Other subclasses provide optimized implementations subject to
// restrictions on the possible network configuration.
class MyelinDynamicComponentBase : public Component {
 public:
  // Partially implements Component.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override;

 protected:
  // Configuration for a fixed feature ID input.
  //
  // TODO(googleuser): Consider making singleton inputs like the feature ID and
  // out-of-bounds indicator into plain value inputs instead of references; it
  // is equally fast to copy the value.
  struct InputId {
    // Tensor to feed with the fixed feature ID.
    sling::myelin::Tensor *id = nullptr;
  };

  // Configuration for a linked feature embedding input.
  struct InputLink {
    // Tensor to feed with the linked activation vector.
    sling::myelin::Tensor *activations = nullptr;

    // Tensor to feed with the linked out-of-bounds indicator, or null if the
    // embedding does not need to be multiplied.
    sling::myelin::Tensor *out_of_bounds = nullptr;
  };

  // Configuration for a recurrent input.
  struct InputRecurrence {
    // Handle of the output layer that is recurrently fed back.
    LayerHandle<float> handle;

    // Tensor to feed with the previous output activation vector.
    sling::myelin::Tensor *previous_output = nullptr;
  };

  // Configuration for an output layer.
  struct OutputLayer {
    // Handle of the output layer.
    LayerHandle<float> handle;

    // Tensor that writes to the layer.
    sling::myelin::Tensor *layer = nullptr;
  };

  // Name of the layer containing logits.  Unlike DynamicComponent, this class
  // does not use the NetworkUnit abstraction and assumes that the logits will
  // be stored in this layer.
  // TODO(googleuser): Make this configurable, if needed.  The logits layer could
  // be given a special alias, for example.
  static constexpr char kLogitsName[] = "logits";

  // Points the cell input |tensor| in the |instance| at the |vector|.
  template <class T>
  static void BindInput(Vector<T> vector, sling::myelin::Tensor *tensor,
                        sling::myelin::Instance *instance);

  // Points the cell output |tensor| in the |instance| at the |vector|.
  template <class T>
  static void BindOutput(MutableVector<T> vector, sling::myelin::Tensor *tensor,
                         sling::myelin::Instance *instance);

  // Binds the feature IDs in the |fixed_embeddings| to the |instance| as
  // configured by the |input_ids_|.
  void BindInputIds(const FixedEmbeddings &fixed_embeddings,
                    sling::myelin::Instance *instance) const;

  // Binds the |embedding| and, if applicable, |is_out_of_bounds| to the
  // |input_link| in the |instance|.
  void BindInputLink(Vector<float> embedding, bool is_out_of_bounds,
                     const InputLink &input_link,
                     sling::myelin::Instance *instance) const;

  // Binds the activation vectors in the |linked_embeddings| to the |instance|
  // as configured by the |input_links_|.
  void BindInputLinks(const LinkedEmbeddings &linked_embeddings,
                      sling::myelin::Instance *instance) const;

  // Binds the output of the step before |step_index| in the |network_states| to
  // the |instance| as configured by the |input_recurrences_|.
  void BindInputRecurrences(size_t step_index,
                            const NetworkStates &network_states,
                            sling::myelin::Instance *instance) const;

  // Binds the output layers for the |step_index| in the |network_states| to the
  // |instance| as configured by the |output_layers_|.
  void BindOutputLayers(size_t step_index, const NetworkStates &network_states,
                        sling::myelin::Instance *instance) const;

  // Returns the reusable fixed and linked embeddings in the |session_state|.
  FixedEmbeddings &GetFixedEmbeddings(SessionState *session_state) const;
  LinkedEmbeddings &GetLinkedEmbeddings(SessionState *session_state) const;

  // Returns the reusable Myelin instance in the |session_state|.
  sling::myelin::Instance &GetInstance(SessionState *session_state) const;

  // If |component_trace| is non-null, ensures that |step_index|+1 steps exist
  // and traces the |instance| in the |step_index|'th step.
  void MaybeTrace(size_t step_index, sling::myelin::Instance *instance,
                  ComponentTrace *component_trace) const;

  // Accessors.
  const string &name() const { return name_; }
  const FixedEmbeddingManager &fixed_embedding_manager() const {
    return fixed_embedding_manager_;
  }
  const LinkedEmbeddingManager &linked_embedding_manager() const {
    return linked_embedding_manager_;
  }
  const sling::myelin::Cell *cell() const { return cell_; }
  const std::vector<InputId> &input_ids() const { return input_ids_; }
  const std::vector<InputLink> &input_links() const { return input_links_; }
  const std::vector<InputRecurrence> &input_recurrences() const {
    return input_recurrences_;
  }
  const std::vector<OutputLayer> &output_layers() const {
    return output_layers_;
  }
  bool deterministic() const { return deterministic_; }
  LayerHandle<float> logits_handle() const { return logits_handle_; }

 private:
  // Returns non-OK if the |component_spec| specifies any unsupported settings.
  // This includes both settings that are not yet implemented and those that are
  // fundamentally incompatible with this class.
  static tensorflow::Status Validate(const ComponentSpec &component_spec);

  // Points the |vector| at the variable in the |network_| named |name|, which
  // must have a vector-like shape (i.e., having at most one dimension > 1) and
  // must match the |type|.  If the |dimension| is >= 0, then the |vector| must
  // be the same size.  On error, returns non-OK and sets |vector| to nullptr.
  // Returns NOT_FOUND iff the |name| does not name a variable.
  tensorflow::Status LookupVector(const string &name, sling::myelin::Type type,
                                  int dimension,
                                  sling::myelin::Tensor **vector) const;

  // Initializes the |input_ids_| based on the |fixed_embedding_manager_| and
  // |network_|.  On error, returns non-OK.
  tensorflow::Status InitializeInputIds();

  // Initializes the |input_links_| based on the |linked_embedding_manager_| and
  // |network_|.  On error, returns non-OK.
  tensorflow::Status InitializeInputLinks();

  // Initializes the |input_recurrences_| based on the |flow|, |manager|, and
  // |network_|.  Requires that layers have been added to the |manager|.  On
  // error, returns non-OK.
  tensorflow::Status InitializeInputRecurrences(
      const sling::myelin::Flow &flow, const NetworkStateManager &manager);

  // Initializes the |output_layers_| based on the |flow|, |manager|, and
  // |network_|. Adds layers to the |manager|.  On error, returns non-OK.
  tensorflow::Status InitializeOutputLayers(const sling::myelin::Flow &flow,
                                            NetworkStateManager *manager);

  // Initializes the constant vectors (|zero_|, |one_|, and |zeros_|) and their
  // backing |array_|.  Requires that the |input_recurrences_| are initialized.
  tensorflow::Status InitializeConstantVectors();

  // Initializes the |logits_handle_| based on the |component_spec| and
  // |manager|, if needed.
  tensorflow::Status MaybeInitializeLogits(const ComponentSpec &component_spec,
                                           const NetworkStateManager &manager);

  // Name of this component.
  string name_;

  // Managers for the fixed and linked embeddings used by the component.
  FixedEmbeddingManager fixed_embedding_manager_;
  LinkedEmbeddingManager linked_embedding_manager_;

  // Fixed and linked embeddings.
  SharedExtensionHandle<FixedEmbeddings> fixed_embeddings_handle_;
  SharedExtensionHandle<LinkedEmbeddings> linked_embeddings_handle_;

  // Library of Myelin kernels and transformations.
  sling::myelin::Library library_;

  // Myelin network that implements the cell computation.
  sling::myelin::Network network_;

  // Cell that contains the compiled code for this component.
  const sling::myelin::Cell *cell_ = nullptr;

  // List of fixed feature ID inputs, aligned with the relevant FixedEmbeddings.
  std::vector<InputId> input_ids_;

  // List of linked feature inputs, aligned with the relevant LinkedEmbeddings.
  std::vector<InputLink> input_links_;

  // List of recurrent input, not ordered.
  std::vector<InputRecurrence> input_recurrences_;

  // List of output layers, not ordered.
  std::vector<OutputLayer> output_layers_;

  // A few constant vectors and their backing array.
  UniqueAlignedArray array_;
  Vector<float> zero_;   // [0.0], for linked out-of-bounds indicators
  Vector<float> one_;    // [1.0], for linked out-of-bounds indicators
  Vector<float> zeros_;  // [0.0...0.0], for linked activation vectors

  // Whether the transition system is deterministic.
  bool deterministic_ = false;

  // Handle to the classification logits.  Valid iff |deterministic_| is false.
  LayerHandle<float> logits_handle_;

  // Instance used to evaluate the network cell.  Local, since each component
  // can have a different cell.
  LocalExtensionHandle<sling::myelin::Instance> instance_handle_;
};

// Implementation details below.

template <class T>
void MyelinDynamicComponentBase::BindInput(Vector<T> vector,
                                           sling::myelin::Tensor *tensor,
                                           sling::myelin::Instance *instance) {
  // Since Myelin only consumes non-const pointers, const_cast() is required.
  // Myelin will not modify the contents of the |vector|, provided it is bound
  // to a cell input.
  DCHECK(tensor->in()) << tensor->name();
  DCHECK(!tensor->out()) << tensor->name();
  DCHECK_LE(tensor->elements(), vector.size()) << tensor->name();
  instance->SetReference(
      tensor,
      const_cast<char *>(reinterpret_cast<const char *>(vector.data())));
}

template <class T>
void MyelinDynamicComponentBase::BindOutput(MutableVector<T> vector,
                                            sling::myelin::Tensor *tensor,
                                            sling::myelin::Instance *instance) {
  DCHECK(tensor->out()) << tensor->name();
  DCHECK_EQ(tensor->elements(), vector.size()) << tensor->name();
  instance->SetReference(tensor, reinterpret_cast<char *>(vector.data()));
  TF_ANNOTATE_MEMORY_IS_INITIALIZED(vector.data(), vector.size() * sizeof(T));
}

inline void MyelinDynamicComponentBase::BindInputIds(
    const FixedEmbeddings &fixed_embeddings,
    sling::myelin::Instance *instance) const {
  for (size_t i = 0; i < input_ids_.size(); ++i) {
    BindInput(fixed_embeddings.ids(i), input_ids_[i].id, instance);
  }
}

inline void MyelinDynamicComponentBase::BindInputLink(
    Vector<float> embedding, bool is_out_of_bounds, const InputLink &input_link,
    sling::myelin::Instance *instance) const {
  BindInput(embedding, input_link.activations, instance);
  if (input_link.out_of_bounds != nullptr) {
    BindInput(is_out_of_bounds ? one_ : zero_, input_link.out_of_bounds,
              instance);
  }
}

inline void MyelinDynamicComponentBase::BindInputLinks(
    const LinkedEmbeddings &linked_embeddings,
    sling::myelin::Instance *instance) const {
  for (size_t channel_id = 0; channel_id < input_links_.size(); ++channel_id) {
    BindInputLink(linked_embeddings.embedding(channel_id),
                  linked_embeddings.is_out_of_bounds(channel_id),
                  input_links_[channel_id], instance);
  }
}

inline void MyelinDynamicComponentBase::BindInputRecurrences(
    size_t step_index, const NetworkStates &network_states,
    sling::myelin::Instance *instance) const {
  for (const InputRecurrence &input : input_recurrences_) {
    if (step_index == 0) {
      // The previous output is out-of-bounds, so feed a zero vector.  Recall
      // that |zeros_| was constructed to be large enough for any recurrence.
      BindInput(zeros_, input.previous_output, instance);
    } else {
      BindInput(Vector<float>(
                    network_states.GetLayer(input.handle).row(step_index - 1)),
                input.previous_output, instance);
    }
  }
}

inline void MyelinDynamicComponentBase::BindOutputLayers(
    size_t step_index, const NetworkStates &network_states,
    sling::myelin::Instance *instance) const {
  for (const OutputLayer &output : output_layers_) {
    BindOutput(network_states.GetLayer(output.handle).row(step_index),
               output.layer, instance);
  }
}

inline FixedEmbeddings &MyelinDynamicComponentBase::GetFixedEmbeddings(
    SessionState *session_state) const {
  return session_state->extensions.Get(fixed_embeddings_handle_);
}

inline LinkedEmbeddings &MyelinDynamicComponentBase::GetLinkedEmbeddings(
    SessionState *session_state) const {
  return session_state->extensions.Get(linked_embeddings_handle_);
}

inline sling::myelin::Instance &MyelinDynamicComponentBase::GetInstance(
    SessionState *session_state) const {
  return session_state->extensions.Get(instance_handle_, cell_);
}

inline void MyelinDynamicComponentBase::MaybeTrace(
    size_t step_index, sling::myelin::Instance *instance,
    ComponentTrace *component_trace) const {
  if (component_trace == nullptr) return;
  while (component_trace->step_trace_size() <= step_index) {
    component_trace->add_step_trace();
  }
  TraceMyelinInstance(instance,
                      component_trace->mutable_step_trace(step_index)
                          ->AddExtension(CellTrace::step_trace_extension));
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_MYELIN_MYELIN_DYNAMIC_COMPONENT_BASE_H_
