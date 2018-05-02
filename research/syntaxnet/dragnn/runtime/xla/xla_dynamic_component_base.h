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

#ifndef DRAGNN_RUNTIME_XLA_XLA_DYNAMIC_COMPONENT_BASE_H_
#define DRAGNN_RUNTIME_XLA_XLA_DYNAMIC_COMPONENT_BASE_H_

#include <stddef.h>
#include <string.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "dragnn/protos/export.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/fixed_embeddings.h"
#include "dragnn/runtime/linked_embeddings.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/type_keyed_set.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Base class for XLA-based versions of DynamicComponent.

//
// Roughly, this is a base class for a version of DynamicComponent where the
// per-transition-step computation is performed by a XLA cell instead of a
// NetworkUnit. This class implements Initialize() and Evaluate(). It has
// the most generality w.r.t. input features and links, but suffers from
// ComputeSession overhead. Subclasses which provide specialized logic that
// replaces the generic ComputeSession should override Evaluate().
//
// XLA JIT and AOT versions of this class must supply appropriate versions
// of InitializeFromComponentSpec() and XlaStaticData().
//
// At initialization time, this class creates lists of configuration structs
// that associate each input or output of the XLA cell with an operand that
// the DRAGNN runtime manages.  See, e.g., InputId and InitializeInputIds().
//
// At inference time, subclasses can bind the relevant DRAGNN runtime operands
// to the inputs and outputs of the XLA instance (see, e.g., BindInputIds())
// and evaluate the XLA cell.  Like DynamicComponent, the cell should be
// evaluated once per transition and the results used to advance the transition
// system state.
//
// Except as noted below, this is a drop-in replacement for DynamicComponent:
// * The name of the logits layer is hard-coded (see kLogitsName).
// * The fixed and linked channels must have embedding_dim=-1, because the fixed
//   lookups and linked multiplications are handled within XLA.
//
// The XlaDynamicComponent subclass provides a general-purpose implementation
// of Evaluate().  Other subclasses provide optimized implementations subject to
// restrictions on the possible network configuration.
class XlaDynamicComponentBase : public Component {
 public:
  // Implements Component.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override;
  tensorflow::Status Evaluate(SessionState *session_state,
                              ComputeSession *compute_session,
                              ComponentTrace *component_trace) const override;

 protected:
  // Initializes the XLA function using the |component_spec|. When successful,
  // the relevant |cell_subgraph_spec| is filled in, and XlaStaticData() is safe
  // to call. On error, returns non-OK.
  virtual tensorflow::Status InitializeFromComponentSpec(
      const ComponentSpec &component_spec,
      CellSubgraphSpec *cell_subgraph_spec) = 0;

  // Returns the StaticData that identifies a specific XLA compiled cell
  // function. It is a fatal error to call this before a successful call to
  // InitializeFromSpec().
  virtual const tensorflow::XlaCompiledCpuFunction::StaticData &XlaStaticData()
      const = 0;

 private:
  // Handle to one of the inputs. The |index| is into an array of
  // pointers used by XlaCompiledCpuFunction. The input vector has
  // the given number of |elements|.
  struct InputHandle {
    int index = -1;
    int elements = 0;
  };

  // Handle to one of the outputs. This |index| is into an array of pointers
  // into the results tuple used by XlaCompiledCpuFunction.
  struct OutputHandle {
    int index = -1;
    int elements = 0;
    int64 bytes = 0;
  };

 protected:
  // Configuration for a fixed feature ID input.
  struct InputId {
    // Tensor to feed with the fixed feature ID.
    InputHandle id;
  };

  // Configuration for a linked feature embedding input.
  struct InputLink {
    // Tensor to feed with the linked activation vector.
    InputHandle activations;

    // Tensor to feed with the linked out-of-bounds indicator, or -1 if the
    // embedding does not need to be multiplied.
    InputHandle out_of_bounds;
  };

  struct InputRecurrence {
    // Handle of the output layer that is recurrently fed back.
    LayerHandle<float> handle;

    // Tensor to feed with the previous output activation vector.
    InputHandle previous_output;
  };

  // Configuration for an output layer.
  struct OutputLayer {
    // Handle of the output layer.
    LayerHandle<float> handle;

    // Tensor that writes to the layer.
    OutputHandle layer;
  };

  // Name of the layer containing logits.  Unlike DynamicComponent, this class
  // does not use the NetworkUnit abstraction and assumes that the logits will
  // be stored in this layer.
  // TODO(googleuser): Make this configurable, if needed.  The logits layer could
  // be given a special alias, for example.
  static constexpr char kLogitsName[] = "logits";

  // Points the cell input |handle| in the |instance| at the |vector|.
  // Must be called before invoking the cell.
  template <class T>
  static void BindInput(Vector<T> vector, const InputHandle &handle,
                        tensorflow::XlaCompiledCpuFunction *instance);

  // Copies the cell output |handle| in the |instance| to the |vector|.
  // Must be called after invoking the cell.
  //
  // TODO(googleuser): Consider wrapping XlaCompiledCpuFunction along with a map
  // from output indices to layer pointers, so this actually binds before the
  // call to Run(). Then add a separate function that realizes the output
  // binding, copying after Run().
  template <class T>
  static void BindOutput(MutableVector<T> vector, const OutputHandle &handle,
                         tensorflow::XlaCompiledCpuFunction *instance);

  // Binds the feature IDs in the |fixed_embeddings| to the |instance| as
  // configured by the |input_ids_|.
  void BindInputIds(const FixedEmbeddings &fixed_embeddings,
                    tensorflow::XlaCompiledCpuFunction *instance) const;

  // Binds the |embedding| and, if applicable, |is_out_of_bounds| to the
  // |input_link| in the |instance|.
  void BindInputLink(Vector<float> embedding, bool is_out_of_bounds,
                     const InputLink &input_link,
                     tensorflow::XlaCompiledCpuFunction *instance) const;

  // Binds the activation vectors in the |linked_embeddings| to the |instance|
  // as configured by the |input_links_|.
  void BindInputLinks(const LinkedEmbeddings &linked_embeddings,
                      tensorflow::XlaCompiledCpuFunction *instance) const;

  // Binds the output of the step before |step_index| in the |network_states| to
  // the |instance| as configured by the |input_recurrences_|.
  void BindInputRecurrences(size_t step_index,
                            const NetworkStates &network_states,
                            tensorflow::XlaCompiledCpuFunction *instance) const;

  // Binds the output layers for the |step_index| in the |network_states| to the
  // |instance| as configured by the |output_layers_|.
  void BindOutputLayers(size_t step_index, const NetworkStates &network_states,
                        tensorflow::XlaCompiledCpuFunction *instance) const;

  // Returns the reusable XLA instance in the |session_state|.
  tensorflow::XlaCompiledCpuFunction &GetInstance(
      SessionState *session_state) const;

  // If |component_trace| is non-null, ensures that |step_index|+1 steps exist
  // and traces the |instance| in the |step_index|'th step.
  void MaybeTrace(size_t step_index,
                  tensorflow::XlaCompiledCpuFunction *instance,
                  ComponentTrace *component_trace) const;

  // Accessors.
  const string &name() const { return name_; }
  const FixedEmbeddingManager &fixed_embedding_manager() const {
    return fixed_embedding_manager_;
  }
  const LinkedEmbeddingManager &linked_embedding_manager() const {
    return linked_embedding_manager_;
  }

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
  // Forbid batches and beams.
  static constexpr int kEvaluateNumItems = 1;

  // Required alignment of pointers to input tensors.
  static constexpr size_t kXlaByteAlignment =
      tensorflow::Allocator::kAllocatorAlignment;

  // Returns non-OK if the |component_spec| specifies any unsupported settings.
  // This includes both settings that are not yet implemented and those that are
  // fundamentally incompatible with this class.
  static tensorflow::Status Validate(const ComponentSpec &component_spec);

  // Returns non-OK if the tensor called |name| isn't compatible with |type| or
  // has an invalid |shape| given |dimension| for use as an input or output.
  // If OK, |elements_out| contains the number of elements in the vector.
  static tensorflow::Status ValidateTensor(const string &name,
                                           const xla::PrimitiveType type,
                                           int dimension,
                                           const xla::Shape &shape,
                                           int *elements_out);

  // Points the |input_handle| or |output_handle| at the variable in the
  // |network_| named |name|, which must have a vector-like shape (i.e., having
  // at most one dimension > 1) and must match the |type|. The |instance| is
  // used to determine the mapping from |name| to the handle. If the |dimension|
  // is >= 0, then the |vector| must be the same size.
  // On error, returns non-OK and sets |vector| to nullptr.
  // Returns NOT_FOUND iff the |name| does not name a variable.
  tensorflow::Status LookupInputVector(
      const string &name, const xla::PrimitiveType type, int dimension,
      const tensorflow::XlaCompiledCpuFunction &instance,
      InputHandle *input_handle) const;
  tensorflow::Status LookupOutputVector(
      const string &name, const xla::PrimitiveType type, int dimension,
      const tensorflow::XlaCompiledCpuFunction &instance,
      OutputHandle *output_handle) const;

  // Initializes the |input_ids_| based on the |fixed_embedding_manager_| and
  // |network_|.  On error, returns non-OK.
  tensorflow::Status InitializeInputIds(
      const tensorflow::XlaCompiledCpuFunction &instance);

  // Initializes the |input_links_| based on the |linked_embedding_manager_| and
  // |network_|.  On error, returns non-OK.
  tensorflow::Status InitializeInputLinks(
      const tensorflow::XlaCompiledCpuFunction &instance);

  // Initializes the |input_recurrences_| based on the |config|, |manager|, and
  // |network_|.  Requires that layers have been added to the |manager|.  On
  // error, returns non-OK.
  tensorflow::Status InitializeInputRecurrences(
      const CellSubgraphSpec &cell_subgraph_spec,
      const NetworkStateManager &manager,
      const tensorflow::XlaCompiledCpuFunction &instance);

  // Initializes the |output_layers_| based on the |config|, |manager|, and
  // |network_|. Adds layers to the |manager|.  On error, returns non-OK.
  tensorflow::Status InitializeOutputLayers(
      const CellSubgraphSpec &cell_subgraph_spec, NetworkStateManager *manager,
      const tensorflow::XlaCompiledCpuFunction &instance);

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

  // The StaticData that identifies the XLA compiled function that implements
  // the network cell.  Cached to reduce virtual call overhead.
  const tensorflow::XlaCompiledCpuFunction::StaticData *static_data_ = nullptr;

  // Description of shapes and types of the compiled function, with indices that
  // correspond to InputHandle and OutputHandle index values.
  const xla::ProgramShape *program_shape_ = nullptr;

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

  // Compiled function that implements the network cell.  Local, since each
  // component can have a different cell.
  LocalExtensionHandle<tensorflow::XlaCompiledCpuFunction> instance_handle_;
};

// Implementation details below.

template <class T>
void XlaDynamicComponentBase::BindInput(
    Vector<T> vector, const InputHandle &handle,
    tensorflow::XlaCompiledCpuFunction *instance) {
  DCHECK_GE(handle.index, 0);
  DCHECK_EQ(reinterpret_cast<size_t>(vector.data()) % kXlaByteAlignment, 0);

  // Since XLA only consumes non-const pointers, const_cast() is required.
  // XLA will not modify the contents of the |vector|, provided it is bound
  // to a cell input.
  instance->set_arg_data(
      handle.index,
      const_cast<void *>(reinterpret_cast<const void *>(vector.data())));
}

template <class T>
void XlaDynamicComponentBase::BindOutput(
    MutableVector<T> vector, const OutputHandle &handle,
    tensorflow::XlaCompiledCpuFunction *instance) {
  DCHECK_GE(handle.index, 0);

  // XLA retains control over the allocation of outputs, and the pointer
  // to the output must be determined using result_data() after every call
  // to Run(). The outputs are copied into the session tensors.
  std::memcpy(vector.data(), instance->result_data(handle.index), handle.bytes);
}

inline void XlaDynamicComponentBase::BindInputIds(
    const FixedEmbeddings &fixed_embeddings,
    tensorflow::XlaCompiledCpuFunction *instance) const {
  for (size_t i = 0; i < input_ids_.size(); ++i) {
    BindInput(fixed_embeddings.ids(i), input_ids_[i].id, instance);
  }
}

inline void XlaDynamicComponentBase::BindInputLink(
    Vector<float> embedding, bool is_out_of_bounds, const InputLink &input_link,
    tensorflow::XlaCompiledCpuFunction *instance) const {
  BindInput(embedding, input_link.activations, instance);
  if (input_link.out_of_bounds.index != -1) {
    BindInput(is_out_of_bounds ? one_ : zero_, input_link.out_of_bounds,
              instance);
  }
}

inline void XlaDynamicComponentBase::BindInputLinks(
    const LinkedEmbeddings &linked_embeddings,
    tensorflow::XlaCompiledCpuFunction *instance) const {
  for (size_t i = 0; i < input_links_.size(); ++i) {
    BindInputLink(linked_embeddings.embedding(i),
                  linked_embeddings.is_out_of_bounds(i), input_links_[i],
                  instance);
  }
}

inline void XlaDynamicComponentBase::BindInputRecurrences(
    size_t step_index, const NetworkStates &network_states,
    tensorflow::XlaCompiledCpuFunction *instance) const {
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

inline void XlaDynamicComponentBase::BindOutputLayers(
    size_t step_index, const NetworkStates &network_states,
    tensorflow::XlaCompiledCpuFunction *instance) const {
  for (const OutputLayer &output : output_layers_) {
    BindOutput(network_states.GetLayer(output.handle).row(step_index),
               output.layer, instance);
  }
}

inline tensorflow::XlaCompiledCpuFunction &XlaDynamicComponentBase::GetInstance(
    SessionState *session_state) const {
  return session_state->extensions.Get(
      instance_handle_, *static_data_,
      tensorflow::XlaCompiledCpuFunction::AllocMode::
          RESULTS_PROFILES_AND_TEMPS_ONLY);
}

inline void XlaDynamicComponentBase::MaybeTrace(
    size_t step_index, tensorflow::XlaCompiledCpuFunction * /*instance*/,
    ComponentTrace *component_trace) const {
  if (component_trace == nullptr) return;
  while (component_trace->step_trace_size() <= step_index) {
    component_trace->add_step_trace();
  }

  // TODO(googleuser): Add once the JIT API supports this.
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_XLA_XLA_DYNAMIC_COMPONENT_BASE_H_
