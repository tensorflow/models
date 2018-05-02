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

// Utils for declaring, allocating, and retrieving network states, similar to
// the "NetworkState" class and the "network_states" argument to the build_*()
// methods of ComponentBuilderBase; see component.py.
//
// In brief, a DRAGNN network consists of a sequence of named components, each
// of which produces a set of named output layers.  Each component can access
// its own layers as well as those of preceding components.  Components can also
// access "local operands", which are like layers but private to that particular
// component.  Local operands can be useful for, e.g., caching an intermediate
// result in a complex computation.
//
// For example, suppose a network has two components: "tagger" and "parser",
// where the parser uses the hidden activations of the tagger.  In this case,
// the tagger can add a layer called "hidden" at init time and fill that layer
// at processing time.  Corespondingly, the parser can look for a layer called
// "hidden" in the "tagger" component at init time, and read the activations at
// processing time.  (Note that for convenience, such links should be handled
// using the utils in linked_embeddings.h).
//
// As another example, suppose we are implementing an LSTM and we wish to keep
// the cell state private.  In this case, the LSTM component could add a layer
// for exporting the hidden activations and a local matrix for the sequence of
// cell states.  A more compact approach is to use two local vectors instead,
// one for even steps and the other for odd steps.

#ifndef DRAGNN_RUNTIME_NETWORK_STATES_H_
#define DRAGNN_RUNTIME_NETWORK_STATES_H_

#include <stddef.h>
#include <stdint.h>
#include <map>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/operands.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Opaque handles used to access typed layers or local operands.
template <class T>
class LayerHandle;
template <class T>
class PairwiseLayerHandle;
template <class T>
class LocalVectorHandle;
template <class T>
class LocalMatrixHandle;

// A class that manages the state of a DRAGNN network and associates each layer
// and local operand with a handle.  Layer and local operand contents can be
// retrieved using these handles; see NetworkStates below.
class NetworkStateManager {
 public:
  // Creates an empty manager.
  NetworkStateManager() = default;

  // Adds a component named |name| and makes it the current component.  The
  // |name| must be unique in the network.  Components are sequenced in the
  // order they are added.  On error, returns non-OK and modifies nothing.
  tensorflow::Status AddComponent(const string &name);

  // Adds a layer named |name| to the current component and sets |handle| to its
  // handle.  The |name| must be unique in the current component.  The layer is
  // realized as a Matrix<T> with one row per step and |dimension| columns.  On
  // error, returns non-OK and modifies nothing.
  template <class T>
  tensorflow::Status AddLayer(const string &name, size_t dimension,
                              LayerHandle<T> *handle);

  // As above, but for pairwise layers.
  template <class T>
  tensorflow::Status AddLayer(const string &name, size_t dimension,
                              PairwiseLayerHandle<T> *handle);

  // As above, but for a local Vector<T> or Matrix<T> operand.  The operand is
  // "local" in the sense that only the caller knows its handle.
  template <class T>
  tensorflow::Status AddLocal(size_t dimension, LocalVectorHandle<T> *handle);
  template <class T>
  tensorflow::Status AddLocal(size_t dimension, LocalMatrixHandle<T> *handle);

  // Makes |alias| an alias of the layer named |name| in the current component,
  // so that lookups of |alias| resolve to |name|.  The |name| must already
  // exist as a layer, and layer names and aliases must be unique within each
  // component.  On error, returns non-OK and modifies nothing.
  tensorflow::Status AddLayerAlias(const string &alias, const string &name);

  // Finds the layer that matches |layer_name_or_alias| in the component named
  // |component_name|.  Sets |dimension| to its dimension and |handle| to its
  // handle.  On error, returns non-OK and modifies nothing.
  template <class T>
  tensorflow::Status LookupLayer(const string &component_name,
                                 const string &layer_name_or_alias,
                                 size_t *dimension,
                                 LayerHandle<T> *handle) const;

  // As above, but for pairwise layers.
  template <class T>
  tensorflow::Status LookupLayer(const string &component_name,
                                 const string &layer_name_or_alias,
                                 size_t *dimension,
                                 PairwiseLayerHandle<T> *handle) const;

 private:
  friend class NetworkStates;

  // Configuration information for a layer.
  struct LayerConfig {
    // Creates a config for a layer with the |name|, |type| ID, and |handle|.
    LayerConfig(const string &name, std::type_index type, OperandHandle handle)
        : name(name), type(type), handle(handle) {}

    // Name of the layer.
    string name;

    // Type ID of the layer contents.
    std::type_index type;

    // Handle of the operand that holds the layer contents.
    OperandHandle handle;
  };

  // Configuration information for a component.
  struct ComponentConfig {
    // Creates an empty config for a component with the |name|.
    explicit ComponentConfig(const string &name) : name(name) {}

    // Name of the component.
    string name;

    // Manager for the operands used by the component.
    OperandManager manager;

    // Configuration of each layer produced by the component.
    std::vector<LayerConfig> layers;

    // Mapping from layer alias to layer name in the component.
    std::map<string, string> aliases;
  };

  // Implements the non-templated part of AddLayer().  Adds a layer with the
  // |name|, |type| ID, and size in |bytes|.  Sets the |component_index| and
  // |operand_handle| according to the containing component and operand.  If
  // |is_pairwise| is true, then the new layer is pairwise (vs stepwise).  On
  // error, returns non-OK and modifies nothing.
  tensorflow::Status AddLayerImpl(const string &name, std::type_index type,
                                  bool is_pairwise, size_t bytes,
                                  size_t *component_index,
                                  OperandHandle *operand_handle);

  // Implements the non-templated portion of AddLocal*().  Adds a local operand
  // with the |spec| and sets |handle| to its handle.  On error, returns non-OK
  // and modifies nothing.
  tensorflow::Status AddLocalImpl(const OperandSpec &spec,
                                  OperandHandle *handle);

  // Implements the non-templated portion of LookupLayer().  Finds the layer
  // that matches the |component_name| and |layer_name_or_alias|.  That layer
  // must match the |type| ID.  Sets |bytes| to its size, |component_index| to
  // the index of its containing component, and |operand_handle| to the handle
  // of its underlying operand.  If |is_pairwise| is true, then the layer must
  // be pairwise (vs stepwise).  On error, returns non-OK and modifies nothing.
  tensorflow::Status LookupLayerImpl(const string &component_name,
                                     const string &layer_name_or_alias,
                                     std::type_index type, bool is_pairwise,
                                     size_t *bytes, size_t *component_index,
                                     OperandHandle *operand_handle) const;

  // Ordered list of configurations for the components in the network.
  std::vector<ComponentConfig> components_;
};

// A set of network states.  The structure of the network is configured by a
// NetworkStateManager, and layer and local operand contents can be accessed
// using the handles produced by the manager.
//
// Multiple NetworkStates instances can share the same NetworkStateManager.  In
// addition, a NetworkStates instance can be reused by repeatedly Reset()-ing
// it, potentially with different NetworkStateManagers.  Such reuse can reduce
// allocation overhead.
class NetworkStates {
 public:
  // Creates an uninitialized set of states.
  NetworkStates() = default;

  // Resets this to an empty set configured by the |manager|.  The |manager|
  // must live until this is destroyed or Reset(), and should not be modified
  // during that time.  No current component is set; call StartNextComponent()
  // to start the first component.
  void Reset(const NetworkStateManager *manager);

  // Starts the next component and makes it the current component.  Initially,
  // the component has zero steps but more can be added using AddStep().  Uses
  // |pre_allocate_num_steps| to pre-allocate storage; see Operands::Reset().
  // On error, returns non-OK and modifies nothing.
  tensorflow::Status StartNextComponent(size_t pre_allocate_num_steps);

  // Adds one or more steps to the current component.  Invalidates all
  // previously-returned matrices of the current component.
  void AddStep() { AddSteps(1); }
  void AddSteps(size_t num_steps);

  // Returns the layer associated with the |handle|.
  template <class T>
  MutableMatrix<T> GetLayer(LayerHandle<T> handle) const;

  // Returns the pairwise layer associated with the |handle|.
  template <class T>
  MutableMatrix<T> GetLayer(PairwiseLayerHandle<T> handle) const;

  // Returns the local vector or matrix associated with the |handle| in the
  // current component.
  template <class T>
  MutableVector<T> GetLocal(LocalVectorHandle<T> handle) const;
  template <class T>
  MutableMatrix<T> GetLocal(LocalMatrixHandle<T> handle) const;

 private:
  // Manager of this set of network states.
  const NetworkStateManager *manager_ = nullptr;

  // Number of active components in the |component_operands_|.
  size_t num_active_components_ = 0;

  // Ordered list of per-component operands.  Only the first
  // |num_active_components_| entries are valid.
  std::vector<Operands> component_operands_;
};

// Implementation details below.

// An opaque handle to a typed layer of some component.
template <class T>
class LayerHandle {
 public:
  static_assert(IsAlignable<T>(), "T must be alignable");

  // Creates an invalid handle.
  LayerHandle() = default;

 private:
  friend class NetworkStateManager;
  friend class NetworkStates;

  // Index of the containing component in the network state manager.
  size_t component_index_ = SIZE_MAX;

  // Handle of the operand holding the layer.
  OperandHandle operand_handle_;
};

// An opaque handle to a typed pairwise layer of some component.
template <class T>
class PairwiseLayerHandle {
 public:
  static_assert(IsAlignable<T>(), "T must be alignable");

  // Creates an invalid handle.
  PairwiseLayerHandle() = default;

 private:
  friend class NetworkStateManager;
  friend class NetworkStates;

  // Index of the containing component in the network state manager.
  size_t component_index_ = SIZE_MAX;

  // Handle of the operand holding the layer.
  OperandHandle operand_handle_;
};

// An opaque handle to a typed local operand of some component.
template <class T>
class LocalVectorHandle {
 public:
  static_assert(IsAlignable<T>(), "T must be alignable");

  // Creates an invalid handle.
  LocalVectorHandle() = default;

 private:
  friend class NetworkStateManager;
  friend class NetworkStates;

  // Handle of the local operand.
  OperandHandle operand_handle_;
};

// An opaque handle to a typed local operand of some component.
template <class T>
class LocalMatrixHandle {
 public:
  static_assert(IsAlignable<T>(), "T must be alignable");

  // Creates an invalid handle.
  LocalMatrixHandle() = default;

 private:
  friend class NetworkStateManager;
  friend class NetworkStates;

  // Handle of the local operand.
  OperandHandle operand_handle_;
};

template <class T>
tensorflow::Status NetworkStateManager::AddLayer(const string &name,
                                                 size_t dimension,
                                                 LayerHandle<T> *handle) {
  return AddLayerImpl(name, std::type_index(typeid(T)), /*is_pairwise=*/false,
                      dimension * sizeof(T), &handle->component_index_,
                      &handle->operand_handle_);
}

template <class T>
tensorflow::Status NetworkStateManager::AddLayer(
    const string &name, size_t dimension, PairwiseLayerHandle<T> *handle) {
  return AddLayerImpl(name, std::type_index(typeid(T)), /*is_pairwise=*/true,
                      dimension * sizeof(T), &handle->component_index_,
                      &handle->operand_handle_);
}

template <class T>
tensorflow::Status NetworkStateManager::AddLocal(size_t dimension,
                                                 LocalVectorHandle<T> *handle) {
  return AddLocalImpl({OperandType::kSingular, dimension * sizeof(T)},
                      &handle->operand_handle_);
}

template <class T>
tensorflow::Status NetworkStateManager::AddLocal(size_t dimension,
                                                 LocalMatrixHandle<T> *handle) {
  return AddLocalImpl({OperandType::kStepwise, dimension * sizeof(T)},
                      &handle->operand_handle_);
}

template <class T>
tensorflow::Status NetworkStateManager::LookupLayer(
    const string &component_name, const string &layer_name_or_alias,
    size_t *dimension, LayerHandle<T> *handle) const {
  TF_RETURN_IF_ERROR(LookupLayerImpl(
      component_name, layer_name_or_alias, std::type_index(typeid(T)),
      /*is_pairwise=*/false, dimension, &handle->component_index_,
      &handle->operand_handle_));
  DCHECK_EQ(*dimension % sizeof(T), 0);
  *dimension /= sizeof(T);  // bytes => Ts
  return tensorflow::Status::OK();
}

template <class T>
tensorflow::Status NetworkStateManager::LookupLayer(
    const string &component_name, const string &layer_name_or_alias,
    size_t *dimension, PairwiseLayerHandle<T> *handle) const {
  TF_RETURN_IF_ERROR(LookupLayerImpl(
      component_name, layer_name_or_alias, std::type_index(typeid(T)),
      /*is_pairwise=*/true, dimension, &handle->component_index_,
      &handle->operand_handle_));
  DCHECK_EQ(*dimension % sizeof(T), 0);
  *dimension /= sizeof(T);  // bytes => Ts
  return tensorflow::Status::OK();
}

inline void NetworkStates::AddSteps(size_t num_steps) {
  component_operands_[num_active_components_ - 1].AddSteps(num_steps);
}

template <class T>
MutableMatrix<T> NetworkStates::GetLayer(LayerHandle<T> handle) const {
  return MutableMatrix<T>(
      component_operands_[handle.component_index_].GetStepwise(
          handle.operand_handle_));
}

template <class T>
MutableMatrix<T> NetworkStates::GetLayer(PairwiseLayerHandle<T> handle) const {
  return MutableMatrix<T>(
      component_operands_[handle.component_index_].GetPairwise(
          handle.operand_handle_));
}

template <class T>
MutableVector<T> NetworkStates::GetLocal(LocalVectorHandle<T> handle) const {
  return MutableVector<T>(
      component_operands_[num_active_components_ - 1].GetSingular(
          handle.operand_handle_));
}

template <class T>
MutableMatrix<T> NetworkStates::GetLocal(LocalMatrixHandle<T> handle) const {
  return MutableMatrix<T>(
      component_operands_[num_active_components_ - 1].GetStepwise(
          handle.operand_handle_));
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_NETWORK_STATES_H_
