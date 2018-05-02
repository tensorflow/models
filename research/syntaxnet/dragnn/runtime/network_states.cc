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

#include "dragnn/runtime/network_states.h"

#include "tensorflow/core/lib/core/errors.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Returns the first value in |container| whose ".name" field is |name|, or null
// if not found.
template <class Container>
const typename Container::value_type *Find(const Container &container,
                                           const string &name) {
  for (auto &value : container) {
    if (value.name == name) return &value;
  }
  return nullptr;
}

}  // namespace

tensorflow::Status NetworkStateManager::AddComponent(const string &name) {
  if (Find(components_, name) != nullptr) {
    return tensorflow::errors::FailedPrecondition("Component '", name,
                                                  "' already exists");
  }

  // Success; make modifications.
  components_.emplace_back(name);
  return tensorflow::Status::OK();
}

tensorflow::Status NetworkStateManager::AddLayerImpl(
    const string &name, std::type_index type, bool is_pairwise, size_t bytes,
    size_t *component_index, OperandHandle *operand_handle) {
  if (components_.empty()) {
    return tensorflow::errors::FailedPrecondition("No current component");
  }
  ComponentConfig &component = components_.back();

  if (Find(component.layers, name) != nullptr) {
    return tensorflow::errors::FailedPrecondition(
        "Layer '", name, "' already exists in component '", component.name,
        "'");
  }

  if (component.aliases.find(name) != component.aliases.end()) {
    return tensorflow::errors::FailedPrecondition(
        "Layer '", name, "' conflicts with an existing alias in component '",
        component.name, "'");
  }

  // Success; make modifications.
  const OperandType operand_type =
      is_pairwise ? OperandType::kPairwise : OperandType::kStepwise;
  *component_index = components_.size() - 1;
  *operand_handle = component.manager.Add({operand_type, bytes});
  component.layers.emplace_back(name, type, *operand_handle);
  return tensorflow::Status::OK();
}

tensorflow::Status NetworkStateManager::AddLayerAlias(const string &alias,
                                                      const string &name) {
  if (components_.empty()) {
    return tensorflow::errors::FailedPrecondition("No current component");
  }
  ComponentConfig &component = components_.back();

  if (Find(component.layers, name) == nullptr) {
    return tensorflow::errors::FailedPrecondition(
        "Target layer '", name, "' of alias '", alias,
        "' does not exist in component '", component.name, "'");
  }

  if (Find(component.layers, alias) != nullptr) {
    return tensorflow::errors::FailedPrecondition(
        "Alias '", alias, "' conflicts with an existing layer in component '",
        component.name, "'");
  }

  if (component.aliases.find(alias) != component.aliases.end()) {
    return tensorflow::errors::FailedPrecondition(
        "Alias '", alias, "' already exists in component '", component.name,
        "'");
  }

  // Success; make modifications.
  component.aliases[alias] = name;
  return tensorflow::Status::OK();
}

tensorflow::Status NetworkStateManager::AddLocalImpl(const OperandSpec &spec,
                                                     OperandHandle *handle) {
  if (components_.empty()) {
    return tensorflow::errors::FailedPrecondition("No current component");
  }
  ComponentConfig &component = components_.back();

  // Success; make modifications.
  *handle = component.manager.Add(spec);
  return tensorflow::Status::OK();
}

tensorflow::Status NetworkStateManager::LookupLayerImpl(
    const string &component_name, const string &layer_name_or_alias,
    std::type_index type, bool is_pairwise, size_t *bytes,
    size_t *component_index, OperandHandle *operand_handle) const {
  const ComponentConfig *component = Find(components_, component_name);
  if (component == nullptr) {
    return tensorflow::errors::FailedPrecondition("Unknown component '",
                                                  component_name, "'");
  }

  // If necessary, resolve a layer alias into a layer name.  Note that aliases
  // are non-transitive, since AddLayerAlias() requires that the target of the
  // alias is a layer.
  const auto it = component->aliases.find(layer_name_or_alias);
  const string &layer_name =
      it != component->aliases.end() ? it->second : layer_name_or_alias;

  const LayerConfig *layer = Find(component->layers, layer_name);
  if (layer == nullptr) {
    return tensorflow::errors::FailedPrecondition(
        "Unknown layer '", layer_name, "' in component '", component_name, "'");
  }

  if (layer->type != type) {
    return tensorflow::errors::InvalidArgument(
        "Layer '", layer_name, "' in component '", component_name,
        "' does not match its expected type");
  }

  const OperandType required_type =
      is_pairwise ? OperandType::kPairwise : OperandType::kStepwise;
  const OperandSpec &operand_spec = component->manager.spec(layer->handle);
  if (operand_spec.type != required_type) {
    return tensorflow::errors::InvalidArgument(
        "Layer '", layer_name, "' in component '", component_name,
        "' does not match its expected OperandType");
  }

  // Success; make modifications.
  *bytes = operand_spec.size;
  *component_index = component - components_.data();
  *operand_handle = layer->handle;
  return tensorflow::Status::OK();
}

void NetworkStates::Reset(const NetworkStateManager *manager) {
  manager_ = manager;
  num_active_components_ = 0;

  // Never shrink the |component_operands_|, to avoid deallocating (and then
  // eventually reallocating) operand arrays.
  if (manager_->components_.size() > component_operands_.size()) {
    component_operands_.resize(manager_->components_.size());
  }
}

tensorflow::Status NetworkStates::StartNextComponent(
    size_t pre_allocate_num_steps) {
  if (manager_ == nullptr) {
    return tensorflow::errors::FailedPrecondition("No manager");
  }

  if (num_active_components_ >= manager_->components_.size()) {
    return tensorflow::errors::OutOfRange("No next component");
  }

  // Success; make modifications.
  const OperandManager *operand_manager =
      &manager_->components_[num_active_components_].manager;
  component_operands_[num_active_components_].Reset(operand_manager,
                                                    pre_allocate_num_steps);
  ++num_active_components_;
  return tensorflow::Status::OK();
}

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
