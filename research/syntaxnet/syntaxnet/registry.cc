/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "syntaxnet/registry.h"

#include <set>
#include <string>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {

// Global list of all component registries.
RegistryMetadata *global_registry_list = nullptr;

void RegistryMetadata::Register(RegistryMetadata *registry) {
  registry->set_link(global_registry_list);
  global_registry_list = registry;
}

string ComponentMetadata::DebugString() const {
  return tensorflow::strings::StrCat("Registered '", name_, "' as class ",
                                     class_name_, " at ", file_, ":", line_);
}

tensorflow::Status RegistryMetadata::Validate() {
  static const tensorflow::Status *const status =
      new tensorflow::Status(ValidateImpl());
  return *status;
}

tensorflow::Status RegistryMetadata::ValidateImpl() {
  // Iterates over the registries for each type.
  for (RegistryMetadata *registry = global_registry_list; registry != nullptr;
       registry = static_cast<RegistryMetadata *>(registry->link())) {
    std::set<string> names;

    // Searches for duplicate names within each component registry.
    for (ComponentMetadata *component = *(registry->components_);
         component != nullptr; component = component->link()) {
      if (!names.insert(component->name()).second) {
        return tensorflow::errors::InvalidArgument(
            "Multiple classes named '", component->name(),
            "' have been registered as ", registry->name(), ": ",
            component->DebugString());
      }
    }
  }
  return tensorflow::Status::OK();
}

}  // namespace syntaxnet
