#include "registry.h"

namespace neurosis {

// Global list of all component registries.
RegistryMetadata *global_registry_list = NULL;

void RegistryMetadata::Register(RegistryMetadata *registry) {
  registry->set_link(global_registry_list);
  global_registry_list = registry;
}

}  // namespace neurosis
