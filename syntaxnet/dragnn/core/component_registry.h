#ifndef NLP_SAFT_OPENSOURCE_DRAGNN_CORE_COMPONENT_REGISTRY_H_
#define NLP_SAFT_OPENSOURCE_DRAGNN_CORE_COMPONENT_REGISTRY_H_

#include "dragnn/core/interfaces/component.h"
#include "syntaxnet/registry.h"

// Macro to add a component to the registry. This macro associates a class with
// its class name as a string, so FooComponent would be associated with the
// string "FooComponent".
#define REGISTER_DRAGNN_COMPONENT(component)                                   \
  REGISTER_SYNTAXNET_CLASS_COMPONENT(syntaxnet::dragnn::Component, #component, \
                                     component)

#endif  // NLP_SAFT_OPENSOURCE_DRAGNN_CORE_COMPONENT_REGISTRY_H_
