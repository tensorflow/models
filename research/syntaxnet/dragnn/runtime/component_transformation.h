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

// Utils for transforming ComponentSpecs, typically (but not necessarily) in
// ways that are intended to improve speed.  For example, a transformer might
// detect a favorable component configuration and replace a generic Component
// implementation with a faster version.

#ifndef DRAGNN_RUNTIME_COMPONENT_TRANSFORMATION_H_
#define DRAGNN_RUNTIME_COMPONENT_TRANSFORMATION_H_

#include <string>

#include "dragnn/protos/spec.pb.h"
#include "syntaxnet/base.h"
#include "syntaxnet/registry.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Loads a MasterSpec from the |input_master_spec_path|, applies all registered
// ComponentTransformers to it (see ComponentTransformer::ApplyAll() below), and
// writes it to the |output_master_spec_path|.  On error, returns non-OK.
//
// Side note: This function has a file-path-based API so it can be easily
// wrapped in a stand-alone binary.

tensorflow::Status TransformComponents(const string &input_master_spec_path,
                                       const string &output_master_spec_path);

// Interface for modules that can transform a ComponentSpec, which allows
// transformations to be plugged in on a decentralized basis.
class ComponentTransformer : public RegisterableClass<ComponentTransformer> {
 public:
  ComponentTransformer(const ComponentTransformer &that) = delete;
  ComponentTransformer &operator=(const ComponentTransformer &that) = delete;
  virtual ~ComponentTransformer() = default;

  // Repeatedly loops through all registered transformers and applies them to
  // the |component_spec| until no more changes occur.  For determinism, each
  // loop applies the transformers in ascending order of registered name.  On
  // error, returns non-OK and modifies nothing.
  static tensorflow::Status ApplyAll(ComponentSpec *component_spec);

 protected:
  ComponentTransformer() = default;

 private:
  // Helps prevent use of the Create() method.
  using RegisterableClass<ComponentTransformer>::Create;

  // Modifies the |component_spec|, which is currently configured to use the
  // |component_type|, if compatible.  On error, returns non-OK and modifies
  // nothing.  Note that it is not an error if the |component_spec| is simply
  // not compatible with the desired transformation.
  virtual tensorflow::Status Transform(const string &component_type,
                                       ComponentSpec *component_spec) = 0;
};

}  // namespace runtime
}  // namespace dragnn

DECLARE_SYNTAXNET_CLASS_REGISTRY("DRAGNN Runtime Component Transformer",
                                 dragnn::runtime::ComponentTransformer);

}  // namespace syntaxnet

#define DRAGNN_RUNTIME_REGISTER_COMPONENT_TRANSFORMER(subclass) \
  REGISTER_SYNTAXNET_CLASS_COMPONENT(                           \
      ::syntaxnet::dragnn::runtime::ComponentTransformer, #subclass, subclass)

#endif  // DRAGNN_RUNTIME_COMPONENT_TRANSFORMATION_H_
