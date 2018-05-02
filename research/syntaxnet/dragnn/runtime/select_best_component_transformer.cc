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

#include <string>

#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/component_transformation.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Transformer that selects the best component subclass for the ComponentSpec.
class SelectBestComponentTransformer : public ComponentTransformer {
 public:
  // Implements ComponentTransformer.
  tensorflow::Status Transform(const string &component_type,
                               ComponentSpec *component_spec) override {
    string best_component_type;
    TF_RETURN_IF_ERROR(
        Component::Select(*component_spec, &best_component_type));
    component_spec->mutable_component_builder()->set_registered_name(
        best_component_type);
    if (component_type != best_component_type) {
      LOG(INFO) << "Component '" << component_spec->name()
                << "' builder updated from " << component_type << " to "
                << best_component_type << ".";
    } else {
      VLOG(2) << "Component '" << component_spec->name() << "' builder type "
              << component_type << " unchanged.";
    }
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_COMPONENT_TRANSFORMER(SelectBestComponentTransformer);

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
