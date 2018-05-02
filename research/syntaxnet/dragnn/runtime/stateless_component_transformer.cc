// Copyright 2018 Google Inc. All Rights Reserved.
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
#include "dragnn/runtime/component_transformation.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Returns true if the |component_type| can be transformed by this.
bool ShouldTransform(const string &component_type) {
  for (const char *supported_type : {


           "SyntaxNetHeadSelectionComponent",  //
           "SyntaxNetMstSolverComponent",      //
       }) {
    if (component_type == supported_type) return true;
  }
  return false;
}

// Changes the backend for some components to StatelessComponent.
class StatelessComponentTransformer : public ComponentTransformer {
 public:
  // Implements ComponentTransformer.
  tensorflow::Status Transform(const string &component_type,
                               ComponentSpec *component_spec) override {
    if (ShouldTransform(component_type)) {
      component_spec->mutable_backend()->set_registered_name(
          "StatelessComponent");
    }
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_COMPONENT_TRANSFORMER(StatelessComponentTransformer);

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
