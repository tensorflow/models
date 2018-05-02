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

#include "dragnn/runtime/network_unit.h"

#include <vector>

#include "tensorflow/core/lib/strings/str_util.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

string NetworkUnit::GetClassName(const ComponentSpec &component_spec) {
  // The Python registration API is based on (relative) module paths, such as
  // "some.module.FooNetwork".  Therefore, we discard the module path prefix and
  // use only the final segment, which is the subclass name.
  const std::vector<string> segments = tensorflow::str_util::Split(
      component_spec.network_unit().registered_name(), ".");
  CHECK_GT(segments.size(), 0) << "No network unit name for component spec: "
                               << component_spec.ShortDebugString();
  return segments.back();
}

}  // namespace runtime
}  // namespace dragnn

REGISTER_SYNTAXNET_CLASS_REGISTRY("DRAGNN Runtime Network Unit",
                                  dragnn::runtime::NetworkUnit);

}  // namespace syntaxnet
