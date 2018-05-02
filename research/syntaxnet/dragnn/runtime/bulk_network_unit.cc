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

#include "dragnn/runtime/bulk_network_unit.h"

#include <vector>

#include "dragnn/runtime/network_unit.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

string BulkNetworkUnit::GetClassName(
    const ComponentSpec &component_spec) {
  // The network unit name specified in the |component_spec| is for the Python
  // registry and cannot be passed directly to the C++ registry.  The function
  // below extracts the C++ registered name; e.g.,
  //   "some.module.FooNetwork" => "FooNetwork".
  // We then prepend "Bulk" to distinguish it from the non-bulk version.
  return tensorflow::strings::StrCat("Bulk",
                                     NetworkUnit::GetClassName(component_spec));
}

}  // namespace runtime
}  // namespace dragnn

REGISTER_SYNTAXNET_CLASS_REGISTRY("DRAGNN Runtime Bulk Network Unit",
                                  dragnn::runtime::BulkNetworkUnit);

}  // namespace syntaxnet
