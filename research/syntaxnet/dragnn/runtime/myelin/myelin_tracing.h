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

#ifndef DRAGNN_RUNTIME_MYELIN_MYELIN_TRACING_H_
#define DRAGNN_RUNTIME_MYELIN_MYELIN_TRACING_H_

#include "dragnn/protos/cell_trace.pb.h"
#include "sling/myelin/compute.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Overwrites the |cell_trace| with traces extracted from the |instance|.  Does
// not modify the |instance|; it is non-const because the relevant accessors are
// declared non-const.
void TraceMyelinInstance(sling::myelin::Instance *instance,
                         CellTrace *cell_trace);

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_MYELIN_MYELIN_TRACING_H_
