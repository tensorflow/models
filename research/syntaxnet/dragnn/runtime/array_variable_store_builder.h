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

#ifndef DRAGNN_RUNTIME_ARRAY_VARIABLE_STORE_BUILDER_H_
#define DRAGNN_RUNTIME_ARRAY_VARIABLE_STORE_BUILDER_H_

#include <map>
#include <string>

#include "dragnn/protos/runtime.pb.h"
#include "dragnn/runtime/variable_store_wrappers.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Utils for converting a set of variables into a byte array that can be loaded
// by ArrayVariableStore.  See that class for details on the required format.
class ArrayVariableStoreBuilder {
 public:
  using Variables = CaptureUsedVariableStoreWrapper::Variables;

  // Forbids instantiation; pure static class.
  ArrayVariableStoreBuilder() = delete;
  ~ArrayVariableStoreBuilder() = delete;

  // Overwrites the |data| with a byte array that represents the |variables|,
  // and overwrites the |spec| with the associated configuration.  On error,
  // returns non-OK.
  static tensorflow::Status Build(const Variables &variables,
                                  ArrayVariableStoreSpec *spec, string *data);
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_ARRAY_VARIABLE_STORE_BUILDER_H_
