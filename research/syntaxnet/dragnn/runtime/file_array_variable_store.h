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

#ifndef DRAGNN_RUNTIME_FILE_ARRAY_VARIABLE_STORE_H_
#define DRAGNN_RUNTIME_FILE_ARRAY_VARIABLE_STORE_H_

#include <string>

#include "dragnn/protos/runtime.pb.h"
#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/array_variable_store.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// An ArrayVariableStore subclass that reads a file into a new-allocated array.
class FileArrayVariableStore : public ArrayVariableStore {
 public:
  // Creates an uninitialized store.
  FileArrayVariableStore() = default;

  // Resets this to represent the variables defined by the |spec|, loading the
  // byte array from the |path|.  On error, returns non-OK and modifies nothing.
  tensorflow::Status Reset(const ArrayVariableStoreSpec &spec,
                           const string &path);

 private:
  // The byte array containing the variables.
  UniqueAlignedArray data_;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_FILE_ARRAY_VARIABLE_STORE_H_
