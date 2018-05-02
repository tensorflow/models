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

#ifndef DRAGNN_RUNTIME_ARRAY_VARIABLE_STORE_H_
#define DRAGNN_RUNTIME_ARRAY_VARIABLE_STORE_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dragnn/protos/runtime.pb.h"
#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// A variable store that groups all variables into a single byte array.  This
// class and its subclasses are intended for use in production.
//
// Each variable occupies a sub-array of the main byte array.  The mapping from
// the name and format of a variable to the sub-array containing its content is
// defined in ArrayVariableStoreSpec.  The variables may appear in any order.
//
// This format allows variables to be mapped directly into memory, which reduces
// initialization time and supports usage on-device, where mmap() is effectively
// obligatory for large data resources.
class ArrayVariableStore : public VariableStore {
 public:
  // Creates an uninitialized store.
  ArrayVariableStore() = default;

  // Resets this to represent the variables defined by the |spec| and |data|.
  // The |data| must remain valid until this is destroyed or Reset().  (Note
  // that subclasses have simpler lifetime requirements).  On error, returns
  // non-OK and modifies nothing.
  tensorflow::Status Reset(const ArrayVariableStoreSpec &spec,
                           AlignedView data);

  // Implements VariableStore.
  using VariableStore::Lookup;  // import Lookup<T>() convenience methods
  tensorflow::Status Lookup(const string &name, VariableSpec::Format format,
                            std::vector<size_t> *dimensions,
                            AlignedArea *area) override;
  tensorflow::Status Close() override;

 private:
  friend class ArrayVariableStoreBuilder;  // for access to kVersion

  // The current version of the serialized format.
  static const uint32 kVersion;

  // A (name,format) key associated with a variable.
  using Key = std::pair<string, VariableSpec::Format>;

  // Dimension vector and aligned area.
  using Value = std::pair<const std::vector<size_t>, AlignedArea>;

  // Mapping from variable key to variable content.  Initially null, filled in
  // Reset(), and deleted in Close().  Wrapped in std::unique_ptr so the entire
  // mapping can be deleted.
  std::unique_ptr<std::map<Key, Value>> variables_;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_ARRAY_VARIABLE_STORE_H_
