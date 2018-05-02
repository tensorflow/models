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

#ifndef DRAGNN_RUNTIME_TRAINED_MODEL_VARIABLE_STORE_H_
#define DRAGNN_RUNTIME_TRAINED_MODEL_VARIABLE_STORE_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dragnn/protos/runtime.pb.h"
#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/trained_model.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// A variable store that extracts variables from a trained DRAGNN model.  This
// should not be used in production (where ArrayVariableStore and its subclasses
// should be used), though it is convenient for experimentation.
class TrainedModelVariableStore : public VariableStore {
 public:
  // Creates an uninitialized store.
  TrainedModelVariableStore() = default;

  // Resets this to represent the variables defined by the TF saved model at the
  // |saved_model_dir|.  On error, returns non-OK and modifies nothing.
  tensorflow::Status Reset(const string &saved_model_dir);

  // Implements VariableStore.
  using VariableStore::Lookup;  // import Lookup<T>() convenience methods
  tensorflow::Status Lookup(const string &name, VariableSpec::Format format,
                            std::vector<size_t> *dimensions,
                            AlignedArea *area) override;
  tensorflow::Status Close() override;

 private:
  // A (name,format) key associated with a variable.
  using Key = std::pair<string, VariableSpec::Format>;

  // Extracted and formatted variable contents, as an aligned byte array and an
  // area that provides a structured interpretation.
  using Variable =
      std::tuple<UniqueAlignedArray, std::vector<size_t>, MutableAlignedArea>;

  // Extracts the contents of the variable named |name| in the |format| and
  // stores the result in the |variable|.  On error, returns non-OK.
  tensorflow::Status GetVariableContents(const string &name,
                                         VariableSpec::Format format,
                                         Variable *variable);

  // Trained DRAGNN model used to extract variables.
  TrainedModel trained_model_;

  // The already-extracted variables.
  std::map<Key, Variable> variables_;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_TRAINED_MODEL_VARIABLE_STORE_H_
