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

#ifndef DRAGNN_RUNTIME_TEST_FAKE_VARIABLE_STORE_H_
#define DRAGNN_RUNTIME_TEST_FAKE_VARIABLE_STORE_H_

#include <map>
#include <string>
#include <vector>

#include "dragnn/protos/runtime.pb.h"
#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/test/helpers.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// A fake variable store with user-specified contents.
class FakeVariableStore : public VariableStore {
 public:
  // Creates an empty store.
  FakeVariableStore() = default;

  // Adds the |data| to this as a variable with the |name| and |format|.  If the
  // |format| is FORMAT_UNKNOWN, adds the data in all formats.  On error, aborts
  // the program.
  void AddOrDie(const string &name, const std::vector<std::vector<float>> &data,
                VariableSpec::Format format = VariableSpec::FORMAT_UNKNOWN);

  // Overrides the default behavior of assuming that there is one block along
  // the major axis of the matrix.
  void SetBlockedDimensionOverride(const string &name,
                                   const std::vector<size_t> &dimensions);

  // Implements VariableStore.
  using VariableStore::Lookup;  // import Lookup<T>() convenience methods
  tensorflow::Status Lookup(const string &name, VariableSpec::Format format,
                            std::vector<size_t> *dimensions,
                            AlignedArea *area) override;
  tensorflow::Status Close() override { return tensorflow::Status::OK(); }

 private:
  using Variable = UniqueMatrix<float>;
  using FormatMap = std::map<VariableSpec::Format, Variable>;

  // Mappings from variable name to format to contents.
  std::map<string, FormatMap> variables_;

  // Overrides blocked dimensions.
  std::map<string, std::vector<size_t>> override_blocked_dimensions_;
};

// Syntactic sugar for replicating data to SimpleFakeVariableStore::MockLookup.
template <typename T>
std::vector<std::vector<T>> ReplicateRows(std::vector<T> values, int times) {
  return std::vector<std::vector<T>>(times, values);
}

// Simpler fake variable store, where the test just sets up the next value to be
// returned.
class SimpleFakeVariableStore : public VariableStore {
 public:
  // Executes cleanup functions (see `cleanup_` comment).
  ~SimpleFakeVariableStore() override;

  // Sets values which store().Lookup() will return.
  template <typename T>
  void MockLookup(const std::vector<size_t> &dimensions,
                  const std::vector<std::vector<T>> &area_values) {
    UniqueMatrix<T> *matrix = new UniqueMatrix<T>(area_values);
    cleanup_.push_back([matrix]() { delete matrix; });
    dimensions_to_return_.reset(new std::vector<size_t>(dimensions));
    area_to_return_.reset(new AlignedArea(matrix->area()));
  }

  using VariableStore::Lookup;  // import Lookup<T>() convenience methods
  tensorflow::Status Lookup(const string &name, VariableSpec::Format format,
                            std::vector<size_t> *dimensions,
                            AlignedArea *area) override;

  tensorflow::Status Close() override { return tensorflow::Status::OK(); }

 private:
  std::unique_ptr<std::vector<size_t>> dimensions_to_return_ = nullptr;
  std::unique_ptr<AlignedArea> area_to_return_ = nullptr;

  // Functions which will delete memory storing mocked arrays. We want to keep
  // the memory accessible until the end of the test. We also can't keep an
  // array of objects to delete, since they are of different types.
  std::vector<std::function<void()>> cleanup_;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_TEST_FAKE_VARIABLE_STORE_H_
