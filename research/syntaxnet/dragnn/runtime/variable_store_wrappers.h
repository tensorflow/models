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

// A set of VariableStore wrappers that provide compositional functionality.
// These are intended for offline processing and experimentation; avoid using
// these in production, where ArrayVariableStore and its subclasses should be
// used instead.

#ifndef DRAGNN_RUNTIME_VARIABLE_STORE_WRAPPERS_H_
#define DRAGNN_RUNTIME_VARIABLE_STORE_WRAPPERS_H_

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

// A wrapper that looks for an averaged version of each variable in the wrapped
// store, and failing that optionally falls back to the non-averaged version.
class TryAveragedVariableStoreWrapper : public VariableStore {
 public:
  // Wraps the |variable_store|.  If |allow_fallback| is true, then when the
  // averaged version is missing the non-averaged version can be substituted.
  explicit TryAveragedVariableStoreWrapper(
      std::unique_ptr<VariableStore> variable_store,
      bool allow_fallback = false);

  // Implements VariableStore.
  using VariableStore::Lookup;  // import Lookup<T>() convenience methods
  tensorflow::Status Lookup(const string &name, VariableSpec::Format format,
                            std::vector<size_t> *dimensions,
                            AlignedArea *area) override;
  tensorflow::Status Close() override;

 private:
  // Wrapped variable store.
  const std::unique_ptr<VariableStore> wrapped_variable_store_;

  // Whether to allow fallback to the non-averaged variable.
  const bool allow_fallback_;
};

// A wrapper that captures each successfully retrieved variable.  Useful for
// finding the exact set of variables used by some set of DRAGNN components.
class CaptureUsedVariableStoreWrapper : public VariableStore {
 public:
  // `Variables` is a list of captured variables, in order that they are
  // captured. We want to preserve the order, so that arrays are sequential in
  // memory. `VariableKey` is name/format metadata used to uniquely identify
  // a variable; duplicate lookups to the same variable will not capture it
  // twice, and its position in the list will be the first position.
  using VariableKey = std::pair<string, VariableSpec::Format>;
  using VariableValue = std::pair<std::vector<size_t>, AlignedArea>;
  using Variables = std::vector<std::pair<VariableKey, VariableValue>>;

  // Wraps the |variable_store|.
  explicit CaptureUsedVariableStoreWrapper(
      std::unique_ptr<VariableStore> variable_store);

  // Implements VariableStore.
  using VariableStore::Lookup;  // import Lookup<T>() convenience methods
  tensorflow::Status Lookup(const string &name, VariableSpec::Format format,
                            std::vector<size_t> *dimensions,
                            AlignedArea *area) override;
  tensorflow::Status Close() override;

  // Returns the current set of captured variables.  The variable content in the
  // returned mapping is valid while this lives.
  const Variables &variables() const { return variables_; }

 private:
  // Wrapped variable store.
  const std::unique_ptr<VariableStore> wrapped_variable_store_;

  // Current set of captured variables.
  Variables variables_;

  // Indexes key --> position in variables_ list.
  std::map<VariableKey, int> index_;
};

// A wrapper that selects a matrix format for the FlexibleMatrixKernel.  This
// could be done in the FlexibleMatrixKernel itself, but factoring it into this
// wrapper allows the selection to occur at model construction time instead of
// at model loading time.
class FlexibleMatrixVariableStoreWrapper : public VariableStore {
 public:
  // Wraps the |variable_store|.
  explicit FlexibleMatrixVariableStoreWrapper(
      std::unique_ptr<VariableStore> variable_store);

  // Looks up the variable named |name| with format |format|, returning its
  // shape in |dimensions| and its data in |area|.  On error, returns non-OK.
  //
  // If the |name| does not end in FlexibleMatrixKernel::kSuffix, passes the
  // request along to the |wrapped_variable_store_|.  Otherwise, if |name| is
  // "foo/<kSuffix>", estimates the throughput of the matrix "foo" in various
  // formats (assuming the workload is matrix-vector multiplications), selects
  // the fastest format, and returns the matrix in that format.
  //
  // It is an error if the selected matrix format does not match the requested
  // variable |format| (e.g., non-blocked vs blocked).  The FlexibleMatrixKernel
  // should request the variable in all relevant variable formats, so eventually
  // it will issue a request in a matching format.
  tensorflow::Status Lookup(const string &name, VariableSpec::Format format,
                            std::vector<size_t> *dimensions,
                            AlignedArea *area) override;
  using VariableStore::Lookup;  // import Lookup<T>() convenience methods

  // Implements VariableStore.
  tensorflow::Status Close() override;

 private:
  // Wrapped variable store.
  const std::unique_ptr<VariableStore> wrapped_variable_store_;
};

}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_RUNTIME_VARIABLE_STORE_WRAPPERS_H_
