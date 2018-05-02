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

#ifndef DRAGNN_RUNTIME_SEQUENCE_PREDICTOR_H_
#define DRAGNN_RUNTIME_SEQUENCE_PREDICTOR_H_

#include <memory>
#include <string>
#include <vector>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/math/types.h"
#include "syntaxnet/base.h"
#include "syntaxnet/registry.h"
#include "tensorflow/core/lib/core/status.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {

// Interface for making predictions on sequences.

//
// This predictor can be used to avoid ComputeSession overhead in simple cases;
// for example, predicting sequences of POS tags.
class SequencePredictor : public RegisterableClass<SequencePredictor> {
 public:
  // Sets |predictor| to an instance of the subclass named |name| initialized
  // from the |component_spec|.  On error, returns non-OK and modifies nothing.
  static tensorflow::Status New(const string &name,
                                const ComponentSpec &component_spec,
                                std::unique_ptr<SequencePredictor> *predictor);

  SequencePredictor(const SequencePredictor &) = delete;
  SequencePredictor &operator=(const SequencePredictor &) = delete;
  virtual ~SequencePredictor() = default;

  // Sets |name| to the registered name of the SequencePredictor that supports
  // the |component_spec|.  On error, returns non-OK and modifies nothing.  The
  // returned statuses include:
  // * OK: If a supporting SequencePredictor was found.
  // * INTERNAL: If an error occurred while searching for a compatible match.
  // * NOT_FOUND: If the search was error-free, but no compatible match was
  //              found.
  static tensorflow::Status Select(const ComponentSpec &component_spec,
                                   string *name);

  // Makes a sequence of predictions using the per-step |logits| and writes
  // annotations to the |input|.
  virtual tensorflow::Status Predict(Matrix<float> logits,
                                     InputBatchCache *input) const = 0;

 protected:
  SequencePredictor() = default;

 private:
  // Helps prevent use of the Create() method; use New() instead.
  using RegisterableClass<SequencePredictor>::Create;

  // Returns true if this supports the |component_spec|.  Implementations must
  // coordinate to ensure that at most one supports any given |component_spec|.
  virtual bool Supports(const ComponentSpec &component_spec) const = 0;

  // Initializes this from the |component_spec|.  On error, returns non-OK.
  virtual tensorflow::Status Initialize(
      const ComponentSpec &component_spec) = 0;
};

}  // namespace runtime
}  // namespace dragnn

DECLARE_SYNTAXNET_CLASS_REGISTRY("DRAGNN Runtime Sequence Predictor",
                                 dragnn::runtime::SequencePredictor);

}  // namespace syntaxnet

#define DRAGNN_RUNTIME_REGISTER_SEQUENCE_PREDICTOR(subclass) \
  REGISTER_SYNTAXNET_CLASS_COMPONENT(                        \
      ::syntaxnet::dragnn::runtime::SequencePredictor, #subclass, subclass)

#endif  // DRAGNN_RUNTIME_SEQUENCE_PREDICTOR_H_
