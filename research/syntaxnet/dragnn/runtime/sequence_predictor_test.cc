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

#include "dragnn/runtime/sequence_predictor.h"

#include <memory>
#include <string>
#include <vector>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/math/types.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Supports components named "success" and initializes successfully.
class Success : public SequencePredictor {
 public:
  // Implements SequencePredictor.
  bool Supports(const ComponentSpec &component_spec) const override {
    return component_spec.name() == "success";
  }
  tensorflow::Status Initialize(const ComponentSpec &) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status Predict(Matrix<float>, InputBatchCache *) const override {
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_PREDICTOR(Success);

// Supports components named "failure" and fails to initialize.
class Failure : public SequencePredictor {
 public:
  // Implements SequencePredictor.
  bool Supports(const ComponentSpec &component_spec) const override {
    return component_spec.name() == "failure";
  }
  tensorflow::Status Initialize(const ComponentSpec &) override {
    return tensorflow::errors::Internal("Boom!");
  }
  tensorflow::Status Predict(Matrix<float>, InputBatchCache *) const override {
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_PREDICTOR(Failure);

// Supports components named "duplicate" and initializes successfully.
class Duplicate : public SequencePredictor {
 public:
  // Implements SequencePredictor.
  bool Supports(const ComponentSpec &component_spec) const override {
    return component_spec.name() == "duplicate";
  }
  tensorflow::Status Initialize(const ComponentSpec &) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status Predict(Matrix<float>, InputBatchCache *) const override {
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_PREDICTOR(Duplicate);

// Duplicate of the above.
using Duplicate2 = Duplicate;
DRAGNN_RUNTIME_REGISTER_SEQUENCE_PREDICTOR(Duplicate2);

// Tests that a component can be successfully created.
TEST(SequencePredictorTest, Success) {
  string name;
  std::unique_ptr<SequencePredictor> predictor;

  ComponentSpec component_spec;
  component_spec.set_name("success");
  TF_ASSERT_OK(SequencePredictor::Select(component_spec, &name));
  ASSERT_EQ(name, "Success");
  TF_EXPECT_OK(SequencePredictor::New(name, component_spec, &predictor));
  EXPECT_NE(predictor, nullptr);
}

// Tests that errors in Initialize() are reported.
TEST(SequencePredictorTest, FailToInitialize) {
  string name;
  std::unique_ptr<SequencePredictor> predictor;

  ComponentSpec component_spec;
  component_spec.set_name("failure");
  TF_ASSERT_OK(SequencePredictor::Select(component_spec, &name));
  EXPECT_EQ(name, "Failure");
  EXPECT_THAT(SequencePredictor::New(name, component_spec, &predictor),
              test::IsErrorWithSubstr("Boom!"));
  EXPECT_EQ(predictor, nullptr);
}

// Tests that unsupported specs are reported as NOT_FOUND errors.
TEST(SequencePredictorTest, UnsupportedSpec) {
  string name = "not overwritten";

  ComponentSpec component_spec;
  component_spec.set_name("unsupported");
  EXPECT_THAT(SequencePredictor::Select(component_spec, &name),
              test::IsErrorWithCodeAndSubstr(
                  tensorflow::error::NOT_FOUND,
                  "No SequencePredictor supports ComponentSpec"));
  EXPECT_EQ(name, "not overwritten");
}

// Tests that unsupported subclass names are reported as errors.
TEST(SequencePredictorTest, UnsupportedSubclass) {
  std::unique_ptr<SequencePredictor> predictor;

  ComponentSpec component_spec;
  EXPECT_THAT(
      SequencePredictor::New("Unsupported", component_spec, &predictor),
      test::IsErrorWithSubstr("Unknown DRAGNN Runtime Sequence Predictor"));
  EXPECT_EQ(predictor, nullptr);
}

// Tests that multiple supporting predictors are reported as INTERNAL errors.
TEST(SequencePredictorTest, Duplicate) {
  string name = "not overwritten";

  ComponentSpec component_spec;
  component_spec.set_name("duplicate");
  EXPECT_THAT(SequencePredictor::Select(component_spec, &name),
              test::IsErrorWithCodeAndSubstr(
                  tensorflow::error::INTERNAL,
                  "Multiple SequencePredictors support ComponentSpec"));
  EXPECT_EQ(name, "not overwritten");
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
