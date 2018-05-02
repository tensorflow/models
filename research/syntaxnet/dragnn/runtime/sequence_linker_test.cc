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

#include "dragnn/runtime/sequence_linker.h"

#include <stddef.h>
#include <memory>
#include <string>
#include <vector>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
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
class Success : public SequenceLinker {
 public:
  // Implements SequenceLinker.
  bool Supports(const LinkedFeatureChannel &,
                const ComponentSpec &component_spec) const override {
    return component_spec.name() == "success";
  }
  tensorflow::Status Initialize(const LinkedFeatureChannel &,
                                const ComponentSpec &) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status GetLinks(size_t, InputBatchCache *,
                              std::vector<int32> *) const override {
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_LINKER(Success);

// Supports components named "failure" and fails to initialize.
class Failure : public SequenceLinker {
 public:
  // Implements SequenceLinker.
  bool Supports(const LinkedFeatureChannel &,
                const ComponentSpec &component_spec) const override {
    return component_spec.name() == "failure";
  }
  tensorflow::Status Initialize(const LinkedFeatureChannel &,
                                const ComponentSpec &) override {
    return tensorflow::errors::Internal("Boom!");
  }
  tensorflow::Status GetLinks(size_t, InputBatchCache *,
                              std::vector<int32> *) const override {
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_LINKER(Failure);

// Supports components named "duplicate" and initializes successfully.
class Duplicate : public SequenceLinker {
 public:
  // Implements SequenceLinker.
  bool Supports(const LinkedFeatureChannel &,
                const ComponentSpec &component_spec) const override {
    return component_spec.name() == "duplicate";
  }
  tensorflow::Status Initialize(const LinkedFeatureChannel &,
                                const ComponentSpec &) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status GetLinks(size_t, InputBatchCache *,
                              std::vector<int32> *) const override {
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_LINKER(Duplicate);

// Duplicate of the above.
using Duplicate2 = Duplicate;
DRAGNN_RUNTIME_REGISTER_SEQUENCE_LINKER(Duplicate2);

// Tests that a component can be successfully created.
TEST(SequenceLinkerTest, Success) {
  string name;
  std::unique_ptr<SequenceLinker> linker;

  ComponentSpec component_spec;
  component_spec.set_name("success");
  TF_ASSERT_OK(SequenceLinker::Select({}, component_spec, &name));
  ASSERT_EQ(name, "Success");
  TF_EXPECT_OK(SequenceLinker::New(name, {}, component_spec, &linker));
  EXPECT_NE(linker, nullptr);
}

// Tests that errors in Initialize() are reported.
TEST(SequenceLinkerTest, FailToInitialize) {
  string name;
  std::unique_ptr<SequenceLinker> linker;

  ComponentSpec component_spec;
  component_spec.set_name("failure");
  TF_ASSERT_OK(SequenceLinker::Select({}, component_spec, &name));
  EXPECT_EQ(name, "Failure");
  EXPECT_THAT(SequenceLinker::New(name, {}, component_spec, &linker),
              test::IsErrorWithSubstr("Boom!"));
  EXPECT_EQ(linker, nullptr);
}

// Tests that unsupported specs are reported as NOT_FOUND errors.
TEST(SequenceLinkerTest, UnsupportedSpec) {
  string name = "not overwritten";

  ComponentSpec component_spec;
  component_spec.set_name("unsupported");
  EXPECT_THAT(
      SequenceLinker::Select({}, component_spec, &name),
      test::IsErrorWithCodeAndSubstr(tensorflow::error::NOT_FOUND,
                                     "No SequenceLinker supports channel"));
  EXPECT_EQ(name, "not overwritten");
}

// Tests that unsupported subclass names are reported as errors.
TEST(SequenceLinkerTest, UnsupportedSubclass) {
  std::unique_ptr<SequenceLinker> linker;

  ComponentSpec component_spec;
  EXPECT_THAT(
      SequenceLinker::New("Unsupported", {}, component_spec, &linker),
      test::IsErrorWithSubstr("Unknown DRAGNN Runtime Sequence Linker"));
  EXPECT_EQ(linker, nullptr);
}

// Tests that multiple supporting linkers are reported as INTERNAL errors.
TEST(SequenceLinkerTest, Duplicate) {
  string name = "not overwritten";

  ComponentSpec component_spec;
  component_spec.set_name("duplicate");
  EXPECT_THAT(SequenceLinker::Select({}, component_spec, &name),
              test::IsErrorWithCodeAndSubstr(
                  tensorflow::error::INTERNAL,
                  "Multiple SequenceLinkers support channel"));
  EXPECT_EQ(name, "not overwritten");
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
