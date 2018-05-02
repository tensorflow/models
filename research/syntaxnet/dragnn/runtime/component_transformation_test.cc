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

#include "dragnn/runtime/component_transformation.h"

#include <string>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Transformer that fails if the component type is "fail".
class MaybeFail : public ComponentTransformer {
 public:
  // Implements ComponentTransformer.
  tensorflow::Status Transform(const string &component_type,
                               ComponentSpec *) override {
    if (component_type == "fail") {
      return tensorflow::errors::InvalidArgument("Boom!");
    }
    return tensorflow::Status::OK();
  }
};

DRAGNN_RUNTIME_REGISTER_COMPONENT_TRANSFORMER(MaybeFail);

// Base class for transformers that change the name of the component, based on
// its current name.
class ChangeNameBase : public ComponentTransformer {
 public:
  // Creates a transformer that changes the component name from |from| to |to|.
  explicit ChangeNameBase(const string &from, const string &to)
      : from_(from), to_(to) {}

  // Implements ComponentTransformer.
  tensorflow::Status Transform(const string &,
                               ComponentSpec *component_spec) override {
    if (component_spec->name() == from_) component_spec->set_name(to_);
    return tensorflow::Status::OK();
  }

 private:
  // Component name to look for.
  const string from_;

  // Component name to change to.
  const string to_;
};

// These will convert chain1 => chain2 => chain3.
class Chain1To2 : public ChangeNameBase {
 public:
  Chain1To2() : ChangeNameBase("chain1", "chain2") {}
};
class Chain2To3 : public ChangeNameBase {
 public:
  Chain2To3() : ChangeNameBase("chain2", "chain3") {}
};

// Adds "." to the name of the component, if it begins with "cycle".
class Cycle : public ComponentTransformer {
 public:
  // Implements ComponentTransformer.
  tensorflow::Status Transform(const string &,
                               ComponentSpec *component_spec) override {
    if (component_spec->name().substr(0, 5) == "cycle") {
      component_spec->mutable_name()->append(".");
    }
    return tensorflow::Status::OK();
  }
};

// Intentionally registered out of order to exercise sorting on registered name.
DRAGNN_RUNTIME_REGISTER_COMPONENT_TRANSFORMER(Chain1To2);
DRAGNN_RUNTIME_REGISTER_COMPONENT_TRANSFORMER(Chain2To3);
DRAGNN_RUNTIME_REGISTER_COMPONENT_TRANSFORMER(Cycle);

// Arbitrary bogus path.
constexpr char kInvalidPath[] = "path/to/some/invalid/file";

// Returns a unique temporary directory for tests.
string GetUniqueTemporaryDir() {
  static int counter = 0;
  const string output_dir =
      tensorflow::io::JoinPath(tensorflow::testing::TmpDir(),
                               tensorflow::strings::StrCat("tmp_", counter++));
  TF_CHECK_OK(tensorflow::Env::Default()->RecursivelyCreateDir(output_dir));
  return output_dir;
}

// Returns a MasterSpec parsed from the |text|.
MasterSpec ParseSpec(const string &text) {
  MasterSpec master_spec;
  CHECK(TextFormat::ParseFromString(text, &master_spec));
  return master_spec;
}

// Tests that TransformComponents() fails if the input master spec path is
// invalid.
TEST(TransformComponentsTest, InvalidInputMasterSpecPath) {
  const string temp_dir = GetUniqueTemporaryDir();
  const string output_path = tensorflow::io::JoinPath(temp_dir, "output");

  EXPECT_FALSE(TransformComponents(kInvalidPath, output_path).ok());
}

// Tests that TransformComponents() fails if the output master spec path is
// invalid.
TEST(TransformComponentsTest, InvalidOutputMasterSpecPath) {
  const string temp_dir = GetUniqueTemporaryDir();
  const string input_path = tensorflow::io::JoinPath(temp_dir, "input");

  const MasterSpec empty_spec;
  TF_ASSERT_OK(tensorflow::WriteTextProto(tensorflow::Env::Default(),
                                          input_path, empty_spec));

  EXPECT_FALSE(TransformComponents(input_path, kInvalidPath).ok());
}

// Tests that TransformComponents() fails if one of the ComponentTransformers
// fails.
TEST(TransformComponentsTest, FailingComponentTransformer) {
  const string temp_dir = GetUniqueTemporaryDir();
  const string input_path = tensorflow::io::JoinPath(temp_dir, "input");
  const string output_path = tensorflow::io::JoinPath(temp_dir, "output");

  const MasterSpec input_spec = ParseSpec(R"(
    component {
      component_builder { registered_name:'foo' }
    }
    component {
      component_builder { registered_name:'fail' }
    }
  )");
  TF_ASSERT_OK(tensorflow::WriteTextProto(tensorflow::Env::Default(),
                                          input_path, input_spec));

  EXPECT_THAT(TransformComponents(input_path, output_path),
              test::IsErrorWithSubstr("Boom!"));
}

// Tests that TransformComponents() properly applies all transformations.
TEST(TransformComponentsTest, Success) {
  const string temp_dir = GetUniqueTemporaryDir();
  const string input_path = tensorflow::io::JoinPath(temp_dir, "input");
  const string output_path = tensorflow::io::JoinPath(temp_dir, "output");

  const MasterSpec input_spec = ParseSpec(R"(
    component {
      name:'chain1'
      component_builder { registered_name:'foo' }
    }
    component {
      name:'irrelevant'
      component_builder { registered_name:'bar' }
    }
  )");
  TF_ASSERT_OK(tensorflow::WriteTextProto(tensorflow::Env::Default(),
                                          input_path, input_spec));

  TF_ASSERT_OK(TransformComponents(input_path, output_path));

  MasterSpec actual_spec;
  TF_ASSERT_OK(tensorflow::ReadTextProto(tensorflow::Env::Default(),
                                         output_path, &actual_spec));

  const MasterSpec expected_spec = ParseSpec(R"(
    component {
      name:'chain3'
      component_builder { registered_name:'foo' }
    }
    component {
      name:'irrelevant'
      component_builder { registered_name:'bar' }
    }
  )");
  EXPECT_THAT(actual_spec, test::EqualsProto(expected_spec));
}

// Tests that ComponentTransformer::ApplyAll() makes the expected modifications,
// including chained modifications.
TEST(ComponentTransformerTest, ApplyAllSuccess) {
  ComponentSpec component_spec;
  component_spec.mutable_component_builder()->set_registered_name("foo");
  component_spec.set_name("chain1");
  ComponentSpec modified_spec = component_spec;

  TF_ASSERT_OK(ComponentTransformer::ApplyAll(&component_spec));

  modified_spec.set_name("chain3");
  EXPECT_THAT(component_spec, test::EqualsProto(modified_spec));
}

// Tests that ComponentTransformer::ApplyAll() limits the number of iterations.
TEST(ComponentTransformerTest, ApplyAllLimitIterations) {
  ComponentSpec component_spec;
  component_spec.mutable_component_builder()->set_registered_name("foo");
  component_spec.set_name("cycle");

  EXPECT_THAT(ComponentTransformer::ApplyAll(&component_spec),
              test::IsErrorWithSubstr("Failed to converge"));
}

// Tests that ComponentTransformer::ApplyAll() fails if one of the
// ComponentTransformers fails.
TEST(ComponentTransformerTest, ApplyAllFailure) {
  ComponentSpec component_spec;
  component_spec.mutable_component_builder()->set_registered_name("fail");

  EXPECT_THAT(ComponentTransformer::ApplyAll(&component_spec),
              test::IsErrorWithSubstr("Boom!"));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
