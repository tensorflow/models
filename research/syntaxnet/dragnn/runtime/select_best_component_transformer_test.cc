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

#include <memory>
#include <string>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/component_transformation.h"
#include "dragnn/runtime/extensions.h"
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Base class for test components.
class TestComponentBase : public Component {
 public:
  // Partially implements Component.
  tensorflow::Status Initialize(const ComponentSpec &, VariableStore *,
                                NetworkStateManager *,
                                ExtensionManager *) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status Evaluate(SessionState *, ComputeSession *,
                              ComponentTrace *) const override {
    return tensorflow::Status::OK();
  }
  bool PreferredTo(const Component &) const override { return false; }
};

// Supports components whose builder name includes "Foo".
class ContainsFoo : public TestComponentBase {
 public:
  // Implements Component.
  bool Supports(const ComponentSpec &,
                const string &normalized_builder_name) const override {
    return normalized_builder_name.find("Foo") != string::npos;
  }
};

DRAGNN_RUNTIME_REGISTER_COMPONENT(ContainsFoo);

// Supports components whose builder name includes "Bar".
class ContainsBar : public TestComponentBase {
 public:
  // Implements Component.
  bool Supports(const ComponentSpec &,
                const string &normalized_builder_name) const override {
    return normalized_builder_name.find("Bar") != string::npos;
  }
};

DRAGNN_RUNTIME_REGISTER_COMPONENT(ContainsBar);

// Tests that a spec with an unknown builder name causes an error.
TEST(SelectBestComponentTransformerTest, Unknown) {
  ComponentSpec component_spec;
  component_spec.mutable_component_builder()->set_registered_name("unknown");

  EXPECT_THAT(ComponentTransformer::ApplyAll(&component_spec),
              test::IsErrorWithSubstr("Could not find a best"));
}

// Tests that a spec with builder "Foo" is changed to "ContainsFoo".
TEST(SelectBestComponentTransformerTest, ChangeToContainsFoo) {
  ComponentSpec component_spec;
  component_spec.mutable_component_builder()->set_registered_name("Foo");

  ComponentSpec expected_spec = component_spec;
  expected_spec.mutable_component_builder()->set_registered_name("ContainsFoo");
  TF_ASSERT_OK(ComponentTransformer::ApplyAll(&component_spec));
  EXPECT_THAT(component_spec, test::EqualsProto(expected_spec));
}

// Tests that a spec with builder "Bar" is changed to "ContainsBar".
TEST(SelectBestComponentTransformerTest, ChangeToContainsBar) {
  ComponentSpec component_spec;
  component_spec.mutable_component_builder()->set_registered_name("Bar");

  ComponentSpec expected_spec = component_spec;
  expected_spec.mutable_component_builder()->set_registered_name("ContainsBar");
  TF_ASSERT_OK(ComponentTransformer::ApplyAll(&component_spec));
  EXPECT_THAT(component_spec, test::EqualsProto(expected_spec));
}

// Tests that a spec with builder "FooBar" causes a conflict.
TEST(SelectBestComponentTransformerTest, Conflict) {
  ComponentSpec component_spec;
  component_spec.mutable_component_builder()->set_registered_name("FooBar");

  EXPECT_THAT(
      ComponentTransformer::ApplyAll(&component_spec),
      test::IsErrorWithSubstr("both think they should be dis-preferred"));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
