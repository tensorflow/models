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

#include "dragnn/runtime/component.h"

#include <memory>
#include <string>

#include "dragnn/core/compute_session.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/variable_store.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Expects that the two pointers have the same address.
void ExpectSameAddress(const void *pointer1, const void *pointer2) {
  EXPECT_EQ(pointer1, pointer2);
}

// A trivial implementation for tests.
class FooComponent : public Component {
 public:
  // Implements Component.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status Evaluate(SessionState *session_state,
                              ComputeSession *compute_session,
                              ComponentTrace *component_trace) const override {
    return tensorflow::Status::OK();
  }
  bool Supports(const ComponentSpec &spec,
                const string &normalized_builder_name) const override {
    return normalized_builder_name == "FooComponent";
  }
  bool PreferredTo(const Component &other) const override { return false; }
};

DRAGNN_RUNTIME_REGISTER_COMPONENT(FooComponent);

// Class that always says it's preferred.
class ImTheBest1 : public FooComponent {
 public:
  bool Supports(const ComponentSpec &spec,
                const string &normalized_builder_name) const override {
    return normalized_builder_name == "ImTheBest";
  }
  bool PreferredTo(const Component &other) const override { return true; }
};
class ImTheBest2 : public ImTheBest1 {};
DRAGNN_RUNTIME_REGISTER_COMPONENT(ImTheBest1);
DRAGNN_RUNTIME_REGISTER_COMPONENT(ImTheBest2);

// Class that always says it's dispreferred.
class ImTheWorst1 : public FooComponent {
 public:
  bool Supports(const ComponentSpec &spec,
                const string &normalized_builder_name) const override {
    return normalized_builder_name == "ImTheWorst";
  }
  bool PreferredTo(const Component &other) const override { return false; }
};
class ImTheWorst2 : public ImTheWorst1 {};
DRAGNN_RUNTIME_REGISTER_COMPONENT(ImTheWorst1);
DRAGNN_RUNTIME_REGISTER_COMPONENT(ImTheWorst2);

// Specialized foo implementation. We use debug-mode down-casting to check that
// the correct sub-class was instantiated.
class SpecializedFooComponent : public Component {
 public:
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status Evaluate(SessionState *session_state,
                              ComputeSession *compute_session,
                              ComponentTrace *component_trace) const override {
    return tensorflow::Status::OK();
  }
  bool Supports(const ComponentSpec &spec,
                const string &normalized_builder_name) const override {
    return normalized_builder_name == "FooComponent" && spec.num_actions() == 1;
  }
  bool PreferredTo(const Component &other) const override { return true; }
};
DRAGNN_RUNTIME_REGISTER_COMPONENT(SpecializedFooComponent);

TEST(ComponentTest, NameResolutionError) {
  ComponentSpec component_spec;
  EXPECT_DEATH(GetNormalizedComponentBuilderName(component_spec),
               "No builder name for component spec");
}

// Tests that Python-esque module specifiers for builders are normalized
// appropriately.
TEST(ComponentTest, VariantsOfComponentBuilderNameResolve) {
  for (const string &registered_name :
       {"FooComponent",
        "FooComponentBuilder",
        "module.FooComponent",
        "module.FooComponentBuilder",
        "some.long.path.to.module.FooComponent",
        "some.long.path.to.module.FooComponentBuilder"}) {
    ComponentSpec component_spec;
    component_spec.mutable_component_builder()->set_registered_name(
        registered_name);

    string result;
    TF_ASSERT_OK(Component::Select(component_spec, &result));
    EXPECT_EQ(result, "FooComponent");
  }
}

TEST(ComponentTest, ErrorWithBothPreferred) {
  ComponentSpec component_spec;
  component_spec.mutable_component_builder()->set_registered_name("ImTheBest");
  string result;
  EXPECT_THAT(
      Component::Select(component_spec, &result),
      test::IsErrorWithCodeAndSubstr(tensorflow::error::FAILED_PRECONDITION,
                                     "Classes 'ImTheBest2' and 'ImTheBest1' "
                                     "both think they should be preferred to "
                                     "each-other. Please add logic to their "
                                     "PreferredTo() methods to avoid this."));
}

TEST(ComponentTest, ErrorWithNeitherPreferred) {
  ComponentSpec component_spec;
  component_spec.mutable_component_builder()->set_registered_name("ImTheWorst");
  string result;
  EXPECT_THAT(Component::Select(component_spec, &result),
              test::IsErrorWithCodeAndSubstr(
                  tensorflow::error::FAILED_PRECONDITION,
                  "Classes 'ImTheWorst2' and 'ImTheWorst1' both think they "
                  "should be dis-preferred to each-other. Please add logic to "
                  "their PreferredTo() methods to avoid this."));
}

TEST(ComponentTest, DefaultComponent) {
  ComponentSpec component_spec;
  component_spec.mutable_component_builder()->set_registered_name(
      "FooComponent");
  component_spec.set_num_actions(45);
  string result;
  TF_EXPECT_OK(Component::Select(component_spec, &result));
  EXPECT_EQ(result, "FooComponent");
}

TEST(ComponentTest, SpecializedComponent) {
  ComponentSpec component_spec;
  component_spec.mutable_component_builder()->set_registered_name(
      "FooComponent");
  component_spec.set_num_actions(1);
  string result;
  TF_EXPECT_OK(Component::Select(component_spec, &result));
  EXPECT_EQ(result, "SpecializedFooComponent");
}

// Tests that Select() returns NOT_FOUND when there is no matching component.
TEST(ComponentTest, NoMatchingComponentNotFound) {
  ComponentSpec component_spec;
  component_spec.mutable_component_builder()->set_registered_name("unknown");
  string result;
  EXPECT_THAT(Component::Select(component_spec, &result),
              test::IsErrorWithCodeAndSubstr(
                  tensorflow::error::NOT_FOUND,
                  "Could not find a best spec for component"));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
