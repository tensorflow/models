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

#include "dragnn/runtime/xla/xla_aot_dynamic_component.h"

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/export.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/test/network_test_base.h"
#include "tensorflow/compiler/tf2xla/xla_compiled_cpu_function.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

using ::testing::_;
using ::testing::InSequence;
using ::testing::Invoke;

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

// Fake AOT class suitable for testing initialization.
class TestComponent {
 public:
  static const tensorflow::XlaCompiledCpuFunction::StaticData &StaticData() {
    static tensorflow::XlaCompiledCpuFunction::StaticData *kStaticData =
        new tensorflow::XlaCompiledCpuFunction::StaticData;
    return *kStaticData;
  }
};

constexpr char kXlaModel[] = "TestModel";
constexpr char kXlaComponent[] = "TestComponent";

class XlaAotDynamicComponent_TestModel_TestComponent
    : public XlaAotDynamicComponent<TestComponent> {
 public:
  XlaAotDynamicComponent_TestModel_TestComponent()
      : XlaAotDynamicComponent<TestComponent>(kXlaModel, kXlaComponent) {}

  using XlaAotDynamicComponent<TestComponent>::Supports;
  using XlaAotDynamicComponent<TestComponent>::InitializeFromComponentSpec;
};
DRAGNN_RUNTIME_REGISTER_COMPONENT(
    XlaAotDynamicComponent_TestModel_TestComponent);

class XlaAotDynamicComponentTest : public ::testing::Test {
 public:
  // Test util that builds a ComponentSpec with |component_name| set (if
  // non-empty). A CompilationSpec extension contains |model_name| (if
  // non-empty) and an empty CellSubgraphSpec if |include_subgraph_spec| is
  // true. No extension is added if |model_name| is empty and
  // |include_subgraph_spec| is false.
  ComponentSpec BuildComponentSpec(const string &model_name,
                                   const string &component_name,
                                   bool include_subgraph_spec) {
    ComponentSpec spec;
    if (!component_name.empty()) spec.set_name(component_name);

    // Add the extension if anything is in it.
    if (!model_name.empty() || include_subgraph_spec) {
      auto *compilation_spec =
          spec.MutableExtension(CompilationSpec::component_spec_extension);

      if (!model_name.empty()) compilation_spec->set_model_name(model_name);

      if (include_subgraph_spec) {
        CellSubgraphSpec cell_subgraph_spec;
        *compilation_spec->mutable_cell_subgraph_spec() = cell_subgraph_spec;
      }
    }
    return spec;
  }

 protected:
  XlaAotDynamicComponent_TestModel_TestComponent component_;
};

TEST_F(XlaAotDynamicComponentTest, Supports) {
  ComponentSpec spec = BuildComponentSpec(kXlaModel, kXlaComponent, true);

  EXPECT_TRUE(component_.Supports(spec, "XlaDynamicComponent"));
  EXPECT_TRUE(component_.Supports(
      spec, "XlaAotDynamicComponent_TestModel_TestComponent"));

  EXPECT_FALSE(component_.Supports(spec, "DynamicComponent"));
  EXPECT_FALSE(component_.Supports(spec, "XlaAotDynamicComponent"));
  EXPECT_FALSE(component_.Supports(
      spec, "XlaAotDynamicComponent_TestModel_OtherComponent"));
}

TEST_F(XlaAotDynamicComponentTest, SupportRequiresMatchingModelName) {
  EXPECT_FALSE(
      component_.Supports(BuildComponentSpec("OtherModel", kXlaComponent, true),
                          "XlaDynamicComponent"));

  EXPECT_FALSE(component_.Supports(BuildComponentSpec("", kXlaComponent, true),
                                   "XlaDynamicComponent"));
}

TEST_F(XlaAotDynamicComponentTest, SupportRequiresSubgraph) {
  EXPECT_FALSE(
      component_.Supports(BuildComponentSpec(kXlaModel, kXlaComponent, false),
                          "XlaDynamicComponent"));
}

TEST_F(XlaAotDynamicComponentTest, InitializeFromComponentSpec) {
  ComponentSpec component_spec;
  auto *compilation_spec = component_spec.MutableExtension(
      CompilationSpec::component_spec_extension);

  // Example spec.
  CellSubgraphSpec expected_cell_subgraph_spec;
  auto *input = expected_cell_subgraph_spec.add_input();
  input->set_name("fixed_channel_0_index_0_ids");
  input->set_tensor("cell/id:0");
  input->set_type(CellSubgraphSpec::Input::TYPE_FEATURE);
  auto *output = expected_cell_subgraph_spec.add_output();
  output->set_name("logits");
  output->set_tensor("cell/lookup:0");

  *compilation_spec->mutable_cell_subgraph_spec() = expected_cell_subgraph_spec;

  CellSubgraphSpec actual_cell_subgraph_spec;
  TF_ASSERT_OK(component_.InitializeFromComponentSpec(
      component_spec, &actual_cell_subgraph_spec));

  EXPECT_THAT(actual_cell_subgraph_spec,
              test::EqualsProto(expected_cell_subgraph_spec));
}

TEST_F(XlaAotDynamicComponentTest, InitializeFromComponentSpecNeedsSubgraph) {
  CellSubgraphSpec cell_subgraph_spec;
  TF_EXPECT_OK(component_.InitializeFromComponentSpec(
      BuildComponentSpec(kXlaModel, kXlaComponent, true), &cell_subgraph_spec));

  EXPECT_THAT(component_.InitializeFromComponentSpec(
                  BuildComponentSpec(kXlaModel, kXlaComponent, false),
                  &cell_subgraph_spec),
              test::IsErrorWithSubstr(
                  "Component TestComponent does not have a CellSubgraphSpec"));
}

// Tests using simple test AOT library.
constexpr int kNumSteps = 50;
constexpr int kVocabularySize = 123;
constexpr char kSimpleComponentSpecPath[] =
    "dragnn/runtime/xla/testdata/simple-component-spec";

class XlaAotDynamicComponentRunTest : public NetworkTestBase {
 public:
  // Creates a component, initializes it based on the |component_spec|,
  // and evaluates it. On error, returns non-OK.
  tensorflow::Status Run(const ComponentSpec &component_spec) {
    AddComponent(kTestComponentName);
    TF_RETURN_IF_ERROR(Component::CreateOrError(
        "XlaAotDynamicComponent_model_v1_test_component", &component_));

    TF_RETURN_IF_ERROR(component_->Initialize(component_spec, &variable_store_,
                                              &network_state_manager_,
                                              &extension_manager_));
    network_states_.Reset(&network_state_manager_);
    StartComponent(0);
    session_state_.extensions.Reset(&extension_manager_);
    TF_RETURN_IF_ERROR(
        component_->Evaluate(&session_state_, &compute_session_, nullptr));

    return tensorflow::Status::OK();
  }

 private:
  std::unique_ptr<Component> component_;
};

// Test that runs a simple deterministic component.
TEST_F(XlaAotDynamicComponentRunTest, Simple) {
  SetupTransitionLoop(kNumSteps);
  EXPECT_CALL(compute_session_, AdvanceFromOracle(kTestComponentName))
      .Times(kNumSteps);

  {  // Extract a sequence of feature IDs equal to 2 * step_index.
    ASSERT_LE(2 * kNumSteps, kVocabularySize);
    InSequence scoped;
    for (int step_index = 0; step_index < kNumSteps; ++step_index) {
      EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
          .WillOnce(Invoke(ExtractFeatures(0, {{2 * step_index, 1.0}})));
    }
  }

  ComponentSpec component_spec;
  TF_ASSERT_OK(tensorflow::ReadTextProto(
      tensorflow::Env::Default(),
      tensorflow::io::JoinPath(test::GetTestDataPrefix(),
                               kSimpleComponentSpecPath),
      &component_spec));
  TF_ASSERT_OK(Run(component_spec));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
