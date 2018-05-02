// Copyright 2018 Google Inc. All Rights Reserved.
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

#include <stddef.h>
#include <memory>
#include <string>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/test/network_test_base.h"
#include "syntaxnet/base.h"
#include <gmock/gmock.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

using ::testing::Return;

constexpr size_t kNumSteps = 33;
constexpr size_t kSourceDim = 44;
constexpr size_t kTargetDim = 55;
constexpr size_t kBadDim = 11;

constexpr float kArcWeight = 1.0;
constexpr float kSourceWeight = 2.0;
constexpr float kRootWeight = 4.0;
constexpr float kRootBias = 8.0;
constexpr float kSourceValue = -0.5;
constexpr float kTargetValue = 1.5;

constexpr char kSourcesComponentName[] = "sources";
constexpr char kTargetsComponentName[] = "targets";
constexpr char kSourcesLayerName[] = "sources";
constexpr char kTargetsLayerName[] = "targets";
constexpr char kBadDimLayerName[] = "bad";

// Configuration for the Run() method.  This makes it easier for tests to
// manipulate breakages.
struct RunConfig {
  // Number of steps in the preceding components.
  size_t sources_num_steps = kNumSteps;
  size_t targets_num_steps = kNumSteps;

  // Dimensions of the variables.
  size_t weights_source_dim = kSourceDim;
  size_t root_weights_dim = kTargetDim;
  size_t root_bias_dim = 1;
};

class BiaffineDigraphComponentTest : public NetworkTestBase {
 protected:
  BiaffineDigraphComponentTest() {
    EXPECT_CALL(compute_session_, GetInputBatchCache())
        .WillRepeatedly(Return(&input_));
  }

  // Returns a working spec.
  static ComponentSpec MakeGoodSpec() {
    ComponentSpec component_spec;
    component_spec.set_name(kTestComponentName);
    component_spec.mutable_component_builder()->set_registered_name(
        "bulk_component.BulkFeatureExtractorComponentBuilder");
    component_spec.mutable_network_unit()->set_registered_name(
        "biaffine_units.BiaffineDigraphNetwork");

    for (const string &name : {kSourcesLayerName, kTargetsLayerName}) {
      LinkedFeatureChannel *link = component_spec.add_linked_feature();
      link->set_name(name);
      link->set_embedding_dim(-1);
      link->set_size(1);
      link->set_source_component(name);
      link->set_source_layer(name);
      link->set_source_translator("identity");
      link->set_fml("input.focus");
    }

    return component_spec;
  }

  // Creates a component, initializes it based on the |component_spec|, and
  // evaluates it.  On error, returns non-OK.
  tensorflow::Status Run(const ComponentSpec &component_spec,
                         const RunConfig &config = RunConfig()) {
    AddComponent(kSourcesComponentName);
    AddLayer(kSourcesLayerName, kSourceDim);
    AddComponent(kTargetsComponentName);
    AddLayer(kTargetsLayerName, kTargetDim);
    AddLayer(kBadDimLayerName, kBadDim);
    AddComponent(kTestComponentName);

    AddMatrixVariable(
        tensorflow::strings::StrCat(kTestComponentName, "/weights_arc"),
        kSourceDim, kTargetDim, kArcWeight);
    AddVectorVariable(
        tensorflow::strings::StrCat(kTestComponentName, "/weights_source"),
        config.weights_source_dim, kSourceWeight);
    AddVectorVariable(
        tensorflow::strings::StrCat(kTestComponentName, "/root_weights"),
        config.root_weights_dim, kRootWeight);
    AddVectorVariable(
        tensorflow::strings::StrCat(kTestComponentName, "/root_bias"),
        config.root_bias_dim, kRootBias);

    TF_RETURN_IF_ERROR(
        Component::CreateOrError("BiaffineDigraphComponent", &component_));
    TF_RETURN_IF_ERROR(component_->Initialize(component_spec, &variable_store_,
                                              &network_state_manager_,
                                              &extension_manager_));

    network_states_.Reset(&network_state_manager_);
    StartComponent(config.sources_num_steps);
    FillLayer(kSourcesComponentName, kSourcesLayerName, kSourceValue);
    StartComponent(config.targets_num_steps);
    FillLayer(kTargetsComponentName, kTargetsLayerName, kTargetValue);
    StartComponent(0);  // BiaffineDigraphComponent will add steps
    session_state_.extensions.Reset(&extension_manager_);

    TF_RETURN_IF_ERROR(
        component_->Evaluate(&session_state_, &compute_session_, nullptr));
    adjacency_ = GetPairwiseLayer(kTestComponentName, "adjacency");
    return tensorflow::Status::OK();
  }

  InputBatchCache input_;
  std::unique_ptr<Component> component_;
  Matrix<float> adjacency_;
};

// Tests that the good spec works properly.
TEST_F(BiaffineDigraphComponentTest, GoodSpec) {
  TF_ASSERT_OK(Run(MakeGoodSpec()));

  constexpr float kExpectedRootScore =
      kRootWeight * kTargetValue * kTargetDim + kRootBias;
  constexpr float kExpectedArcScore =
      kSourceDim * kSourceValue * kArcWeight * kTargetValue * kTargetDim +
      kSourceWeight * kSourceValue * kSourceDim;

  ASSERT_EQ(adjacency_.num_rows(), kNumSteps);
  ASSERT_EQ(adjacency_.num_columns(), kNumSteps);
  for (size_t row = 0; row < kNumSteps; ++row) {
    for (size_t column = 0; column < kNumSteps; ++column) {
      if (row == column) {
        ASSERT_EQ(adjacency_.row(row)[column], kExpectedRootScore);
      } else {
        ASSERT_EQ(adjacency_.row(row)[column], kExpectedArcScore);
      }
    }
  }
}

// Tests the set of supported components.
TEST_F(BiaffineDigraphComponentTest, Supports) {
  ComponentSpec component_spec = MakeGoodSpec();
  string component_name;

  TF_ASSERT_OK(Component::Select(component_spec, &component_name));
  EXPECT_EQ(component_name, "BiaffineDigraphComponent");

  component_spec.mutable_network_unit()->set_registered_name("bad");
  EXPECT_THAT(Component::Select(component_spec, &component_name),
              test::IsErrorWithSubstr("Could not find a best spec"));

  component_spec = MakeGoodSpec();
  component_spec.mutable_component_builder()->set_registered_name(
      "BiaffineDigraphComponent");
  TF_ASSERT_OK(Component::Select(component_spec, &component_name));
  EXPECT_EQ(component_name, "BiaffineDigraphComponent");

  component_spec.mutable_component_builder()->set_registered_name("bad");
  EXPECT_THAT(Component::Select(component_spec, &component_name),
              test::IsErrorWithSubstr("Could not find a best spec"));
}

// Tests that fixed features are rejected.
TEST_F(BiaffineDigraphComponentTest, FixedFeatures) {
  ComponentSpec component_spec = MakeGoodSpec();
  component_spec.add_fixed_feature();

  EXPECT_THAT(Run(component_spec),
              test::IsErrorWithSubstr("fixed features are forbidden"));
}

// Tests that too few linked features are rejected.
TEST_F(BiaffineDigraphComponentTest, TooFewLinkedFeatures) {
  ComponentSpec component_spec = MakeGoodSpec();
  component_spec.mutable_linked_feature()->RemoveLast();

  EXPECT_THAT(Run(component_spec),
              test::IsErrorWithSubstr("two linked features are required"));
}

// Tests that too many linked features are rejected.
TEST_F(BiaffineDigraphComponentTest, TooManyLinkedFeatures) {
  ComponentSpec component_spec = MakeGoodSpec();
  component_spec.add_linked_feature();

  EXPECT_THAT(Run(component_spec),
              test::IsErrorWithSubstr("two linked features are required"));
}

// Tests that a spec with no "sources" link is rejected.
TEST_F(BiaffineDigraphComponentTest, MissingSources) {
  ComponentSpec component_spec = MakeGoodSpec();
  component_spec.mutable_linked_feature(0)->set_name("bad");

  EXPECT_THAT(Run(component_spec),
              test::IsErrorWithSubstr("link 'sources' does not exist"));
}

// Tests that a spec with no "targets" link is rejected.
TEST_F(BiaffineDigraphComponentTest, MissingTargets) {
  ComponentSpec component_spec = MakeGoodSpec();
  component_spec.mutable_linked_feature(1)->set_name("bad");

  EXPECT_THAT(Run(component_spec),
              test::IsErrorWithSubstr("link 'targets' does not exist"));
}

// Tests that a spec with transformed links is rejected.
TEST_F(BiaffineDigraphComponentTest, TransformedLink) {
  ComponentSpec component_spec = MakeGoodSpec();
  component_spec.mutable_linked_feature(1)->set_embedding_dim(123);

  EXPECT_THAT(Run(component_spec),
              test::IsErrorWithSubstr("transformed links are forbidden"));
}

// Tests that a spec with multi-embedding links is rejected.
TEST_F(BiaffineDigraphComponentTest, MultiEmbeddingLink) {
  ComponentSpec component_spec = MakeGoodSpec();
  component_spec.mutable_linked_feature(1)->set_size(2);

  EXPECT_THAT(Run(component_spec),
              test::IsErrorWithSubstr("multi-embedding links are forbidden"));
}

// Tests that a spec with recurrent links is rejected.
TEST_F(BiaffineDigraphComponentTest, RecurrentLink) {
  ComponentSpec component_spec = MakeGoodSpec();
  component_spec.mutable_linked_feature(1)->set_source_component(
      kTestComponentName);

  EXPECT_THAT(Run(component_spec),
              test::IsErrorWithSubstr("recurrent links are forbidden"));
}

// Tests that a spec with improper FML is rejected.
TEST_F(BiaffineDigraphComponentTest, BadFML) {
  ComponentSpec component_spec = MakeGoodSpec();
  component_spec.mutable_linked_feature(1)->set_fml("bad");

  EXPECT_THAT(
      Run(component_spec),
      test::IsErrorWithSubstr("non-trivial link translation is forbidden"));
}

// Tests that a spec with non-identity links is rejected.
TEST_F(BiaffineDigraphComponentTest, NonIdentityLink) {
  ComponentSpec component_spec = MakeGoodSpec();
  component_spec.mutable_linked_feature(1)->set_source_translator("bad");

  EXPECT_THAT(
      Run(component_spec),
      test::IsErrorWithSubstr("non-trivial link translation is forbidden"));
}

// Tests that a link with the wrong dimension is rejected.
TEST_F(BiaffineDigraphComponentTest, WrongLinkDimension) {
  ComponentSpec component_spec = MakeGoodSpec();
  component_spec.mutable_linked_feature(1)->set_source_layer(kBadDimLayerName);

  EXPECT_THAT(
      Run(component_spec),
      test::IsErrorWithSubstr("link 'targets' has dimension 11 instead of 55"));
}

// Tests that a mismatched weights_source dimension is rejected.
TEST_F(BiaffineDigraphComponentTest, WeightsSourceDimensionMismatch) {
  RunConfig config;
  config.weights_source_dim = 999;

  EXPECT_THAT(Run(MakeGoodSpec(), config),
              test::IsErrorWithSubstr("dimension mismatch between weights_arc "
                                      "[44,55] and weights_source [999]"));
}

// Tests that a mismatched root_weights dimension is rejected.
TEST_F(BiaffineDigraphComponentTest, RootWeightsDimensionMismatch) {
  RunConfig config;
  config.root_weights_dim = 999;

  EXPECT_THAT(Run(MakeGoodSpec(), config),
              test::IsErrorWithSubstr("dimension mismatch between weights_arc "
                                      "[44,55] and root_weights [999]"));
}

// Tests that a mismatched root_bias dimension is rejected.
TEST_F(BiaffineDigraphComponentTest, RootBiasDimensionMismatch) {
  RunConfig config;
  config.root_bias_dim = 999;

  EXPECT_THAT(Run(MakeGoodSpec(), config),
              test::IsErrorWithSubstr("root_bias must be a singleton"));
}

// Tests that a mismatched number of steps is rejected.
TEST_F(BiaffineDigraphComponentTest, StepCountMismatch) {
  RunConfig config;
  config.targets_num_steps = 999;

  EXPECT_THAT(
      Run(MakeGoodSpec(), config),
      test::IsErrorWithSubstr(
          "step count mismatch between sources (33) and targets (999)"));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
