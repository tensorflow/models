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

#include <stddef.h>
#include <memory>
#include <string>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/bulk_network_unit.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/test/network_test_base.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include <gmock/gmock.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

using ::testing::_;
using ::testing::Invoke;
using ::testing::Return;

constexpr size_t kNumSteps = 50;
constexpr size_t kFixedDim = 11;
constexpr size_t kFixedVocabularySize = 123;
constexpr float kFixedValue = 0.5;
constexpr size_t kLinkedDim = 13;
constexpr float kLinkedValue = 1.25;
constexpr char kPreviousComponentName[] = "previous_component";
constexpr char kPreviousLayerName[] = "previous_layer";
constexpr char kOutputsName[] = "outputs";
constexpr size_t kOutputsDim = kFixedDim + kLinkedDim;

// Adds one to all inputs.
class BulkAddOne : public BulkNetworkUnit {
 public:
  // Implements BulkNetworkUnit.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override {
    return network_state_manager->AddLayer(kOutputsName, kOutputsDim,
                                           &outputs_handle_);
  }
  tensorflow::Status ValidateInputDimension(size_t dimension) const override {
    return tensorflow::Status::OK();
  }
  string GetLogitsName() const override { return ""; }
  tensorflow::Status Evaluate(Matrix<float> inputs,
                              SessionState *session_state) const override {
    const MutableMatrix<float> outputs =
        session_state->network_states.GetLayer(outputs_handle_);
    if (outputs.num_rows() != inputs.num_rows() ||
        outputs.num_columns() != inputs.num_columns()) {
      return tensorflow::errors::InvalidArgument("Dimension mismatch");
    }

    for (size_t row = 0; row < inputs.num_rows(); ++row) {
      for (size_t column = 0; column < inputs.num_columns(); ++column) {
        outputs.row(row)[column] = inputs.row(row)[column] + 1.0;
      }
    }

    return tensorflow::Status::OK();
  }

 private:
  // Output outputs.
  LayerHandle<float> outputs_handle_;
};

DRAGNN_RUNTIME_REGISTER_BULK_NETWORK_UNIT(BulkAddOne);

// A component that also prefers itself but is triggered on a certain backend.
// This can be used to cause a component selection conflict.
class ImTheBest : public Component {
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
  bool Supports(const ComponentSpec &component_spec,
                const string &normalized_builder_name) const override {
    return component_spec.backend().registered_name() == "CauseConflict";
  }
  bool PreferredTo(const Component &other) const override { return true; }
};

DRAGNN_RUNTIME_REGISTER_COMPONENT(ImTheBest);

class BulkDynamicComponentTest : public NetworkTestBase {
 protected:
  // Returns a spec that the network supports.
  ComponentSpec GetSupportedSpec() {
    ComponentSpec component_spec;
    component_spec.set_name(kTestComponentName);
    component_spec.set_num_actions(1);

    component_spec.mutable_network_unit()->set_registered_name("AddOne");
    component_spec.mutable_component_builder()->set_registered_name(
        "DynamicComponent");

    FixedFeatureChannel *fixed_feature = component_spec.add_fixed_feature();
    fixed_feature->set_size(1);
    fixed_feature->set_embedding_dim(kFixedDim);
    fixed_feature->set_vocabulary_size(kFixedVocabularySize);

    LinkedFeatureChannel *linked_feature = component_spec.add_linked_feature();
    linked_feature->set_size(1);
    linked_feature->set_embedding_dim(-1);
    linked_feature->set_source_component(kPreviousComponentName);
    linked_feature->set_source_layer(kPreviousLayerName);

    return component_spec;
  }

  // Adds mock call expectations to the |compute_session_| for the transition
  // system traversal and feature extraction.
  void AddComputeSessionMocks() {
    SetupTransitionLoop(kNumSteps);
    EXPECT_CALL(compute_session_, AdvanceFromOracle(_)).Times(kNumSteps);
    EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
        .Times(kNumSteps)
        .WillRepeatedly(Invoke(ExtractFeatures(0, {{0, 1.0}})));
    EXPECT_CALL(compute_session_, GetTranslatedLinkFeatures(_, _))
        .Times(kNumSteps)
        .WillRepeatedly(Invoke(ExtractLinks(0, {"step_idx: 0"})));
    EXPECT_CALL(compute_session_, SourceComponentBeamSize(_, _))
        .Times(kNumSteps)
        .WillRepeatedly(Return(1));
  }

  // Creates a network unit, initializes it based on the |component_spec_text|,
  // and evaluates it.  On error, returns non-OK.
  tensorflow::Status Run(const ComponentSpec &component_spec) {
    AddComponent(kPreviousComponentName);
    AddLayer(kPreviousLayerName, kLinkedDim);
    AddComponent(kTestComponentName);
    AddFixedEmbeddingMatrix(0, kFixedVocabularySize, kFixedDim, kFixedValue);

    TF_RETURN_IF_ERROR(
        Component::CreateOrError("BulkDynamicComponent", &component_));
    TF_RETURN_IF_ERROR(component_->Initialize(component_spec, &variable_store_,
                                              &network_state_manager_,
                                              &extension_manager_));

    // Allocates network states for a few steps.
    network_states_.Reset(&network_state_manager_);
    StartComponent(kNumSteps);
    FillLayer(kPreviousComponentName, kPreviousLayerName, kLinkedValue);
    StartComponent(0);
    session_state_.extensions.Reset(&extension_manager_);

    TF_RETURN_IF_ERROR(
        component_->Evaluate(&session_state_, &compute_session_, nullptr));
    outputs_ = GetLayer(kTestComponentName, kOutputsName);

    return tensorflow::Status::OK();
  }

  std::unique_ptr<Component> component_;
  Matrix<float> outputs_;
};

// Tests that the supported spec is supported.
TEST_F(BulkDynamicComponentTest, Supported) {
  const ComponentSpec component_spec = GetSupportedSpec();

  string component_type;
  TF_ASSERT_OK(Component::Select(component_spec, &component_type));
  EXPECT_EQ(component_type, "BulkDynamicComponent");

  AddComputeSessionMocks();
  TF_ASSERT_OK(Run(component_spec));

  ASSERT_EQ(outputs_.num_rows(), kNumSteps);
  ASSERT_EQ(outputs_.num_columns(), kFixedDim + kLinkedDim);

  for (size_t row = 0; row < kNumSteps; ++row) {
    size_t column = 0;
    for (; column < kFixedDim; ++column) {
      EXPECT_EQ(outputs_.row(row)[column], kFixedValue + 1.0);
    }
    for (; column < kFixedDim + kLinkedDim; ++column) {
      EXPECT_EQ(outputs_.row(row)[column], kLinkedValue + 1.0);
    }
  }
}

// Tests that the BulkDynamicComponent also supports its own name.
TEST_F(BulkDynamicComponentTest, SupportsBulkName) {
  ComponentSpec component_spec = GetSupportedSpec();
  component_spec.mutable_component_builder()->set_registered_name(
      "BulkDynamicComponent");

  string component_type;
  TF_ASSERT_OK(Component::Select(component_spec, &component_type));
  EXPECT_EQ(component_type, "BulkDynamicComponent");
}

// Tests that the transition system must be deterministic.
TEST_F(BulkDynamicComponentTest, ForbidNonDeterminism) {
  ComponentSpec component_spec = GetSupportedSpec();
  component_spec.set_num_actions(100);

  string component_type;
  EXPECT_THAT(
      Component::Select(component_spec, &component_type),
      test::IsErrorWithSubstr("Could not find a best spec for component"));

  EXPECT_THAT(Run(component_spec),
              test::IsErrorWithSubstr(
                  "BulkFeatureExtractorNetwork does not support component"));
}

// Tests that links cannot be recurrent.
TEST_F(BulkDynamicComponentTest, ForbidRecurrences) {
  ComponentSpec component_spec = GetSupportedSpec();
  component_spec.mutable_linked_feature(0)->set_source_component(
      kTestComponentName);

  string component_type;
  EXPECT_THAT(
      Component::Select(component_spec, &component_type),
      test::IsErrorWithSubstr("Could not find a best spec for component"));

  EXPECT_THAT(Run(component_spec),
              test::IsErrorWithSubstr(
                  "BulkFeatureExtractorNetwork does not support component"));
}

// Tests that the component prefers itself.
TEST_F(BulkDynamicComponentTest, PrefersItself) {
  ComponentSpec component_spec = GetSupportedSpec();
  component_spec.mutable_backend()->set_registered_name("CauseConflict");

  // The "CauseConflict" backend triggers the ImTheBest component, which also
  // prefers itself and leads to a selection conflict.
  string component_type;
  EXPECT_THAT(Component::Select(component_spec, &component_type),
              test::IsErrorWithSubstr("both think they should be preferred"));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
