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
#include <algorithm>
#include <limits>
#include <memory>
#include <string>

#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/network_unit.h"
#include "dragnn/runtime/session_state.h"
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
using ::testing::Return;

constexpr size_t kStepsDim = 41;
constexpr size_t kNumSteps = 23;

// Fills each row of its logits with the step index.
class StepsNetwork : public NetworkUnit {
 public:
  // Implements NetworkUnit.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override {
    return network_state_manager->AddLayer("steps", kStepsDim, &handle_);
  }
  string GetLogitsName() const override { return "steps"; }
  tensorflow::Status Evaluate(size_t step_index, SessionState *session_state,
                              ComputeSession *compute_session) const override {
    const MutableVector<float> logits =
        session_state->network_states.GetLayer(handle_).row(step_index);
    for (float &logit : logits) logit = step_index;
    return tensorflow::Status::OK();
  }

 private:
  // Handle to the logits layer.
  LayerHandle<float> handle_;
};

DRAGNN_RUNTIME_REGISTER_NETWORK_UNIT(StepsNetwork);

// As above, but does not report a logits layer.
class NoLogitsNetwork : public StepsNetwork {
 public:
  // Implements NetworkUnit.
  string GetLogitsName() const override { return ""; }
};

DRAGNN_RUNTIME_REGISTER_NETWORK_UNIT(NoLogitsNetwork);

class DynamicComponentTest : public NetworkTestBase {
 protected:
  // Creates a component, initializes it based on the |component_spec_text| and
  // |network_unit_name|, and evaluates it.  On error, returns non-OK.
  tensorflow::Status Run(const string &component_spec_text,
                         const string &network_unit_name) {
    ComponentSpec component_spec;
    CHECK(TextFormat::ParseFromString(component_spec_text, &component_spec));
    component_spec.set_name(kTestComponentName);
    component_spec.mutable_network_unit()->set_registered_name(
        network_unit_name);

    // Neither DynamicComponent nor the test networks use linked embeddings, so
    // a trivial network suffices.
    AddComponent(kTestComponentName);

    TF_RETURN_IF_ERROR(
        Component::CreateOrError("DynamicComponent", &component_));
    TF_RETURN_IF_ERROR(component_->Initialize(component_spec, &variable_store_,
                                              &network_state_manager_,
                                              &extension_manager_));

    network_states_.Reset(&network_state_manager_);
    StartComponent(0);  // DynamicComponent will add steps
    session_state_.extensions.Reset(&extension_manager_);

    TF_RETURN_IF_ERROR(
        component_->Evaluate(&session_state_, &compute_session_, nullptr));
    steps_ = GetLayer(kTestComponentName, "steps");
    return tensorflow::Status::OK();
  }

  std::unique_ptr<Component> component_;
  Matrix<float> steps_;
};

// Tests that DynamicComponent fails if the spec uses attention.
TEST_F(DynamicComponentTest, UnsupportedAttention) {
  EXPECT_THAT(Run("attention_component: 'foo'", "NoLogitsNetwork"),
              test::IsErrorWithSubstr("Attention is not supported"));
}

// Tests that DynamicComponent fails if the network does not produce logits.
TEST_F(DynamicComponentTest, NoLogits) {
  EXPECT_THAT(Run("", "NoLogitsNetwork"),
              test::IsErrorWithSubstr("Network unit does not produce logits"));
}

// Tests that DynamicComponent fails if the logits do not have the required
// dimension.
TEST_F(DynamicComponentTest, MismatchedLogitsDimension) {
  EXPECT_THAT(
      Run("num_actions: 42", "StepsNetwork"),
      test::IsErrorWithSubstr("Dimension mismatch between network unit logits "
                              "(41) and ComponentSpec.num_actions (42)"));
}

// Tests that DynamicComponent fails if ComputeSession::AdvanceFromPrediction()
// returns false.
TEST_F(DynamicComponentTest, FailToAdvanceFromPrediction) {
  EXPECT_CALL(compute_session_, IsTerminal(_)).WillRepeatedly(Return(false));
  EXPECT_CALL(compute_session_, AdvanceFromPrediction(_, _, _, _))
      .WillOnce(Return(false));

  EXPECT_THAT(Run("num_actions: 41", "StepsNetwork"),
              test::IsErrorWithSubstr(
                  "Error in ComputeSession::AdvanceFromPrediction()"));
}

// Tests that DynamicComponent evaluates its network unit once per transition,
// each time passing the proper step index.
TEST_F(DynamicComponentTest, Steps) {
  SetupTransitionLoop(kNumSteps);

  // Accept |num_steps| transition steps.
  EXPECT_CALL(compute_session_, AdvanceFromPrediction(_, _, _, _))
      .Times(kNumSteps)
      .WillRepeatedly(Return(true));

  TF_ASSERT_OK(Run("num_actions: 41", "StepsNetwork"));

  ASSERT_EQ(steps_.num_rows(), kNumSteps);
  for (size_t step_index = 0; step_index < kNumSteps; ++step_index) {
    ExpectVector(steps_.row(step_index), kStepsDim, step_index);
  }
}

// Tests that DynamicComponent calls ComputeSession::AdvanceFromOracle() and
// does not use logits when the component is deterministic.
TEST_F(DynamicComponentTest, Determinstic) {
  SetupTransitionLoop(kNumSteps);

  // Take the oracle transition instead of predicting from logits.
  EXPECT_CALL(compute_session_, AdvanceFromOracle(_)).Times(kNumSteps);

  TF_EXPECT_OK(Run("num_actions: 1", "NoLogitsNetwork"));

  // The NoLogitsNetwork still produces the "steps" layer, even if it does not
  // mark them as its logits.
  ASSERT_EQ(steps_.num_rows(), kNumSteps);
  for (size_t step_index = 0; step_index < kNumSteps; ++step_index) {
    ExpectVector(steps_.row(step_index), kStepsDim, step_index);
  }
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
