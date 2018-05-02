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

#include "dragnn/runtime/master.h"

#include <stddef.h>
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "dragnn/core/test/generic.h"
#include "dragnn/core/test/mock_compute_session.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/test/fake_variable_store.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include <gmock/gmock.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

using ::testing::_;
using ::testing::InSequence;
using ::testing::Invoke;
using ::testing::Return;

// Number of steps to take in each component.
constexpr size_t kNumSteps = 123;

// Outputs a layer of all 1s.
class Ones : public Component {
 public:
  // Implements Component.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override {
    return network_state_manager->AddLayer("ones", 1, &output_handle_);
  }
  tensorflow::Status Evaluate(SessionState *session_state,
                              ComputeSession *compute_session,
                              ComponentTrace *component_trace) const override {
    NetworkStates *network_states = &session_state->network_states;
    for (size_t step = 0; step < kNumSteps; ++step) {
      network_states->AddStep();
      network_states->GetLayer(output_handle_).row(step)[0] = 1.0;
    }
    return tensorflow::Status::OK();
  }
  bool Supports(const ComponentSpec &spec,
                const string &normalized_builder_name) const override {
    return normalized_builder_name == "Ones";
  }
  bool PreferredTo(const Component &other) const override { return false; }

 private:
  // Handle to the output layer.
  LayerHandle<float> output_handle_;
};

DRAGNN_RUNTIME_REGISTER_COMPONENT(Ones);

// Extends its input layer with the step-wise cumulative sum of the final entry
// in each row of the input.  E.g.,
//   [[0, 1],      [[0, 1, 1 (= 1)],
//    [2, 3],  =>   [2, 3, 4 (= 1 + 3)],
//    [4, 5]]       [4, 5, 9 (= 1 + 3 + 5)]]
class ExtendWithCumulativeSum : public Component {
 public:
  // Implements Component.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override {
    // NB: In a real Component implementation, linked embeddings are accessed
    // using the LinkedEmbeddingManager and LinkedEmbeddings.  Here, we set up
    // the link manually because it's simple and makes the test self-contained.
    CHECK_EQ(component_spec.linked_feature_size(), 1);
    const LinkedFeatureChannel &link = component_spec.linked_feature(0);
    size_t dimension = 0;
    TF_RETURN_IF_ERROR(network_state_manager->LookupLayer(
        link.source_component(), link.source_layer(), &dimension,
        &input_handle_));
    CHECK_GT(dimension, 0);
    return network_state_manager->AddLayer("sums", dimension + 1,
                                           &output_handle_);
  }

  tensorflow::Status Evaluate(SessionState *session_state,
                              ComputeSession *compute_session,
                              ComponentTrace *component_trace) const override {
    NetworkStates *network_states = &session_state->network_states;
    float sum = 0.0;
    for (size_t step = 0; step < kNumSteps; ++step) {
      network_states->AddStep();
      const Vector<float> inputs(
          network_states->GetLayer(input_handle_).row(step));
      const MutableVector<float> outputs(
          network_states->GetLayer(output_handle_).row(step));
      CHECK_EQ(outputs.size(), inputs.size() + 1);
      sum += inputs[inputs.size() - 1];
      *std::copy(inputs.begin(), inputs.end(), outputs.begin()) = sum;
    }
    return tensorflow::Status::OK();
  }

  bool Supports(const ComponentSpec &spec,
                const string &normalized_builder_name) const override {
    return normalized_builder_name == "ExtendWithCumulativeSum";
  }

  bool PreferredTo(const Component &other) const override { return false; }

 private:
  // Handles to the input and output layers.
  LayerHandle<float> input_handle_;
  LayerHandle<float> output_handle_;
};

DRAGNN_RUNTIME_REGISTER_COMPONENT(ExtendWithCumulativeSum);

// Makes predictions using its inputs.
class MakePredictions : public Component {
 public:
  // Implements Component.
  tensorflow::Status Initialize(const ComponentSpec &component_spec,
                                VariableStore *variable_store,
                                NetworkStateManager *network_state_manager,
                                ExtensionManager *extension_manager) override {
    name_ = component_spec.name();
    CHECK_EQ(component_spec.linked_feature_size(), 1);
    const LinkedFeatureChannel &link = component_spec.linked_feature(0);
    size_t dimension = 0;
    return network_state_manager->LookupLayer(link.source_component(),
                                              link.source_layer(), &dimension,
                                              &input_handle_);
  }

  tensorflow::Status Evaluate(SessionState *session_state,
                              ComputeSession *compute_session,
                              ComponentTrace *component_trace) const override {
    NetworkStates *network_states = &session_state->network_states;
    Matrix<float> inputs(network_states->GetLayer(input_handle_));
    for (size_t step = 0; step < kNumSteps; ++step) {
      const Vector<float> logits = inputs.row(step);
      if (!compute_session->AdvanceFromPrediction(name_, logits.data(), 1,
                                                  logits.size())) {
        return tensorflow::errors::Internal(
            "Error in ComputeSession::AdvanceFromPrediction() at step ", step);
      }
    }
    return tensorflow::Status::OK();
  }

  bool Supports(const ComponentSpec &spec,
                const string &normalized_builder_name) const override {
    return normalized_builder_name == "MakePredictions";
  }

  bool PreferredTo(const Component &other) const override { return false; }

 private:
  // Name of this component.
  string name_;

  // Handle to the input layer, which is treated as prediction logits.
  LayerHandle<float> input_handle_;
};

DRAGNN_RUNTIME_REGISTER_COMPONENT(MakePredictions);

// Component whose Evaluate() always fails.
class AlwaysFails : public Component {
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
    return tensorflow::errors::Internal("I always fail!");
  }

  bool Supports(const ComponentSpec &spec,
                const string &normalized_builder_name) const override {
    return normalized_builder_name == "AlwaysFails";
  }

  bool PreferredTo(const Component &other) const override { return false; }
};

DRAGNN_RUNTIME_REGISTER_COMPONENT(AlwaysFails);

class MasterTest : public ::testing::Test {
 protected:
  // Returns a new VariableStore.
  static std::unique_ptr<VariableStore> NewVariableStore() {
    // None of the tests or components look at the pre-trained variables, so
    // return an empty store.
    return std::unique_ptr<VariableStore>(new FakeVariableStore());
  }

  // Initializes and runs the |master_| using the text-format MasterSpec in
  // |master_spec_text|.  The |master_trace| is overwritten with traces, if
  // specified.  If |expect_success| is false, then EXPECT_CALLs that assume
  // success are disabled.  On error, returns non-OK.
  tensorflow::Status TryRun(const string &master_spec_text, bool expect_success,
                            MasterTrace *master_trace = nullptr) {
    MasterSpec master_spec;
    CHECK(TextFormat::ParseFromString(master_spec_text, &master_spec));

    TF_RETURN_IF_ERROR(master_.Initialize(master_spec, NewVariableStore()));

    {  // Add call expectations for initializing each component, in order.
      InSequence ordered_calls;
      for (const ComponentSpec &component_spec : master_spec.component()) {
        EXPECT_CALL(compute_session_,
                    InitializeComponentData(component_spec.name(), 1))
            .Times(1);
      }
    }

    // If applicable, add call expectations for making "predictions" in the
    // final component that capture the prediction logits for inspection.
    if (master_spec.component_size() > 0 && expect_success) {
      const string &last_component_name =
          master_spec.component(master_spec.component_size() - 1).name();
      EXPECT_CALL(compute_session_,
                  AdvanceFromPrediction(last_component_name, _, 1, _))
          .Times(kNumSteps)
          .WillRepeatedly(
              Invoke([this](const string &, const float *data, int, int size) {
                logits_.emplace_back(data, data + size);
                return true;
              }));
    }

    // Add call expectations for finalizing data in all components.
    if (expect_success) {
      for (const ComponentSpec &component_spec : master_spec.component()) {
        EXPECT_CALL(compute_session_, FinalizeData(component_spec.name()))
            .Times(1);
      }
    }

    return master_.Evaluate(&compute_session_, master_trace);
  }

  // As above, but asserts that all operations succeed.
  void Run(const string &master_spec_text,
           MasterTrace *master_trace = nullptr) {
    TF_ASSERT_OK(
        TryRun(master_spec_text, /*expect_success=*/true, master_trace));
  }

  ::testing::StrictMock<MockComputeSession> compute_session_;
  std::vector<std::vector<float>> logits_;
  Master master_;
};

// Tests that Master cannot be initialized multiple times.
TEST_F(MasterTest, InitializeTwice) {
  TF_ASSERT_OK(master_.Initialize(MasterSpec(), NewVariableStore()));
  EXPECT_THAT(master_.Initialize(MasterSpec(), NewVariableStore()),
              test::IsErrorWithSubstr("Can't initialize twice"));
}

// Tests that Master requires a variable store.
TEST_F(MasterTest, NoVariableStore) {
  EXPECT_THAT(master_.Initialize(MasterSpec(), nullptr),
              test::IsErrorWithSubstr("No VariableStore"));
}

// Tests that Master must be initialized prior to session.
TEST_F(MasterTest, EvaluateWithoutInitializing) {
  EXPECT_THAT(master_.Evaluate(&compute_session_, nullptr),
              test::IsErrorWithSubstr("Not initialized"));
}

// Tests that Master requires a compute session.
TEST_F(MasterTest, NoComputeSession) {
  TF_ASSERT_OK(master_.Initialize(MasterSpec(), NewVariableStore()));
  EXPECT_THAT(master_.Evaluate(nullptr, nullptr),
              test::IsErrorWithSubstr("No ComputeSession"));
}

// Tests that Master works with an empty spec and does nothing (StrictMock would
// raise an error if any methods on the ComputeSession were called).
TEST_F(MasterTest, EmptySpec) {
  Run("");

  EXPECT_TRUE(logits_.empty());
}

// Tests that Master can run a simple pipeline that generates ones.
TEST_F(MasterTest, Ones) {
  Run(R"(component {
           name: 'component1'
           component_builder {
             registered_name: 'Ones'
           }
         }
         component {
           name: 'component2'
           component_builder {
             registered_name: 'MakePredictions'
           }
           linked_feature {
             source_component: 'component1'
             source_layer: 'ones'
           }
         })");

  EXPECT_EQ(logits_.size(), kNumSteps);
  const std::vector<float> expected_row = {1.0};
  for (const auto &row : logits_) EXPECT_EQ(row, expected_row);
}

// Tests that Master can run a pipeline with a cumulative summation.
TEST_F(MasterTest, SingleSummation) {
  Run(R"(component {
           name: 'component1'
           component_builder {
             registered_name: 'Ones'
           }
         }
         component {
           name: 'component2'
           component_builder {
             registered_name: 'ExtendWithCumulativeSum'
           }
           linked_feature {
             source_component: 'component1'
             source_layer: 'ones'
           }
         }
         component {
           name: 'component3'
           component_builder {
             registered_name: 'MakePredictions'
           }
           linked_feature {
             source_component: 'component2'
             source_layer: 'sums'
           }
         })");

  EXPECT_EQ(logits_.size(), kNumSteps);
  float sum = 0.0;
  for (const auto &row : logits_) {
    ++sum;
    const std::vector<float> expected_row = {1.0, sum};
    EXPECT_EQ(row, expected_row);
  }
}

// Tests that Master can run a pipeline with multiple summations.
TEST_F(MasterTest, MultiSummation) {
  Run(R"(component {
           name: 'component1'
           component_builder {
             registered_name: 'Ones'
           }
         }
         component {
           name: 'component2'
           component_builder {
             registered_name: 'ExtendWithCumulativeSum'
           }
           linked_feature {
             source_component: 'component1'
             source_layer: 'ones'
           }
         }
         component {
           name: 'component3'
           component_builder {
             registered_name: 'ExtendWithCumulativeSum'
           }
           linked_feature {
             source_component: 'component2'
             source_layer: 'sums'
           }
         }
         component {
           name: 'component4'
           component_builder {
             registered_name: 'ExtendWithCumulativeSum'
           }
           linked_feature {
             source_component: 'component3'
             source_layer: 'sums'
           }
         }
         component {
           name: 'component5'
           component_builder {
             registered_name: 'MakePredictions'
           }
           linked_feature {
             source_component: 'component4'
             source_layer: 'sums'
           }
         })");

  EXPECT_EQ(logits_.size(), kNumSteps);
  float sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
  for (const auto &row : logits_) {
    sum3 += sum2 += ++sum1;
    const std::vector<float> expected_row = {1.0, sum1, sum2, sum3};
    EXPECT_EQ(row, expected_row);
  }
}

// Tests that Master can run a pipeline with tracing.
TEST_F(MasterTest, SingleSummationWithTracing) {
  { // Expect to enable and then disable tracing, in that order.
    InSequence ordered_calls;
    EXPECT_CALL(compute_session_, SetTracing(true));
    EXPECT_CALL(compute_session_, SetTracing(false));
  }

  // Build a set of traces for the compute session to return.
  std::vector<MasterTrace> traces(1);
  traces.back().add_component_trace()->add_step_trace()->set_caption("A");
  traces.back().add_component_trace()->add_step_trace()->set_caption("B");
  traces.back().add_component_trace()->add_step_trace()->set_caption("C");
  traces.back().add_component_trace()->add_step_trace()->set_caption("D");
  EXPECT_CALL(compute_session_, GetTraceProtos()).WillOnce(Return(traces));

  MasterTrace master_trace;
  Run(R"(component {
           name: 'component1'
           component_builder {
             registered_name: 'Ones'
           }
         }
         component {
           name: 'component2'
           component_builder {
             registered_name: 'ExtendWithCumulativeSum'
           }
           linked_feature {
             source_component: 'component1'
             source_layer: 'ones'
           }
         }
         component {
           name: 'component3'
           component_builder {
             registered_name: 'MakePredictions'
           }
           linked_feature {
             source_component: 'component2'
             source_layer: 'sums'
           }
         })",
      &master_trace);

  const string kExpectedTraceText = R"(
    component_trace { name: 'component1' step_trace { caption: 'A' } }
    component_trace { name: 'component2' step_trace { caption: 'B' } }
    component_trace { name: 'component3' step_trace { caption: 'C' } }
    component_trace {                    step_trace { caption: 'D' } }
  )";
  MasterTrace expected_trace;
  ASSERT_TRUE(TextFormat::ParseFromString(kExpectedTraceText, &expected_trace));

  EXPECT_THAT(master_trace, test::EqualsProto(expected_trace));
}

// Tests that Master disables tracing even on error.
TEST_F(MasterTest, DisablesTracingOnFailure) {
  { // Expect to enable and then disable tracing, in that order.
    InSequence ordered_calls;
    EXPECT_CALL(compute_session_, SetTracing(true));
    EXPECT_CALL(compute_session_, SetTracing(false));
  }

  const string kMasterSpec = R"(component {
                                name: 'component1'
                                component_builder {
                                  registered_name: 'AlwaysFails'
                                }
                              })";
  MasterTrace master_trace;
  EXPECT_THAT(TryRun(kMasterSpec, /*expect_success=*/false, &master_trace),
              test::IsErrorWithSubstr("I always fail!"));

  const string kExpectedTraceText = "component_trace { name: 'component1' }";
  MasterTrace expected_trace;
  ASSERT_TRUE(TextFormat::ParseFromString(kExpectedTraceText, &expected_trace));

  EXPECT_THAT(master_trace, test::EqualsProto(expected_trace));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
