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
#include "dragnn/protos/cell_trace.pb.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/myelin/myelin_spec_utils.h"
#include "dragnn/runtime/test/network_test_base.h"
#include "syntaxnet/base.h"
#include "sling/file/file.h"
#include "sling/myelin/flow.h"
#include <gmock/gmock.h>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

using ::testing::_;
using ::testing::InSequence;
using ::testing::Invoke;
using ::testing::Return;

constexpr int kFlowVersion = 4;
constexpr int kVocabularySize = 123;
constexpr int kLogitsDim = 11;
constexpr int kNumSteps = 50;

class MyelinDynamicComponentTest : public NetworkTestBase {
 protected:
  // Options for building a Flow file for tests.  By default, this specifies a
  // working Flow file, but settings can be perturbed to trigger errors.
  struct FlowFileOptions {
    FlowFileOptions() = default;

    // Name of the function to create.
    string function_name = kTestComponentName;

    // Dimension of the classification logits.
    int logits_dim = kLogitsDim;

    // Name of the variable containing the classification logits.
    string logits_name = "logits";

    // Type of the feature ID input.
    sling::myelin::Type id_type = sling::myelin::DT_INT32;

    // Dimension of the feature ID input.
    int id_dim = 1;
  };

  // Builds and writes a simple Flow file.  By default it produces a valid Flow,
  // but arguments can be overridden for error testing.  Returns the path to the
  // Flow file.
  static string WriteFlowFile() { return WriteFlowFile(FlowFileOptions()); }
  static string WriteFlowFile(const FlowFileOptions &options) {
    sling::myelin::Flow flow;

    // A fixed feature ID input.
    sling::myelin::Flow::Variable *id =
        flow.AddVariable("id", options.id_type, {options.id_dim});
    id->ref = true;
    id->aliases.push_back(MakeMyelinInputFixedFeatureIdName(0, 0));

    // An embedding matrix constant.  Each embedding is filled with its index.
    sling::myelin::Flow::Variable *embeddings =
        flow.AddVariable("embeddings", sling::myelin::DT_FLOAT,
                         {kVocabularySize, options.logits_dim});
    std::vector<float> data(kVocabularySize * options.logits_dim);
    for (int row = 0; row < kVocabularySize; ++row) {
      for (int column = 0; column < options.logits_dim; ++column) {
        data[row * options.logits_dim + column] = row;
      }
    }
    embeddings->SetData(data.data(), data.size() * sizeof(float));

    // The retrieved embedding row, as logits.
    sling::myelin::Flow::Variable *logits =
        flow.AddVariable(options.logits_name, sling::myelin::DT_FLOAT,
                         {options.id_dim, options.logits_dim});
    logits->ref = true;
    logits->aliases.push_back(MakeMyelinOutputLayerName(options.logits_name));

    // A Gather op that looks up the |id| in the |embeddings|, and returns the
    // result in the |logits|.
    flow.AddOperation(flow.AddFunction(options.function_name), "gather",
                      "Gather", {embeddings, id}, {logits});

    const string flow_path =
        tensorflow::io::JoinPath(tensorflow::testing::TmpDir(), "foo.flow");
    sling::File::Init();
    flow.Save(flow_path, kFlowVersion);
    return flow_path;
  }

  // Creates a component, initializes it based on the |component_spec_text| and
  // |flow_path|, and evaluates it.  The |component_trace| is overwritten with
  // traces, if non-null.  On error, returns non-OK.
  tensorflow::Status Run(const string &component_spec_text = "",
                         const string &flow_path = WriteFlowFile(),
                         ComponentTrace *component_trace = nullptr) {
    ComponentSpec component_spec;
    CHECK(TextFormat::ParseFromString(component_spec_text, &component_spec));
    if (!component_spec.has_num_actions()) {
      component_spec.set_num_actions(kLogitsDim);
    }
    component_spec.set_name(kTestComponentName);

    auto *fixed_feature = component_spec.add_fixed_feature();
    fixed_feature->set_embedding_dim(-1);
    fixed_feature->set_size(1);

    TF_RETURN_IF_ERROR(AddMyelinFlowResource(flow_path, &component_spec));

    AddComponent(kTestComponentName);
    TF_RETURN_IF_ERROR(
        Component::CreateOrError("MyelinDynamicComponent", &component_));
    TF_RETURN_IF_ERROR(component_->Initialize(component_spec, &variable_store_,
                                              &network_state_manager_,
                                              &extension_manager_));

    network_states_.Reset(&network_state_manager_);
    StartComponent(0);  // MyelinDynamicComponent will add steps
    session_state_.extensions.Reset(&extension_manager_);

    TF_RETURN_IF_ERROR(component_->Evaluate(&session_state_, &compute_session_,
                                            component_trace));
    return tensorflow::Status::OK();
  }

  std::unique_ptr<Component> component_;
};

// Tests that MyelinDynamicComponent fails if the spec uses attention.
TEST_F(MyelinDynamicComponentTest, UnsupportedAttention) {
  EXPECT_THAT(Run("attention_component:'foo'"),
              test::IsErrorWithSubstr("Attention is not supported"));
}

// Tests that MyelinDynamicComponent fails if the spec has embedded fixed
// features.
TEST_F(MyelinDynamicComponentTest, InvalidFixedFeatureIsEmbedded) {
  EXPECT_THAT(
      Run("fixed_feature { embedding_dim:1 }"),
      test::IsErrorWithSubstr("Myelin requires non-embedded fixed features"));
}

// Tests that MyelinDynamicComponent fails if the ComponentSpec has a fixed
// feature that does not appear in the Flow.
TEST_F(MyelinDynamicComponentTest, InvalidFixedFeatureNotInFlow) {
  EXPECT_THAT(Run("fixed_feature { embedding_dim:-1 size:1 }"),
              test::IsErrorWithSubstr(tensorflow::strings::StrCat(
                  "No Myelin tensor named '",
                  MakeMyelinInputFixedFeatureIdName(1, 0), "'")));
}

// Tests that MyelinDynamicComponent fails if the spec has multipled linked
// features.
TEST_F(MyelinDynamicComponentTest, InvalidLinkedFeatureIsMultiplied) {
  EXPECT_THAT(Run("linked_feature { embedding_dim:1 }"),
              test::IsErrorWithSubstr(
                  "Myelin requires non-multiplied linked features"));
}

// Tests that MyelinDynamicComponent fails if the ComponentSpec has a linked
// feature that does not appear in the Flow.
TEST_F(MyelinDynamicComponentTest, InvalidLinkedFeatureNotInFlow) {
  const string kSpec = tensorflow::strings::StrCat(
      "linked_feature { source_component:'", kTestComponentName,
      "' source_layer:'logits' embedding_dim:-1 size:1 }");

  EXPECT_THAT(Run(kSpec),
              test::IsErrorWithSubstr(tensorflow::strings::StrCat(
                  "No Myelin tensor named '",
                  MakeMyelinInputLinkedActivationVectorName(0), "'")));
}

// Tests that MyelinDynamicComponent fails if the Flow file does not exist.
TEST_F(MyelinDynamicComponentTest, InvalidFlowFilePath) {
  EXPECT_THAT(Run("", "/invalid/path"),
              test::IsErrorWithSubstr("Failed to load Myelin Flow"));
}

// Tests that MyelinDynamicComponent fails if the function in the Flow file has
// the wrong name.
TEST_F(MyelinDynamicComponentTest, WrongFunctionName) {
  FlowFileOptions options;
  options.function_name = "wrong_function";

  EXPECT_THAT(
      Run("", WriteFlowFile(options)),
      test::IsErrorWithSubstr(tensorflow::strings::StrCat(
          "No function named '", kTestComponentName, "' in Myelin network")));
}

// Tests that MyelinDynamicComponent fails if the logits dimension does not
// match ComponentSpec.num_actions.
TEST_F(MyelinDynamicComponentTest, WrongLogitsDimension) {
  FlowFileOptions options;
  options.logits_dim = kLogitsDim + 1;

  EXPECT_THAT(Run("", WriteFlowFile(options)),
              test::IsErrorWithSubstr(
                  "Dimension mismatch between classification logits"));
}

// Tests that MyelinDynamicComponent fails if there is no "logits" layer.
TEST_F(MyelinDynamicComponentTest, WrongLogitsName) {
  FlowFileOptions options;
  options.logits_name = "not_logits";

  EXPECT_THAT(Run("", WriteFlowFile(options)),
              test::IsErrorWithSubstr("Unknown layer 'logits'"));
}

// Tests that MyelinDynamicComponent fails to compile if one of the Myelin
// tensors has the wrong type.
TEST_F(MyelinDynamicComponentTest, FailToCompile) {
  FlowFileOptions options;
  options.id_type = sling::myelin::DT_FLOAT;

  EXPECT_THAT(Run("", WriteFlowFile(options)),
              test::IsErrorWithSubstr("Failed to compile Myelin network"));
}

// Tests that MyelinDynamicComponent fails if one of the Myelin tensors is not
// vector-like.
TEST_F(MyelinDynamicComponentTest, NotVectorLike) {
  FlowFileOptions options;
  options.id_dim = 2;

  EXPECT_THAT(
      Run("", WriteFlowFile(options)),
      test::IsErrorWithSubstr("Myelin tensor has non-vector-like shape"));
}

// Tests that MyelinDynamicComponent fails if AdvanceFromPrediction() fails.
TEST_F(MyelinDynamicComponentTest, FailToAdvanceFromPrediction) {
  EXPECT_CALL(compute_session_, IsTerminal(_)).WillRepeatedly(Return(false));
  EXPECT_CALL(compute_session_, AdvanceFromPrediction(_, _, _, _))
      .WillOnce(Return(false));
  EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
      .WillOnce(Invoke(ExtractFeatures(0, {{10, 1.0}})));

  EXPECT_THAT(Run(), test::IsErrorWithSubstr(
                         "Error in ComputeSession::AdvanceFromPrediction()"));
}

// Tests that MyelinDynamicComponent can run a simple non-deterministic Flow.
TEST_F(MyelinDynamicComponentTest, SimpleNonDeterministicFlow) {
  SetupTransitionLoop(kNumSteps);
  EXPECT_CALL(compute_session_, AdvanceFromPrediction(_, _, _, _))
      .Times(kNumSteps)
      .WillRepeatedly(Return(true));

  {  // Extract a sequence of feature IDs equal to 2 * step_index.
    ASSERT_LE(2 * kNumSteps, kVocabularySize);
    InSequence scoped;
    for (int step_index = 0; step_index < kNumSteps; ++step_index) {
      EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
          .WillOnce(Invoke(ExtractFeatures(0, {{2 * step_index, 1.0}})));
    }
  }

  TF_ASSERT_OK(Run());

  const Matrix<float> logits(GetLayer(kTestComponentName, "logits"));
  ASSERT_EQ(logits.num_rows(), kNumSteps);
  ASSERT_EQ(logits.num_columns(), kLogitsDim);

  // Since each row of the embedding matrix is filled with its index, the logits
  // should be equal to the feature IDs.
  for (int step_index = 0; step_index < kNumSteps; ++step_index) {
    ExpectVector(logits.row(step_index), kLogitsDim, 2 * step_index);
  }
}

// Tests that MyelinDynamicComponent can run a simple deterministic Flow.
TEST_F(MyelinDynamicComponentTest, SimpleDeterministicFlow) {
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

  FlowFileOptions options;
  options.logits_dim = 1;
  TF_ASSERT_OK(Run("num_actions:1", WriteFlowFile(options)));
}

// Tests that MyelinDynamicComponent can run a simple Flow with tracing enabled.
TEST_F(MyelinDynamicComponentTest, SimpleFlowWithTracing) {
  SetupTransitionLoop(kNumSteps);
  EXPECT_CALL(compute_session_, AdvanceFromPrediction(_, _, _, _))
      .Times(kNumSteps)
      .WillRepeatedly(Return(true));

  { // Extract a sequence of feature IDs equal to 2 * step_index.
    ASSERT_LE(2 * kNumSteps, kVocabularySize);
    InSequence scoped;
    for (int step_index = 0; step_index < kNumSteps; ++step_index) {
      EXPECT_CALL(compute_session_, GetInputFeatures(_, _, _, _, _))
          .WillOnce(Invoke(ExtractFeatures(0, {{2 * step_index, 1.0}})));
    }
  }

  ComponentTrace component_trace;
  TF_ASSERT_OK(Run("", WriteFlowFile(), &component_trace));

  // Each step trace should have a cell trace from the Myelin instance.
  ASSERT_EQ(component_trace.step_trace_size(), kNumSteps);
  for (const ComponentStepTrace &step_trace : component_trace.step_trace()) {
    ASSERT_EQ(step_trace.ExtensionSize(CellTrace::step_trace_extension), 1);
    const CellTrace &cell_trace =
        step_trace.GetExtension(CellTrace::step_trace_extension, 0);
    EXPECT_EQ(cell_trace.name(), kTestComponentName);
  }
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
