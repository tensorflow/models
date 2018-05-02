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
#include <vector>

#include "dragnn/core/compute_session.h"
#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/protos/trace.pb.h"
#include "dragnn/runtime/component.h"
#include "dragnn/runtime/extensions.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/myelin/myelin_spec_utils.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/sequence_backend.h"
#include "dragnn/runtime/sequence_extractor.h"
#include "dragnn/runtime/sequence_linker.h"
#include "dragnn/runtime/sequence_predictor.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/test/network_test_base.h"
#include "dragnn/runtime/variable_store.h"
#include "syntaxnet/base.h"
#include <gmock/gmock.h>
#include "sling/file/file.h"
#include "sling/myelin/flow.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

using ::testing::Return;

constexpr int kFlowVersion = 4;
constexpr int kNumSteps = 50;
constexpr int kVocabularySize = 123;
constexpr int kFixedDim = 6;
constexpr int kLinkedDim = 4;
constexpr int kLogitsDim = kFixedDim + kLinkedDim;
constexpr char kLogitsName[] = "logits";
constexpr char kPreviousComponentName[] = "previous_component";
constexpr char kPreviousLayerName[] = "previous_layer";
constexpr float kPreviousLayerValue = -1.0;

// Builds and writes a simple Flow file with a function named |function_name|
// that gathers the rows of a matrix, concatenates that with a linked embedding,
// and outputs the result as the classification logits.  Each row is filled with
// its index, so we can infer which indices were gathered.
string WriteFlowFile(const string &function_name) {
  sling::myelin::Flow flow;

  // A fixed feature ID input.
  sling::myelin::Flow::Variable *id =
      flow.AddVariable("id", sling::myelin::DT_INT32, {1});
  id->ref = true;
  id->aliases.push_back(MakeMyelinInputFixedFeatureIdName(0, 0));

  // A linked feature embedding input.
  sling::myelin::Flow::Variable *link =
      flow.AddVariable("link", sling::myelin::DT_FLOAT, {1, kLinkedDim});
  link->ref = true;
  link->aliases.push_back(MakeMyelinInputLinkedActivationVectorName(0));

  // An embedding matrix constant.  Each embedding is filled with its index.
  sling::myelin::Flow::Variable *embeddings = flow.AddVariable(
      "embeddings", sling::myelin::DT_FLOAT, {kVocabularySize, kFixedDim});
  std::vector<float> data(kVocabularySize * kLogitsDim);
  for (int row = 0; row < kVocabularySize; ++row) {
    for (int column = 0; column < kFixedDim; ++column) {
      data[row * kFixedDim + column] = row;
    }
  }
  embeddings->SetData(data.data(), data.size() * sizeof(float));

  // The retrieved embedding row.
  sling::myelin::Flow::Variable *row =
      flow.AddVariable("row", sling::myelin::DT_FLOAT, {1, kFixedDim});

  // A concatenation axis constant.
  sling::myelin::Flow::Variable *axis =
      flow.AddVariable("axis", sling::myelin::DT_INT32, {1});
  const int32 axis_value = 1;
  axis->SetData(&axis_value, sizeof(int32));

  // The classification logits output.
  sling::myelin::Flow::Variable *logits =
      flow.AddVariable(kLogitsName, sling::myelin::DT_FLOAT, {1, kLogitsDim});
  logits->ref = true;
  logits->aliases.push_back(MakeMyelinOutputLayerName(kLogitsName));

  // Function that contains the ops and variables.
  sling::myelin::Flow::Function *function = flow.AddFunction(function_name);

  // A Gather op that looks up the |id| in the |embeddings|, and returns the
  // result in the |row|.
  flow.AddOperation(function, "gather", "Gather", {embeddings, id}, {row});

  // A Concat op that concatenates the |row| and |link| along the |axis|,
  // placing the result in the |logits| output.
  flow.AddOperation(function, "concat", "ConcatV2", {row, link, axis},
                    {logits});

  const string flow_path =
      tensorflow::io::JoinPath(tensorflow::testing::TmpDir(), "foo.flow");
  sling::File::Init();
  flow.Save(flow_path, kFlowVersion);
  return flow_path;
}

// Sequence extractor that extracts [0, 2, 4, ...].
class EvenNumbers : public SequenceExtractor {
 public:
  // Implements SequenceExtractor.
  bool Supports(const FixedFeatureChannel &,
                const ComponentSpec &) const override {
    return true;
  }
  tensorflow::Status Initialize(const FixedFeatureChannel &,
                                const ComponentSpec &) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status GetIds(InputBatchCache *,
                            std::vector<int32> *ids) const override {
    ids->clear();
    for (int i = 0; i < num_steps_; ++i) ids->push_back(2 * i);
    return tensorflow::Status::OK();
  }

  // Sets the number of steps to emit.
  static void SetNumSteps(int num_steps) { num_steps_ = num_steps; }

 private:
  // The number of steps to produce.
  static int num_steps_;
};

int EvenNumbers::num_steps_ = kNumSteps;

DRAGNN_RUNTIME_REGISTER_SEQUENCE_EXTRACTOR(EvenNumbers);

// Component that supports a particular component name and is not preferred.
// Used to exercise PreferredTo().
class NotPreferred : public Component {
 public:
  // Implements Component.
  tensorflow::Status Initialize(const ComponentSpec &, VariableStore *,
                                NetworkStateManager *,
                                ExtensionManager *) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status Evaluate(SessionState *, ComputeSession *,
                              ComponentTrace *) const override {
    return tensorflow::Status::OK();
  }
  bool Supports(const ComponentSpec &spec, const string &) const override {
    return spec.name() == "InSupportsConflictTest";
  }
  bool PreferredTo(const Component &) const override { return false; }
};

DRAGNN_RUNTIME_REGISTER_COMPONENT(NotPreferred);

// Trivial linker that links everything to step 0.
class LinkToZero : public SequenceLinker {
 public:
  // Implements SequenceLinker.
  bool Supports(const LinkedFeatureChannel &,
                const ComponentSpec &) const override {
    return true;
  }
  tensorflow::Status Initialize(const LinkedFeatureChannel &,
                                const ComponentSpec &) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status GetLinks(size_t, InputBatchCache *,
                              std::vector<int32> *links) const override {
    links->assign(num_steps_, 0);
    return tensorflow::Status::OK();
  }

  // Sets the number of steps to emit.
  static void SetNumSteps(int num_steps) { num_steps_ = num_steps; }

 private:
  // The number of steps to produce.
  static int num_steps_;
};

int LinkToZero::num_steps_ = kNumSteps;

DRAGNN_RUNTIME_REGISTER_SEQUENCE_LINKER(LinkToZero);

// Trivial predictor that captures the prediction logits.
class CaptureLogits : public SequencePredictor {
 public:
  // Implements SequenceLinker.
  bool Supports(const ComponentSpec &) const override { return true; }
  tensorflow::Status Initialize(const ComponentSpec &) override {
    return tensorflow::Status::OK();
  }
  tensorflow::Status Predict(Matrix<float> logits,
                             InputBatchCache *) const override {
    GetLogits() = logits;
    return tensorflow::Status::OK();
  }

  // Returns the captured logits.
  static Matrix<float> &GetLogits() {
    static auto *logits = new Matrix<float>();
    return *logits;
  }
};

DRAGNN_RUNTIME_REGISTER_SEQUENCE_PREDICTOR(CaptureLogits);

class SequenceMyelinDynamicComponentTest : public NetworkTestBase {
 protected:
  // Adds default call expectations.  Since these are added first, they can be
  // overridden by call expectations in individual tests.
  SequenceMyelinDynamicComponentTest() {
    EXPECT_CALL(compute_session_, GetInputBatchCache())
        .WillRepeatedly(Return(&input_));
    EXPECT_CALL(compute_session_, GetReadiedComponent(kTestComponentName))
        .WillRepeatedly(Return(&backend_));
    TF_CHECK_OK(Component::CreateOrError("SequenceMyelinDynamicComponent",
                                         &component_));

    // Some tests overwrite these; ensure that they are restored to the normal
    // values at the start of each test.
    EvenNumbers::SetNumSteps(kNumSteps);
    LinkToZero::SetNumSteps(kNumSteps);
    CaptureLogits::GetLogits() = Matrix<float>();
  }

  // Build and write the flow file once.
  static void SetUpTestCase() {
    flow_path_ = new string(WriteFlowFile(kTestComponentName));
  }

  // Cleans up the flow file path.
  static void TearDownTestCase() {
    delete flow_path_;
    flow_path_ = nullptr;
  }

  // Creates a component, initializes it based on the |component_spec|, and
  // evaluates it.  On error, returns non-OK.
  tensorflow::Status Run(ComponentSpec component_spec) {
    component_spec.set_name(kTestComponentName);
    TF_RETURN_IF_ERROR(AddMyelinFlowResource(*flow_path_, &component_spec));

    AddComponent(kPreviousComponentName);
    AddLayer(kPreviousLayerName, kLinkedDim);
    AddComponent(kTestComponentName);
    TF_RETURN_IF_ERROR(component_->Initialize(component_spec, &variable_store_,
                                              &network_state_manager_,
                                              &extension_manager_));

    network_states_.Reset(&network_state_manager_);
    session_state_.extensions.Reset(&extension_manager_);
    StartComponent(kNumSteps);
    FillLayer(kPreviousComponentName, kPreviousLayerName, kPreviousLayerValue);
    StartComponent(0);
    TF_RETURN_IF_ERROR(
        component_->Evaluate(&session_state_, &compute_session_, nullptr));

    return tensorflow::Status::OK();
  }

  // Returns the sequence size passed to the |backend_|.
  int GetBackendSequenceSize() {
    // The sequence size is not directly exposed, but can be inferred using one
    // of the reverse step translators.
    return backend_.GetStepLookupFunction("reverse-token")(0, 0, 0) + 1;
  }

  // Path to a simple Myelin Flow file.
  static const string *flow_path_;

  // Component used in the test.
  std::unique_ptr<Component> component_;

  // Input batch injected into Evaluate() by default.
  InputBatchCache input_;

  // Backend injected into Evaluate().
  SequenceBackend backend_;
};

const string *SequenceMyelinDynamicComponentTest::flow_path_ = nullptr;

// Returns a ComponentSpec that is supported.
ComponentSpec MakeSupportedSpec() {
  ComponentSpec component_spec;
  component_spec.set_num_actions(kLogitsDim);

  component_spec.mutable_component_builder()->set_registered_name(
      "SequenceMyelinDynamicComponent");
  component_spec.mutable_component_builder()->mutable_parameters()->insert(
      {"sequence_extractors", "EvenNumbers"});
  component_spec.mutable_component_builder()->mutable_parameters()->insert(
      {"sequence_linkers", "LinkToZero"});
  component_spec.mutable_component_builder()->mutable_parameters()->insert(
      {"sequence_predictor", "CaptureLogits"});

  component_spec.mutable_backend()->set_registered_name("SequenceBackend");

  FixedFeatureChannel *fixed_feature = component_spec.add_fixed_feature();
  fixed_feature->set_size(1);
  fixed_feature->set_embedding_dim(-1);

  LinkedFeatureChannel *linked_feature = component_spec.add_linked_feature();
  linked_feature->set_source_component(kPreviousComponentName);
  linked_feature->set_source_layer(kPreviousLayerName);
  linked_feature->set_size(1);
  linked_feature->set_embedding_dim(-1);

  return component_spec;
}

// Tests that the component supports a supported spec.
TEST_F(SequenceMyelinDynamicComponentTest, Supported) {
  string component_type;

  const ComponentSpec component_spec = MakeSupportedSpec();
  TF_ASSERT_OK(Component::Select(component_spec, &component_type));
}

// Tests that the component does not support a spec with the wrong component
// builder.
TEST_F(SequenceMyelinDynamicComponentTest, UnsupportedComponentBuilder) {
  string component_type;

  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.mutable_component_builder()->set_registered_name("bad");
  EXPECT_THAT(Component::Select(component_spec, &component_type),
              test::IsErrorWithSubstr("Could not find a best"));
}

// Tests that the component
TEST_F(SequenceMyelinDynamicComponentTest, SupportsConflict) {
  string component_type;

  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.set_name("InSupportsConflictTest");  // see NotPreferred
  EXPECT_THAT(
      Component::Select(component_spec, &component_type),
      test::IsErrorWithSubstr("both think they should be dis-preferred"));
}

// Asserts that the vector starts with |kFixedDim| copies of |value| and ends
// with |kLinkedDim| copies of |kPreviousLayerValue|.
void AssertOutputRow(Vector<float> row, float value) {
  ASSERT_EQ(row.size(), kLogitsDim);
  for (int i = 0; i < row.size(); ++i) {
    if (i < kFixedDim) {
      ASSERT_EQ(row[i], value);
    } else {
      ASSERT_EQ(row[i], kPreviousLayerValue);
    }
  }
}

// Tests that the component extracts a left-to-right sequence by default.
TEST_F(SequenceMyelinDynamicComponentTest, LeftToRightByDefault) {
  TF_ASSERT_OK(Run(MakeSupportedSpec()));

  EXPECT_EQ(GetBackendSequenceSize(), kNumSteps);

  const Matrix<float> logits = CaptureLogits::GetLogits();
  ASSERT_EQ(logits.num_rows(), kNumSteps);
  ASSERT_EQ(logits.num_columns(), kLogitsDim);
  for (int i = 0; i < kNumSteps; ++i) {
    AssertOutputRow(logits.row(i), 2.0 * i);
  }
}

// Tests that the component can be explicitly configured for a left-to-right
// sequence.
TEST_F(SequenceMyelinDynamicComponentTest, LeftToRightExplicitly) {
  ComponentSpec component_spec = MakeSupportedSpec();
  (*component_spec.mutable_transition_system()
        ->mutable_parameters())["left_to_right"] = "true";

  TF_ASSERT_OK(Run(component_spec));

  EXPECT_EQ(GetBackendSequenceSize(), kNumSteps);

  const Matrix<float> logits = CaptureLogits::GetLogits();
  ASSERT_EQ(logits.num_rows(), kNumSteps);
  ASSERT_EQ(logits.num_columns(), kLogitsDim);
  for (int i = 0; i < kNumSteps; ++i) {
    AssertOutputRow(logits.row(i), 2.0 * i);
  }
}

// Tests that the component can be explicitly configured for a right-to-left
// sequence.
TEST_F(SequenceMyelinDynamicComponentTest, RightToLeft) {
  ComponentSpec component_spec = MakeSupportedSpec();
  (*component_spec.mutable_transition_system()
        ->mutable_parameters())["left_to_right"] = "false";

  TF_ASSERT_OK(Run(component_spec));

  EXPECT_EQ(GetBackendSequenceSize(), kNumSteps);

  const Matrix<float> logits = CaptureLogits::GetLogits();
  ASSERT_EQ(logits.num_rows(), kNumSteps);
  ASSERT_EQ(logits.num_columns(), kLogitsDim);
  for (int i = 0; i < kNumSteps; ++i) {
    const int reversed = kNumSteps - i - 1;
    AssertOutputRow(logits.row(i), 2.0 * reversed);
  }
}

// Tests that the component can handle an empty sequence.
TEST_F(SequenceMyelinDynamicComponentTest, EmptySequence) {
  EvenNumbers::SetNumSteps(0);
  LinkToZero::SetNumSteps(0);

  TF_ASSERT_OK(Run(MakeSupportedSpec()));

  EXPECT_EQ(GetBackendSequenceSize(), 0);

  const Matrix<float> logits = CaptureLogits::GetLogits();
  ASSERT_EQ(logits.num_rows(), 0);
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
