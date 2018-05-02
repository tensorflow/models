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

#include "dragnn/runtime/sequence_model.h"

#include <string>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/fixed_embeddings.h"
#include "dragnn/runtime/linked_embeddings.h"
#include "dragnn/runtime/math/types.h"
#include "dragnn/runtime/network_states.h"
#include "dragnn/runtime/sequence_backend.h"
#include "dragnn/runtime/sequence_extractor.h"
#include "dragnn/runtime/sequence_linker.h"
#include "dragnn/runtime/sequence_predictor.h"
#include "dragnn/runtime/session_state.h"
#include "dragnn/runtime/test/network_test_base.h"
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

using ::testing::Return;

constexpr int kNumSteps = 50;
constexpr int kVocabularySize = 123;
constexpr int kLinkedDim = 11;
constexpr int kLogitsDim = 17;
constexpr char kLogitsName[] = "oddly_named_logits";
constexpr char kPreviousComponentName[] = "previous_component";
constexpr char kPreviousLayerName[] = "previous_layer";
constexpr float kPreviousLayerValue = -1.0;

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

// Trivial linker that links each index to the previous one.
class LinkToPrevious : public SequenceLinker {
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
    links->clear();
    for (int i = 0; i < num_steps_; ++i) links->push_back(i - 1);
    return tensorflow::Status::OK();
  }

  // Sets the number of steps to emit.
  static void SetNumSteps(int num_steps) { num_steps_ = num_steps; }

 private:
  // The number of steps to produce.
  static int num_steps_;
};

int LinkToPrevious::num_steps_ = kNumSteps;

DRAGNN_RUNTIME_REGISTER_SEQUENCE_LINKER(LinkToPrevious);

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

class SequenceModelTest : public NetworkTestBase {
 protected:
  // Adds default call expectations.  Since these are added first, they can be
  // overridden by call expectations in individual tests.
  SequenceModelTest() {
    EXPECT_CALL(compute_session_, GetInputBatchCache())
        .WillRepeatedly(Return(&input_));
    EXPECT_CALL(compute_session_, GetReadiedComponent(kTestComponentName))
        .WillRepeatedly(Return(&backend_));

    // Some tests overwrite these; ensure that they are restored to the normal
    // values at the start of each test.
    EvenNumbers::SetNumSteps(kNumSteps);
    LinkToPrevious::SetNumSteps(kNumSteps);
    CaptureLogits::GetLogits() = Matrix<float>();
  }

  // Initializes the |model_| and its underlying feature managers from the
  // |component_spec|, then uses the |model_| to preprocess and predict the
  // |input_|.  Also sets each row of the logits to twice its row index.  On
  // error, returns non-OK.
  tensorflow::Status Run(ComponentSpec component_spec) {
    component_spec.set_name(kTestComponentName);

    AddComponent(kPreviousComponentName);
    AddLayer(kPreviousLayerName, kLinkedDim);
    AddComponent(kTestComponentName);
    AddLayer(kLogitsName, kLogitsDim);

    TF_RETURN_IF_ERROR(fixed_embedding_manager_.Reset(
        component_spec, &variable_store_, &network_state_manager_));
    TF_RETURN_IF_ERROR(linked_embedding_manager_.Reset(
        component_spec, &variable_store_, &network_state_manager_));

    TF_RETURN_IF_ERROR(model_.Initialize(
        component_spec, kLogitsName, &fixed_embedding_manager_,
        &linked_embedding_manager_, &network_state_manager_));

    network_states_.Reset(&network_state_manager_);
    StartComponent(kNumSteps);
    FillLayer(kPreviousComponentName, kPreviousLayerName, kPreviousLayerValue);
    StartComponent(0);

    TF_RETURN_IF_ERROR(model_.Preprocess(&session_state_, &compute_session_,
                                         &evaluate_state_));

    MutableMatrix<float> logits = GetLayer(kTestComponentName, kLogitsName);
    for (int row = 0; row < logits.num_rows(); ++row) {
      for (int column = 0; column < logits.num_columns(); ++column) {
        logits.row(row)[column] = 2.0 * row;
      }
    }

    return model_.Predict(network_states_, &evaluate_state_);
  }

  // Returns the sequence size passed to the |backend_|.
  int GetBackendSequenceSize() {
    // The sequence size is not directly exposed, but can be inferred using one
    // of the reverse step translators.
    return backend_.GetStepLookupFunction("reverse-token")(0, 0, 0) + 1;
  }

  // Fixed and linked embedding managers.
  FixedEmbeddingManager fixed_embedding_manager_;
  LinkedEmbeddingManager linked_embedding_manager_;

  // Input batch injected into Preprocess() by default.
  InputBatchCache input_;

  // Backend injected into Preprocess().
  SequenceBackend backend_;

  // Sequence-based model.
  SequenceModel model_;

  // Per-evaluation state.
  SequenceModel::EvaluateState evaluate_state_;
};

// Returns a ComponentSpec that is supported.
ComponentSpec MakeSupportedSpec() {
  ComponentSpec component_spec;
  component_spec.set_num_actions(kLogitsDim);

  component_spec.mutable_component_builder()->mutable_parameters()->insert(
      {"sequence_extractors", "EvenNumbers"});
  component_spec.mutable_component_builder()->mutable_parameters()->insert(
      {"sequence_linkers", "LinkToPrevious"});
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

// Tests that the model supports a supported spec.
TEST_F(SequenceModelTest, Supported) {
  const ComponentSpec component_spec = MakeSupportedSpec();

  EXPECT_TRUE(SequenceModel::Supports(component_spec));
}

// Tests that the model rejects a spec with the wrong backend.
TEST_F(SequenceModelTest, UnsupportedBackend) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.mutable_backend()->set_registered_name("bad");

  EXPECT_FALSE(SequenceModel::Supports(component_spec));
}

// Tests that the model rejects a spec with no features.
TEST_F(SequenceModelTest, UnsupportedNoFeatures) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.clear_fixed_feature();
  component_spec.clear_linked_feature();

  EXPECT_FALSE(SequenceModel::Supports(component_spec));
}

// Tests that the model rejects a spec with a multi-embedding fixed feature.
TEST_F(SequenceModelTest, UnsupportedMultiEmbeddingFixedFeature) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.mutable_fixed_feature(0)->set_size(2);

  EXPECT_FALSE(SequenceModel::Supports(component_spec));
}

// Tests that the model rejects a spec with a multi-embedding linked feature.
TEST_F(SequenceModelTest, UnsupportedMultiEmbeddingLinkedFeature) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.mutable_linked_feature(0)->set_size(2);

  EXPECT_FALSE(SequenceModel::Supports(component_spec));
}

// Tests that the model rejects a spec with only recurrent links.
TEST_F(SequenceModelTest, UnsupportedOnlyRecurrentLinks) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.set_name("foo");
  component_spec.clear_fixed_feature();
  component_spec.mutable_linked_feature(0)->set_source_component("foo");

  EXPECT_FALSE(SequenceModel::Supports(component_spec));
}

// Tests that Initialize() succeeds on a supported spec.
TEST_F(SequenceModelTest, InitializeSupported) {
  const ComponentSpec component_spec = MakeSupportedSpec();

  TF_ASSERT_OK(Run(component_spec));

  EXPECT_FALSE(model_.deterministic());
  EXPECT_TRUE(model_.left_to_right());
  EXPECT_EQ(model_.sequence_feature_manager().num_channels(), 1);
  EXPECT_EQ(model_.sequence_link_manager().num_channels(), 1);
}

// Tests that Initialize() detects deterministic components.
TEST_F(SequenceModelTest, InitializeDeterministic) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.set_num_actions(1);

  TF_ASSERT_OK(Run(component_spec));

  EXPECT_TRUE(model_.deterministic());
  EXPECT_TRUE(model_.left_to_right());
  EXPECT_EQ(model_.sequence_feature_manager().num_channels(), 1);
  EXPECT_EQ(model_.sequence_link_manager().num_channels(), 1);
}

// Tests that Initialize() detects right-to-left components.
TEST_F(SequenceModelTest, InitializeLeftToRight) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.mutable_transition_system()->mutable_parameters()->insert(
      {"left_to_right", "false"});

  TF_ASSERT_OK(Run(component_spec));

  EXPECT_FALSE(model_.deterministic());
  EXPECT_FALSE(model_.left_to_right());
  EXPECT_EQ(model_.sequence_feature_manager().num_channels(), 1);
  EXPECT_EQ(model_.sequence_link_manager().num_channels(), 1);
}

// Tests that Initialize() fails if the backend is wrong.
TEST_F(SequenceModelTest, WrongBackend) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.mutable_backend()->set_registered_name("bad");

  EXPECT_THAT(Run(component_spec),
              test::IsErrorWithSubstr("Invalid component backend"));
}

// Tests that Initialize() fails if the number of actions in the ComponentSpec
// does not match the logits.
TEST_F(SequenceModelTest, WrongNumActions) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.set_num_actions(kLogitsDim + 1);

  EXPECT_THAT(Run(component_spec),
              test::IsErrorWithSubstr("Logits dimension mismatch"));
}

// Tests that Initialize() fails if an unknown sequence extractor is specified.
TEST_F(SequenceModelTest, UnknownSequenceExtractor) {
  ComponentSpec component_spec = MakeSupportedSpec();
  (*component_spec.mutable_component_builder()
        ->mutable_parameters())["sequence_extractors"] = "bad";

  EXPECT_THAT(
      Run(component_spec),
      test::IsErrorWithSubstr("Unknown DRAGNN Runtime Sequence Extractor"));
}

// Tests that Initialize() fails if an unknown sequence linker is specified.
TEST_F(SequenceModelTest, UnknownSequenceLinker) {
  ComponentSpec component_spec = MakeSupportedSpec();
  (*component_spec.mutable_component_builder()
        ->mutable_parameters())["sequence_linkers"] = "bad";

  EXPECT_THAT(
      Run(component_spec),
      test::IsErrorWithSubstr("Unknown DRAGNN Runtime Sequence Linker"));
}

// Tests that Initialize() fails if an unknown sequence predictor is specified.
TEST_F(SequenceModelTest, UnknownSequencePredictor) {
  ComponentSpec component_spec = MakeSupportedSpec();
  (*component_spec.mutable_component_builder()
        ->mutable_parameters())["sequence_predictor"] = "bad";

  EXPECT_THAT(
      Run(component_spec),
      test::IsErrorWithSubstr("Unknown DRAGNN Runtime Sequence Predictor"));
}

// Tests that Initialize() fails on an unknown component builder parameter.
TEST_F(SequenceModelTest, UnknownComponentBuilderParameter) {
  ComponentSpec component_spec = MakeSupportedSpec();
  (*component_spec.mutable_component_builder()->mutable_parameters())["bad"] =
      "bad";

  EXPECT_THAT(Run(component_spec),
              test::IsErrorWithSubstr("Unknown attribute"));
}

// Tests that Initialize() fails if there are no fixed or linked features.
TEST_F(SequenceModelTest, InitializeRequiresFeatures) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.clear_fixed_feature();
  component_spec.clear_linked_feature();
  (*component_spec.mutable_component_builder()
        ->mutable_parameters())["sequence_extractors"] = "";
  (*component_spec.mutable_component_builder()
        ->mutable_parameters())["sequence_linkers"] = "";

  EXPECT_THAT(Run(component_spec),
              test::IsErrorWithSubstr("No fixed or linked features"));
}

// Tests that the model fails if a null batch is returned.
TEST_F(SequenceModelTest, NullBatch) {
  EXPECT_CALL(compute_session_, GetInputBatchCache()).WillOnce(Return(nullptr));

  EXPECT_THAT(Run(MakeSupportedSpec()),
              test::IsErrorWithSubstr("Null input batch"));
}

// Tests that the model properly sets up the EvaluateState and logits.
TEST_F(SequenceModelTest, Success) {
  TF_ASSERT_OK(Run(MakeSupportedSpec()));

  EXPECT_EQ(GetBackendSequenceSize(), kNumSteps);
  EXPECT_EQ(evaluate_state_.num_steps, kNumSteps);
  EXPECT_EQ(evaluate_state_.input, &input_);

  EXPECT_EQ(evaluate_state_.features.num_channels(), 1);
  EXPECT_EQ(evaluate_state_.features.num_steps(), kNumSteps);

  EXPECT_EQ(evaluate_state_.features.GetId(0, 0), 0);
  EXPECT_EQ(evaluate_state_.features.GetId(0, 1), 2);
  EXPECT_EQ(evaluate_state_.features.GetId(0, 2), 4);

  EXPECT_EQ(evaluate_state_.links.num_channels(), 1);
  EXPECT_EQ(evaluate_state_.links.num_steps(), kNumSteps);

  Vector<float> embedding;
  bool is_out_of_bounds = false;
  evaluate_state_.links.Get(0, 0, &embedding, &is_out_of_bounds);
  ExpectVector(embedding, kLinkedDim, 0.0);
  EXPECT_TRUE(is_out_of_bounds);
  evaluate_state_.links.Get(0, 1, &embedding, &is_out_of_bounds);
  ExpectVector(embedding, kLinkedDim, kPreviousLayerValue);
  EXPECT_FALSE(is_out_of_bounds);

  const Matrix<float> logits = CaptureLogits::GetLogits();
  ASSERT_EQ(logits.num_rows(), kNumSteps);
  ASSERT_EQ(logits.num_columns(), kLogitsDim);
  for (int i = 0; i < kNumSteps; ++i) {
    ExpectVector(logits.row(i), kLogitsDim, 2.0 * i);
  }
}

// Tests that the model works with only fixed features.
TEST_F(SequenceModelTest, FixedFeaturesOnly) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.clear_linked_feature();
  (*component_spec.mutable_component_builder()
        ->mutable_parameters())["sequence_linkers"] = "";

  TF_ASSERT_OK(Run(component_spec));

  EXPECT_EQ(GetBackendSequenceSize(), kNumSteps);
  EXPECT_EQ(evaluate_state_.num_steps, kNumSteps);
  EXPECT_EQ(evaluate_state_.input, &input_);

  EXPECT_EQ(evaluate_state_.features.num_channels(), 1);
  EXPECT_EQ(evaluate_state_.features.num_steps(), kNumSteps);

  EXPECT_EQ(evaluate_state_.features.GetId(0, 0), 0);
  EXPECT_EQ(evaluate_state_.features.GetId(0, 1), 2);
  EXPECT_EQ(evaluate_state_.features.GetId(0, 2), 4);

  EXPECT_EQ(evaluate_state_.links.num_channels(), 0);
  EXPECT_EQ(evaluate_state_.links.num_steps(), 0);

  const Matrix<float> logits = CaptureLogits::GetLogits();
  ASSERT_EQ(logits.num_rows(), kNumSteps);
  ASSERT_EQ(logits.num_columns(), kLogitsDim);
  for (int i = 0; i < kNumSteps; ++i) {
    ExpectVector(logits.row(i), kLogitsDim, 2.0 * i);
  }
}

// Tests that the model works with only linked features.
TEST_F(SequenceModelTest, LinkedFeaturesOnly) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.clear_fixed_feature();
  (*component_spec.mutable_component_builder()
        ->mutable_parameters())["sequence_extractors"] = "";

  TF_ASSERT_OK(Run(component_spec));

  EXPECT_EQ(GetBackendSequenceSize(), kNumSteps);
  EXPECT_EQ(evaluate_state_.num_steps, kNumSteps);
  EXPECT_EQ(evaluate_state_.input, &input_);

  EXPECT_EQ(evaluate_state_.features.num_channels(), 0);
  EXPECT_EQ(evaluate_state_.features.num_steps(), 0);

  EXPECT_EQ(evaluate_state_.links.num_channels(), 1);
  EXPECT_EQ(evaluate_state_.links.num_steps(), kNumSteps);

  Vector<float> embedding;
  bool is_out_of_bounds = false;
  evaluate_state_.links.Get(0, 0, &embedding, &is_out_of_bounds);
  ExpectVector(embedding, kLinkedDim, 0.0);
  EXPECT_TRUE(is_out_of_bounds);
  evaluate_state_.links.Get(0, 1, &embedding, &is_out_of_bounds);
  ExpectVector(embedding, kLinkedDim, kPreviousLayerValue);
  EXPECT_FALSE(is_out_of_bounds);

  const Matrix<float> logits = CaptureLogits::GetLogits();
  ASSERT_EQ(logits.num_rows(), kNumSteps);
  ASSERT_EQ(logits.num_columns(), kLogitsDim);
  for (int i = 0; i < kNumSteps; ++i) {
    ExpectVector(logits.row(i), kLogitsDim, 2.0 * i);
  }
}

// Tests that the model fails if the fixed and linked features disagree on the
// number of steps.
TEST_F(SequenceModelTest, FixedAndLinkedDisagree) {
  EvenNumbers::SetNumSteps(5);
  LinkToPrevious::SetNumSteps(6);

  EXPECT_THAT(Run(MakeSupportedSpec()),
              test::IsErrorWithSubstr("Sequence length mismatch between fixed "
                                      "features (5) and linked features (6)"));
}

// Tests that the model can handle an empty sequence.
TEST_F(SequenceModelTest, EmptySequence) {
  EvenNumbers::SetNumSteps(0);
  LinkToPrevious::SetNumSteps(0);

  TF_ASSERT_OK(Run(MakeSupportedSpec()));

  EXPECT_EQ(GetBackendSequenceSize(), 0);

  const Matrix<float> logits = CaptureLogits::GetLogits();
  ASSERT_EQ(logits.num_rows(), 0);
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
