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

#include <string>
#include <vector>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/protos/spec.pb.h"
#include "dragnn/runtime/alignment.h"
#include "dragnn/runtime/sequence_predictor.h"
#include "dragnn/runtime/test/helpers.h"
#include "dragnn/runtime/test/term_map_helpers.h"
#include "syntaxnet/base.h"
#include "syntaxnet/sentence.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

constexpr char kResourceName[] = "tag-map";

// Writes a default tag map and returns a path to it.
string GetTagMapPath() {
  static string *const kPath =
      new string(WriteTermMap({{"NOUN", 3}, {"VERB", 2}, {"DET", 1}}));
  return *kPath;
}

// Returns a ComponentSpec parsed from the |text| that contains a term map
// resource pointing at the |path|.
ComponentSpec MakeSpec(const string &text, const string &path) {
  ComponentSpec component_spec;
  CHECK(TextFormat::ParseFromString(text, &component_spec));
  AddTermMapResource(kResourceName, path, &component_spec);
  return component_spec;
}

// Returns a ComponentSpec that the predictor will support.
ComponentSpec MakeSupportedSpec() {
  return MakeSpec(R"(transition_system { registered_name: 'tagger' }
                     backend { registered_name: 'SyntaxNetComponent' }
                     num_actions: 3)",
                  GetTagMapPath());
}

// Returns per-token tag logits.
UniqueMatrix<float> MakeLogits() {
  return UniqueMatrix<float>({{0.0, 0.0, 1.0},    // predict 2 = DET
                              {1.0, 0.0, 0.0},    // predict 0 = NOUN
                              {0.0, 1.0, 0.0},    // predict 1 = VERB
                              {0.0, 0.0, 1.0},    // predict 2 = DET
                              {1.0, 0.0, 0.0}});  // predict 0 = NOUN
}

// Returns a default sentence.
Sentence MakeSentence() {
  Sentence sentence;
  for (const string &word : {"the", "cat", "chased", "a", "mouse"}) {
    Token *token = sentence.add_token();
    token->set_start(0);  // never used; set because required field
    token->set_end(0);  // never used; set because required field
    token->set_word(word);
  }
  return sentence;
}

// Tests that the predictor supports an appropriate spec.
TEST(SyntaxNetTagSequencePredictorTest, Supported) {
  const ComponentSpec component_spec = MakeSupportedSpec();

  string name;
  TF_ASSERT_OK(SequencePredictor::Select(component_spec, &name));
  EXPECT_EQ(name, "SyntaxNetTagSequencePredictor");
}

// Tests that the predictor requires the proper backend.
TEST(SyntaxNetTagSequencePredictorTest, WrongBackend) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.mutable_backend()->set_registered_name("bad");

  string name;
  EXPECT_THAT(
      SequencePredictor::Select(component_spec, &name),
      test::IsErrorWithSubstr("No SequencePredictor supports ComponentSpec"));
}

// Tests that the predictor requires the proper transition system.
TEST(SyntaxNetTagSequencePredictorTest, WrongTransitionSystem) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.mutable_transition_system()->set_registered_name("bad");

  string name;
  EXPECT_THAT(
      SequencePredictor::Select(component_spec, &name),
      test::IsErrorWithSubstr("No SequencePredictor supports ComponentSpec"));
}

// Tests that the predictor can be initialized and used to add POS tags to a
// sentence.
TEST(SyntaxNetTagSequencePredictorTest, InitializeAndPredict) {
  const ComponentSpec component_spec = MakeSupportedSpec();

  std::unique_ptr<SequencePredictor> predictor;
  TF_ASSERT_OK(SequencePredictor::New("SyntaxNetTagSequencePredictor",
                                      component_spec, &predictor));

  UniqueMatrix<float> logits = MakeLogits();
  const Sentence sentence = MakeSentence();
  InputBatchCache input(sentence.SerializeAsString());
  TF_ASSERT_OK(predictor->Predict(Matrix<float>(*logits), &input));

  const std::vector<string> predictions = input.SerializedData();
  ASSERT_EQ(predictions.size(), 1);
  Sentence tagged;
  ASSERT_TRUE(tagged.ParseFromString(predictions[0]));

  ASSERT_EQ(tagged.token_size(), 5);
  EXPECT_EQ(tagged.token(0).tag(), "DET");   // the
  EXPECT_EQ(tagged.token(1).tag(), "NOUN");  // cat
  EXPECT_EQ(tagged.token(2).tag(), "VERB");  // chased
  EXPECT_EQ(tagged.token(3).tag(), "DET");   // a
  EXPECT_EQ(tagged.token(4).tag(), "NOUN");  // mouse
}

// Tests that the predictor works on an empty sentence.
TEST(SyntaxNetTagSequencePredictorTest, EmptySentence) {
  const ComponentSpec component_spec = MakeSupportedSpec();

  std::unique_ptr<SequencePredictor> predictor;
  TF_ASSERT_OK(SequencePredictor::New("SyntaxNetTagSequencePredictor",
                                      component_spec, &predictor));

  AlignedView view;
  AlignedArea area;
  TF_ASSERT_OK(area.Reset(view, 0, 3 * sizeof(float)));
  Matrix<float> logits(area);
  const Sentence sentence;
  InputBatchCache input(sentence.SerializeAsString());
  TF_ASSERT_OK(predictor->Predict(logits, &input));

  const std::vector<string> predictions = input.SerializedData();
  ASSERT_EQ(predictions.size(), 1);
  Sentence tagged;
  ASSERT_TRUE(tagged.ParseFromString(predictions[0]));

  ASSERT_EQ(tagged.token_size(), 0);
}

// Tests that the predictor fails on an empty term map.
TEST(SyntaxNetTagSequencePredictorTest, EmptyTermMap) {
  const string path = WriteTermMap({});
  const ComponentSpec component_spec = MakeSpec("", path);

  std::unique_ptr<SequencePredictor> predictor;
  EXPECT_THAT(SequencePredictor::New("SyntaxNetTagSequencePredictor",
                                     component_spec, &predictor),
              test::IsErrorWithSubstr("Empty tag map"));
}

// Tests that Predict() fails if the batch is the wrong size.
TEST(SyntaxNetTagSequencePredictorTest, WrongBatchSize) {
  const ComponentSpec component_spec = MakeSupportedSpec();

  std::unique_ptr<SequencePredictor> predictor;
  TF_ASSERT_OK(SequencePredictor::New("SyntaxNetTagSequencePredictor",
                                      component_spec, &predictor));

  UniqueMatrix<float> logits = MakeLogits();
  const Sentence sentence = MakeSentence();
  const std::vector<string> data = {sentence.SerializeAsString(),
                                    sentence.SerializeAsString()};
  InputBatchCache input(data);
  EXPECT_THAT(predictor->Predict(Matrix<float>(*logits), &input),
              test::IsErrorWithSubstr("Non-singleton batch: got 2 elements"));
}

// Tests that Initialize() fails if the term map doesn't match the specified
// number of actions.
TEST(SyntaxNetTagSequencePredictorTest, WrongNumActions) {
  ComponentSpec component_spec = MakeSupportedSpec();
  component_spec.set_num_actions(1000);

  std::unique_ptr<SequencePredictor> predictor;
  EXPECT_THAT(
      SequencePredictor::New("SyntaxNetTagSequencePredictor", component_spec,
                             &predictor),
      test::IsErrorWithSubstr(
          "Tag count mismatch between term map (3) and ComponentSpec (1000)"));
}

// Tests that Predict() fails if the logits don't match the term map.
TEST(SyntaxNetTagSequencePredictorTest, WrongLogitsColumns) {
  const string path = WriteTermMap({{"a", 1}, {"b", 1}});
  const ComponentSpec component_spec = MakeSpec("num_actions: 2", path);

  std::unique_ptr<SequencePredictor> predictor;
  TF_ASSERT_OK(SequencePredictor::New("SyntaxNetTagSequencePredictor",
                                      component_spec, &predictor));

  UniqueMatrix<float> logits = MakeLogits();
  Sentence sentence = MakeSentence();
  InputBatchCache input(sentence.SerializeAsString());
  EXPECT_THAT(predictor->Predict(Matrix<float>(*logits), &input),
              test::IsErrorWithSubstr(
                  "Logits shape mismatch: expected 2 columns but got 3"));
}

// Tests that Predict() fails if the logits don't match the number of tokens.
TEST(SyntaxNetTagSequencePredictorTest, WrongLogitsRows) {
  const ComponentSpec component_spec = MakeSupportedSpec();

  std::unique_ptr<SequencePredictor> predictor;
  TF_ASSERT_OK(SequencePredictor::New("SyntaxNetTagSequencePredictor",
                                      component_spec, &predictor));

  UniqueMatrix<float> logits = MakeLogits();
  Sentence sentence = MakeSentence();
  sentence.mutable_token()->RemoveLast();  // bad
  InputBatchCache input(sentence.SerializeAsString());
  EXPECT_THAT(predictor->Predict(Matrix<float>(*logits), &input),
              test::IsErrorWithSubstr(
                  "Logits shape mismatch: expected 4 rows but got 5"));
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
