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

#include "dragnn/components/syntaxnet/syntaxnet_transition_state.h"

#include "dragnn/components/syntaxnet/syntaxnet_component.h"
#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/core/test/mock_transition_state.h"
#include "dragnn/io/sentence_input_batch.h"
#include "dragnn/protos/spec.pb.h"
#include "syntaxnet/sentence.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

// This test suite is intended to validate the contracts that the DRAGNN
// system expects from all transition state subclasses. Developers creating
// new TransitionStates should copy this test and modify it as necessary,
// using it to ensure their state conforms to DRAGNN expectations.

namespace syntaxnet {
namespace dragnn {

namespace {

const char kSentence0[] = R"(
token {
  word: "Sentence" start: 0 end: 7 tag: "NN" category: "NOUN" label: "ROOT"
  break_level: NO_BREAK
}
token {
  word: "0" start: 9 end: 9 head: 0 tag: "CD" category: "NUM" label: "num"
  break_level: SPACE_BREAK
}
token {
  word: "." start: 10 end: 10 head: 0 tag: "." category: "." label: "punct"
  break_level: NO_BREAK
}
)";

}  // namespace

using testing::Return;

class SyntaxNetTransitionStateTest : public ::testing::Test {
 public:
  std::unique_ptr<SyntaxNetTransitionState> CreateState() {
    // Get the master spec proto from the test data directory.
    MasterSpec master_spec;
    string file_name = tensorflow::io::JoinPath(
        test::GetTestDataPrefix(), "dragnn/components/syntaxnet/testdata",
        "master_spec.textproto");
    TF_CHECK_OK(tensorflow::ReadTextProto(tensorflow::Env::Default(), file_name,
                                          &master_spec));

    // Get all the resource protos from the test data directory.
    for (Resource &resource :
         *(master_spec.mutable_component(0)->mutable_resource())) {
      resource.mutable_part(0)->set_file_pattern(tensorflow::io::JoinPath(
          test::GetTestDataPrefix(), "dragnn/components/syntaxnet/testdata",
          resource.part(0).file_pattern()));
    }

    // Create an empty input batch and beam vector to initialize the parser.
    Sentence sentence_0;
    TextFormat::ParseFromString(kSentence0, &sentence_0);
    string sentence_0_str;
    sentence_0.SerializeToString(&sentence_0_str);
    data_.reset(new InputBatchCache(sentence_0_str));
    SentenceInputBatch *sentences = data_->GetAs<SentenceInputBatch>();

    // Create a parser comoponent that will generate a parser state for this
    // test.
    SyntaxNetComponent component;
    component.InitializeComponent(*(master_spec.mutable_component(0)));
    std::vector<std::vector<const TransitionState *>> states;
    constexpr int kBeamSize = 1;
    component.InitializeData(states, kBeamSize, data_.get());

    // Get a transition state from the component.
    std::unique_ptr<SyntaxNetTransitionState> test_state =
        component.CreateState(&(sentences->data()->at(0)));
    return test_state;
  }

  std::unique_ptr<InputBatchCache> data_;
};

// Validates the consistency of the beam index setter and getter.
TEST_F(SyntaxNetTransitionStateTest, CanSetAndGetBeamIndex) {
  // Create and initialize a test state.
  MockTransitionState mock_state;
  auto test_state = CreateState();
  test_state->Init(mock_state);

  constexpr int kOldBeamIndex = 12;
  test_state->SetBeamIndex(kOldBeamIndex);
  EXPECT_EQ(test_state->GetBeamIndex(), kOldBeamIndex);

  constexpr int kNewBeamIndex = 7;
  test_state->SetBeamIndex(kNewBeamIndex);
  EXPECT_EQ(test_state->GetBeamIndex(), kNewBeamIndex);
}

// Validates the consistency of the score setter and getter.
TEST_F(SyntaxNetTransitionStateTest, CanSetAndGetScore) {
  // Create and initialize a test state.
  MockTransitionState mock_state;
  auto test_state = CreateState();
  test_state->Init(mock_state);

  constexpr float kOldScore = 12.1;
  test_state->SetScore(kOldScore);
  EXPECT_EQ(test_state->GetScore(), kOldScore);

  constexpr float kNewScore = 7.2;
  test_state->SetScore(kNewScore);
  EXPECT_EQ(test_state->GetScore(), kNewScore);
}

// Validates the consistency of the goldness setter and getter.
TEST_F(SyntaxNetTransitionStateTest, CanSetAndGetGold) {
  // Create and initialize a test state.
  MockTransitionState mock_state;
  auto test_state = CreateState();
  test_state->Init(mock_state);

  constexpr bool kOldGold = true;
  test_state->SetGold(kOldGold);
  EXPECT_EQ(test_state->IsGold(), kOldGold);

  constexpr bool kNewGold = false;
  test_state->SetGold(kNewGold);
  EXPECT_EQ(test_state->IsGold(), kNewGold);
}

// This test ensures that the initializing state's current index is saved
// as the parent beam index of the state being initialized.
TEST_F(SyntaxNetTransitionStateTest, ReportsParentBeamIndex) {
  // Create a mock transition state that wil report a specific current index.
  // This index should become the parent state index for the test state.
  MockTransitionState mock_state;
  constexpr int kParentBeamIndex = 1138;
  EXPECT_CALL(mock_state, GetBeamIndex())
      .WillRepeatedly(Return(kParentBeamIndex));

  auto test_state = CreateState();
  test_state->Init(mock_state);
  EXPECT_EQ(test_state->ParentBeamIndex(), kParentBeamIndex);
}

// This test ensures that the initializing state's current score is saved
// as the current score of the state being initialized.
TEST_F(SyntaxNetTransitionStateTest, InitializationCopiesParentScore) {
  // Create a mock transition state that wil report a specific current index.
  // This index should become the parent state index for the test state.
  MockTransitionState mock_state;
  constexpr float kParentScore = 24.12;
  EXPECT_CALL(mock_state, GetScore()).WillRepeatedly(Return(kParentScore));

  auto test_state = CreateState();
  test_state->Init(mock_state);
  EXPECT_EQ(test_state->GetScore(), kParentScore);
}

// This test ensures that calling Clone maintains the state data (parent beam
// index, beam index, score, etc.) of the state that was cloned.
TEST_F(SyntaxNetTransitionStateTest, CloningMaintainsState) {
  // Create and initialize the state->
  MockTransitionState mock_state;
  constexpr int kParentBeamIndex = 1138;
  EXPECT_CALL(mock_state, GetBeamIndex())
      .WillRepeatedly(Return(kParentBeamIndex));
  auto test_state = CreateState();
  test_state->Init(mock_state);

  // Validate the internal state of the test state.
  constexpr float kOldScore = 20.0;
  test_state->SetScore(kOldScore);
  EXPECT_EQ(test_state->GetScore(), kOldScore);
  constexpr int kOldBeamIndex = 12;
  test_state->SetBeamIndex(kOldBeamIndex);
  EXPECT_EQ(test_state->GetBeamIndex(), kOldBeamIndex);

  auto clone = test_state->Clone();

  // The clone should have identical state to the old state.
  EXPECT_EQ(clone->ParentBeamIndex(), kParentBeamIndex);
  EXPECT_EQ(clone->GetScore(), kOldScore);
  EXPECT_EQ(clone->GetBeamIndex(), kOldBeamIndex);
}

// Validates the consistency of the step_for_token setter and getter.
TEST_F(SyntaxNetTransitionStateTest, CanSetAndGetStepForToken) {
  // Create and initialize a test state.
  MockTransitionState mock_state;
  auto test_state = CreateState();
  test_state->Init(mock_state);

  constexpr int kStepForTokenZero = 12;
  constexpr int kStepForTokenTwo = 34;
  test_state->set_step_for_token(0, kStepForTokenZero);
  test_state->set_step_for_token(2, kStepForTokenTwo);

  // Expect that the set tokens return values and the unset steps return the
  // default.
  constexpr int kDefaultValue = -1;
  EXPECT_EQ(kStepForTokenZero, test_state->step_for_token(0));
  EXPECT_EQ(kDefaultValue, test_state->step_for_token(1));
  EXPECT_EQ(kStepForTokenTwo, test_state->step_for_token(2));

  // Expect that out of bound accesses will return the default. (There are only
  // 3 tokens in the backing sentence, so token 3 and greater are out of bound.)
  EXPECT_EQ(kDefaultValue, test_state->step_for_token(-1));
  EXPECT_EQ(kDefaultValue, test_state->step_for_token(3));
}

// Validates the consistency of the parent_step_for_token setter and getter.
TEST_F(SyntaxNetTransitionStateTest, CanSetAndGetParentStepForToken) {
  // Create and initialize a test state.
  MockTransitionState mock_state;
  auto test_state = CreateState();
  test_state->Init(mock_state);

  constexpr int kStepForTokenZero = 12;
  constexpr int kStepForTokenTwo = 34;
  test_state->set_parent_step_for_token(0, kStepForTokenZero);
  test_state->set_parent_step_for_token(2, kStepForTokenTwo);

  // Expect that the set tokens return values and the unset steps return the
  // default.
  constexpr int kDefaultValue = -1;
  EXPECT_EQ(kStepForTokenZero, test_state->parent_step_for_token(0));
  EXPECT_EQ(kDefaultValue, test_state->parent_step_for_token(1));
  EXPECT_EQ(kStepForTokenTwo, test_state->parent_step_for_token(2));

  // Expect that out of bound accesses will return the default. (There are only
  // 3 tokens in the backing sentence, so token 3 and greater are out of bound.)
  EXPECT_EQ(kDefaultValue, test_state->parent_step_for_token(-1));
  EXPECT_EQ(kDefaultValue, test_state->parent_step_for_token(3));
}

// Validates the consistency of the parent_for_token setter and getter.
TEST_F(SyntaxNetTransitionStateTest, CanSetAndGetParentForToken) {
  // Create and initialize a test state.
  MockTransitionState mock_state;
  auto test_state = CreateState();
  test_state->Init(mock_state);

  constexpr int kParentForTokenZero = 12;
  constexpr int kParentForTokenTwo = 34;
  test_state->set_parent_for_token(0, kParentForTokenZero);
  test_state->set_parent_for_token(2, kParentForTokenTwo);

  // Expect that the set tokens return values and the unset steps return the
  // default.
  constexpr int kDefaultValue = -1;
  EXPECT_EQ(kParentForTokenZero, test_state->parent_for_token(0));
  EXPECT_EQ(kDefaultValue, test_state->parent_for_token(1));
  EXPECT_EQ(kParentForTokenTwo, test_state->parent_for_token(2));

  // Expect that out of bound accesses will return the default. (There are only
  // 3 tokens in the backing sentence, so token 3 and greater are out of bound.)
  EXPECT_EQ(kDefaultValue, test_state->parent_for_token(-1));
  EXPECT_EQ(kDefaultValue, test_state->parent_for_token(3));
}

// Validates the consistency of trace proto setter/getter.
TEST_F(SyntaxNetTransitionStateTest, CanSetAndGetTrace) {
  // Create and initialize a test state.
  MockTransitionState mock_state;
  auto test_state = CreateState();
  test_state->Init(mock_state);

  const string kTestComponentName = "test";
  std::unique_ptr<ComponentTrace> trace;
  trace.reset(new ComponentTrace());
  trace->set_name(kTestComponentName);
  test_state->set_trace(std::move(trace));

  EXPECT_EQ(trace.get(), nullptr);
  EXPECT_EQ(test_state->mutable_trace()->name(), kTestComponentName);

  // Should be preserved when cloing.
  auto cloned_state = test_state->Clone();
  EXPECT_EQ(cloned_state->mutable_trace()->name(), kTestComponentName);
  EXPECT_EQ(test_state->mutable_trace()->name(), kTestComponentName);
}

}  // namespace dragnn
}  // namespace syntaxnet
