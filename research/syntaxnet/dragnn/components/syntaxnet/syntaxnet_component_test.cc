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

#include "dragnn/components/syntaxnet/syntaxnet_component.h"

#include <limits>

#include "dragnn/core/input_batch_cache.h"
#include "dragnn/core/test/generic.h"
#include "dragnn/core/test/mock_transition_state.h"
#include "dragnn/io/sentence_input_batch.h"
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
text: "Sentence 0."
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

const char kSentence1[] = R"(
text: "Sentence 1."
token {
  word: "Sentence" start: 0 end: 7 tag: "NN" category: "NOUN" label: "ROOT"
  break_level: NO_BREAK
}
token {
  word: "1" start: 9 end: 9 head: 0 tag: "CD" category: "NUM" label: "num"
  break_level: SPACE_BREAK
}
token {
  word: "." start: 10 end: 10 head: 0 tag: "." category: "." label: "punct"
  break_level: NO_BREAK
}
)";

const char kLongSentence[] = R"(
text: "Sentence 123."
token {
  word: "Sentence" start: 0 end: 7 tag: "NN" category: "NOUN" label: "ROOT"
  break_level: NO_BREAK
}
token {
  word: "1" start: 9 end: 9 head: 0 tag: "CD" category: "NUM" label: "num"
  break_level: SPACE_BREAK
}
token {
  word: "2" start: 10 end: 10 head: 0 tag: "CD" category: "NUM" label: "num"
  break_level: SPACE_BREAK
}
token {
  word: "3" start: 11 end: 11 head: 0 tag: "CD" category: "NUM" label: "num"
  break_level: SPACE_BREAK
}
token {
  word: "." start: 12 end: 12 head: 0 tag: "." category: "." label: "punct"
  break_level: NO_BREAK
}
)";

}  // namespace

using testing::Return;

class SyntaxNetComponentTest : public ::testing::Test {
 public:
  std::unique_ptr<SyntaxNetComponent> CreateParser(
      const std::vector<std::vector<const TransitionState *>> &states,
      const std::vector<string> &data) {
    constexpr int kBeamSize = 2;
    return CreateParserWithBeamSize(kBeamSize, states, data);
  }
  std::unique_ptr<SyntaxNetComponent> CreateParserWithBeamSize(
      int beam_size,
      const std::vector<std::vector<const TransitionState *>> &states,
      const std::vector<string> &data) {
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

    data_.reset(new InputBatchCache(data));

    // Create a parser component with the specified beam size.
    std::unique_ptr<SyntaxNetComponent> parser_component(
        new SyntaxNetComponent());
    parser_component->InitializeComponent(*(master_spec.mutable_component(0)));
    parser_component->InitializeData(states, beam_size, data_.get());
    return parser_component;
  }

  const std::vector<Beam<SyntaxNetTransitionState> *> GetBeams(
      SyntaxNetComponent *component) const {
    std::vector<Beam<SyntaxNetTransitionState> *> return_vector;
    for (const auto &beam : component->batch_) {
      return_vector.push_back(beam.get());
    }
    return return_vector;
  }

  std::unique_ptr<InputBatchCache> data_;
};

TEST_F(SyntaxNetComponentTest, AdvancesFromOracleAndTerminates) {
  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  auto test_parser = CreateParser({}, {sentence_0_str});
  constexpr int kNumTokensInSentence = 3;

  // The master spec will initialize a parser, so expect 2*N transitions.
  constexpr int kExpectedNumTransitions = kNumTokensInSentence * 2;
  for (int i = 0; i < kExpectedNumTransitions; ++i) {
    EXPECT_FALSE(test_parser->IsTerminal());
    test_parser->AdvanceFromOracle();
  }

  // At this point, the test parser should be terminal.
  EXPECT_TRUE(test_parser->IsTerminal());

  // Check that the component is reporting 2N steps taken.
  EXPECT_EQ(test_parser->StepsTaken(0), kExpectedNumTransitions);

  // Make sure the parser doesn't segfault.
  test_parser->FinalizeData();
}

TEST_F(SyntaxNetComponentTest, AdvancesFromPredictionAndTerminates) {
  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  auto test_parser = CreateParser({}, {sentence_0_str});
  constexpr int kNumTokensInSentence = 3;

  // The master spec will initialize a parser, so expect 2*N transitions.
  constexpr int kExpectedNumTransitions = kNumTokensInSentence * 2;

  // There are 93 possible transitions for any given state. Create a transition
  // array with a score of 10.0 for each transition.
  constexpr int kBeamSize = 2;
  constexpr int kNumPossibleTransitions = 93;
  constexpr float kTransitionValue = 10.0;
  float transition_matrix[kNumPossibleTransitions * kBeamSize];
  for (int i = 0; i < kNumPossibleTransitions * kBeamSize; ++i) {
    transition_matrix[i] = kTransitionValue;
  }

  // Transition the expected number of times.
  for (int i = 0; i < kExpectedNumTransitions; ++i) {
    EXPECT_FALSE(test_parser->IsTerminal());
    EXPECT_TRUE(test_parser->AdvanceFromPrediction(transition_matrix, kBeamSize,
                                                   kNumPossibleTransitions));
  }

  // At this point, the test parser should be terminal.
  EXPECT_TRUE(test_parser->IsTerminal());

  // Check that the component is reporting 2N steps taken.
  EXPECT_EQ(test_parser->StepsTaken(0), kExpectedNumTransitions);

  // Prepare to validate the batched beams.
  auto beam = test_parser->GetBeam();

  // All beams should only have one element.
  for (const auto &per_beam : beam) {
    EXPECT_EQ(per_beam.size(), 1);
  }

  // The final states should have kExpectedNumTransitions * kTransitionValue.
  EXPECT_EQ(beam.at(0).at(0)->GetScore(),
            kTransitionValue * kExpectedNumTransitions);

  // Make sure the parser doesn't segfault.
  test_parser->FinalizeData();

  // TODO(googleuser): What should the finalized data look like?
}

TEST_F(SyntaxNetComponentTest, AdvancesFromPredictionFailsWithNanWeights) {
  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  auto test_parser = CreateParser({}, {sentence_0_str});

  // There are 93 possible transitions for any given state. Create a transition
  // array with a score of 10.0 for each transition.
  constexpr int kBeamSize = 2;
  constexpr int kNumPossibleTransitions = 93;
  float transition_matrix[kNumPossibleTransitions * kBeamSize];
  for (int i = 0; i < kNumPossibleTransitions * kBeamSize; ++i) {
    transition_matrix[i] = std::numeric_limits<double>::quiet_NaN();
  }

  EXPECT_FALSE(test_parser->IsTerminal());
  EXPECT_FALSE(test_parser->AdvanceFromPrediction(transition_matrix, kBeamSize,
                                                  kNumPossibleTransitions));
}

TEST_F(SyntaxNetComponentTest, RetainsPassedTransitionStateData) {
  // Create and initialize the state->
  MockTransitionState mock_state_one;
  constexpr int kParentBeamIndexOne = 1138;
  constexpr float kParentScoreOne = 7.2;
  EXPECT_CALL(mock_state_one, GetBeamIndex())
      .WillRepeatedly(Return(kParentBeamIndexOne));
  EXPECT_CALL(mock_state_one, GetScore())
      .WillRepeatedly(Return(kParentScoreOne));

  MockTransitionState mock_state_two;
  constexpr int kParentBeamIndexTwo = 1123;
  constexpr float kParentScoreTwo = 42.03;
  EXPECT_CALL(mock_state_two, GetBeamIndex())
      .WillRepeatedly(Return(kParentBeamIndexTwo));
  EXPECT_CALL(mock_state_two, GetScore())
      .WillRepeatedly(Return(kParentScoreTwo));

  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  auto test_parser =
      CreateParser({{&mock_state_one, &mock_state_two}}, {sentence_0_str});
  constexpr int kNumTokensInSentence = 3;

  // The master spec will initialize a parser, so expect 2*N transitions.
  constexpr int kExpectedNumTransitions = kNumTokensInSentence * 2;

  // There are 93 possible transitions for any given state. Create a transition
  // array with a score of 10.0 for each transition.
  constexpr int kBeamSize = 2;
  constexpr int kNumPossibleTransitions = 93;
  constexpr float kTransitionValue = 10.0;
  float transition_matrix[kNumPossibleTransitions * kBeamSize];
  for (int i = 0; i < kNumPossibleTransitions * kBeamSize; ++i) {
    transition_matrix[i] = kTransitionValue;
  }

  // Transition the expected number of times
  for (int i = 0; i < kExpectedNumTransitions; ++i) {
    EXPECT_FALSE(test_parser->IsTerminal());
    EXPECT_TRUE(test_parser->AdvanceFromPrediction(transition_matrix, kBeamSize,
                                                   kNumPossibleTransitions));
  }

  // At this point, the test parser should be terminal.
  EXPECT_TRUE(test_parser->IsTerminal());

  // Check that the component is reporting 2N steps taken.
  EXPECT_EQ(test_parser->StepsTaken(0), kExpectedNumTransitions);

  // The final states should have kExpectedNumTransitions * kTransitionValue,
  // plus the higher parent state score (from state two).
  auto beam = test_parser->GetBeam();
  EXPECT_EQ(beam.at(0).at(0)->GetScore(),
            kTransitionValue * kExpectedNumTransitions + kParentScoreTwo);

  // Make sure that the parent state is reported correctly.
  EXPECT_EQ(test_parser->GetSourceBeamIndex(0, 0), kParentBeamIndexTwo);

  // Make sure the parser doesn't segfault.
  test_parser->FinalizeData();

  // TODO(googleuser): What should the finalized data look like?
}

TEST_F(SyntaxNetComponentTest, AdvancesFromPredictionForMultiSentenceBatches) {
  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  Sentence sentence_1;
  TextFormat::ParseFromString(kSentence1, &sentence_1);
  string sentence_1_str;
  sentence_1.SerializeToString(&sentence_1_str);

  auto test_parser = CreateParser({}, {sentence_0_str, sentence_1_str});
  constexpr int kNumTokensInSentence = 3;

  // The master spec will initialize a parser, so expect 2*N transitions.
  constexpr int kExpectedNumTransitions = kNumTokensInSentence * 2;

  // There are 93 possible transitions for any given state. Create a transition
  // array with a score of 10.0 for each transition.
  constexpr int kBatchSize = 2;
  constexpr int kBeamSize = 2;
  constexpr int kNumPossibleTransitions = 93;
  constexpr float kTransitionValue = 10.0;
  float transition_matrix[kNumPossibleTransitions * kBeamSize * kBatchSize];
  for (int i = 0; i < kNumPossibleTransitions * kBeamSize * kBatchSize; ++i) {
    transition_matrix[i] = kTransitionValue;
  }

  // Transition the expected number of times.
  for (int i = 0; i < kExpectedNumTransitions; ++i) {
    EXPECT_FALSE(test_parser->IsTerminal());
    EXPECT_TRUE(test_parser->AdvanceFromPrediction(
        transition_matrix, kBeamSize * kBatchSize, kNumPossibleTransitions));
  }

  // At this point, the test parser should be terminal.
  EXPECT_TRUE(test_parser->IsTerminal());

  // Check that the component is reporting 2N steps taken.
  EXPECT_EQ(test_parser->StepsTaken(0), kExpectedNumTransitions);
  EXPECT_EQ(test_parser->StepsTaken(1), kExpectedNumTransitions);

  // The final states should have kExpectedNumTransitions * kTransitionValue.
  auto beam = test_parser->GetBeam();
  EXPECT_EQ(beam.at(0).at(0)->GetScore(),
            kTransitionValue * kExpectedNumTransitions);
  EXPECT_EQ(beam.at(1).at(0)->GetScore(),
            kTransitionValue * kExpectedNumTransitions);

  // Make sure the parser doesn't segfault.
  test_parser->FinalizeData();

  // TODO(googleuser): What should the finalized data look like?
}

TEST_F(SyntaxNetComponentTest,
       AdvancesFromPredictionForVaryingLengthSentences) {
  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  Sentence long_sentence;
  TextFormat::ParseFromString(kLongSentence, &long_sentence);
  string long_sentence_str;
  long_sentence.SerializeToString(&long_sentence_str);

  auto test_parser = CreateParser({}, {sentence_0_str, long_sentence_str});
  constexpr int kNumTokensInSentence = 3;
  constexpr int kNumTokensInLongSentence = 5;

  // There are 93 possible transitions for any given state. Create a transition
  // array with a score of 10.0 for each transition.
  constexpr int kBatchSize = 2;
  constexpr int kBeamSize = 2;
  constexpr int kNumPossibleTransitions = 93;
  constexpr float kTransitionValue = 10.0;
  float transition_matrix[kNumPossibleTransitions * kBeamSize * kBatchSize];
  for (int i = 0; i < kNumPossibleTransitions * kBeamSize * kBatchSize; ++i) {
    transition_matrix[i] = kTransitionValue;
  }

  // Transition the expected number of times.
  constexpr int kExpectedNumTransitions = kNumTokensInLongSentence * 2;
  for (int i = 0; i < kExpectedNumTransitions; ++i) {
    EXPECT_FALSE(test_parser->IsTerminal());
    EXPECT_TRUE(test_parser->AdvanceFromPrediction(
        transition_matrix, kBeamSize * kBatchSize, kNumPossibleTransitions));
  }

  // At this point, the test parser should be terminal.
  EXPECT_TRUE(test_parser->IsTerminal());

  // Check that the component is reporting 2N steps taken.
  EXPECT_EQ(test_parser->StepsTaken(0), kNumTokensInSentence * 2);
  EXPECT_EQ(test_parser->StepsTaken(1), kNumTokensInLongSentence * 2);

  // The final states should have kExpectedNumTransitions * kTransitionValue.
  auto beam = test_parser->GetBeam();

  // The first sentence is shorter, so it should have a lower final score.
  EXPECT_EQ(beam.at(0).at(0)->GetScore(),
            kTransitionValue * kNumTokensInSentence * 2);
  EXPECT_EQ(beam.at(1).at(0)->GetScore(),
            kTransitionValue * kNumTokensInLongSentence * 2);

  // Make sure the parser doesn't segfault.
  test_parser->FinalizeData();

  // TODO(googleuser): What should the finalized data look like?
}

TEST_F(SyntaxNetComponentTest, ResetAllowsReductionInBatchSize) {
  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  Sentence long_sentence;
  TextFormat::ParseFromString(kLongSentence, &long_sentence);
  string long_sentence_str;
  long_sentence.SerializeToString(&long_sentence_str);

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

  // Create an input batch cache with a large batch size.
  constexpr int kBeamSize = 2;
  std::unique_ptr<InputBatchCache> large_batch_data(new InputBatchCache(
      {sentence_0_str, sentence_0_str, sentence_0_str, sentence_0_str}));
  std::unique_ptr<SyntaxNetComponent> parser_component(
      new SyntaxNetComponent());
  parser_component->InitializeComponent(*(master_spec.mutable_component(0)));
  parser_component->InitializeData({}, kBeamSize, large_batch_data.get());

  // Reset the component and pass in a new input batch that is smaller.
  parser_component->ResetComponent();
  std::unique_ptr<InputBatchCache> small_batch_data(new InputBatchCache(
      {long_sentence_str, long_sentence_str, long_sentence_str}));
  parser_component->InitializeData({}, kBeamSize, small_batch_data.get());

  // There are 93 possible transitions for any given state. Create a transition
  // array with a score of 10.0 for each transition.
  constexpr int kBatchSize = 3;
  constexpr int kNumPossibleTransitions = 93;
  constexpr float kTransitionValue = 10.0;
  float transition_matrix[kNumPossibleTransitions * kBeamSize * kBatchSize];
  for (int i = 0; i < kNumPossibleTransitions * kBeamSize * kBatchSize; ++i) {
    transition_matrix[i] = kTransitionValue;
  }

  // Transition the expected number of times.
  constexpr int kNumTokensInSentence = 5;
  constexpr int kExpectedNumTransitions = kNumTokensInSentence * 2;
  for (int i = 0; i < kExpectedNumTransitions; ++i) {
    EXPECT_FALSE(parser_component->IsTerminal());
    parser_component->AdvanceFromPrediction(
        transition_matrix, kBeamSize * kBatchSize, kNumPossibleTransitions);
  }

  // At this point, the test parser should be terminal.
  EXPECT_TRUE(parser_component->IsTerminal());

  // Check that the component is reporting 2N steps taken.
  EXPECT_EQ(parser_component->StepsTaken(0), kExpectedNumTransitions);
  EXPECT_EQ(parser_component->StepsTaken(1), kExpectedNumTransitions);
  EXPECT_EQ(parser_component->StepsTaken(2), kExpectedNumTransitions);

  // The final states should have kExpectedNumTransitions * kTransitionValue.
  auto beam = parser_component->GetBeam();

  // The beam should be of batch size 3.
  EXPECT_EQ(beam.size(), 3);
  EXPECT_EQ(beam.at(0).at(0)->GetScore(),
            kTransitionValue * kExpectedNumTransitions);
  EXPECT_EQ(beam.at(1).at(0)->GetScore(),
            kTransitionValue * kExpectedNumTransitions);
  EXPECT_EQ(beam.at(2).at(0)->GetScore(),
            kTransitionValue * kExpectedNumTransitions);

  // Make sure the parser doesn't segfault.
  parser_component->FinalizeData();
}

TEST_F(SyntaxNetComponentTest, ResetAllowsIncreaseInBatchSize) {
  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  Sentence long_sentence;
  TextFormat::ParseFromString(kLongSentence, &long_sentence);
  string long_sentence_str;
  long_sentence.SerializeToString(&long_sentence_str);

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

  // Create an input batch cache with a small batch size.
  constexpr int kBeamSize = 2;
  std::unique_ptr<InputBatchCache> small_batch_data(
      new InputBatchCache(sentence_0_str));
  std::unique_ptr<SyntaxNetComponent> parser_component(
      new SyntaxNetComponent());
  parser_component->InitializeComponent(*(master_spec.mutable_component(0)));
  parser_component->InitializeData({}, kBeamSize, small_batch_data.get());

  // Reset the component and pass in a new input batch that is larger.
  parser_component->ResetComponent();
  std::unique_ptr<InputBatchCache> large_batch_data(new InputBatchCache(
      {long_sentence_str, long_sentence_str, long_sentence_str}));
  parser_component->InitializeData({}, kBeamSize, large_batch_data.get());

  // There are 93 possible transitions for any given state. Create a transition
  // array with a score of 10.0 for each transition.
  constexpr int kBatchSize = 3;
  constexpr int kNumPossibleTransitions = 93;
  constexpr float kTransitionValue = 10.0;
  float transition_matrix[kNumPossibleTransitions * kBeamSize * kBatchSize];
  for (int i = 0; i < kNumPossibleTransitions * kBeamSize * kBatchSize; ++i) {
    transition_matrix[i] = kTransitionValue;
  }

  // Transition the expected number of times.
  constexpr int kNumTokensInSentence = 5;
  constexpr int kExpectedNumTransitions = kNumTokensInSentence * 2;
  for (int i = 0; i < kExpectedNumTransitions; ++i) {
    EXPECT_FALSE(parser_component->IsTerminal());
    parser_component->AdvanceFromPrediction(
        transition_matrix, kBeamSize * kBatchSize, kNumPossibleTransitions);
  }

  // At this point, the test parser should be terminal.
  EXPECT_TRUE(parser_component->IsTerminal());

  // Check that the component is reporting 2N steps taken.
  EXPECT_EQ(parser_component->StepsTaken(0), kExpectedNumTransitions);
  EXPECT_EQ(parser_component->StepsTaken(1), kExpectedNumTransitions);
  EXPECT_EQ(parser_component->StepsTaken(2), kExpectedNumTransitions);

  // The final states should have kExpectedNumTransitions * kTransitionValue.
  auto beam = parser_component->GetBeam();

  // The beam should be of batch size 3.
  EXPECT_EQ(beam.size(), 3);
  EXPECT_EQ(beam.at(0).at(0)->GetScore(),
            kTransitionValue * kExpectedNumTransitions);
  EXPECT_EQ(beam.at(1).at(0)->GetScore(),
            kTransitionValue * kExpectedNumTransitions);
  EXPECT_EQ(beam.at(2).at(0)->GetScore(),
            kTransitionValue * kExpectedNumTransitions);

  // Make sure the parser doesn't segfault.
  parser_component->FinalizeData();
}

TEST_F(SyntaxNetComponentTest, ResetCausesBeamToReset) {
  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  Sentence long_sentence;
  TextFormat::ParseFromString(kLongSentence, &long_sentence);
  string long_sentence_str;
  long_sentence.SerializeToString(&long_sentence_str);

  auto test_parser = CreateParser({}, {sentence_0_str});
  constexpr int kNumTokensInSentence = 3;

  // The master spec will initialize a parser, so expect 2*N transitions.
  constexpr int kExpectedNumTransitions = kNumTokensInSentence * 2;

  // There are 93 possible transitions for any given state. Create a transition
  // array with a score of 10.0 for each transition.
  constexpr int kBeamSize = 2;
  constexpr int kNumPossibleTransitions = 93;
  constexpr float kTransitionValue = 10.0;
  float transition_matrix[kNumPossibleTransitions * kBeamSize];
  for (int i = 0; i < kNumPossibleTransitions * kBeamSize; ++i) {
    transition_matrix[i] = kTransitionValue;
  }

  // Transition the expected number of times.
  for (int i = 0; i < kExpectedNumTransitions; ++i) {
    EXPECT_FALSE(test_parser->IsTerminal());
    EXPECT_TRUE(test_parser->AdvanceFromPrediction(transition_matrix, kBeamSize,
                                                   kNumPossibleTransitions));
  }

  // At this point, the test parser should be terminal.
  EXPECT_TRUE(test_parser->IsTerminal());

  // Check that the component is reporting 2N steps taken.
  EXPECT_EQ(test_parser->StepsTaken(0), kExpectedNumTransitions);

  // The final states should have kExpectedNumTransitions * kTransitionValue.
  auto beam = test_parser->GetBeam();
  EXPECT_EQ(beam.at(0).at(0)->GetScore(),
            kTransitionValue * kExpectedNumTransitions);

  // Reset the test parser and give it new data.
  test_parser->ResetComponent();
  std::unique_ptr<InputBatchCache> new_data(
      new InputBatchCache(long_sentence_str));
  test_parser->InitializeData({}, kBeamSize, new_data.get());

  // Check that the component is not terminal.
  EXPECT_FALSE(test_parser->IsTerminal());

  // Check that the component is reporting 0 steps taken.
  EXPECT_EQ(test_parser->StepsTaken(0), 0);

  // The  states should have 0 as their score.
  auto new_beam = test_parser->GetBeam();
  EXPECT_EQ(new_beam.at(0).at(0)->GetScore(), 0);
}

TEST_F(SyntaxNetComponentTest, AdjustingMaxBeamSizeAdjustsSizeForAllBeams) {
  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  Sentence long_sentence;
  TextFormat::ParseFromString(kLongSentence, &long_sentence);
  string long_sentence_str;
  long_sentence.SerializeToString(&long_sentence_str);

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

  // Create an input batch cache with a small batch size.
  constexpr int kBeamSize = 2;
  std::unique_ptr<InputBatchCache> small_batch_data(
      new InputBatchCache(sentence_0_str));
  std::unique_ptr<SyntaxNetComponent> parser_component(
      new SyntaxNetComponent());
  parser_component->InitializeComponent(*(master_spec.mutable_component(0)));
  parser_component->InitializeData({}, kBeamSize, small_batch_data.get());

  // Make sure all the beams in the batch have max size 2.
  for (const auto &beam : GetBeams(parser_component.get())) {
    EXPECT_EQ(beam->max_size(), kBeamSize);
  }

  // Reset the component and pass in a new input batch that is larger, with
  // a higher beam size.
  constexpr int kNewBeamSize = 5;
  parser_component->ResetComponent();
  std::unique_ptr<InputBatchCache> large_batch_data(new InputBatchCache(
      {long_sentence_str, long_sentence_str, long_sentence_str}));
  parser_component->InitializeData({}, kNewBeamSize, large_batch_data.get());

  // Make sure all the beams in the batch now have max size 5.
  for (const auto &beam : GetBeams(parser_component.get())) {
    EXPECT_EQ(beam->max_size(), kNewBeamSize);
  }
}

TEST_F(SyntaxNetComponentTest, SettingBeamSizeZeroFails) {
  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  Sentence long_sentence;
  TextFormat::ParseFromString(kLongSentence, &long_sentence);
  string long_sentence_str;
  long_sentence.SerializeToString(&long_sentence_str);

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

  // Create an input batch cache with a small batch size.
  constexpr int kBeamSize = 0;
  std::unique_ptr<InputBatchCache> small_batch_data(
      new InputBatchCache(sentence_0_str));
  std::unique_ptr<SyntaxNetComponent> parser_component(
      new SyntaxNetComponent());
  parser_component->InitializeComponent(*(master_spec.mutable_component(0)));
  EXPECT_DEATH(
      parser_component->InitializeData({}, kBeamSize, small_batch_data.get()),
      "must be greater than 0");
}

TEST_F(SyntaxNetComponentTest, ExportsFixedFeaturesWithPadding) {
  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  Sentence sentence_1;
  TextFormat::ParseFromString(kSentence1, &sentence_1);
  string sentence_1_str;
  sentence_1.SerializeToString(&sentence_1_str);

  constexpr int kBeamSize = 3;

  auto test_parser =
      CreateParserWithBeamSize(kBeamSize, {}, {sentence_0_str, sentence_1_str});

  // Get and check the raw link features.
  vector<int32> indices;
  auto indices_fn = [&indices](int size) {
    indices.resize(size);
    return indices.data();
  };
  vector<int64> ids;
  auto ids_fn = [&ids](int size) {
    ids.resize(size);
    return ids.data();
  };
  vector<float> weights;
  auto weights_fn = [&weights](int size) {
    weights.resize(size);
    return weights.data();
  };
  constexpr int kChannelId = 0;
  const int num_features =
      test_parser->GetFixedFeatures(indices_fn, ids_fn, weights_fn, kChannelId);

  // The raw features for each beam object should be [single, single].
  // There is also padding expected in this beam - there is only one
  // element in each beam (so two elements total; batch is two). Thus, we expect
  // 0,1 and 6,7 to be filled with one element each.
  constexpr int kExpectedOutputSize = 4;
  const vector<int32> expected_indices({0, 1, 6, 7});
  const vector<int64> expected_ids({0, 12, 0, 12});
  const vector<float> expected_weights({1.0, 1.0, 1.0, 1.0});

  EXPECT_EQ(expected_indices.size(), kExpectedOutputSize);
  EXPECT_EQ(expected_ids.size(), kExpectedOutputSize);
  EXPECT_EQ(expected_weights.size(), kExpectedOutputSize);
  EXPECT_EQ(num_features, kExpectedOutputSize);

  EXPECT_EQ(expected_indices, indices);
  EXPECT_EQ(expected_ids, ids);
  EXPECT_EQ(expected_weights, weights);
}

TEST_F(SyntaxNetComponentTest, ExportsFixedFeatures) {
  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  Sentence sentence_1;
  TextFormat::ParseFromString(kSentence1, &sentence_1);
  string sentence_1_str;
  sentence_1.SerializeToString(&sentence_1_str);

  constexpr int kBeamSize = 3;

  auto test_parser =
      CreateParserWithBeamSize(kBeamSize, {}, {sentence_0_str, sentence_1_str});

  // There are 93 possible transitions for any given state. Create a transition
  // array with a score of 10.0 for each transition.
  constexpr int kBatchSize = 2;
  constexpr int kNumPossibleTransitions = 93;
  constexpr float kTransitionValue = 10.0;
  float transition_matrix[kNumPossibleTransitions * kBeamSize * kBatchSize];
  for (int i = 0; i < kNumPossibleTransitions * kBeamSize * kBatchSize; ++i) {
    transition_matrix[i] = kTransitionValue;
  }

  // Advance twice, so that the underlying parser fills the beam.
  EXPECT_TRUE(test_parser->AdvanceFromPrediction(
      transition_matrix, kBeamSize * kBatchSize, kNumPossibleTransitions));
  EXPECT_TRUE(test_parser->AdvanceFromPrediction(
      transition_matrix, kBeamSize * kBatchSize, kNumPossibleTransitions));

  // Get and check the raw link features.
  vector<int32> indices;
  auto indices_fn = [&indices](int size) {
    indices.resize(size);
    return indices.data();
  };
  vector<int64> ids;
  auto ids_fn = [&ids](int size) {
    ids.resize(size);
    return ids.data();
  };
  vector<float> weights;
  auto weights_fn = [&weights](int size) {
    weights.resize(size);
    return weights.data();
  };
  constexpr int kChannelId = 0;
  const int num_features =
      test_parser->GetFixedFeatures(indices_fn, ids_fn, weights_fn, kChannelId);

  constexpr int kExpectedOutputSize = 12;
  const vector<int32> expected_indices({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  const vector<int64> expected_ids({7, 50, 12, 7, 12, 7, 7, 50, 12, 7, 12, 7});
  const vector<float> expected_weights(
      {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});

  EXPECT_EQ(expected_indices.size(), kExpectedOutputSize);
  EXPECT_EQ(expected_ids.size(), kExpectedOutputSize);
  EXPECT_EQ(expected_weights.size(), kExpectedOutputSize);
  EXPECT_EQ(num_features, kExpectedOutputSize);

  EXPECT_EQ(expected_indices, indices);
  EXPECT_EQ(expected_ids, ids);
  EXPECT_EQ(expected_weights, weights);
}

TEST_F(SyntaxNetComponentTest, AdvancesAccordingToHighestWeightedInputOption) {
  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  Sentence sentence_1;
  TextFormat::ParseFromString(kSentence1, &sentence_1);
  string sentence_1_str;
  sentence_1.SerializeToString(&sentence_1_str);

  constexpr int kBeamSize = 3;

  auto test_parser =
      CreateParserWithBeamSize(kBeamSize, {}, {sentence_0_str, sentence_1_str});

  // There are 93 possible transitions for any given state. Create a transition
  // array with a score of 10.0 for each transition.
  constexpr int kBatchSize = 2;
  constexpr int kNumPossibleTransitions = 93;
  constexpr float kTransitionValue = 10.0;
  float transition_matrix[kNumPossibleTransitions * kBeamSize * kBatchSize];
  for (int i = 0; i < kNumPossibleTransitions * kBeamSize * kBatchSize; ++i) {
    transition_matrix[i] = kTransitionValue;
  }

  // Replace the first several options with varying scores to test sorting.
  constexpr int kBatchOffset = kNumPossibleTransitions * kBeamSize;
  transition_matrix[0] = 3 * kTransitionValue;
  transition_matrix[1] = 3 * kTransitionValue;
  transition_matrix[2] = 4 * kTransitionValue;
  transition_matrix[3] = 4 * kTransitionValue;
  transition_matrix[4] = 2 * kTransitionValue;
  transition_matrix[5] = 2 * kTransitionValue;
  transition_matrix[kBatchOffset + 0] = 3 * kTransitionValue;
  transition_matrix[kBatchOffset + 1] = 3 * kTransitionValue;
  transition_matrix[kBatchOffset + 2] = 4 * kTransitionValue;
  transition_matrix[kBatchOffset + 3] = 4 * kTransitionValue;
  transition_matrix[kBatchOffset + 4] = 2 * kTransitionValue;
  transition_matrix[kBatchOffset + 5] = 2 * kTransitionValue;

  // Advance twice, so that the underlying parser fills the beam.
  EXPECT_TRUE(test_parser->AdvanceFromPrediction(
      transition_matrix, kBeamSize * kBatchSize, kNumPossibleTransitions));
  EXPECT_TRUE(test_parser->AdvanceFromPrediction(
      transition_matrix, kBeamSize * kBatchSize, kNumPossibleTransitions));

  // Get and check the raw link features.
  vector<int32> indices;
  auto indices_fn = [&indices](int size) {
    indices.resize(size);
    return indices.data();
  };
  vector<int64> ids;
  auto ids_fn = [&ids](int size) {
    ids.resize(size);
    return ids.data();
  };
  vector<float> weights;
  auto weights_fn = [&weights](int size) {
    weights.resize(size);
    return weights.data();
  };
  constexpr int kChannelId = 0;
  const int num_features =
      test_parser->GetFixedFeatures(indices_fn, ids_fn, weights_fn, kChannelId);

  // In this case, all even features and all odd features are identical.
  constexpr int kExpectedOutputSize = 12;
  const vector<int32> expected_indices({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  const vector<int64> expected_ids({12, 7, 7, 50, 12, 7, 12, 7, 7, 50, 12, 7});
  const vector<float> expected_weights(
      {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});

  EXPECT_EQ(expected_indices.size(), kExpectedOutputSize);
  EXPECT_EQ(expected_ids.size(), kExpectedOutputSize);
  EXPECT_EQ(expected_weights.size(), kExpectedOutputSize);
  EXPECT_EQ(num_features, kExpectedOutputSize);

  EXPECT_EQ(expected_indices, indices);
  EXPECT_EQ(expected_ids, ids);
  EXPECT_EQ(expected_weights, weights);
}

TEST_F(SyntaxNetComponentTest, ExportsBulkFixedFeatures) {
  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  Sentence sentence_1;
  TextFormat::ParseFromString(kSentence1, &sentence_1);
  string sentence_1_str;
  sentence_1.SerializeToString(&sentence_1_str);

  constexpr int kBeamSize = 3;
  auto test_parser =
      CreateParserWithBeamSize(kBeamSize, {}, {sentence_0_str, sentence_1_str});

  // Get and check the raw link features.
  vector<vector<int32>> indices;
  auto indices_fn = [&indices](int channel, int size) {
    indices.resize(channel + 1);
    indices[channel].resize(size);
    return indices[channel].data();
  };
  vector<vector<int64>> ids;
  auto ids_fn = [&ids](int channel, int size) {
    ids.resize(channel + 1);
    ids[channel].resize(size);
    return ids[channel].data();
  };
  vector<vector<float>> weights;
  auto weights_fn = [&weights](int channel, int size) {
    weights.resize(channel + 1);
    weights[channel].resize(size);
    return weights[channel].data();
  };

  BulkFeatureExtractor extractor(indices_fn, ids_fn, weights_fn);
  const int num_steps = test_parser->BulkGetFixedFeatures(extractor);

  // There should be 6 steps (2N, where N is the longest number of tokens).
  EXPECT_EQ(num_steps, 6);

  // These are empirically derived.
  const vector<int32> expected_ch0_indices({0, 36, 18, 54, 1, 37, 19, 55,
                                            2, 38, 20, 56, 3, 39, 21, 57,
                                            4, 40, 22, 58, 5, 41, 23, 59});
  const vector<int64> expected_ch0_ids({0,  12, 0,  12, 12, 7,  12, 7,
                                        7,  50, 7,  50, 7,  50, 7,  50,
                                        50, 50, 50, 50, 50, 50, 50, 50});
  const vector<float> expected_ch0_weights(
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  const vector<int32> expected_ch1_indices(
      {0, 36, 72, 18, 54, 90, 1, 37, 73, 19, 55, 91, 2, 38, 74, 20, 56, 92,
       3, 39, 75, 21, 57, 93, 4, 40, 76, 22, 58, 94, 5, 41, 77, 23, 59, 95});
  const vector<int64> expected_ch1_ids(
      {51, 0, 12, 51, 0, 12, 0, 12, 7,  0, 12, 7,  12, 7,  50, 12, 7,  50,
       12, 7, 50, 12, 7, 50, 7, 50, 50, 7, 50, 50, 7,  50, 50, 7,  50, 50});
  const vector<float> expected_ch1_weights(
      {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});

  EXPECT_EQ(indices[0], expected_ch0_indices);
  EXPECT_EQ(ids[0], expected_ch0_ids);
  EXPECT_EQ(weights[0], expected_ch0_weights);
  EXPECT_EQ(indices[1], expected_ch1_indices);
  EXPECT_EQ(ids[1], expected_ch1_ids);
  EXPECT_EQ(weights[1], expected_ch1_weights);
}

TEST_F(SyntaxNetComponentTest, ExportsRawLinkFeaturesWithPadding) {
  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  Sentence sentence_1;
  TextFormat::ParseFromString(kSentence1, &sentence_1);
  string sentence_1_str;
  sentence_1.SerializeToString(&sentence_1_str);

  constexpr int kBeamSize = 3;
  constexpr int kBatchSize = 2;
  auto test_parser =
      CreateParserWithBeamSize(kBeamSize, {}, {sentence_0_str, sentence_1_str});

  // Get and check the raw link features.
  constexpr int kNumLinkFeatures = 2;
  auto link_features = test_parser->GetRawLinkFeatures(0);
  EXPECT_EQ(link_features.size(), kBeamSize * kBatchSize * kNumLinkFeatures);

  EXPECT_EQ(link_features.at(0).feature_value(), -1);
  EXPECT_EQ(link_features.at(0).batch_idx(), 0);
  EXPECT_EQ(link_features.at(0).beam_idx(), 0);

  EXPECT_EQ(link_features.at(1).feature_value(), -2);
  EXPECT_EQ(link_features.at(1).batch_idx(), 0);
  EXPECT_EQ(link_features.at(1).beam_idx(), 0);

  // These are padding, so we do not expect them to have a feature value.
  EXPECT_FALSE(link_features.at(2).has_feature_value());
  EXPECT_FALSE(link_features.at(2).has_batch_idx());
  EXPECT_FALSE(link_features.at(2).has_beam_idx());
  EXPECT_FALSE(link_features.at(3).has_feature_value());
  EXPECT_FALSE(link_features.at(3).has_batch_idx());
  EXPECT_FALSE(link_features.at(3).has_beam_idx());
  EXPECT_FALSE(link_features.at(4).has_feature_value());
  EXPECT_FALSE(link_features.at(4).has_batch_idx());
  EXPECT_FALSE(link_features.at(4).has_beam_idx());
  EXPECT_FALSE(link_features.at(5).has_feature_value());
  EXPECT_FALSE(link_features.at(5).has_batch_idx());
  EXPECT_FALSE(link_features.at(5).has_beam_idx());

  EXPECT_EQ(link_features.at(6).feature_value(), -1);
  EXPECT_EQ(link_features.at(6).batch_idx(), 1);
  EXPECT_EQ(link_features.at(6).beam_idx(), 0);

  EXPECT_EQ(link_features.at(7).feature_value(), -2);
  EXPECT_EQ(link_features.at(7).batch_idx(), 1);
  EXPECT_EQ(link_features.at(7).beam_idx(), 0);

  // These are padding, so we do not expect them to have a feature value.
  EXPECT_FALSE(link_features.at(8).has_feature_value());
  EXPECT_FALSE(link_features.at(8).has_batch_idx());
  EXPECT_FALSE(link_features.at(8).has_beam_idx());
  EXPECT_FALSE(link_features.at(9).has_feature_value());
  EXPECT_FALSE(link_features.at(9).has_batch_idx());
  EXPECT_FALSE(link_features.at(9).has_beam_idx());
  EXPECT_FALSE(link_features.at(10).has_feature_value());
  EXPECT_FALSE(link_features.at(10).has_batch_idx());
  EXPECT_FALSE(link_features.at(10).has_beam_idx());
  EXPECT_FALSE(link_features.at(11).has_feature_value());
  EXPECT_FALSE(link_features.at(11).has_batch_idx());
  EXPECT_FALSE(link_features.at(11).has_beam_idx());
}

TEST_F(SyntaxNetComponentTest, ExportsRawLinkFeatures) {
  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  Sentence sentence_1;
  TextFormat::ParseFromString(kSentence1, &sentence_1);
  string sentence_1_str;
  sentence_1.SerializeToString(&sentence_1_str);

  constexpr int kBeamSize = 3;
  auto test_parser =
      CreateParserWithBeamSize(kBeamSize, {}, {sentence_0_str, sentence_1_str});

  // There are 93 possible transitions for any given state. Create a transition
  // array with a score of 10.0 for each transition.
  constexpr int kBatchSize = 2;
  constexpr int kNumPossibleTransitions = 93;
  constexpr float kTransitionValue = 10.0;
  float transition_matrix[kNumPossibleTransitions * kBeamSize * kBatchSize];
  for (int i = 0; i < kNumPossibleTransitions * kBeamSize * kBatchSize; ++i) {
    transition_matrix[i] = kTransitionValue;
  }

  // Advance twice, so that the underlying parser fills the beam.
  EXPECT_TRUE(test_parser->AdvanceFromPrediction(
      transition_matrix, kBeamSize * kBatchSize, kNumPossibleTransitions));
  EXPECT_TRUE(test_parser->AdvanceFromPrediction(
      transition_matrix, kBeamSize * kBatchSize, kNumPossibleTransitions));

  // Get and check the raw link features.
  constexpr int kNumLinkFeatures = 2;
  auto link_features = test_parser->GetRawLinkFeatures(0);
  EXPECT_EQ(link_features.size(), kBeamSize * kBatchSize * kNumLinkFeatures);

  // These should index into batch 0.
  EXPECT_EQ(link_features.at(0).feature_value(), 1);
  EXPECT_EQ(link_features.at(0).batch_idx(), 0);
  EXPECT_EQ(link_features.at(0).beam_idx(), 0);

  EXPECT_EQ(link_features.at(1).feature_value(), 0);
  EXPECT_EQ(link_features.at(1).batch_idx(), 0);
  EXPECT_EQ(link_features.at(1).beam_idx(), 0);

  EXPECT_EQ(link_features.at(2).feature_value(), -1);
  EXPECT_EQ(link_features.at(2).batch_idx(), 0);
  EXPECT_EQ(link_features.at(2).beam_idx(), 1);

  EXPECT_EQ(link_features.at(3).feature_value(), -2);
  EXPECT_EQ(link_features.at(3).batch_idx(), 0);
  EXPECT_EQ(link_features.at(3).beam_idx(), 1);

  EXPECT_EQ(link_features.at(4).feature_value(), -1);
  EXPECT_EQ(link_features.at(4).batch_idx(), 0);
  EXPECT_EQ(link_features.at(4).beam_idx(), 2);

  EXPECT_EQ(link_features.at(5).feature_value(), -2);
  EXPECT_EQ(link_features.at(5).batch_idx(), 0);
  EXPECT_EQ(link_features.at(5).beam_idx(), 2);

  // These should index into batch 1.
  EXPECT_EQ(link_features.at(6).feature_value(), 1);
  EXPECT_EQ(link_features.at(6).batch_idx(), 1);
  EXPECT_EQ(link_features.at(6).beam_idx(), 0);

  EXPECT_EQ(link_features.at(7).feature_value(), 0);
  EXPECT_EQ(link_features.at(7).batch_idx(), 1);
  EXPECT_EQ(link_features.at(7).beam_idx(), 0);

  EXPECT_EQ(link_features.at(8).feature_value(), -1);
  EXPECT_EQ(link_features.at(8).batch_idx(), 1);
  EXPECT_EQ(link_features.at(8).beam_idx(), 1);

  EXPECT_EQ(link_features.at(9).feature_value(), -2);
  EXPECT_EQ(link_features.at(9).batch_idx(), 1);
  EXPECT_EQ(link_features.at(9).beam_idx(), 1);

  EXPECT_EQ(link_features.at(10).feature_value(), -1);
  EXPECT_EQ(link_features.at(10).batch_idx(), 1);
  EXPECT_EQ(link_features.at(10).beam_idx(), 2);

  EXPECT_EQ(link_features.at(11).feature_value(), -2);
  EXPECT_EQ(link_features.at(11).batch_idx(), 1);
  EXPECT_EQ(link_features.at(11).beam_idx(), 2);
}

TEST_F(SyntaxNetComponentTest, AdvancesFromOracleWithTracing) {
  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  constexpr int kBeamSize = 1;
  auto test_parser = CreateParserWithBeamSize(kBeamSize, {}, {sentence_0_str});
  test_parser->InitializeTracing();

  constexpr int kNumTokensInSentence = 3;

  // The master spec will initialize a parser, so expect 2*N transitions.
  constexpr int kExpectedNumTransitions = kNumTokensInSentence * 2;
  constexpr int kFixedFeatureChannels = 1;
  for (int i = 0; i < kExpectedNumTransitions; ++i) {
    EXPECT_FALSE(test_parser->IsTerminal());
    vector<int32> indices;
    auto indices_fn = [&indices](int size) {
      indices.resize(size);
      return indices.data();
    };
    vector<int64> ids;
    auto ids_fn = [&ids](int size) {
      ids.resize(size);
      return ids.data();
    };
    vector<float> weights;
    auto weights_fn = [&weights](int size) {
      weights.resize(size);
      return weights.data();
    };
    for (int j = 0; j < kFixedFeatureChannels; ++j) {
      test_parser->GetFixedFeatures(indices_fn, ids_fn, weights_fn, j);
    }
    auto features = test_parser->GetRawLinkFeatures(0);

    // Make some fake translations to test visualization.
    for (int j = 0; j < features.size(); ++j) {
      features[j].set_step_idx(j < i ? j : -1);
    }
    test_parser->AddTranslatedLinkFeaturesToTrace(features, 0);
    test_parser->AdvanceFromOracle();
  }

  // At this point, the test parser should be terminal.
  EXPECT_TRUE(test_parser->IsTerminal());

  // TODO(googleuser): Add EXPECT_EQ here instead of printing.
  std::vector<std::vector<ComponentTrace>> traces =
      test_parser->GetTraceProtos();
  for (auto &batch_trace : traces) {
    for (auto &trace : batch_trace) {
      LOG(INFO) << "trace:" << std::endl << trace.DebugString();
    }
  }
}

TEST_F(SyntaxNetComponentTest, NoTracingDropsFeatureNames) {
  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  constexpr int kBeamSize = 1;
  const auto test_parser =
      CreateParserWithBeamSize(kBeamSize, {}, {sentence_0_str});
  const auto link_features = test_parser->GetRawLinkFeatures(0);

  // The fml associated with the channel is "stack.focus stack(1).focus".
  // Both features should lack the feature_name field.
  EXPECT_EQ(link_features.size(), 2);
  EXPECT_FALSE(link_features.at(0).has_feature_name());
  EXPECT_FALSE(link_features.at(1).has_feature_name());
}

TEST_F(SyntaxNetComponentTest, TracingOutputsFeatureNames) {
  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  constexpr int kBeamSize = 1;
  auto test_parser = CreateParserWithBeamSize(kBeamSize, {}, {sentence_0_str});
  test_parser->InitializeTracing();
  const auto link_features = test_parser->GetRawLinkFeatures(0);

  // The fml associated with the channel is "stack.focus stack(1).focus".
  EXPECT_EQ(link_features.size(), 2);
  EXPECT_EQ(link_features.at(0).feature_name(), "stack.focus");
  EXPECT_EQ(link_features.at(1).feature_name(), "stack(1).focus");
}

TEST_F(SyntaxNetComponentTest, BulkEmbedFixedFeaturesIsNotSupported) {
  // Create an empty input batch and beam vector to initialize the parser.
  Sentence sentence_0;

  // TODO(googleuser): Wrap this in a lint-friendly helper function.
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  constexpr int kBeamSize = 1;
  auto test_parser = CreateParserWithBeamSize(kBeamSize, {}, {sentence_0_str});
  EXPECT_TRUE(test_parser->IsReady());
  EXPECT_DEATH(test_parser->BulkEmbedFixedFeatures(0, 0, 0, {nullptr}, nullptr),
               "Method not supported");
}

TEST_F(SyntaxNetComponentTest, GetStepLookupFunction) {
  Sentence sentence_0;
  TextFormat::ParseFromString(kSentence0, &sentence_0);
  string sentence_0_str;
  sentence_0.SerializeToString(&sentence_0_str);

  constexpr int kBeamSize = 1;
  auto test_parser = CreateParserWithBeamSize(kBeamSize, {}, {sentence_0_str});
  ASSERT_TRUE(test_parser->IsReady());

  const auto reverse_token_lookup =
      test_parser->GetStepLookupFunction("reverse-token");
  const int kNumTokens = sentence_0.token_size();
  for (int i = 0; i < kNumTokens; ++i) {
    EXPECT_EQ(i, reverse_token_lookup(0, 0, kNumTokens - i - 1));
  }

  const auto reverse_char_lookup =
      test_parser->GetStepLookupFunction("reverse-char");
  const int kNumChars = sentence_0.text().size();  // assumes ASCII
  for (int i = 0; i < kNumChars; ++i) {
    EXPECT_EQ(i, reverse_char_lookup(0, 0, kNumChars - i - 1));
  }
}

}  // namespace dragnn
}  // namespace syntaxnet
