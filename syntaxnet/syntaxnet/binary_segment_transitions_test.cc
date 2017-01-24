/* Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "syntaxnet/binary_segment_state.h"
#include "syntaxnet/parser_features.h"
#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/term_frequency_map.h"
#include "syntaxnet/workspace.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {

class SegmentationTransitionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    transition_system_ = std::unique_ptr<ParserTransitionSystem>(
        ParserTransitionSystem::Create("binary-segment-transitions"));

    // Prepare a sentence.
    const char *str_sentence = "text: '因为 有 这样' "
        "token { word: '因' start: 0 end: 2 break_level: SPACE_BREAK } "
        "token { word: '为' start: 3 end: 5 break_level: NO_BREAK } "
        "token { word: ' ' start: 6 end: 6 break_level: SPACE_BREAK } "
        "token { word: '有' start: 7 end: 9 break_level: SPACE_BREAK } "
        "token { word: ' ' start: 10 end: 10 break_level: SPACE_BREAK } "
        "token { word: '这' start: 11 end: 13 break_level: SPACE_BREAK } "
        "token { word: '样' start: 14 end: 16 break_level: NO_BREAK } ";
    sentence_ = std::unique_ptr<Sentence>(new Sentence());
    TextFormat::ParseFromString(str_sentence, sentence_.get());

    word_map_.Increment("因为");
    word_map_.Increment("因为");
    word_map_.Increment("有");
    word_map_.Increment("这");
    word_map_.Increment("这");
    word_map_.Increment("样");
    word_map_.Increment("样");
    word_map_.Increment("这样");
    word_map_.Increment("这样");
    string filename = tensorflow::strings::StrCat(
        tensorflow::testing::TmpDir(), "word-map");
    word_map_.Save(filename);

    // Re-load in sorted order, ignore words that only occurs once.
    word_map_.Load(filename, 2, -1);

    // Prepare task context.
    context_ = std::unique_ptr<TaskContext>(new TaskContext());
    AddInputToContext("word-map", filename, "text", "");
    registry_ = std::unique_ptr<WorkspaceRegistry>( new WorkspaceRegistry());
  }

  // Adds an input to the task context.
  void AddInputToContext(const string &name,
                         const string &file_pattern,
                         const string &file_format,
                         const string &record_format) {
    TaskInput *input = context_->GetInput(name);
    TaskInput::Part *part = input->add_part();
    part->set_file_pattern(file_pattern);
    part->set_file_format(file_format);
    part->set_record_format(record_format);
  }

  // Prepares a feature for computations.
  void PrepareFeature(const string &feature_name, ParserState *state) {
    feature_extractor_ = std::unique_ptr<ParserFeatureExtractor>(
        new ParserFeatureExtractor());
    feature_extractor_->Parse(feature_name);
    feature_extractor_->Setup(context_.get());
    feature_extractor_->Init(context_.get());
    feature_extractor_->RequestWorkspaces(registry_.get());
    workspace_.Reset(*registry_);
    feature_extractor_->Preprocess(&workspace_, state);
  }

  // Computes the feature value for the parser state.
  FeatureValue ComputeFeature(const ParserState &state) const {
    FeatureVector result;
    feature_extractor_->ExtractFeatures(workspace_, state, &result);
    return result.size() > 0 ? result.value(0) : -1;
  }

  void CheckStarts(const ParserState &state, const std::vector<int> &target) {
    ASSERT_EQ(state.StackSize(), target.size());
    std::vector<int> starts;
    for (int i = 0; i < state.StackSize(); ++i) {
      EXPECT_EQ(state.Stack(i), target[i]);
    }
  }

  // The test sentence.
  std::unique_ptr<Sentence> sentence_;

  // Members for testing features.
  std::unique_ptr<ParserFeatureExtractor> feature_extractor_;
  std::unique_ptr<TaskContext> context_;
  std::unique_ptr<WorkspaceRegistry> registry_;
  WorkspaceSet workspace_;

  std::unique_ptr<ParserTransitionSystem> transition_system_;
  TermFrequencyMap label_map_;
  TermFrequencyMap word_map_;
};

TEST_F(SegmentationTransitionTest, GoldNextActionTest) {
  BinarySegmentState *segment_state = static_cast<BinarySegmentState *>(
      transition_system_->NewTransitionState(true));
  ParserState state(sentence_.get(), segment_state, &label_map_);

  // Do segmentation by following the gold actions.
  while (transition_system_->IsFinalState(state) == false) {
    ParserAction action = transition_system_->GetNextGoldAction(state);
    transition_system_->PerformActionWithoutHistory(action, &state);
  }

  // Test STARTs.
  CheckStarts(state, {5, 4, 3, 2, 0});

  // Test the annotated tokens.
  segment_state->AddParseToDocument(state, false, sentence_.get());
  ASSERT_EQ(sentence_->token_size(), 3);
  EXPECT_EQ(sentence_->token(0).word(), "因为");
  EXPECT_EQ(sentence_->token(1).word(), "有");
  EXPECT_EQ(sentence_->token(2).word(), "这样");

  // Test start/end annotation of each token.
  EXPECT_EQ(sentence_->token(0).start(), 0);
  EXPECT_EQ(sentence_->token(0).end(), 5);
  EXPECT_EQ(sentence_->token(1).start(), 7);
  EXPECT_EQ(sentence_->token(1).end(), 9);
  EXPECT_EQ(sentence_->token(2).start(), 11);
  EXPECT_EQ(sentence_->token(2).end(), 16);
}

TEST_F(SegmentationTransitionTest, DefaultActionTest) {
  BinarySegmentState *segment_state = static_cast<BinarySegmentState *>(
      transition_system_->NewTransitionState(true));
  ParserState state(sentence_.get(), segment_state, &label_map_);

  // Do segmentation, tagging and parsing by following the gold actions.
  while (transition_system_->IsFinalState(state) == false) {
    ParserAction action = transition_system_->GetDefaultAction(state);
    transition_system_->PerformActionWithoutHistory(action, &state);
  }

  // Every character should be START.
  CheckStarts(state, {6, 5, 4, 3, 2, 1, 0});

  // Every non-space character should be a word.
  segment_state->AddParseToDocument(state, false, sentence_.get());
  ASSERT_EQ(sentence_->token_size(), 5);
  EXPECT_EQ(sentence_->token(0).word(), "因");
  EXPECT_EQ(sentence_->token(1).word(), "为");
  EXPECT_EQ(sentence_->token(2).word(), "有");
  EXPECT_EQ(sentence_->token(3).word(), "这");
  EXPECT_EQ(sentence_->token(4).word(), "样");
}

TEST_F(SegmentationTransitionTest, LastWordFeatureTest) {
  const int unk_id = word_map_.Size();
  const int outside_id = unk_id + 1;

  // Prepare a parser state.
  BinarySegmentState *segment_state = new BinarySegmentState();
  auto state = std::unique_ptr<ParserState>(new ParserState(
      sentence_.get(), segment_state, &label_map_));

  // Test initial state which contains no words.
  PrepareFeature("last-word(1,min-freq=2)", state.get());
  EXPECT_EQ(outside_id, ComputeFeature(*state));
  PrepareFeature("last-word(2,min-freq=2)", state.get());
  EXPECT_EQ(outside_id, ComputeFeature(*state));
  PrepareFeature("last-word(3,min-freq=2)", state.get());
  EXPECT_EQ(outside_id, ComputeFeature(*state));

  // Test when the state contains only one start.
  segment_state->AddStart(0, state.get());
  PrepareFeature("last-word(1,min-freq=2)", state.get());
  EXPECT_EQ(outside_id, ComputeFeature(*state));
  PrepareFeature("last-word(2,min-freq=2)", state.get());
  EXPECT_EQ(outside_id, ComputeFeature(*state));

  // Test when the state contains two starts, which forms a complete word and
  // the start of another new word.
  segment_state->AddStart(2, state.get());
  EXPECT_NE(word_map_.LookupIndex("因为", unk_id), unk_id);
  PrepareFeature("last-word(1)", state.get());
  EXPECT_EQ(word_map_.LookupIndex("因为", unk_id), ComputeFeature(*state));

  // The last-word still points to outside.
  PrepareFeature("last-word(2,min-freq=2)", state.get());
  EXPECT_EQ(outside_id, ComputeFeature(*state));

  // Adding more starts that leads to the following words:
  // 因为 ‘ ’ 有 ‘ ’
  segment_state->AddStart(3, state.get());
  segment_state->AddStart(4, state.get());

  // Note 有 is pruned from the map since its frequency is less than 2.
  EXPECT_EQ(word_map_.LookupIndex("有", unk_id), unk_id);
  PrepareFeature("last-word(1,min-freq=2)", state.get());
  EXPECT_EQ(unk_id, ComputeFeature(*state));

  // Note that last-word(2) points to ' ' which is also a unk.
  PrepareFeature("last-word(2,min-freq=2)", state.get());
  EXPECT_EQ(unk_id, ComputeFeature(*state));
  PrepareFeature("last-word(3,min-freq=2)", state.get());
  EXPECT_EQ(word_map_.LookupIndex("因为", unk_id), ComputeFeature(*state));

  // Adding two words: "这" and "样".
  segment_state->AddStart(5, state.get());
  segment_state->AddStart(6, state.get());
  PrepareFeature("last-word(1,min-freq=2)", state.get());
  EXPECT_EQ(word_map_.LookupIndex("这", unk_id), ComputeFeature(*state));
}

}  // namespace syntaxnet
