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
#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"
#include "syntaxnet/term_frequency_map.h"
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
  }

  void CheckStarts(const ParserState &state, const vector<int> &target) {
    ASSERT_EQ(state.StackSize(), target.size());
    vector<int> starts;
    for (int i = 0; i < state.StackSize(); ++i) {
      EXPECT_EQ(state.Stack(i), target[i]);
    }
  }

  // The test document, parse tree, and sentence with tags and partial parses.
  std::unique_ptr<Sentence> sentence_;
  std::unique_ptr<ParserTransitionSystem> transition_system_;
  TermFrequencyMap label_map_;
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

}  // namespace syntaxnet
