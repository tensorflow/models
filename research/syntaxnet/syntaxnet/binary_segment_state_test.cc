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

#include <memory>

#include "syntaxnet/base.h"
#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/term_frequency_map.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {

class BinarySegmentStateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Prepare a sentence.
    const char *str_sentence = "text: '测试 的 句子' "
        "token { word: '测' start: 0 end: 2 } "
        "token { word: '试' start: 3 end: 5 } "
        "token { word: ' ' start: 6 end: 6 } "
        "token { word: '的' start: 7 end: 9 } "
        "token { word: ' ' start: 10 end: 10 } "
        "token { word: '句' start: 11 end: 13 } "
        "token { word: '子' start: 14 end: 16 } ";
    sentence_ = std::unique_ptr<Sentence>(new Sentence());
    TextFormat::ParseFromString(str_sentence, sentence_.get());
  }

  // The test document, parse tree, and sentence.
  std::unique_ptr<Sentence> sentence_;
  TermFrequencyMap label_map_;
};

TEST_F(BinarySegmentStateTest, AddStartLastStartNumStartsTest) {
  BinarySegmentState *segment_state = new BinarySegmentState();
  ParserState state(sentence_.get(), segment_state, &label_map_);

  // Test segment_state initialized with zero starts.
  EXPECT_EQ(0, segment_state->NumStarts(state));

  // Adding the first token as a start token.
  segment_state->AddStart(0, &state);
  ASSERT_EQ(1, segment_state->NumStarts(state));
  EXPECT_EQ(0, segment_state->LastStart(0, state));

  // Adding more starts.
  segment_state->AddStart(2, &state);
  segment_state->AddStart(3, &state);
  segment_state->AddStart(4, &state);
  segment_state->AddStart(5, &state);
  ASSERT_EQ(5, segment_state->NumStarts(state));
  EXPECT_EQ(5, segment_state->LastStart(0, state));
  EXPECT_EQ(4, segment_state->LastStart(1, state));
  EXPECT_EQ(3, segment_state->LastStart(2, state));
  EXPECT_EQ(2, segment_state->LastStart(3, state));
  EXPECT_EQ(0, segment_state->LastStart(4, state));
}

TEST_F(BinarySegmentStateTest, AddParseToDocumentTest) {
  BinarySegmentState *segment_state = new BinarySegmentState();
  ParserState state(sentence_.get(), segment_state, &label_map_);

  // Test gold segmentation.
  // 0   1   2    3   4   5   6
  // 测  试  ' '  的  ' '  句  子
  // S   M   S    S   S   S   M
  segment_state->AddStart(0, &state);
  segment_state->AddStart(2, &state);
  segment_state->AddStart(3, &state);
  segment_state->AddStart(4, &state);
  segment_state->AddStart(5, &state);
  Sentence sentence_with_annotation = *sentence_;
  segment_state->AddParseToDocument(state, false, &sentence_with_annotation);

  // Test the number of tokens as well as the start/end byte-offsets of each
  // token.
  ASSERT_EQ(3, sentence_with_annotation.token_size());

  // The first token is 测试.
  EXPECT_EQ(0, sentence_with_annotation.token(0).start());
  EXPECT_EQ(5, sentence_with_annotation.token(0).end());

  // The second token is 的.
  EXPECT_EQ(7, sentence_with_annotation.token(1).start());
  EXPECT_EQ(9, sentence_with_annotation.token(1).end());

  // The third token is 句子.
  EXPECT_EQ(11, sentence_with_annotation.token(2).start());
  EXPECT_EQ(16, sentence_with_annotation.token(2).end());

  // Test merge space to other tokens. Since spaces, or more generally break
  // characters, should never be a part of any word, they are skipped no matter
  // how they are tagged.
  // 0   1   2    3   4   5   6
  // 测  试  ' '  的  ' '  句  子
  // S   M   M    S   M   M   M
  while (!state.StackEmpty()) state.Pop();
  segment_state->AddStart(0, &state);
  segment_state->AddStart(3, &state);
  sentence_with_annotation = *sentence_;
  segment_state->AddParseToDocument(state, false, &sentence_with_annotation);

  ASSERT_EQ(2, sentence_with_annotation.token_size());

  // The first token is 测试. Note even a space is tagged as "merge", it is not
  // attached to its previous word.
  EXPECT_EQ(0, sentence_with_annotation.token(0).start());
  EXPECT_EQ(5, sentence_with_annotation.token(0).end());

  // The second token is 的句子.
  EXPECT_EQ(7, sentence_with_annotation.token(1).start());
  EXPECT_EQ(16, sentence_with_annotation.token(1).end());

  // Test merge a token to space tokens. In such case, the current token would
  // be merged to the first non-space token on its left side.
  // 0   1   2    3   4   5   6
  // 测  试  ' '  的  ' '  句  子
  // S   M   S    M   S   M   M
  while (!state.StackEmpty()) state.Pop();
  segment_state->AddStart(0, &state);
  segment_state->AddStart(2, &state);
  segment_state->AddStart(4, &state);
  sentence_with_annotation = *sentence_;
  segment_state->AddParseToDocument(state, false, &sentence_with_annotation);
  ASSERT_EQ(1, sentence_with_annotation.token_size());
  EXPECT_EQ(0, sentence_with_annotation.token(0).start());
  EXPECT_EQ(16, sentence_with_annotation.token(0).end());
}

TEST_F(BinarySegmentStateTest, SpaceDocumentTest) {
  const char *str_sentence = "text: ' \t\t' "
      "token { word: ' ' start: 0 end: 0 } "
      "token { word: '\t' start: 1 end: 1 } "
      "token { word: '\t' start: 2 end: 2 } ";
  TextFormat::ParseFromString(str_sentence, sentence_.get());
  BinarySegmentState *segment_state = new BinarySegmentState();
  ParserState state(sentence_.get(), segment_state, &label_map_);

  // Break-chars should always be skipped, no matter how they are tagged.
  // 0    1     2
  //' '   '\t'  '\t'
  // M    M     M
  Sentence sentence_with_annotation = *sentence_;
  segment_state->AddParseToDocument(state, false, &sentence_with_annotation);
  ASSERT_EQ(0, sentence_with_annotation.token_size());

  // 0    1     2
  //' '   '\t'  '\t'
  // S    S     S
  segment_state->AddStart(0, &state);
  segment_state->AddStart(1, &state);
  segment_state->AddStart(2, &state);
  sentence_with_annotation = *sentence_;
  segment_state->AddParseToDocument(state, false, &sentence_with_annotation);
  ASSERT_EQ(0, sentence_with_annotation.token_size());
}

TEST_F(BinarySegmentStateTest, DocumentBeginWithSpaceTest) {
  const char *str_sentence = "text: ' 空格' "
      "token { word: ' ' start: 0 end: 0 } "
      "token { word: '空' start: 1 end: 3 } "
      "token { word: '格' start: 4 end: 6 } ";
  TextFormat::ParseFromString(str_sentence, sentence_.get());
  BinarySegmentState *segment_state = new BinarySegmentState();
  ParserState state(sentence_.get(), segment_state, &label_map_);

  // 0    1    2
  //' '   空   格
  // M    M    M
  Sentence sentence_with_annotation = *sentence_;
  segment_state->AddParseToDocument(state, false, &sentence_with_annotation);

  ASSERT_EQ(1, sentence_with_annotation.token_size());

  // The first token is 空格.
  EXPECT_EQ(1, sentence_with_annotation.token(0).start());
  EXPECT_EQ(6, sentence_with_annotation.token(0).end());

  // 0    1    2
  //' '   空   格
  // S    M    M
  while (!state.StackEmpty()) state.Pop();
  segment_state->AddStart(0, &state);
  sentence_with_annotation = *sentence_;
  segment_state->AddParseToDocument(state, false, &sentence_with_annotation);

  ASSERT_EQ(1, sentence_with_annotation.token_size());

  // The first token is 空格.
  EXPECT_EQ(1, sentence_with_annotation.token(0).start());
  EXPECT_EQ(6, sentence_with_annotation.token(0).end());
}

TEST_F(BinarySegmentStateTest, EmptyDocumentTest) {
  const char *str_sentence = "text: '' ";
  TextFormat::ParseFromString(str_sentence, sentence_.get());
  BinarySegmentState *segment_state = new BinarySegmentState();
  ParserState state(sentence_.get(), segment_state, &label_map_);
  Sentence sentence_with_annotation = *sentence_;
  segment_state->AddParseToDocument(state, false, &sentence_with_annotation);
  ASSERT_EQ(0, sentence_with_annotation.token_size());
}

}  // namespace syntaxnet
