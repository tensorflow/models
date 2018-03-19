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

#include "syntaxnet/segmenter_utils.h"

#include <string>
#include <vector>

#include "syntaxnet/char_properties.h"
#include "syntaxnet/sentence.pb.h"
#include <gmock/gmock.h>
#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {

namespace {

// Returns a Sentence proto constructed by parsing the given prototxt.
Sentence ParseASCII(const string &prototxt) {
  Sentence document;
  CHECK(TextFormat::ParseFromString(prototxt, &document));
  return document;
}

// Creates a Korean senence.
Sentence GetKoSentence() {
  return ParseASCII(
      "text: '서울시는 2012년부터' "
      "token { word: '서울시' start: 0 end: 8 } "
      "token { word: '는' start: 9 end: 11 } "
      "token { word: '2012' start: 13 end: 16 } "
      "token { word: '년' start: 17 end: 19 } "
      "token { word: '부터' start: 20 end: 25 } "
  );
}

Sentence GetUTF8InconsistentKoSentence1() {
  return ParseASCII(
      "text: '서울시는 2012년부터' "
      "token { word: '서울시' start: 0 end: 7 } "  // End should be 8.
      "token { word: '는' start: 9 end: 11 } "
      "token { word: '2012' start: 13 end: 16 } "
      "token { word: '년' start: 17 end: 19 } "
      "token { word: '부터' start: 20 end: 25 } "
  );
}

Sentence GetUTF8InconsistentKoSentence2() {
  return ParseASCII(
      "text: '서울시는  2012년부터' "  // Extra space in text field.
      "token { word: '서울시' start: 0 end: 8 } "
      "token { word: '는' start: 9 end: 11 } "
      "token { word: '2012' start: 13 end: 16 } "
      "token { word: '년' start: 17 end: 19 } "
      "token { word: '부터' start: 20 end: 25 } "
  );
}

// Gets the start end bytes of the given chars in the given text.
void GetStartEndBytes(const string &text,
                             const std::vector<tensorflow::StringPiece> &chars,
                             std::vector<int> *starts,
                             std::vector<int> *ends) {
  SegmenterUtils segment_utils;
  for (const tensorflow::StringPiece &c : chars) {
    int start; int end;
    segment_utils.GetCharStartEndBytes(text, c, &start, &end);
    starts->push_back(start);
    ends->push_back(end);
  }
}

}  // namespace

TEST(SegmenterUtilsTest, DocTokensUTF8ConsistentTest) {
  Sentence consistent_sentence = GetKoSentence();
  std::vector<tensorflow::StringPiece> chars;
  SegmenterUtils::GetUTF8Chars(consistent_sentence.text(), &chars);
  EXPECT_TRUE(SegmenterUtils::DocTokensUTF8Consistent(
      chars, consistent_sentence));

  // Test an inconsistent sentence.
  Sentence inconsistent_sentence = GetUTF8InconsistentKoSentence1();
  chars.clear();
  SegmenterUtils::GetUTF8Chars(inconsistent_sentence.text(), &chars);
  EXPECT_FALSE(SegmenterUtils::DocTokensUTF8Consistent(
      chars, inconsistent_sentence));

  // Test another inconsistent sentence.
  inconsistent_sentence = GetUTF8InconsistentKoSentence2();
  chars.clear();
  SegmenterUtils::GetUTF8Chars(inconsistent_sentence.text(), &chars);
  EXPECT_FALSE(SegmenterUtils::DocTokensUTF8Consistent(
      chars, inconsistent_sentence));
}

TEST(SegmenterUtilsTest, ConvertToCharTokenDocTest) {
  Sentence sentence = GetKoSentence();
  Sentence char_sentence;
  EXPECT_TRUE(SegmenterUtils::ConvertToCharTokenDoc(sentence, &char_sentence));
  std::vector<int> starts, ends;
  std::vector<Token::BreakLevel> breaks;
  for (const auto &token : char_sentence.token()) {
    starts.push_back(token.start());
    ends.push_back(token.end());
    breaks.push_back(token.break_level());
  }
  EXPECT_THAT(starts,
              testing::ContainerEq<std::vector<int>>(
                  {0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 20, 23}));
  EXPECT_THAT(ends,
              testing::ContainerEq<std::vector<int>>(
                  {2, 5, 8, 11, 12, 13, 14, 15, 16, 19, 22, 25}));
  EXPECT_THAT(breaks,
              testing::ContainerEq<std::vector<Token::BreakLevel>>(
                  {
                    // Token '서울시'
                    Token::SPACE_BREAK,
                    Token::NO_BREAK,
                    Token::NO_BREAK,

                    // Token '는'
                    Token::SPACE_BREAK,

                    // Token ' '
                    Token::SPACE_BREAK,

                    // Token '2012'
                    Token::SPACE_BREAK,
                    Token::NO_BREAK,
                    Token::NO_BREAK,
                    Token::NO_BREAK,

                    // Token '년'
                    Token::SPACE_BREAK,

                    // Token '부터'
                    Token::SPACE_BREAK,
                    Token::NO_BREAK
                   }));

  // Another test where tokens of the sentence is not consistent with the text
  // of the sentence.
  Sentence inconsistent_sentence = GetUTF8InconsistentKoSentence1();
  EXPECT_FALSE(SegmenterUtils::ConvertToCharTokenDoc(
      inconsistent_sentence, &char_sentence));
}

// Test the GetChars function.
TEST(SegmenterUtilsTest, GetCharsTest) {
  // Create test sentence.
  const Sentence sentence = GetKoSentence();
  std::vector<tensorflow::StringPiece> chars;
  SegmenterUtils::GetUTF8Chars(sentence.text(), &chars);

  // Check the number of characters is correct.
  CHECK_EQ(chars.size(), 12);

  std::vector<int> starts;
  std::vector<int> ends;
  GetStartEndBytes(sentence.text(), chars, &starts, &ends);

  // Check start positions.
  CHECK_EQ(starts[0], 0);
  CHECK_EQ(starts[1], 3);
  CHECK_EQ(starts[2], 6);
  CHECK_EQ(starts[3], 9);
  CHECK_EQ(starts[4], 12);
  CHECK_EQ(starts[5], 13);
  CHECK_EQ(starts[6], 14);
  CHECK_EQ(starts[7], 15);
  CHECK_EQ(starts[8], 16);
  CHECK_EQ(starts[9], 17);
  CHECK_EQ(starts[10], 20);
  CHECK_EQ(starts[11], 23);

  // Check end positions.
  CHECK_EQ(ends[0], 2);
  CHECK_EQ(ends[1], 5);
  CHECK_EQ(ends[2], 8);
  CHECK_EQ(ends[3], 11);
  CHECK_EQ(ends[4], 12);
  CHECK_EQ(ends[5], 13);
  CHECK_EQ(ends[6], 14);
  CHECK_EQ(ends[7], 15);
  CHECK_EQ(ends[8], 16);
  CHECK_EQ(ends[9], 19);
  CHECK_EQ(ends[10], 22);
  CHECK_EQ(ends[11], 25);
}

// Test the SetCharsAsTokens function.
TEST(SegmenterUtilsTest, SetCharsAsTokensTest) {
  // Create test sentence.
  const Sentence sentence = GetKoSentence();
  std::vector<tensorflow::StringPiece> chars;
  SegmenterUtils segment_utils;
  segment_utils.GetUTF8Chars(sentence.text(), &chars);

  std::vector<int> starts;
  std::vector<int> ends;
  GetStartEndBytes(sentence.text(), chars, &starts, &ends);

  // Check that the new docs word, start and end positions are properly set.
  Sentence new_sentence;
  segment_utils.SetCharsAsTokens(sentence.text(), chars, &new_sentence);
  CHECK_EQ(new_sentence.token_size(), chars.size());
  for (int t = 0; t < sentence.token_size(); ++t) {
    CHECK_EQ(new_sentence.token(t).word(), chars[t]);
    CHECK_EQ(new_sentence.token(t).start(), starts[t]);
    CHECK_EQ(new_sentence.token(t).end(), ends[t]);
  }

  // Re-running should remove the old tokens.
  segment_utils.SetCharsAsTokens(sentence.text(), chars, &new_sentence);
  CHECK_EQ(new_sentence.token_size(), chars.size());
  for (int t = 0; t < sentence.token_size(); ++t) {
    CHECK_EQ(new_sentence.token(t).word(), chars[t]);
    CHECK_EQ(new_sentence.token(t).start(), starts[t]);
    CHECK_EQ(new_sentence.token(t).end(), ends[t]);
  }
}

}  // namespace syntaxnet
