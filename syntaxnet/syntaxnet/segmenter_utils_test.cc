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

// Creates a Korean senence and also initializes the token field.
static Sentence GetKoSentence() {
  Sentence sentence;

  string text = "서울시는 2012년부터";

  // Add tokens.
  sentence.set_text(text);
  Token *tok = sentence.add_token();
  tok->set_word("서울시");
  tok->set_start(0);
  tok->set_end(8);
  tok = sentence.add_token();
  tok->set_word("는");
  tok->set_start(9);
  tok->set_end(11);
  tok = sentence.add_token();
  tok->set_word("2012");
  tok->set_start(13);
  tok->set_end(16);
  tok = sentence.add_token();
  tok->set_word("년");
  tok->set_start(17);
  tok->set_end(19);
  tok = sentence.add_token();
  tok->set_word("부터");
  tok->set_start(20);
  tok->set_end(25);

  return sentence;
}

// Gets the start end bytes of the given chars in the given text.
static void GetStartEndBytes(const string &text,
                             const vector<tensorflow::StringPiece> &chars,
                             vector<int> *starts,
                             vector<int> *ends) {
  SegmenterUtils segment_utils;
  for (const tensorflow::StringPiece &c : chars) {
    int start; int end;
    segment_utils.GetCharStartEndBytes(text, c, &start, &end);
    starts->push_back(start);
    ends->push_back(end);
  }
}

// Test the GetChars function.
TEST(SegmenterUtilsTest, GetCharsTest) {
  // Create test sentence.
  const Sentence sentence = GetKoSentence();
  vector<tensorflow::StringPiece> chars;
  SegmenterUtils::GetUTF8Chars(sentence.text(), &chars);

  // Check the number of characters is correct.
  CHECK_EQ(chars.size(), 12);

  vector<int> starts;
  vector<int> ends;
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
  vector<tensorflow::StringPiece> chars;
  SegmenterUtils segment_utils;
  segment_utils.GetUTF8Chars(sentence.text(), &chars);

  vector<int> starts;
  vector<int> ends;
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
