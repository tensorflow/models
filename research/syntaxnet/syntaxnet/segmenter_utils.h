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

#ifndef SYNTAXNET_SEGMENTER_UTILS_H_
#define SYNTAXNET_SEGMENTER_UTILS_H_

#include <string>
#include <unordered_set>
#include <vector>

#include "syntaxnet/sentence.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "util/utf8/unicodetext.h"

namespace syntaxnet {

// A set of common convenience functions.
class SegmenterUtils {
 public:
  // Takes a text and convert it into a vector, where each element is a utf8
  // character.
  static void GetUTF8Chars(const string &text,
                           std::vector<tensorflow::StringPiece> *chars);

  // Sets tokens in the sentence so that each token is a single character.
  // Assigns the start/end byte offsets.
  //
  // If the sentence is not empty, the current tokens will be cleared.
  static void SetCharsAsTokens(
      const string &text, const std::vector<tensorflow::StringPiece> &chars,
      Sentence *sentence);

  // Takes a sentence with its original text and gold tokens and outputs a new
  // sentence with the original text, but have a single utf8 character per token
  // and a break level that is set to:
  //   0 (NO_BREAK) iff in the gold tokenization there was no break from the
  //                    last token and merge with previous token.
  //   1 (SPACE_BREAK) iff in the gold tokenization there was a break from the
  //                       last token. SPACE_BREAK represents all breaks.
  // Returns true if sentence token start/end bytes are consistent with UTF-8
  // characters.
  static bool ConvertToCharTokenDoc(const Sentence &sentence,
                                    Sentence *char_sentence);

  // Returns true if the start/end byte offsets of a sentence's tokens are
  // consistent with UTF8 character boundaries.
  // Note: it must be the case that chars was constructed from sentence.text().
  static bool DocTokensUTF8Consistent(
      const std::vector<tensorflow::StringPiece> &chars,
      const Sentence &document);

  // Returns true for UTF-8 characters that cannot be 'real' tokens. This is
  // defined as any whitespace, line break or paragraph break.
  static bool IsBreakChar(const string &word) {
    if (word == "\n" || word == "\t") return true;
    UnicodeText text;
    text.PointToUTF8(word.c_str(), word.length());
    CHECK_EQ(text.size(), 1);
    return kBreakChars.find(*text.begin()) != kBreakChars.end();
  }

  // Returns the break level for the next token based on the current character.
  static Token::BreakLevel BreakLevel(const string &word) {
    UnicodeText text;
    text.PointToUTF8(word.c_str(), word.length());
    auto point = *text.begin();
    if (word == "\n" || point == kLineSeparator) {
      return Token::LINE_BREAK;
    } else if (point == kParagraphSeparator) {
      return Token::SENTENCE_BREAK;  // No PARAGRAPH_BREAK in sentence proto.
    } else if (word == "\t" || kBreakChars.find(point) != kBreakChars.end()) {
      return Token::SPACE_BREAK;
    }
    return Token::NO_BREAK;
  }

  // Convenience function for computing start/end byte offsets of a character
  // StringPiece relative to original text.
  static void GetCharStartEndBytes(const string &text,
                                   tensorflow::StringPiece c,
                                   int *start,
                                   int *end) {
    *start = c.data() - text.data();
    *end = *start + c.size() - 1;
  }

  // Returns true if this segment is a valid segment. Currently checks:
  // 1) It is non-empty
  // 2) It is valid UTF8
  static bool IsValidSegment(const Sentence &sentence, const Token &token);

  // Set for utf8 break characters.
  static const std::unordered_set<int> kBreakChars;
  static const int kLineSeparator = 0x2028;
  static const int kParagraphSeparator = 0x2029;
};

}  // namespace syntaxnet

#endif  // SYNTAXNET_SEGMENTER_UTILS_H_
