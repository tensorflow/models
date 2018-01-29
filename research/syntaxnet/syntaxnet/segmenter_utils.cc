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
#include "util/utf8/unicodetext.h"
#include "util/utf8/unilib.h"
#include "util/utf8/unilib_utf8_utils.h"

namespace syntaxnet {

// Separators, code Zs from http://www.unicode.org/Public/UNIDATA/PropList.txt
// NB: This list is not necessarily exhaustive.
const std::unordered_set<int> SegmenterUtils::kBreakChars({
  0x2028,  // line separator
  0x2029,  // paragraph separator
  0x0020,  // space
  0x00a0,  // no-break space
  0x1680,  // Ogham space mark
  0x180e,  // Mongolian vowel separator
  0x202f,  // narrow no-break space
  0x205f,  // medium mathematical space
  0x3000,  // ideographic space
  0xe5e5,  // Google addition
  0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008,
  0x2009, 0x200a
});

bool SegmenterUtils::ConvertToCharTokenDoc(const Sentence &sentence,
                                           Sentence *char_sentence) {
  CHECK(char_sentence);

  // Extracts tokens and byte offsets.
  std::vector<tensorflow::StringPiece> orig_chars;
  GetUTF8Chars(sentence.text(), &orig_chars);

  // If sentence's token start/end bytes are not consistent with UTF-8
  // characters, then do not process the sentence.
  if (!DocTokensUTF8Consistent(orig_chars, sentence)) {
    LOG(WARNING) << "Document token boundaries not UTF8 consistent.";
    return false;
  }

  // Create a mapping from text byte to token index or -1.
  // token_ids[i] = index of the token byte i is contained in and -1 o.w.
  std::vector<int> token_ids;
  for (int t = 0; t < sentence.token_size(); ++t) {
    const Token &token = sentence.token(t);
    while (token_ids.size() < token.start()) token_ids.push_back(-1);
    while (token_ids.size() <= token.end()) token_ids.push_back(t);
  }
  while (token_ids.size() < sentence.text().size()) token_ids.push_back(-1);

  // Infer SPLIT/MERGE for each UTF-8 char.
  std::vector<Token::BreakLevel> break_levels;
  break_levels.push_back(Token::SPACE_BREAK);  // first token is always a split.
  for (int c = 1; c < orig_chars.size(); ++c) {
    int char_start, char_end;
    GetCharStartEndBytes(sentence.text(), orig_chars[c], &char_start,
                         &char_end);
    int prev_char_start, prev_char_end;
    GetCharStartEndBytes(sentence.text(), orig_chars[c - 1], &prev_char_start,
                         &prev_char_end);

    // We split if this character is a break token (token_ids = -1) or if this
    // character is part of a different token from the previous.
    const bool is_split =
        token_ids[char_start] == -1 ||
        token_ids[char_start] != token_ids[prev_char_end];
    break_levels.push_back(is_split ? Token::SPACE_BREAK : Token::NO_BREAK);
  }

  // Initialize character sentence.
  SetCharsAsTokens(sentence.text(), orig_chars, char_sentence);
  CHECK_EQ(break_levels.size(), char_sentence->token_size());
  for (int i = 0 ; i < break_levels.size(); ++i) {
    char_sentence->mutable_token(i)->set_break_level(break_levels[i]);
  }
  return true;
}

bool SegmenterUtils::DocTokensUTF8Consistent(
    const std::vector<tensorflow::StringPiece> &chars,
    const Sentence &sentence) {
  std::set<int> starts;
  std::set<int> ends;
  for (const tensorflow::StringPiece c : chars) {
    int start_byte, end_byte;
    GetCharStartEndBytes(sentence.text(), c, &start_byte, &end_byte);
    starts.insert(start_byte);
    ends.insert(end_byte);
  }
  for (const Token &t : sentence.token()) {
    if (starts.find(t.start()) == starts.end()) return false;
    if (ends.find(t.end()) == ends.end()) return false;
  }
  return true;
}

void SegmenterUtils::GetUTF8Chars(const string &text,
                                  std::vector<tensorflow::StringPiece> *chars) {
  const char *start = text.c_str();
  const char *end = text.c_str() + text.size();
  while (start < end) {
    int char_length = UniLib::OneCharLen(start);
    chars->emplace_back(start, char_length);
    start += char_length;
  }
}

void SegmenterUtils::SetCharsAsTokens(
    const string &text,
    const std::vector<tensorflow::StringPiece> &chars,
    Sentence *sentence) {
  sentence->clear_token();
  sentence->set_text(text);
  for (int i = 0; i < chars.size(); ++i) {
    Token *tok = sentence->add_token();
    tok->set_word(chars[i].ToString());  // NOLINT
    int start_byte, end_byte;
    GetCharStartEndBytes(text, chars[i], &start_byte, &end_byte);
    tok->set_start(start_byte);
    tok->set_end(end_byte);
  }
}

bool SegmenterUtils::IsValidSegment(const Sentence &sentence,
                                    const Token &token) {
  // Check that the token is not empty, both by string and by bytes.
  if (token.word().empty()) return false;
  if (token.start() > token.end()) return false;

  // Check token boudaries inside of text.
  if (token.start() < 0) return false;
  if (token.end() >= sentence.text().size()) return false;

  // Check that token string is valid UTF8, by bytes.
  const char s = sentence.text()[token.start()];
  const char e = sentence.text()[token.end() + 1];
  if (UniLib::IsTrailByte(s)) return false;
  if (UniLib::IsTrailByte(e)) return false;
  return true;
}

}  // namespace syntaxnet
