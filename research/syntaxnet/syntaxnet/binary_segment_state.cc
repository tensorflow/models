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

#include <string>
#include "syntaxnet/segmenter_utils.h"
#include "syntaxnet/sentence.pb.h"

namespace syntaxnet {

ParserTransitionState *BinarySegmentState::Clone() const {
  return new BinarySegmentState();
}

string BinarySegmentState::ToString(const ParserState &state) const {
  string str("[");
  for (int i = NumStarts(state) - 1; i >=0; --i) {
    int start = LastStart(i, state);
    int end = 0;
    if (i - 1 >= 0) {
      end = LastStart(i - 1, state) - 1;
    } else if (state.EndOfInput()) {
      end = state.sentence().token_size() - 1;
    } else {
      end = state.Next() - 1;
    }
    for (int k = start; k <= end; ++k) {
      str.append(state.GetToken(k).word());
    }
    if (i >= 1) str.append(" ");
  }

  str.append("] ");
  for (int i = state.Next(); i < state.NumTokens(); ++i) {
    str.append(state.GetToken(i).word());
  }
  return str;
}

void BinarySegmentState::AddParseToDocument(const ParserState &state,
                                            bool rewrite_root_labels,
                                            Sentence *sentence) const {
  if (sentence->token_size() == 0) return;
  std::vector<bool> is_starts(sentence->token_size(), false);
  for (int i = 0; i < NumStarts(state); ++i) {
    is_starts[LastStart(i, state)] = true;
  }

  // Break level of the current token is determined based on its previous token.
  Token::BreakLevel break_level = Token::NO_BREAK;
  bool is_first_token = true;
  Sentence new_sentence;
  for (int i = 0; i < sentence->token_size(); ++i) {
    const Token &token = sentence->token(i);
    const string &word = token.word();
    bool is_break = SegmenterUtils::IsBreakChar(word);
    if (is_starts[i] || is_first_token) {
      if (!is_break) {
        // The current character is the first char of a new token/word.
        Token *new_token = new_sentence.add_token();
        new_token->set_start(token.start());
        new_token->set_end(token.end());
        new_token->set_word(word);

        // For the first token, keep the old break level to make sure that the
        // number of sentences stays unchanged.
        new_token->set_break_level(break_level);
        is_first_token = false;
      }
    } else {
      // Append the character to the previous token.
      if (!is_break) {
        int index = new_sentence.token_size() - 1;
        auto *last_token = new_sentence.mutable_token(index);
        last_token->mutable_word()->append(word);
        last_token->set_end(token.end());
      }
    }

    // Update break level. Note we do not introduce new sentences in the
    // transition system, thus anything goes beyond line break would be reduced
    // to line break.
    break_level = is_break ? SegmenterUtils::BreakLevel(word) : Token::NO_BREAK;
    if (break_level >= Token::LINE_BREAK) break_level = Token::LINE_BREAK;
  }
  sentence->mutable_token()->Swap(new_sentence.mutable_token());
}

}  // namespace syntaxnet
