/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tf_ops/subsequence_finder.h"  // seq_flow_lite

#include <deque>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "icu4c/source/common/unicode/uchar.h"
#include "icu4c/source/common/unicode/utf8.h"

namespace seq_flow_lite {

void SubsequenceFinder::AddSubsequence(absl::string_view subsequence,
                                       int category) {
  const char* s = subsequence.data();
  int32_t length = subsequence.length();
  int32_t n = 0;
  TrieNode* trie = &subsequence_trie_;
  bool new_word = true;
  while (n < length) {
    UChar32 c;
    U8_NEXT(s, n, length, c);

    if (c < 0) return;

    c = u_tolower(c);
    if (c == ' ') {
      new_word = true;
    } else if (!new_word) {
      trie = &trie->continue_token[c];
    } else {
      trie = &trie->next_token[c];
      new_word = false;
    }
  }
  trie->categories.insert(category);
}

// Given a UChar32 and a trie node representing an in-progress subsequence,
// determine if we can use the UChar32 to continue the subsequence, and
// update `categories`, `next_tokens`, and `continue_tokens` if needed.
void SubsequenceFinder::ProcessUChar32AndTrieNode(
    int index, UChar32 c,
    const absl::flat_hash_map<UChar32, TrieNode>& token_map,
    absl::flat_hash_set<int>* categories,
    std::deque<std::pair<int, const TrieNode*>>* next_tokens,
    std::vector<const TrieNode*>* continue_tokens) const {
  auto iter = token_map.find(c);
  if (iter != token_map.end()) {
    categories->insert(iter->second.categories.begin(),
                       iter->second.categories.end());
    if (!iter->second.continue_token.empty()) {
      continue_tokens->push_back(&iter->second);
    }
    if (!iter->second.next_token.empty()) {
      next_tokens->emplace_back(index, &iter->second);
    }
  }
}

absl::flat_hash_set<int> SubsequenceFinder::FindSubsequences(
    absl::string_view input) const {
  absl::flat_hash_set<int> categories;

  // Tracks subsequences in progress that are starting the next token,
  // as well as the index of their last character.
  std::deque<std::pair<int, const TrieNode*>> next_tokens;

  // Tracks subsequences in progress that are looking for the next character
  // in their corrent token.  `current_continue_tokens` is the current set of
  // subsequences being processed, while `future_continue_tokens` is the set
  // of subsequences to process for the next character.
  std::vector<const TrieNode*> current_continue_tokens;
  std::vector<const TrieNode*> future_continue_tokens;

  const char* s = input.data();
  int32_t length = input.length();
  int32_t n = 0;
  int index = 0;
  while (n < length) {
    UChar32 c;
    U8_NEXT(s, n, length, c);

    if (c < 0) return categories;

    c = u_tolower(c);

    // Drop any subsequences which would need to skip more than `max_skip_size_`
    // characters between the end of their last token and the current character.
    while (!next_tokens.empty()) {
      if (next_tokens.front().first + max_skip_size_ + 1 < index) {
        next_tokens.pop_front();
      } else {
        break;
      }
    }

    // Check subsequences starting a new token.
    size_t size = next_tokens.size();
    for (size_t i = 0; i < size; i++) {
      ProcessUChar32AndTrieNode(index, c, next_tokens[i].second->next_token,
                                &categories, &next_tokens,
                                &future_continue_tokens);
    }

    // Check subsequences continuing a token.
    for (const TrieNode* continue_token : current_continue_tokens) {
      ProcessUChar32AndTrieNode(index, c, continue_token->continue_token,
                                &categories, &next_tokens,
                                &future_continue_tokens);
    }

    // Check if we can start a new subsequence.
    ProcessUChar32AndTrieNode(index, c, subsequence_trie_.next_token,
                              &categories, &next_tokens,
                              &future_continue_tokens);

    current_continue_tokens.swap(future_continue_tokens);
    future_continue_tokens.clear();

    index++;
  }

  return categories;
}

}  // namespace seq_flow_lite
