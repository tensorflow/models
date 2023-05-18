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
#include "tf_ops/skipgram_finder.h"  // seq_flow_lite

#include <cctype>
#include <deque>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "icu4c/source/common/unicode/uchar.h"
#include "icu4c/source/common/unicode/utf8.h"

namespace seq_flow_lite {
namespace {

void PreprocessToken(std::string& token) {
  char* s = const_cast<char*>(token.data());
  int32_t size = token.size();
  int32_t in = 0;
  int32_t out = 0;
  while (in < size) {
    UChar32 c;
    int32_t old_in = in;
    U8_NEXT(s, in, size, c);
    if (c < 0) {
      break;
    }
    if (u_ispunct(c)) continue;
    UChar32 cl = u_tolower(c);
    // This is a hack, but there are exactly two unicode characters whose
    // lowercase versions have longer UTF-8 encodings (0x23a to 0x2c65,
    // 0x23e to 0x2c66).  So, to avoid sizing issues, they're not lowercased.
    if (U8_LENGTH(cl) > (in - old_in)) {
      cl = c;
    }
    U8_APPEND_UNSAFE(s, out, cl);
  }

  size_t remaining = token.size() - in;
  if (remaining > 0) {
    memmove(s + out, s + in, remaining);
    out += remaining;
  }
  token.resize(out);
}

}  // namespace

void SkipgramFinder::AddSkipgram(absl::string_view skipgram, int category) {
  std::vector<std::string> tokens = absl::StrSplit(skipgram, ' ');

  // Store the skipgram in a trie-like structure that uses tokens as the
  // edge labels, instead of characters.  Each node represents a skipgram made
  // from the tokens used to reach the node, and stores the categories the
  // skipgram is associated with.
  TrieNode* cur = &skipgram_trie_;
  for (auto& token : tokens) {
    if (absl::EndsWith(token, ".*")) {
      token.resize(token.size() - 2);
      PreprocessToken(token);
      auto iter = cur->prefix_to_node.find(token);
      if (iter != cur->prefix_to_node.end()) {
        cur = &iter->second;
      } else {
        cur = &cur->prefix_to_node
                   .emplace(std::piecewise_construct,
                            std::forward_as_tuple(token), std::make_tuple<>())
                   .first->second;
      }
      continue;
    }

    PreprocessToken(token);
    auto iter = cur->token_to_node.find(token);
    if (iter != cur->token_to_node.end()) {
      cur = &iter->second;
    } else {
      cur = &cur->token_to_node
                 .emplace(std::piecewise_construct,
                          std::forward_as_tuple(token), std::make_tuple<>())
                 .first->second;
    }
  }
  cur->categories.insert(category);
}

absl::flat_hash_set<int> SkipgramFinder::FindSkipgrams(
    absl::string_view input) const {
  std::vector<std::string> tokens = absl::StrSplit(input, ' ');
  std::vector<absl::string_view> sv_tokens;
  sv_tokens.reserve(tokens.size());
  for (auto& token : tokens) {
    PreprocessToken(token);
    sv_tokens.emplace_back(token.data(), token.size());
  }
  return FindSkipgrams(sv_tokens);
}

absl::flat_hash_set<int> SkipgramFinder::FindSkipgrams(
    const std::vector<absl::string_view>& tokens) const {
  absl::flat_hash_set<int> categories;

  // Tracks skipgram prefixes and the index of their last token.
  std::deque<std::pair<int, const TrieNode*>> indices_and_skipgrams;

  for (int token_i = 0; token_i < tokens.size(); token_i++) {
    const absl::string_view& token = tokens[token_i];

    std::vector<absl::string_view> token_prefixes;
    {
      const char* s = token.data();
      int32_t l = token.size();
      int32_t n = 0;
      while (n < l) {
        int32_t n_old = n;
        U8_FWD_1(s, n, l);
        if (n == n_old) break;
        token_prefixes.emplace_back(s, n);
      }
    }

    // Drop any skipgrams prefixes which would skip more than `max_skip_size_`
    // tokens between the end of the prefix and the current token.
    while (!indices_and_skipgrams.empty()) {
      if (indices_and_skipgrams.front().first + max_skip_size_ + 1 < token_i) {
        indices_and_skipgrams.pop_front();
      } else {
        break;
      }
    }

    // Check if we can form a valid skipgram prefix (or skipgram) by adding
    // the current token to any of the existing skipgram prefixes, or
    // if the current token is a valid skipgram prefix (or skipgram).
    size_t size = indices_and_skipgrams.size();
    for (size_t skipgram_i = 0; skipgram_i <= size; skipgram_i++) {
      const auto& node = skipgram_i < size
                             ? *indices_and_skipgrams[skipgram_i].second
                             : skipgram_trie_;

      auto iter = node.token_to_node.find(token);
      if (iter != node.token_to_node.end()) {
        categories.insert(iter->second.categories.begin(),
                          iter->second.categories.end());
        indices_and_skipgrams.push_back(std::make_pair(token_i, &iter->second));
      }

      for (auto token_prefix = token_prefixes.rbegin();
           token_prefix != token_prefixes.rend(); token_prefix++) {
        auto iter = node.prefix_to_node.find(*token_prefix);
        if (iter != node.prefix_to_node.end()) {
          categories.insert(iter->second.categories.begin(),
                            iter->second.categories.end());
          indices_and_skipgrams.push_back(
              std::make_pair(token_i, &iter->second));
        }
      }
    }
  }

  return categories;
}

}  // namespace seq_flow_lite
