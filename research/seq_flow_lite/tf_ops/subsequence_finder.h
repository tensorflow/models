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
#ifndef TENSORFLOW_MODELS_SEQ_FLOW_LITE_TF_OPS_SUBSEQUENCE_FINDER_H_
#define TENSORFLOW_MODELS_SEQ_FLOW_LITE_TF_OPS_SUBSEQUENCE_FINDER_H_

#include <deque>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "icu4c/source/common/unicode/uchar.h"

namespace seq_flow_lite {

// SubsequenceFinder finds subsequences in UTF-8 strings.
//
// Specifically, given a subsequence t_1 t_2 ... t_n, we will check if a
// string matches '.*t_1.{0,N}t_2.{0,N} ... .{0,N}t_n.*', where N is the
// maximum skip size.
//
// To use: First, add subsequences using AddSubsequence() - each subsequence
// is associated with some category.  Then call FindSubsequences() on a string,
// which will return the set of categories of the subsesequences in the string.
//
// The subsequences will be tokenized by splitting on spaces.  Both subsequences
// and input strings will be normalized by lowercasing.
class SubsequenceFinder {
 public:
  explicit SubsequenceFinder(int max_skip_size)
      : max_skip_size_(max_skip_size) {}

  // Adds a subsequence that SubsequenceFinder should look for in input strings.
  void AddSubsequence(absl::string_view subsequence, int category);

  // Find all of the subsequences in `input`, and return their categories.
  absl::flat_hash_set<int> FindSubsequences(absl::string_view input) const;

 private:
  // This trie tracks the next character needed to:
  // * continue the current token
  // * start the next token
  struct TrieNode {
    absl::flat_hash_set<int> categories;
    absl::flat_hash_map<UChar32, TrieNode> continue_token;
    absl::flat_hash_map<UChar32, TrieNode> next_token;
  };

  void ProcessUChar32AndTrieNode(
      int index, UChar32 c,
      const absl::flat_hash_map<UChar32, TrieNode>& token_map,
      absl::flat_hash_set<int>* categories,
      std::deque<std::pair<int, const TrieNode*>>* next_tokens,
      std::vector<const TrieNode*>* continue_tokens) const;

  TrieNode subsequence_trie_;

  int max_skip_size_;
};

}  // namespace seq_flow_lite

#endif  // TENSORFLOW_MODELS_SEQ_FLOW_LITE_TF_OPS_SUBSEQUENCE_FINDER_H_
