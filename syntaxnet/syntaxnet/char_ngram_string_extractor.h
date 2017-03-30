/* Copyright 2017 Google Inc. All Rights Reserved.

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

#ifndef SYNTAXNET_CHAR_NGRAM_STRING_EXTRACTOR_H_
#define SYNTAXNET_CHAR_NGRAM_STRING_EXTRACTOR_H_

#include <algorithm>
#include <string>
#include <vector>

#include "syntaxnet/segmenter_utils.h"
#include "syntaxnet/task_context.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {

// Extracts character n-grams from words.
class CharNgramStringExtractor {
 public:
  // Creates a default-configured extractor.
  CharNgramStringExtractor() = default;

  // Configures the extractor.
  void set_min_length(int length) { min_length_ = length; }
  void set_max_length(int length) { max_length_ = length; }
  void set_add_terminators(bool add) { add_terminators_ = add; }
  void set_mark_boundaries(bool mark) { mark_boundaries_ = mark; }

  // Returns a compact string that encodes the configuration of this extractor.
  string GetConfigId() const;

  // Applies any configuration settings defined in the |context|, overwriting
  // current settings.  The configuration setters above should not be called
  // afterwards.  Uses the following task parameters:
  //   int lexicon_min_char_ngram_length
  //   int lexicon_max_char_ngram_length
  //   bool lexicon_char_ngram_include_terminators
  void Setup(const TaskContext &context);

  // Calls the |consumer| on each character n-gram of the |word|.  If the word
  // contains space (' ') characters, n-grams that include the spaces will not
  // be extracted.
  template <class Consumer>
  void Extract(const string &word, Consumer consumer) const;

 private:
  // Minimum and maximum length of n-grams to extract.
  int min_length_ = 1;
  int max_length_ = 3;

  // Whether to pad the word with "^" and "$" before extracting n-grams.  Note
  // that the terminators count towards the n-gram length.  Incompatible with
  // |mark_boundaries_|.
  bool add_terminators_ = false;

  // Whether to mark the first and last characters with "^ " and " $" before
  // extracting n-grams.  Since space (' ') is otherwise forbidden in n-grams,
  // boundary markers are unambiguous.  Incompatible with |add_terminators_|.
  bool mark_boundaries_ = false;
};

template <class Consumer>
void CharNgramStringExtractor::Extract(const string &word,
                                       Consumer consumer) const {
  std::vector<tensorflow::StringPiece> char_sp;

  if (add_terminators_) char_sp.push_back("^");
  SegmenterUtils::GetUTF8Chars(word, &char_sp);
  if (add_terminators_) char_sp.push_back("$");

  const int num_chars = char_sp.size();
  string char_ngram;
  for (int start = 0; start < num_chars; ++start) {
    char_ngram.clear();
    if (mark_boundaries_ && start == 0) char_ngram = "^ ";  // mark first char
    const int max_length = std::min(max_length_, num_chars - start);
    for (int index = 0; index < max_length; ++index) {
      const int char_index = start + index;
      tensorflow::StringPiece curr = char_sp[char_index];
      if (curr == " ") break;  // Never add char ngrams containing spaces.
      tensorflow::strings::StrAppend(&char_ngram, curr);
      if (mark_boundaries_ && char_index + 1 == num_chars) {
        tensorflow::strings::StrAppend(&char_ngram, " $");  // mark last char
      }
      if (index + 1 >= min_length_) consumer(char_ngram);
    }
  }
}

}  // namespace syntaxnet

#endif  // SYNTAXNET_CHAR_NGRAM_STRING_EXTRACTOR_H_
