/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_MODELS_SEQUENCE_PROJECTION_TF_OPS_PROJECTION_TOKENIZER_UTIL_H_
#define TENSORFLOW_MODELS_SEQUENCE_PROJECTION_TF_OPS_PROJECTION_TOKENIZER_UTIL_H_

#include <string>
#include <unordered_set>
#include <vector>

#include "libutf/utf.h"

// Tokenizes the input with the given separators. To properly tokenize a text
// containing contractions in English (e.g. I'm), it combines the apostrophe
// with the token coming after it. For example, the text "I'm happy" is
// tokenized into three tokens: "I", "'m", "happy". When |separators| is not
// given, use the space to tokenize the input.
// Note) This tokenization supports only English.
class ProjectionTokenizer {
 public:
  explicit ProjectionTokenizer(const std::string& separators) {
    InitializeSeparators(separators);
  }

  // Tokenizes the input by separators_. Limit to max_tokens, when it is not -1.
  std::vector<std::string> Tokenize(const std::string& input, size_t max_input,
                                    size_t max_tokens) const {
    return Tokenize(input.c_str(), input.size(), max_input, max_tokens);
  }

  std::vector<std::string> Tokenize(const char* input_ptr, size_t len,
                                    size_t max_input, size_t max_tokens) const;

 private:
  // Parses and extracts supported separators.
  void InitializeSeparators(const std::string& separators);

  // Starting from input_ptr[from], search for the next occurrence of
  // separators_. Don't search beyond input_ptr[length](non-inclusive). Return
  // -1 if not found.
  size_t FindNextSeparator(const char* input_ptr, size_t from,
                           size_t length) const;

  std::unordered_set<char> separators_;
};

#endif  // TENSORFLOW_MODELS_SEQUENCE_PROJECTION_TF_OPS_PROJECTION_TOKENIZER_UTIL_H_
