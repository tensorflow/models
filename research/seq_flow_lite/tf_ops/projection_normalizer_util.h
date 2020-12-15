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
#ifndef TENSORFLOW_MODELS_SEQUENCE_PROJECTION_TF_OPS_PROJECTION_NORMALIZER_UTIL_H_
#define TENSORFLOW_MODELS_SEQUENCE_PROJECTION_TF_OPS_PROJECTION_NORMALIZER_UTIL_H_

#include <string>
#include <unordered_set>
#include <vector>

#include "libutf/utf.h"

// Normalizes the input with the given |separators| by adding a space before and
// after each separator. When |normalize_repetition| is true, it removes the
// repeated characters (except numbers) which consecutively appeared more than
// twice in a word.
// Examples: arwwwww -> arww, good!!!!! -> good!!, hahaha => haha.
class ProjectionNormalizer {
 public:
  explicit ProjectionNormalizer(const std::string& separators,
                                bool normalize_repetition = false) {
    InitializeSeparators(separators);
    normalize_repetition_ = normalize_repetition;
  }

  // Normalizes the repeated characters (except numbers) which consecutively
  // appeared more than twice in a word.
  std::string Normalize(const std::string& input, size_t max_input = 300);
  std::string Normalize(const char* input_ptr, size_t len,
                        size_t max_input = 300);

 private:
  // Parses and extracts supported separators.
  void InitializeSeparators(const std::string& separators);

  // Removes repeated chars.
  std::string NormalizeInternal(const char* input_ptr, size_t len);

  std::unordered_set<char> separators_;
  bool normalize_repetition_;
};

#endif  // TENSORFLOW_MODELS_SEQUENCE_PROJECTION_TF_OPS_PROJECTION_NORMALIZER_UTIL_H_
