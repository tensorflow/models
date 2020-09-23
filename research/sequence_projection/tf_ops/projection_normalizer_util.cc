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
#include "tf_ops/projection_normalizer_util.h"  // sequence_projection

#include <algorithm>
#include <cstddef>
#include <memory>
#include <sstream>
#include <utility>

#include "tf_ops/projection_util.h"  // sequence_projection

// Returns true if the given text contains a number.
bool IsDigit(const std::string& text) {
  Rune rune;
  for (size_t i = 0; i < text.length();) {
    const int bytes_read = charntorune(&rune, text.data(), 1);
    if (rune == Runeerror || bytes_read == 0) break;
    if (rune >= static_cast<Rune>('0') && rune <= static_cast<Rune>('9')) {
      return true;
    }
    i += bytes_read;
  }
  return false;
}

// Gets the string containing |num_chars| characters from |start| position.
std::string GetCharToken(const std::vector<std::string>& char_tokens,
                         size_t start, size_t num_chars) {
  std::string char_token = "";
  if (start + num_chars <= char_tokens.size()) {
    for (size_t i = 0; i < num_chars; ++i) {
      char_token.append(char_tokens[start + i]);
    }
  }
  return char_token;
}

// Counts how many times |pattern| appeared from |start| position.
int GetNumPattern(const std::vector<std::string>& char_tokens, size_t start,
                  size_t num_chars, const std::string& pattern) {
  int count = 0;
  for (size_t i = start; i < char_tokens.size(); i += num_chars) {
    std::string cur_pattern = GetCharToken(char_tokens, i, num_chars);
    if (pattern == cur_pattern) {
      ++count;
    } else {
      break;
    }
  }
  return count;
}

std::string ContractToken(const char* input_ptr, size_t len, size_t num_chars) {
  // This function contracts patterns whose length is |num_chars| and appeared
  // more than twice. So if the input is shorter than 3 * |num_chars|, do not
  // apply any contraction.
  if (len < 3 * num_chars) {
    return input_ptr;
  }
  std::vector<std::string> char_tokens = SplitByChar(input_ptr, len, len);

  std::string token;
  token.reserve(len);
  for (size_t i = 0; i < char_tokens.size();) {
    std::string cur_pattern = GetCharToken(char_tokens, i, num_chars);

    // Count how many times this pattern appeared.
    int num_cur_patterns = 0;
    if (cur_pattern.find(" ") == std::string::npos && !IsDigit(cur_pattern)) {
      num_cur_patterns =
          GetNumPattern(char_tokens, i + num_chars, num_chars, cur_pattern);
    }

    if (num_cur_patterns >= 2) {
      // If this pattern is repeated, store it only twice.
      token.append(cur_pattern);
      token.append(cur_pattern);
      i += (num_cur_patterns + 1) * num_chars;
    } else {
      token.append(char_tokens[i]);
      ++i;
    }
  }

  return token;
}

void ProjectionNormalizer::InitializeSeparators(const std::string& separators) {
  for (size_t i = 0; i < separators.length(); ++i) {
    if (separators[i] != ' ') {
      separators_.insert(separators[i]);
    }
  }
}

std::string ProjectionNormalizer::NormalizeInternal(const char* input_ptr,
                                                    size_t len) {
  std::string normalized;
  normalized.reserve(len * 2);
  for (size_t i = 0; i < len; ++i) {
    char c = input_ptr[i];
    bool matched_separator = separators_.find(c) != separators_.end();
    if (matched_separator) {
      if (i > 0 && input_ptr[i - 1] != ' ' && normalized.back() != ' ') {
        normalized.append(" ");
      }
    }
    normalized.append(1, c);
    if (matched_separator) {
      if (i + 1 < len && input_ptr[i + 1] != ' ' && c != '\'') {
        normalized.append(" ");
      }
    }
  }
  return normalized;
}

std::string ProjectionNormalizer::Normalize(const std::string& input,
                                            size_t max_input) {
  return Normalize(input.c_str(), input.size(), max_input);
}

std::string ProjectionNormalizer::Normalize(const char* input_ptr, size_t len,
                                            size_t max_input) {
  std::string normalized(input_ptr, std::min(len, max_input));

  if (normalize_repetition_) {
    // Remove repeated 1 char (e.g. soooo => soo)
    normalized = ContractToken(normalized.data(), normalized.length(), 1);

    // Remove repeated 2 chars from the beginning (e.g. hahaha =>
    // haha, xhahaha => xhaha, xyhahaha => xyhaha).
    normalized = ContractToken(normalized.data(), normalized.length(), 2);

    // Remove repeated 3 chars from the beginning
    // (e.g. wowwowwow => wowwow, abcdbcdbcd => abcdbcd).
    normalized = ContractToken(normalized.data(), normalized.length(), 3);
  }

  if (!separators_.empty()) {
    // Add space around separators_.
    normalized = NormalizeInternal(normalized.data(), normalized.length());
  }
  return normalized;
}
