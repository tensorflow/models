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
#include "tf_ops/projection_tokenizer_util.h"  // seq_flow_lite

#include <cstddef>
#include <iostream>
#include <memory>
#include <sstream>
#include <utility>

#include "tf_ops/projection_util.h"  // seq_flow_lite


namespace {
constexpr char kApostrophe = '\'';
constexpr char kSpace = ' ';
constexpr char kComma = ',';
constexpr char kDot = '.';
constexpr size_t kInvalid = -1;
}  // namespace

// Returns true if the input |c| is ascii number.
bool is_numeric(char c) { return c >= '0' && c <= '9'; }

// Returns true if we want to prepend the separator to the next token.
bool prepend_separator(char separator) { return separator == kApostrophe; }

void ProjectionTokenizer::InitializeSeparators(const std::string& separators) {
  for (size_t i = 0; i < separators.length(); ++i) {
    separators_.insert(separators[i]);
  }
}

size_t ProjectionTokenizer::FindNextSeparator(const char* input_ptr,
                                              size_t from,
                                              size_t length) const {
  auto index = from;
  while (index < length) {
    char c = input_ptr[index];
    // Do not break a number (e.g. "10,000", "0.23").
    if (c == kComma || c == kDot) {
      if (index + 1 < length && is_numeric(input_ptr[index + 1])) {
        c = input_ptr[++index];
      }
    }
    if (separators_.find(c) != separators_.end()) {
      break;
    }
    ++index;
  }
  return index == length ? kInvalid : index;
}

std::vector<std::string> ProjectionTokenizer::Tokenize(
    const char* input_ptr, size_t len, size_t max_input,
    size_t max_tokens) const {
  // If separators_ is not given, tokenize the input with a space.
  if (separators_.empty()) {
    return SplitBySpace(input_ptr, len, max_input, max_tokens);
  }

  std::vector<std::string> tokens;
  size_t last_index =
      max_input == kEntireString ? len : (len < max_input ? len : max_input);
  size_t start = 0;
  // Skip leading spaces.
  while (start < last_index && input_ptr[start] == kSpace) {
    start++;
  }
  auto end = FindNextSeparator(input_ptr, start, last_index);

  while (end != kInvalid &&
         (max_tokens == kAllTokens || tokens.size() < max_tokens - 1)) {
    auto length = end - start;
    if (length > 0) tokens.emplace_back(input_ptr + start, length);

    // Add the separator (except space and apostrophe) as a token
    char separator = input_ptr[end];
    if (separator != kSpace && separator != kApostrophe) {
      tokens.emplace_back(input_ptr + end, 1);
    }

    start = end + (prepend_separator(separator) ? 0 : 1);
    end = FindNextSeparator(input_ptr, end + 1, last_index);
  }
  auto length = end == kInvalid ? (last_index - start) : (end - start);
  if (length > 0) tokens.emplace_back(input_ptr + start, length);
  return tokens;
}
