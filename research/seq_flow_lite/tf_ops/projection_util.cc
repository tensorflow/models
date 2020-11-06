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
#include "tf_ops/projection_util.h"  // seq_flow_lite

#include <cstddef>
#include <iostream>
#include <memory>
#include <sstream>

namespace {
constexpr size_t kInvalid = -1;
constexpr char kSpace = ' ';
}  // namespace

std::string ProjectionUnicodeHandler::LowerCaseUTF8WithSupportedUnicodes(
    const std::pair<const char*, size_t>& source) const {
  // Ideally the size of target should be less than or equal to source. But
  // when we do to_lower the number of bytes needed to encode a unicode
  // character could increase. To account for this 4 times the source length
  // is allocated for target.
  const char* csource = source.first;
  int len = source.second;
  auto target = std::unique_ptr<char[]>(new char[len * 4]);
  auto target_ptr = target.get();
  int i = 0;
  while (i < len) {
    Rune rune;
    const int bytes_read = charntorune(&rune, csource + i, len - i);
    if (bytes_read == 0) {
      break;
    }
    i += bytes_read;
    if (rune != Runeerror) {
      Rune lower = tolowerrune(rune);
      // Skip processing the unicode if exclude_nonalphaspace_unicodes_ is true
      // and the unicode is not alpha and not space.
      const Rune kSpaceRune = ' ';
      if (exclude_nonalphaspace_unicodes_ && !isalpharune(lower) &&
          lower != kSpaceRune) {
        continue;
      }
      if (IsUnrestrictedVocabulary() || IsValidUnicode(lower)) {
        const int bytes_written = runetochar(target_ptr, &lower);
        target_ptr += bytes_written;
      }
    }
  }
  return std::string(target.get(), target_ptr);
}

void ProjectionUnicodeHandler::InitializeVocabulary(
    const std::string& vocabulary) {
  for (size_t i = 0, index = 0; i < vocabulary.length();) {
    Rune rune;
    const int bytes_read =
        charntorune(&rune, vocabulary.c_str() + i, vocabulary.length() - i);
    if (!bytes_read) {
      break;
    }
    i += bytes_read;
    // Include novel lower case unicode segments as part of valid chars.
    if (rune == Runeerror) {
      std::clog << "Invalid rune in vocabulary.";
    } else if (IsValidUnicode(rune)) {
      std::clog << "Duplicate rune " << rune << " found in vocabulary.";
    } else if (rune != tolowerrune(rune)) {
      std::clog << "Upper case rune " << rune << " found in vocabulary.";
    } else {
      valid_chars_[rune] = index++;
    }
  }
}

// Starting from input_ptr[from], search for the next occurrence of ' ',
// Don't search beyond input_ptr[length](non-inclusive), return -1 if not found.
inline size_t FindNextSpace(const char* input_ptr, size_t from, size_t length) {
  size_t space_index;
  for (space_index = from; space_index < length; space_index++) {
    if (input_ptr[space_index] == kSpace) {
      break;
    }
  }
  return space_index == length ? kInvalid : space_index;
}

template <typename T>
void SplitBySpaceInternal(std::vector<T>* tokens, const char* input_ptr,
                          size_t len, size_t max_input, size_t max_tokens) {
  size_t last_index =
      max_input == kEntireString ? len : (len < max_input ? len : max_input);
  size_t start = 0;
  // skip leading spaces
  while (start < last_index && input_ptr[start] == kSpace) {
    start++;
  }
  auto end = FindNextSpace(input_ptr, start, last_index);
  while (end != kInvalid &&
         (max_tokens == kAllTokens || tokens->size() < max_tokens - 1)) {
    auto length = end - start;
    if (length > 0) {
      tokens->emplace_back(input_ptr + start, length);
    }

    start = end + 1;
    end = FindNextSpace(input_ptr, start, last_index);
  }
  auto length = end == kInvalid ? (last_index - start) : (end - start);
  if (length > 0) {
    tokens->emplace_back(input_ptr + start, length);
  }
}

std::vector<std::pair<const char*, size_t>> SplitBySpaceAsPairs(
    const char* input_ptr, size_t len, size_t max_tokens) {
  std::vector<std::pair<const char*, size_t>> tokens;
  SplitBySpaceInternal(&tokens, input_ptr, len, kEntireString, max_tokens);
  return tokens;
}

std::vector<std::string> SplitBySpace(const char* input_ptr, size_t len,
                                      size_t max_input, size_t max_tokens) {
  std::vector<std::string> tokens;
  SplitBySpaceInternal(&tokens, input_ptr, len, max_input, max_tokens);
  return tokens;
}

template <typename T>
void SplitByCharInternal(std::vector<T>* tokens, const char* input_ptr,
                         size_t len, size_t max_tokens) {
  Rune rune;
  for (size_t i = 0; i < len;) {
    auto bytes_read = charntorune(&rune, input_ptr + i, len - i);
    if (bytes_read == 0) break;
    tokens->emplace_back(input_ptr + i, bytes_read);
    if (max_tokens != kInvalid && tokens->size() == max_tokens) {
      break;
    }
    i += bytes_read;
  }
}

std::vector<std::pair<const char*, size_t>> SplitByCharAsPairs(
    const char* input_ptr, size_t len, size_t max_tokens) {
  std::vector<std::pair<const char*, size_t>> tokens;
  SplitByCharInternal(&tokens, input_ptr, len, max_tokens);
  return tokens;
}

std::vector<std::string> SplitByChar(const char* input_ptr, size_t len,
                                     size_t max_tokens) {
  std::vector<std::string> tokens;
  SplitByCharInternal(&tokens, input_ptr, len, max_tokens);
  return tokens;
}

std::string JoinPairsBySpace(
    std::vector<std::pair<const char*, size_t>> words) {
  std::stringstream ss;
  bool first = true;
  for (auto& str_pair : words) {
    if (first) {
      ss << std::string(str_pair.first, str_pair.second);
      first = false;
    } else {
      ss << kSpace << std::string(str_pair.first, str_pair.second);
    }
  }
  return ss.str();
}

std::vector<std::pair<const char*, size_t>> ProjectionUnicodeHandler::Tokenize(
    const char* str, size_t len, bool by_space, int max_tokens) const {
  return by_space ? SplitBySpaceAsPairs(str, len, max_tokens)
                  : SplitByCharAsPairs(str, len, max_tokens);
}
