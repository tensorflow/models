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

#ifndef $TARGETDIR_UTILS_H_
#define $TARGETDIR_UTILS_H_

#include <functional>
#include <string>
#include <vector>
#include <unordered_set>
#include "syntaxnet/base.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/default/integral_types.h"
#include "tensorflow/core/platform/mutex.h"
#include "util/utf8/unicodetext.h"

namespace syntaxnet {
namespace utils {

bool ParseInt32(const char *c_str, int *value);
bool ParseInt64(const char *c_str, int64 *value);
bool ParseDouble(const char *c_str, double *value);

template <typename T>
T ParseUsing(const string &str, std::function<bool(const char *, T *)> func) {
  T value;
  CHECK(func(str.c_str(), &value)) << "Failed to convert: " << str;
  return value;
}

template <typename T>
T ParseUsing(const string &str, T defval,
             std::function<bool(const char *, T *)> func) {
  return str.empty() ? defval : ParseUsing<T>(str, func);
}

string CEscape(const string &src);

std::vector<string> Split(const string &text, char delim);

template <typename T>
string Join(const std::vector<T> &s, const char *sep) {
  string result;
  bool first = true;
  for (const auto &x : s) {
    tensorflow::strings::StrAppend(&result, (first ? "" : sep), x);
    first = false;
  }
  return result;
}

string JoinPath(std::initializer_list<StringPiece> paths);

size_t RemoveLeadingWhitespace(tensorflow::StringPiece *text);

size_t RemoveTrailingWhitespace(tensorflow::StringPiece *text);

size_t RemoveWhitespaceContext(tensorflow::StringPiece *text);

uint32 Hash32(const char *data, size_t n, uint32 seed);

// Deletes all the elements in an STL container and clears the container. This
// function is suitable for use with a vector, set, hash_set, or any other STL
// container which defines sensible begin(), end(), and clear() methods.
// If container is NULL, this function is a no-op.
template <typename T>
void STLDeleteElements(T *container) {
  if (!container) return;
  auto it = container->begin();
  while (it != container->end()) {
    auto temp = it;
    ++it;
    delete *temp;
  }
  container->clear();
}

// Returns lower-cased version of s.
string Lowercase(tensorflow::StringPiece s);

class PunctuationUtil {
 public:
  // Unicode character ranges for punctuation characters according to CoNLL.
  struct CharacterRange {
    int first;
    int last;
  };
  static CharacterRange kPunctuation[];

  // Returns true if Unicode character is a punctuation character.
  static bool IsPunctuation(int u) {
    int i = 0;
    while (kPunctuation[i].first > 0) {
      if (u < kPunctuation[i].first) return false;
      if (u <= kPunctuation[i].last) return true;
      ++i;
    }
    return false;
  }

  // Determine if tag is a punctuation tag.
  static bool IsPunctuationTag(const string &tag) {
    for (size_t i = 0; i < tag.length(); ++i) {
      int c = tag[i];
      if (c != ',' && c != ':' && c != '.' && c != '\'' && c != '`') {
        return false;
      }
    }
    return true;
  }

  // Returns true if word consists of punctuation characters.
  static bool IsPunctuationToken(const string &word) {
    UnicodeText text;
    text.PointToUTF8(word.c_str(), word.length());
    UnicodeText::const_iterator it;
    for (it = text.begin(); it != text.end(); ++it) {
      if (!IsPunctuation(*it)) return false;
    }
    return true;
  }

  // Returns true if tag is non-empty and has only punctuation or parens
  // symbols.
  static bool IsPunctuationTagOrParens(const string &tag) {
    if (tag.empty()) return false;
    for (size_t i = 0; i < tag.length(); ++i) {
      int c = tag[i];
      if (c != '(' && c != ')' && c != ',' && c != ':' && c != '.' &&
          c != '\'' && c != '`') {
        return false;
      }
    }
    return true;
  }

  // Decides whether to score a token, given the word, the POS tag and
  // and the scoring type.
  static bool ScoreToken(const string &word, const string &tag,
                         const string &scoring_type) {
    if (scoring_type == "default") {
      return tag.empty() || !IsPunctuationTag(tag);
    } else if (scoring_type == "conllx") {
      return !IsPunctuationToken(word);
    } else if (scoring_type == "ignore_parens") {
      return !IsPunctuationTagOrParens(tag);
    }
    CHECK(scoring_type.empty()) << "Unknown scoring strategy " << scoring_type;
    return true;
  }
};

void NormalizeDigits(string *form);

}  // namespace utils
}  // namespace syntaxnet

#endif  // $TARGETDIR_UTILS_H_
