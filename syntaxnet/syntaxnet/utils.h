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

#ifndef SYNTAXNET_UTILS_H_
#define SYNTAXNET_UTILS_H_

#include <functional>
#include <string>
#include <unordered_set>
#include <vector>
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

// Splits the given string on every occurrence of the given delimiter char.
std::vector<string> Split(const string &text, char delim);

// Splits the given string on the first occurrence of the given delimiter char,
// or returns the given string if the given delimiter is not found.
std::vector<string> SplitOne(const string &text, char delim);

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

string JoinPath(std::initializer_list<tensorflow::StringPiece> paths);

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

// Helper type to mark missing c-tor argument types
// for Type's c-tor in LazyStaticPtr<Type, ...>.
struct NoArg {};

template <typename Type, typename Arg1 = NoArg, typename Arg2 = NoArg,
          typename Arg3 = NoArg>
class LazyStaticPtr {
 public:
  typedef Type element_type;  // per smart pointer convention

  // Pretend to be a pointer to Type (never NULL due to on-demand creation):
  Type &operator*() const { return *get(); }
  Type *operator->() const { return get(); }

  // Named accessor/initializer:
  Type *get() const {
    mutex_lock l(mu_);
    if (!ptr_) Initialize(this);
    return ptr_;
  }

 public:
  // All the data is public and LazyStaticPtr has no constructors so that we can
  // initialize LazyStaticPtr objects with the "= { arg_value, ... }" syntax.
  // Clients of LazyStaticPtr must not access the data members directly.

  // Arguments for Type's c-tor
  // (unused NoArg-typed arguments consume either no space, or 1 byte to
  //  ensure address uniqueness):
  Arg1 arg1_;
  Arg2 arg2_;
  Arg3 arg3_;

  // The object we create and show.
  mutable Type *ptr_;

  // Ensures the ptr_ is initialized only once.
  mutable mutex mu_;

 private:
  template <typename A1, typename A2, typename A3>
  static Type *Factory(const A1 &a1, const A2 &a2, const A3 &a3) {
    return new Type(a1, a2, a3);
  }

  template <typename A1, typename A2>
  static Type *Factory(const A1 &a1, const A2 &a2, NoArg a3) {
    return new Type(a1, a2);
  }

  template <typename A1>
  static Type *Factory(const A1 &a1, NoArg a2, NoArg a3) {
    return new Type(a1);
  }

  static Type *Factory(NoArg a1, NoArg a2, NoArg a3) { return new Type(); }

  static void Initialize(const LazyStaticPtr *lsp) {
    lsp->ptr_ = Factory(lsp->arg1_, lsp->arg2_, lsp->arg3_);
  }
};

}  // namespace utils
}  // namespace syntaxnet

#endif  // SYNTAXNET_UTILS_H_
