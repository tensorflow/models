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

#include "syntaxnet/char_ngram_string_extractor.h"

#include <set>
#include <string>

#include "syntaxnet/task_context.h"
#include <gmock/gmock.h>
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace {

TEST(CharNgramStringExtractorTest, GetConfigId) {
  // The 0'th extractor is default-configured; others differ in one of the four
  // configuration settings.
  std::vector<CharNgramStringExtractor> extractors(5);
  extractors[1].set_min_length(2);
  extractors[2].set_max_length(4);
  extractors[3].set_add_terminators(true);
  extractors[4].set_mark_boundaries(true);

  // Assert that all config IDs are unique.
  std::set<string> ids;
  for (CharNgramStringExtractor &extractor : extractors) {
    extractor.Setup(TaskContext());
    const string id = extractor.GetConfigId();
    EXPECT_TRUE(ids.emplace(id).second) << "Duplicate id: " << id;
  }
  EXPECT_EQ(extractors.size(), ids.size());
}

// Returns the character n-grams extracted from the |word| based on the
// configuration settings.
std::multiset<string> ExtractCharNgramSet(const string &word,
                                          const int min_length,
                                          const int max_length,
                                          const bool add_terminators,
                                          const bool mark_boundaries) {
  std::multiset<string> ngrams;
  CharNgramStringExtractor extractor;
  extractor.set_min_length(min_length);
  extractor.set_max_length(max_length);
  extractor.set_add_terminators(add_terminators);
  extractor.set_mark_boundaries(mark_boundaries);
  extractor.Setup(TaskContext());
  extractor.Extract(word, [&](const string &ngram) { ngrams.insert(ngram); });
  return ngrams;
}

TEST(CharNgramStringExtractorTest, Normal) {
  const std::multiset<string> expected = {"h", "he", "hel", "e", "el", "ell",
                                          "l", "ll", "llo", "l", "lo", "o"};
  EXPECT_EQ(expected, ExtractCharNgramSet("hello", 1, 3, false, false));
}

TEST(CharNgramStringExtractorTest, MinLength) {
  const std::multiset<string> expected = {"he", "hel", "el", "ell",
                                          "ll", "llo", "lo"};
  EXPECT_EQ(expected, ExtractCharNgramSet("hello", 2, 3, false, false));
}

TEST(CharNgramStringExtractorTest, SpaceInMiddle) {
  const std::multiset<string> expected = {"h", "he", "e", "l", "lo", "o"};
  EXPECT_EQ(expected, ExtractCharNgramSet("he lo", 1, 3, false, false));
}

TEST(CharNgramStringExtractorTest, AddTerminators) {
  const std::multiset<string> expected = {"^", "^h", "^he", "h", "he", "hel",
                                          "e", "el", "ell", "l", "ll", "llo",
                                          "l", "lo", "lo$", "o", "o$", "$"};
  EXPECT_EQ(expected, ExtractCharNgramSet("hello", 1, 3, true, false));
}

TEST(CharNgramStringExtractorTest, MarkBoundaries) {
  const std::multiset<string> expected = {"^ h",   "^ he", "^ hel", "e",
                                          "el",    "ell",  "l",     "ll",
                                          "llo $", "l",    "lo $",  "o $"};
  EXPECT_EQ(expected, ExtractCharNgramSet("hello", 1, 3, false, true));
}

TEST(CharNgramStringExtractorTest, MarkBoundariesSingleton) {
  const std::multiset<string> expected = {"^ h $"};
  EXPECT_EQ(expected, ExtractCharNgramSet("h", 1, 3, false, true));
}

TEST(CharNgramStringExtractorTest, MarkBoundariesLeadingSpace) {
  const std::multiset<string> expected = {"e",     "el", "ell",  "l",  "ll",
                                          "llo $", "l",  "lo $", "o $"};
  EXPECT_EQ(expected, ExtractCharNgramSet(" ello", 1, 3, false, true));
}

TEST(CharNgramStringExtractorTest, MarkBoundariesTrailingSpace) {
  const std::multiset<string> expected = {"^ h", "^ he", "^ hel", "e", "el",
                                          "ell", "l",    "ll",    "l"};
  EXPECT_EQ(expected, ExtractCharNgramSet("hell ", 1, 3, false, true));
}

TEST(CharNgramStringExtractorTest, MarkBoundariesLeadingAndTrailingSpace) {
  const std::multiset<string> expected = {"e", "el", "ell", "l", "ll", "l"};
  EXPECT_EQ(expected, ExtractCharNgramSet(" ell ", 1, 3, false, true));
}

}  // namespace
}  // namespace syntaxnet
