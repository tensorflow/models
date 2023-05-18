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
#include "tf_ops/skipgram_finder.h"  // seq_flow_lite

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "icu4c/source/common/unicode/uchar.h"
#include "icu4c/source/common/unicode/utf8.h"

namespace seq_flow_lite {
namespace {

using ::testing::UnorderedElementsAreArray;

void TestFindSkipgrams(const SkipgramFinder& skipgram_finder,
                       const std::vector<std::string>& tokens,
                       const std::vector<int>& categories,
                       const std::vector<int>& token_categories) {
  EXPECT_THAT(skipgram_finder.FindSkipgrams(absl::StrJoin(tokens, " ")),
              UnorderedElementsAreArray(categories));

  std::vector<absl::string_view> sv_tokens;
  sv_tokens.reserve(tokens.size());
  for (const auto& token : tokens) {
    sv_tokens.emplace_back(token.data(), token.size());
  }
  EXPECT_THAT(skipgram_finder.FindSkipgrams(sv_tokens),
              UnorderedElementsAreArray(token_categories));
}

// Test that u_tolower() will only increase the number of bytes in the
// UTF-8 encoding in two specific cases.
TEST(SkipgramFinderTest, UCharToLower) {
  for (UChar32 c = 0; c < 0x10000; c++) {
    if (c == 0x23a || c == 0x23e) continue;
    UChar32 l = u_tolower(c);
    EXPECT_GE(U8_LENGTH(c), U8_LENGTH(l)) << c << " lowercases to " << l;
  }
}

TEST(SkipgramFinderTest, SingleExists) {
  SkipgramFinder skipgram_finder(1);
  std::string s("q r s");
  skipgram_finder.AddSkipgram(s, 0);
  TestFindSkipgrams(skipgram_finder, {"a", "q", "r", "s", "c"}, {0}, {0});
  TestFindSkipgrams(skipgram_finder, {"a", "q", "xyz", "R!", "xy", "s", "c"},
                    {0}, {});
  TestFindSkipgrams(skipgram_finder, {"a", "q", "r", "q", "R", "s.", "c"}, {0},
                    {});
}

TEST(SkipgramFinderTest, SingleNotExists) {
  SkipgramFinder skipgram_finder(1);
  std::string s("q r s");
  skipgram_finder.AddSkipgram(s, 0);
  TestFindSkipgrams(skipgram_finder, {"a", "q", "x", "x", "r", "x", "s", "c"},
                    {}, {});
  TestFindSkipgrams(skipgram_finder, {"a", "q", "x", "r", "x", "c"}, {}, {});
  TestFindSkipgrams(skipgram_finder, {"a", "r", "x", "s", "q", "c"}, {}, {});
}

TEST(SkipgramFinderTest, SinglePrefixExists) {
  SkipgramFinder skipgram_finder(1);
  std::string s("q.* r s");
  skipgram_finder.AddSkipgram(s, 0);
  TestFindSkipgrams(skipgram_finder, {"a", "qa", "r", "s", "c"}, {0}, {0});
  TestFindSkipgrams(skipgram_finder, {"a", "q", "xyz", "R!", "xy", "s", "c"},
                    {0}, {});
  TestFindSkipgrams(skipgram_finder, {"a", "qc", "r", "qd", "R", "s.", "c"},
                    {0}, {});
}

TEST(SkipgramFinderTest, SinglePrefixNotExists) {
  SkipgramFinder skipgram_finder(1);
  std::string s("q.* r s");
  skipgram_finder.AddSkipgram(s, 0);
  TestFindSkipgrams(skipgram_finder, {"a", "aq", "r", "s", "c"}, {}, {});
  TestFindSkipgrams(skipgram_finder, {"a", "aqc", "xyz", "R!", "xy", "s", "c"},
                    {}, {});
  TestFindSkipgrams(skipgram_finder, {"a", "q", "ar", "q", "aR", "s.", "c"}, {},
                    {});
}

TEST(SkipgramFinderTest, Punctuation) {
  SkipgramFinder skipgram_finder(1);
  std::string s("a-b-c def");
  skipgram_finder.AddSkipgram(s, 0);
  TestFindSkipgrams(skipgram_finder, {"q", "abc", "q", "d-e-f", "q"}, {0}, {});
  TestFindSkipgrams(skipgram_finder, {"a", "'abc'", "q", "'def'", "q"}, {0},
                    {});
  TestFindSkipgrams(skipgram_finder, {"q", "abc", "q", "def", "q"}, {0}, {0});
}

TEST(SkipgramFinderTest, HandlesMultibyteInput) {
  SkipgramFinder skipgram_finder(1);
  std::string s("hello\363\243\243\243!");
  skipgram_finder.AddSkipgram(s, 0);
}

TEST(SkipgramFinderTest, Multiple) {
  SkipgramFinder skipgram_finder(1);
  std::string s1("a b c");
  std::string s2("D e. F!");
  std::string s3("ghi jkl mno");
  std::string s4("S T U");
  std::string s5("x. y, z!");
  std::string s6("d.* e f");
  skipgram_finder.AddSkipgram(s1, 0);
  skipgram_finder.AddSkipgram(s2, 2);
  skipgram_finder.AddSkipgram(s3, 4);
  skipgram_finder.AddSkipgram(s4, 6);
  skipgram_finder.AddSkipgram(s5, 8);
  skipgram_finder.AddSkipgram(s6, 10);
  TestFindSkipgrams(skipgram_finder, {"a", "d", "b", "e", "c", "f"}, {0, 2, 10},
                    {0, 2, 10});
  TestFindSkipgrams(skipgram_finder, {"a", "dq", "b", "e", "c", "f"}, {0, 10},
                    {0, 10});
  TestFindSkipgrams(skipgram_finder, {"a", "d", "b", "eq", "c", "f"}, {0}, {0});
  TestFindSkipgrams(skipgram_finder, {"a", "ghi", "b", "jkl", "c", "x", "mno"},
                    {0}, {0});
  TestFindSkipgrams(skipgram_finder, {"ghi", "d", "jkl", "e", "mno", "f"},
                    {2, 4, 10}, {2, 4, 10});
  TestFindSkipgrams(skipgram_finder, {"s", "x", "t", "y", "u", "z"}, {6, 8},
                    {6, 8});
}

TEST(SkipgramFinderTest, UnicodeLowercase) {
  // Check that the lowercase has a smaller UTF-8 encoding than the uppercase.
  UChar32 cu;
  U8_GET_UNSAFE("Ɦ", 0, cu);
  UChar32 cl = u_tolower(cu);
  EXPECT_GT(U8_LENGTH(cu), U8_LENGTH(cl));

  SkipgramFinder skipgram_finder(1);
  std::string s("Ɦ");
  skipgram_finder.AddSkipgram(s, 0);
  TestFindSkipgrams(skipgram_finder, {"Ɦ"}, {0}, {});
  TestFindSkipgrams(skipgram_finder, {"ɦ"}, {0}, {0});
  TestFindSkipgrams(skipgram_finder, {"h"}, {}, {});
}

}  // namespace
}  // namespace seq_flow_lite
