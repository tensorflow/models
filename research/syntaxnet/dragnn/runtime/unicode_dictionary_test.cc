// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "dragnn/runtime/unicode_dictionary.h"

#include "dragnn/core/test/generic.h"
#include "dragnn/runtime/test/term_map_helpers.h"
#include "syntaxnet/base.h"
#include "syntaxnet/term_frequency_map.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "third_party/utf/utf.h"
#include "util/utf8/unilib.h"
#include "util/utf8/unilib_utf8_utils.h"

namespace syntaxnet {
namespace dragnn {
namespace runtime {
namespace {

constexpr char kInvalidUtf8[] = "\xff\xff\xff\xff";
constexpr char k1ByteCharacter[] = "a";
constexpr char k2ByteCharacter[] = "¼";
constexpr char k3ByteCharacter[] = "好";
constexpr char k4ByteCharacter[] = "𠜎";

// NB: String sizes are one more than expected from the trailing NUL.
static_assert(sizeof(k1ByteCharacter) / sizeof(char) == 2,
              "1-byte character has the wrong size");
static_assert(sizeof(k2ByteCharacter) / sizeof(char) == 3,
              "2-byte character has the wrong size");
static_assert(sizeof(k3ByteCharacter) / sizeof(char) == 4,
              "3-byte character has the wrong size");
static_assert(sizeof(k4ByteCharacter) / sizeof(char) == 5,
              "4-byte character has the wrong size");

// Tests that the dictionary is empty by default.
TEST(UnicodeDictionaryTest, EmptyByDefault) {
  UnicodeDictionary dictionary;

  EXPECT_EQ(dictionary.size(), 0);
  EXPECT_EQ(dictionary.Lookup(k1ByteCharacter, 1, -123), -123);
  EXPECT_EQ(dictionary.Lookup(k2ByteCharacter, 2, -123), -123);
  EXPECT_EQ(dictionary.Lookup(k3ByteCharacter, 3, -123), -123);
  EXPECT_EQ(dictionary.Lookup(k4ByteCharacter, 4, -123), -123);
}

// Tests that the dictionary can be reset to a copy of a term map.
TEST(UnicodeDictionaryTest, Reset) {
  TermFrequencyMap character_map;
  ASSERT_EQ(character_map.Increment(k1ByteCharacter), 0);
  ASSERT_EQ(character_map.Increment(k2ByteCharacter), 1);
  ASSERT_EQ(character_map.Increment(k3ByteCharacter), 2);
  ASSERT_EQ(character_map.Increment(k4ByteCharacter), 3);

  UnicodeDictionary dictionary;
  TF_ASSERT_OK(dictionary.Reset(character_map));

  EXPECT_EQ(dictionary.size(), 4);
  EXPECT_EQ(dictionary.Lookup(k1ByteCharacter, 1, -123), 0);
  EXPECT_EQ(dictionary.Lookup(k2ByteCharacter, 2, -123), 1);
  EXPECT_EQ(dictionary.Lookup(k3ByteCharacter, 3, -123), 2);
  EXPECT_EQ(dictionary.Lookup(k4ByteCharacter, 4, -123), 3);
}

// Tests that the dictionary fails if a character is empty.
TEST(UnicodeDictionaryTest, EmptyCharacter) {
  TermFrequencyMap character_map;
  ASSERT_EQ(character_map.Increment(""), 0);

  UnicodeDictionary dictionary;
  EXPECT_THAT(dictionary.Reset(character_map),
              test::IsErrorWithSubstr("Term 0 is empty"));
}

// Tests that the dictionary fails if a term contains more than one character.
TEST(UnicodeDictionaryTest, MultipleCharacters) {
  TermFrequencyMap character_map;
  ASSERT_EQ(character_map.Increment("1234"), 0);

  UnicodeDictionary dictionary;
  EXPECT_THAT(dictionary.Reset(character_map),
              test::IsErrorWithSubstr("Term 0 should have size 1"));
}

// Tests that the dictionary fails if a character is invalid.
TEST(UnicodeDictionaryTest, InvalidUtf8) {
  TermFrequencyMap character_map;
  ASSERT_EQ(character_map.Increment(kInvalidUtf8), 0);

  UnicodeDictionary dictionary;
  EXPECT_THAT(dictionary.Reset(character_map),
              test::IsErrorWithSubstr("Term 0 is not valid UTF-8"));
}

// Tests that the dictionary can be constructed from a file.
TEST(UnicodeDictionaryTest, ConstructFromFile) {
  // Recall that terms are loaded in order of descending frequency.
  const string character_map_path = WriteTermMap({{"too-infrequent", 1},
                                                  {k1ByteCharacter, 2},
                                                  {k2ByteCharacter, 3},
                                                  {k3ByteCharacter, 4},
                                                  {k4ByteCharacter, 5}});

  const UnicodeDictionary dictionary(character_map_path, 2, 0);

  EXPECT_EQ(dictionary.size(), 4);
  EXPECT_EQ(dictionary.Lookup(k1ByteCharacter, 1, -123), 3);
  EXPECT_EQ(dictionary.Lookup(k2ByteCharacter, 2, -123), 2);
  EXPECT_EQ(dictionary.Lookup(k3ByteCharacter, 3, -123), 1);
  EXPECT_EQ(dictionary.Lookup(k4ByteCharacter, 4, -123), 0);
}

// Tests that the dictionary constructor dies on error.
TEST(UnicodeDictionaryTest, ConstructorDiesOnError) {
  const string bad_path = WriteTermMap({{"1234", 1}});

  EXPECT_DEATH(UnicodeDictionary dictionary(bad_path, 0, 0),
               "Term 0 should have size 1");
}

// Tests that the dictionary can map all valid codepoints.
TEST(UnicodeDictionaryTest, AllValidCodepoints) {
  TermFrequencyMap character_map;
  for (Rune rune = 0; rune < Runemax; ++rune) {
    // Some codepoints are considered invalid, and UnicodeDictionary::Reset()
    // will fail if it encounters them (see the InvalidUtf8 test).  Skip those
    // since we've already tested this in the "InvalidUtf8" test.
    if (!UniLib::IsValidCodepoint(rune)) continue;
    char data[UTFmax];
    const int size = runetochar(data, &rune);
    const string character(data, size);
    const int index = character_map.Size();
    ASSERT_EQ(character_map.Increment(character), index);
  }

  UnicodeDictionary dictionary;
  TF_ASSERT_OK(dictionary.Reset(character_map));
  for (int index = 0; index < character_map.Size(); ++index) {
    const string &character = character_map.GetTerm(index);
    EXPECT_EQ(dictionary.Lookup(character.data(), character.size(), -1), index);
  }
}

}  // namespace
}  // namespace runtime
}  // namespace dragnn
}  // namespace syntaxnet
