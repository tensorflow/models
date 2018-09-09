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

// Tests for char_properties.cc:
//
// (1) Test the DEFINE_CHAR_PROPERTY_AS_SET and DEFINE_CHAR_PROPERTY macros
//     by defining a few fake char properties and verifying their contents.
//
// (2) Test the char properties defined in char_properties.cc by spot-checking
//     a few chars.
//

#include "syntaxnet/char_properties.h"

#include <ctype.h>  // for ispunct, isspace
#include <map>
#include <set>
#include <utility>
#include <vector>

#include <gmock/gmock.h>  // for ContainerEq, EXPECT_THAT
#include "tensorflow/core/platform/test.h"
#include "third_party/utf/utf.h"
#include "util/utf8/unilib.h"  // for IsValidCodepoint, etc
#include "util/utf8/unilib_utf8_utils.h"

using ::testing::ContainerEq;

namespace syntaxnet {

// Invalid UTF-8 bytes are decoded as the Replacement Character, U+FFFD
// (which is also Runeerror). Invalid code points are encoded in UTF-8
// with the UTF-8 representation of the Replacement Character.
static const char ReplacementCharacterUTF8[3] = {'\xEF', '\xBF', '\xBD'};

// ====================================================================
// CharPropertiesTest
//

class CharPropertiesTest : public testing::Test {
 protected:
  // Collect a set of chars.
  void CollectChars(const std::set<char32> &chars) {
    collected_set_.insert(chars.begin(), chars.end());
  }

  // Collect an array of chars.
  void CollectArray(const char32 arr[], int len) {
    collected_set_.insert(arr, arr + len);
  }

  // Collect the chars for which the named CharProperty holds.
  void CollectCharProperty(const char *name) {
    const CharProperty *prop = CharProperty::Lookup(name);
    ASSERT_TRUE(prop != nullptr) << "for " << name;

    for (char32 c = 0; c <= 0x10FFFF; ++c) {
      if (UniLib::IsValidCodepoint(c) && prop->HoldsFor(c)) {
        collected_set_.insert(c);
      }
    }
  }

  // Collect the chars for which an ascii predicate holds.
  void CollectAsciiPredicate(AsciiPredicate *pred) {
    for (char32 c = 0; c < 256; ++c) {
      if ((*pred)(c)) {
        collected_set_.insert(c);
      }
    }
  }

  // Expect the named char property to be true for precisely the chars in
  // the collected set.
  void ExpectCharPropertyEqualsCollectedSet(const char *name) {
    const CharProperty *prop = CharProperty::Lookup(name);
    ASSERT_TRUE(prop != nullptr) << "for " << name;

    // Test that char property holds for all collected chars.  Exercises both
    // signatures of CharProperty::HoldsFor().
    for (std::set<char32>::const_iterator it = collected_set_.begin();
         it != collected_set_.end(); ++it) {
      // Test utf8 version of is_X().
      const char32 c = *it;
      string utf8_char = EncodeAsUTF8(&c, 1);
      EXPECT_TRUE(prop->HoldsFor(utf8_char.c_str(), utf8_char.size()));

      // Test ucs-2 version of is_X().
      EXPECT_TRUE(prop->HoldsFor(static_cast<int>(c)));
    }

    // Test that the char property holds for precisely the collected chars.
    // Somewhat redundant with previous test, but exercises
    // CharProperty::NextElementAfter().
    std::set<char32> actual_chars;
    int c = -1;
    while ((c = prop->NextElementAfter(c)) >= 0) {
      actual_chars.insert(static_cast<char32>(c));
    }
    EXPECT_THAT(actual_chars, ContainerEq(collected_set_))
        << " for " << name;
  }

  // Expect the named char property to be true for at least the chars in
  // the collected set.
  void ExpectCharPropertyContainsCollectedSet(const char *name) {
    const CharProperty *prop = CharProperty::Lookup(name);
    ASSERT_TRUE(prop != nullptr) << "for " << name;

    for (std::set<char32>::const_iterator it = collected_set_.begin();
         it != collected_set_.end(); ++it) {
      EXPECT_TRUE(prop->HoldsFor(static_cast<int>(*it)));
    }
  }

  string EncodeAsUTF8(const char32 *in, int size) {
    string out;
    out.reserve(size);
    for (int i = 0; i < size; ++i) {
      char buf[UTFmax];
      int len = EncodeAsUTF8Char(*in++, buf);
      out.append(buf, len);
    }
    return out;
  }

  int EncodeAsUTF8Char(char32 in, char *out) {
    if (UniLib::IsValidCodepoint(in)) {
      return runetochar(out, &in);
    } else {
      memcpy(out, ReplacementCharacterUTF8, 3);
      return 3;
    }
  }

 private:
  std::set<char32> collected_set_;
};

//======================================================================
// Declarations of the sample character sets below
// (to test the DECLARE_CHAR_PROPERTY() macro)
//

DECLARE_CHAR_PROPERTY(test_digit);
DECLARE_CHAR_PROPERTY(test_wavy_dash);
DECLARE_CHAR_PROPERTY(test_digit_or_wavy_dash);
DECLARE_CHAR_PROPERTY(test_punctuation_plus);

//======================================================================
// Definitions of sample character sets
//

// Digits.
DEFINE_CHAR_PROPERTY_AS_SET(test_digit,
  RANGE('0', '9'),
)

// Wavy dashes.
DEFINE_CHAR_PROPERTY_AS_SET(test_wavy_dash,
  '~',
  0x301C,  // wave dash
  0x3030,  // wavy dash
)

// Digits or wavy dashes.
DEFINE_CHAR_PROPERTY(test_digit_or_wavy_dash, prop) {
  prop->AddCharProperty("test_digit");
  prop->AddCharProperty("test_wavy_dash");
}

// Punctuation plus a few extraneous chars.
DEFINE_CHAR_PROPERTY(test_punctuation_plus, prop) {
  prop->AddChar('a');
  prop->AddCharRange('b', 'b');
  prop->AddCharRange('c', 'e');
  static const int kUnicodes[] = {'f', RANGE('g', 'i'), 'j'};
  prop->AddCharSpec(kUnicodes, ABSL_ARRAYSIZE(kUnicodes));
  prop->AddCharProperty("punctuation");
}

//====================================================================
// Another form of the character sets above -- for verification
//

const char32 kTestDigit[] = {
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
};

const char32 kTestWavyDash[] = {
  '~',
  0x301C,  // wave dash,
  0x3030,  // wavy dash
};

const char32 kTestPunctuationPlusExtras[] = {
  'a',
  'b',
  'c',
  'd',
  'e',
  'f',
  'g',
  'h',
  'i',
  'j',
};

// ====================================================================
// Tests
//

TEST_F(CharPropertiesTest, TestDigit) {
  CollectArray(kTestDigit, ABSL_ARRAYSIZE(kTestDigit));
  ExpectCharPropertyEqualsCollectedSet("test_digit");
}

TEST_F(CharPropertiesTest, TestWavyDash) {
  CollectArray(kTestWavyDash, ABSL_ARRAYSIZE(kTestWavyDash));
  ExpectCharPropertyEqualsCollectedSet("test_wavy_dash");
}

TEST_F(CharPropertiesTest, TestDigitOrWavyDash) {
  CollectArray(kTestDigit, ABSL_ARRAYSIZE(kTestDigit));
  CollectArray(kTestWavyDash, ABSL_ARRAYSIZE(kTestWavyDash));
  ExpectCharPropertyEqualsCollectedSet("test_digit_or_wavy_dash");
}

TEST_F(CharPropertiesTest, TestPunctuationPlus) {
  CollectCharProperty("punctuation");
  CollectArray(kTestPunctuationPlusExtras,
               ABSL_ARRAYSIZE(kTestPunctuationPlusExtras));
  ExpectCharPropertyEqualsCollectedSet("test_punctuation_plus");
}

// ====================================================================
// Spot-check predicates in char_properties.cc
//

TEST_F(CharPropertiesTest, StartSentencePunc) {
  CollectChars({0x00A1, 0x00BF});
  ExpectCharPropertyContainsCollectedSet("start_sentence_punc");
}

TEST_F(CharPropertiesTest, EndSentencePunc) {
  CollectChars({'.', '!', '?'});
  ExpectCharPropertyContainsCollectedSet("end_sentence_punc");
}

TEST_F(CharPropertiesTest, OpenExprPunc) {
  CollectChars({'(', '['});
  ExpectCharPropertyContainsCollectedSet("open_expr_punc");
}

TEST_F(CharPropertiesTest, CloseExprPunc) {
  CollectChars({')', ']'});
  ExpectCharPropertyContainsCollectedSet("close_expr_punc");
}

TEST_F(CharPropertiesTest, OpenQuote) {
  CollectChars({'\'', '"'});
  ExpectCharPropertyContainsCollectedSet("open_quote");
}

TEST_F(CharPropertiesTest, CloseQuote) {
  CollectChars({'\'', '"'});
  ExpectCharPropertyContainsCollectedSet("close_quote");
}

TEST_F(CharPropertiesTest, OpenBookquote) {
  CollectChars({0x300A});
  ExpectCharPropertyContainsCollectedSet("open_bookquote");
}

TEST_F(CharPropertiesTest, CloseBookquote) {
  CollectChars({0x300B});
  ExpectCharPropertyContainsCollectedSet("close_bookquote");
}

TEST_F(CharPropertiesTest, OpenPunc) {
  CollectChars({'(', '['});
  CollectChars({'\'', '"'});
  ExpectCharPropertyContainsCollectedSet("open_punc");
}

TEST_F(CharPropertiesTest, ClosePunc) {
  CollectChars({')', ']'});
  CollectChars({'\'', '"'});
  ExpectCharPropertyContainsCollectedSet("close_punc");
}

TEST_F(CharPropertiesTest, LeadingSentencePunc) {
  CollectChars({'(', '['});
  CollectChars({'\'', '"'});
  CollectChars({0x00A1, 0x00BF});
  ExpectCharPropertyContainsCollectedSet("leading_sentence_punc");
}

TEST_F(CharPropertiesTest, TrailingSentencePunc) {
  CollectChars({')', ']'});
  CollectChars({'\'', '"'});
  CollectChars({'.', '!', '?'});
  ExpectCharPropertyContainsCollectedSet("trailing_sentence_punc");
}

TEST_F(CharPropertiesTest, NoncurrencyTokenPrefixSymbol) {
  CollectChars({'#'});
  ExpectCharPropertyContainsCollectedSet("noncurrency_token_prefix_symbol");
}

TEST_F(CharPropertiesTest, TokenSuffixSymbol) {
  CollectChars({'%', 0x2122, 0x00A9, 0x00B0});
  ExpectCharPropertyContainsCollectedSet("token_suffix_symbol");
}

TEST_F(CharPropertiesTest, TokenPrefixSymbol) {
  CollectChars({'#'});
  CollectChars({'$', 0x00A5, 0x20AC});
  ExpectCharPropertyContainsCollectedSet("token_prefix_symbol");
}

TEST_F(CharPropertiesTest, SubscriptSymbol) {
  CollectChars({0x2082, 0x2083});
  ExpectCharPropertyContainsCollectedSet("subscript_symbol");
}

TEST_F(CharPropertiesTest, SuperscriptSymbol) {
  CollectChars({0x00B2, 0x00B3});
  ExpectCharPropertyContainsCollectedSet("superscript_symbol");
}

TEST_F(CharPropertiesTest, CurrencySymbol) {
  CollectChars({'$', 0x00A5, 0x20AC});
  ExpectCharPropertyContainsCollectedSet("currency_symbol");
}

TEST_F(CharPropertiesTest, DirectionalFormattingCode) {
  CollectChars({0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E});
  ExpectCharPropertyContainsCollectedSet("directional_formatting_code");
}

TEST_F(CharPropertiesTest, Punctuation) {
  CollectAsciiPredicate(ispunct);
  ExpectCharPropertyContainsCollectedSet("punctuation");
}

TEST_F(CharPropertiesTest, Separator) {
  CollectAsciiPredicate(isspace);
  ExpectCharPropertyContainsCollectedSet("separator");
}

}  // namespace syntaxnet
