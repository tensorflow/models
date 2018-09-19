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

// char_properties.cc - define is_X() tests for various character properties
//
// See char_properties.h for how to write a character property.
//
// References for the char sets below:
//
// . http://www.unicode.org/Public/UNIDATA/PropList.txt
//
//   Large (but not exhaustive) list of Unicode chars and their "properties"
//   (e.g., the property "Pi" = an initial quote punctuation char).
//
// . http://www.unicode.org/Public/UNIDATA/PropertyValueAliases.txt
//
//   Defines the list of properties, such as "Pi", used in the above list.
//
// . http://www.unipad.org/unimap/index.php?param_char=XXXX&page=detail
//
//   Gives detail about a particular character code.
//   XXXX is a 4-hex-digit Unicode character code.
//
// . http://www.unicode.org/Public/UNIDATA/UCD.html
//
//   General reference for Unicode characters.
//

#include "syntaxnet/char_properties.h"

#include <ctype.h>  // for ispunct, isspace
#include <memory>
#include <utility>
#include <vector>  // for vector

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "third_party/utf/utf.h"      // for runetochar, ::UTFmax, Rune
#include "util/utf8/unilib.h"  // for IsValidCodepoint, etc
#include "util/utf8/unilib_utf8_utils.h"

//============================================================
// CharPropertyImplementation
//

// A CharPropertyImplementation stores a set of Unicode characters,
// encoded in UTF-8, as a trie.  The trie is represented as a vector
// of nodes.  Each node is a 256-element array that specifies what to
// do with one byte of the UTF-8 sequence.  Each element n of a node
// is one of:
//  n = 0,  indicating that the Property is not true of any
//          character whose UTF-8 encoding includes this byte at
//          this position
//  n = -1, indicating that the Property is true for the UTF-8 sequence
//          that ends with this byte.
//  n > 0,  indicating the index of the row that describes the
//          remaining bytes in the UTF-8 sequence.
//
// The only operation that needs to be fast is HoldsFor, which tests
// whether a character has a given property. We use each byte of the
// character's UTF-8 encoding to index into a row. If the value is 0,
// then the property is not true for the character. (We might discover
// this even before getting to the end of the sequence.) If the value
// is -1, then the property is true for this character. Otherwise,
// the value is the index of another row, which we index using the next
// byte in the sequence, and so on. The design of UTF-8 prevents
// ambiguities here; no prefix of a UTF-8 sequence is a valid UTF-8
// sequence.
//
// While it is possible to implement an iterator for this representation,
// it is much easier to use set<char32> for this purpose. In fact, we
// would use that as the entire representation, were it not for concerns
// that HoldsFor might be slower.

namespace syntaxnet {

struct CharPropertyImplementation {
  unordered_set<char32> chars;
  std::vector<std::vector<int> > rows;
  CharPropertyImplementation() {
    rows.reserve(10);
    rows.resize(1);
    rows[0].resize(256, 0);
  }
  void AddChar(char *buf, int len) {
    int n = 0;  // row index
    for (int i = 0; i < len; ++i) {
      int ch = reinterpret_cast<unsigned char *>(buf)[i];
      int m = rows[n][ch];
      if (m > 0) {
        CHECK_LT(i, len - 1)
            << " : " << (i + 1) << "-byte UTF-8 sequence "
            << "(" << tensorflow::str_util::CEscape(string(buf, i + 1)) << ")"
            << " is prefix of previously-seen UTF-8 sequence(s)";
        n = m;
      } else if (i == len - 1) {
        rows[n][ch] = -1;
      } else {
        CHECK_EQ(m, 0) << " : UTF-8 sequence is extension of previously-seen "
                       << (i + 1) << "-byte UTF-8 sequence "
                       << "("
                       << tensorflow::str_util::CEscape(string(buf, i + 1))
                       << ")";
        int a = rows.size();
        rows.resize(a + 1);
        rows[a].resize(256, 0);
        rows[n][ch] = a;
        n = a;
      }
    }
  }

  bool HoldsFor(const char *buf) const {
    const unsigned char *bytes = reinterpret_cast<const unsigned char *>(buf);

    // Lookup each byte of the UTF-8 sequence, starting in row 0.
    int n = rows[0][*bytes];
    if (n == 0) return false;
    if (n == -1) return true;

    // If the value is not 0 or -1, then it is the index of the row for the
    // second byte in the sequence.
    n = rows[n][*++bytes];
    if (n == 0) return false;
    if (n == -1) return true;
    n = rows[n][*++bytes];  // Likewise for the third byte.
    if (n == 0) return false;
    if (n == -1) return true;
    n = rows[n][*++bytes];  // Likewise for the fourth byte.
    if (n == 0) return false;

    // Since there can be at most 4 bytes in the sequence, n must be -1.
    return true;

    // Implementation note: it is possible (and perhaps clearer) to write this
    // code as a loop, "for (int i = 0; i < 4; ++i) ...", but the TestHoldsFor
    // benchmark results indicate that doing so produces slower code for
    // anything other than short 7-bit ASCII strings (< 512 bytes). This is
    // mysterious, since the compiler unrolls the loop, producing code that
    // is almost the same as what we have here, except for the shortcut on
    // the 4th byte.
  }
};

//============================================================
// CharProperty - a property that holds for selected Unicode chars
//

CharProperty::CharProperty(const char *name,
                           const int *unicodes,
                           int num_unicodes)
    : name_(name),
      impl_(new CharPropertyImplementation) {
  // Initialize CharProperty to its char set.
  AddCharSpec(unicodes, num_unicodes);
}

CharProperty::CharProperty(const char *name, CharPropertyInitializer *init_fn)
    : name_(name),
      impl_(new CharPropertyImplementation) {
  (*init_fn)(this);
}

CharProperty::~CharProperty() {
  delete impl_;
}

void CharProperty::AddChar(int c) {
  CheckUnicodeVal(c);
  impl_->chars.insert(c);

  char buf[UTFmax];
  Rune r = c;
  int len = runetochar(buf, &r);
  impl_->AddChar(buf, len);
}

void CharProperty::AddCharRange(int c1, int c2) {
  for (int c = c1; c <= c2; ++c) {
    AddChar(c);
  }
}

void CharProperty::AddAsciiPredicate(AsciiPredicate *pred) {
  for (int c = 0; c < 256; ++c) {
    if ((*pred)(c)) {
      AddChar(c);
    }
  }
}

void CharProperty::AddCharProperty(const char *propname) {
  const CharProperty *prop = CharProperty::Lookup(propname);
  CHECK(prop != nullptr) << ": unknown char property \"" << propname << "\" in "
                         << name_;
  int c = -1;
  while ((c = prop->NextElementAfter(c)) >= 0) {
    AddChar(c);
  }
}

void CharProperty::AddCharSpec(const int *unicodes, int num_unicodes) {
  for (int i = 0; i < num_unicodes; ++i) {
    if (i + 3 < num_unicodes && unicodes[i] == kPreUnicodeRange &&
        unicodes[i + 3] == kPostUnicodeRange) {
      // Range of unicode values
      int lower = unicodes[i + 1];
      int upper = unicodes[i + 2];
      i += 3;  // i will be incremented once more at top of loop
      CHECK(lower <= upper) << ": invalid char range in " << name_
                            << ": [" << UnicodeToString(lower) << ", "
                            << UnicodeToString(upper) << "]";
      AddCharRange(lower, upper);
    } else {
      AddChar(unicodes[i]);
    }
  }
}

bool CharProperty::HoldsFor(int c) const {
  if (!UniLib::IsValidCodepoint(c)) return false;
  char buf[UTFmax];
  Rune r = c;
  runetochar(buf, &r);
  return impl_->HoldsFor(buf);
}

bool CharProperty::HoldsFor(const char *str, int len) const {
  // UniLib::IsUTF8ValidCodepoint also checks for structural validity.
  return len > 0 && UniLib::IsUTF8ValidCodepoint(StringPiece(str, len)) &&
         impl_->HoldsFor(str);
}

// Return -1 or the smallest Unicode char greater than c for which
// the CharProperty holds.  Expects c == -1 or HoldsFor(c).
int CharProperty::NextElementAfter(int c) const {
  DCHECK(c == -1 || HoldsFor(c));
  unordered_set<char32>::const_iterator end = impl_->chars.end();
  if (c < 0) {
    unordered_set<char32>::const_iterator it = impl_->chars.begin();
    if (it == end) return -1;
    return *it;
  }
  char32 r = c;
  unordered_set<char32>::const_iterator it = impl_->chars.find(r);
  if (it == end) return -1;
  it++;
  if (it == end) return -1;
  return *it;
}

REGISTER_SYNTAXNET_CLASS_REGISTRY("char property wrapper", CharPropertyWrapper);

const CharProperty *CharProperty::Lookup(const char *subclass) {
  // Create a CharPropertyWrapper object and delete it.  We only care about
  // the CharProperty it provides.
  std::unique_ptr<CharPropertyWrapper> wrapper(
      CharPropertyWrapper::Create(subclass));
  if (wrapper == nullptr) {
    LOG(ERROR) << "CharPropertyWrapper not found for subclass: "
               << "\"" << subclass << "\"";
    return nullptr;
  }
  return wrapper->GetCharProperty();
}

// Check that a given Unicode value is in range.
void CharProperty::CheckUnicodeVal(int c) const {
  CHECK(UniLib::IsValidCodepoint(c))
      << "Unicode in " << name_ << " out of range: " << UnicodeToString(c);
}

// Converts a Unicode value to a string (for error messages).
string CharProperty::UnicodeToString(int c) {
  const char *fmt;

  if (c < 0) {
    fmt = "%d";      // out-of-range
  } else if (c <= 0x7f) {
    fmt = "'%c'";    // ascii
  } else if (c <= 0xffff) {
    fmt = "0x%04X";  // 4 hex digits
  } else {
    fmt = "0x%X";    // also out-of-range
  }

  return tensorflow::strings::Printf(fmt, c);
}

//======================================================================
// Expression-level punctuation
//

// Punctuation that starts a sentence.
DEFINE_CHAR_PROPERTY_AS_SET(start_sentence_punc,
  0x00A1,  // Spanish inverted exclamation mark
  0x00BF,  // Spanish inverted question mark
)

// Punctuation that ends a sentence.
// Based on: http://www.unicode.org/unicode/reports/tr29/#Sentence_Boundaries
DEFINE_CHAR_PROPERTY_AS_SET(end_sentence_punc,
  '.',
  '!',
  '?',
  0x055C,  // Armenian exclamation mark
  0x055E,  // Armenian question mark
  0x0589,  // Armenian full stop
  0x061F,  // Arabic question mark
  0x06D4,  // Arabic full stop
  0x0700,  // Syriac end of paragraph
  0x0701,  // Syriac supralinear full stop
  0x0702,  // Syriac sublinear full stop
  RANGE(0x0964, 0x0965),  // Devanagari danda..Devanagari double danda
  0x1362,  // Ethiopic full stop
  0x1367,  // Ethiopic question mark
  0x1368,  // Ethiopic paragraph separator
  0x104A,  // Myanmar sign little section
  0x104B,  // Myanmar sign section
  0x166E,  // Canadian syllabics full stop
  0x17d4,  // Khmer sign khan
  0x1803,  // Mongolian full stop
  0x1809,  // Mongolian Manchu full stop
  0x1944,  // Limbu exclamation mark
  0x1945,  // Limbu question mark
  0x203C,  // double exclamation mark
  0x203D,  // interrobang
  0x2047,  // double question mark
  0x2048,  // question exclamation mark
  0x2049,  // exclamation question mark
  0x3002,  // ideographic full stop
  0x037E,  // Greek question mark
  0xFE52,  // small full stop
  0xFE56,  // small question mark
  0xFE57,  // small exclamation mark
  0xFF01,  // fullwidth exclamation mark
  0xFF0E,  // fullwidth full stop
  0xFF1F,  // fullwidth question mark
  0xFF61,  // halfwidth ideographic full stop
  0x2026,  // ellipsis
)

// Punctuation, such as parens, that opens a "nested expression" of text.
DEFINE_CHAR_PROPERTY_AS_SET(open_expr_punc,
  '(',
  '[',
  '<',
  '{',
  0x207D,  // superscript left parenthesis
  0x208D,  // subscript left parenthesis
  0x27E6,  // mathematical left white square bracket
  0x27E8,  // mathematical left angle bracket
  0x27EA,  // mathematical left double angle bracket
  0x2983,  // left white curly bracket
  0x2985,  // left white parenthesis
  0x2987,  // Z notation left image bracket
  0x2989,  // Z notation left binding bracket
  0x298B,  // left square bracket with underbar
  0x298D,  // left square bracket with tick in top corner
  0x298F,  // left square bracket with tick in bottom corner
  0x2991,  // left angle bracket with dot
  0x2993,  // left arc less-than bracket
  0x2995,  // double left arc greater-than bracket
  0x2997,  // left black tortoise shell bracket
  0x29D8,  // left wiggly fence
  0x29DA,  // left double wiggly fence
  0x29FC,  // left-pointing curved angle bracket
  0x3008,  // CJK left angle bracket
  0x300A,  // CJK left double angle bracket
  0x3010,  // CJK left black lenticular bracket
  0x3014,  // CJK left tortoise shell bracket
  0x3016,  // CJK left white lenticular bracket
  0x3018,  // CJK left white tortoise shell bracket
  0x301A,  // CJK left white square bracket
  0xFD3E,  // Ornate left parenthesis
  0xFE59,  // small left parenthesis
  0xFE5B,  // small left curly bracket
  0xFF08,  // fullwidth left parenthesis
  0xFF3B,  // fullwidth left square bracket
  0xFF5B,  // fullwidth left curly bracket
)

// Punctuation, such as parens, that closes a "nested expression" of text.
DEFINE_CHAR_PROPERTY_AS_SET(close_expr_punc,
  ')',
  ']',
  '>',
  '}',
  0x207E,  // superscript right parenthesis
  0x208E,  // subscript right parenthesis
  0x27E7,  // mathematical right white square bracket
  0x27E9,  // mathematical right angle bracket
  0x27EB,  // mathematical right double angle bracket
  0x2984,  // right white curly bracket
  0x2986,  // right white parenthesis
  0x2988,  // Z notation right image bracket
  0x298A,  // Z notation right binding bracket
  0x298C,  // right square bracket with underbar
  0x298E,  // right square bracket with tick in top corner
  0x2990,  // right square bracket with tick in bottom corner
  0x2992,  // right angle bracket with dot
  0x2994,  // right arc greater-than bracket
  0x2996,  // double right arc less-than bracket
  0x2998,  // right black tortoise shell bracket
  0x29D9,  // right wiggly fence
  0x29DB,  // right double wiggly fence
  0x29FD,  // right-pointing curved angle bracket
  0x3009,  // CJK right angle bracket
  0x300B,  // CJK right double angle bracket
  0x3011,  // CJK right black lenticular bracket
  0x3015,  // CJK right tortoise shell bracket
  0x3017,  // CJK right white lenticular bracket
  0x3019,  // CJK right white tortoise shell bracket
  0x301B,  // CJK right white square bracket
  0xFD3F,  // Ornate right parenthesis
  0xFE5A,  // small right parenthesis
  0xFE5C,  // small right curly bracket
  0xFF09,  // fullwidth right parenthesis
  0xFF3D,  // fullwidth right square bracket
  0xFF5D,  // fullwidth right curly bracket
)

// Chars that open a quotation.
// Based on: http://www.unicode.org/uni2book/ch06.pdf
DEFINE_CHAR_PROPERTY_AS_SET(open_quote,
  '"',
  '\'',
  '`',
  0xFF07,  // fullwidth apostrophe
  0xFF02,  // fullwidth quotation mark
  0x2018,  // left single quotation mark (English, others)
  0x201C,  // left double quotation mark (English, others)
  0x201B,  // single high-reveresed-9 quotation mark (PropList.txt)
  0x201A,  // single low-9 quotation mark (Czech, German, Slovak)
  0x201E,  // double low-9 quotation mark (Czech, German, Slovak)
  0x201F,  // double high-reversed-9 quotation mark (PropList.txt)
  0x2019,  // right single quotation mark (Danish, Finnish, Swedish, Norw.)
  0x201D,  // right double quotation mark (Danish, Finnish, Swedish, Norw.)
  0x2039,  // single left-pointing angle quotation mark (French, others)
  0x00AB,  // left-pointing double angle quotation mark (French, others)
  0x203A,  // single right-pointing angle quotation mark (Slovenian, others)
  0x00BB,  // right-pointing double angle quotation mark (Slovenian, others)
  0x300C,  // left corner bracket (East Asian languages)
  0xFE41,  // presentation form for vertical left corner bracket
  0xFF62,  // halfwidth left corner bracket (East Asian languages)
  0x300E,  // left white corner bracket (East Asian languages)
  0xFE43,  // presentation form for vertical left white corner bracket
  0x301D,  // reversed double prime quotation mark (East Asian langs, horiz.)
)

// Chars that close a quotation.
// Based on: http://www.unicode.org/uni2book/ch06.pdf
DEFINE_CHAR_PROPERTY_AS_SET(close_quote,
  '\'',
  '"',
  '`',
  0xFF07,  // fullwidth apostrophe
  0xFF02,  // fullwidth quotation mark
  0x2019,  // right single quotation mark (English, others)
  0x201D,  // right double quotation mark (English, others)
  0x2018,  // left single quotation mark (Czech, German, Slovak)
  0x201C,  // left double quotation mark (Czech, German, Slovak)
  0x203A,  // single right-pointing angle quotation mark (French, others)
  0x00BB,  // right-pointing double angle quotation mark (French, others)
  0x2039,  // single left-pointing angle quotation mark (Slovenian, others)
  0x00AB,  // left-pointing double angle quotation mark (Slovenian, others)
  0x300D,  // right corner bracket (East Asian languages)
  0xfe42,  // presentation form for vertical right corner bracket
  0xFF63,  // halfwidth right corner bracket (East Asian languages)
  0x300F,  // right white corner bracket (East Asian languages)
  0xfe44,  // presentation form for vertical right white corner bracket
  0x301F,  // low double prime quotation mark (East Asian languages)
  0x301E,  // close double prime (East Asian languages written horizontally)
)

// Punctuation chars that open an expression or a quotation.
DEFINE_CHAR_PROPERTY(open_punc, prop) {
  prop->AddCharProperty("open_expr_punc");
  prop->AddCharProperty("open_quote");
}

// Punctuation chars that close an expression or a quotation.
DEFINE_CHAR_PROPERTY(close_punc, prop) {
  prop->AddCharProperty("close_expr_punc");
  prop->AddCharProperty("close_quote");
}

// Punctuation chars that can come at the beginning of a sentence.
DEFINE_CHAR_PROPERTY(leading_sentence_punc, prop) {
  prop->AddCharProperty("open_punc");
  prop->AddCharProperty("start_sentence_punc");
}

// Punctuation chars that can come at the end of a sentence.
DEFINE_CHAR_PROPERTY(trailing_sentence_punc, prop) {
  prop->AddCharProperty("close_punc");
  prop->AddCharProperty("end_sentence_punc");
}

//======================================================================
// Special symbols
//

// Currency symbols.
// From: http://www.unicode.org/charts/PDF/U20A0.pdf
DEFINE_CHAR_PROPERTY_AS_SET(currency_symbol,
  '$',
  // 0x00A2,  // cents (NB: typically FOLLOWS the amount)
  0x00A3,  // pounds and liras
  0x00A4,  // general currency sign
  0x00A5,  // yen or yuan
  0x0192,  // Dutch florin (latin small letter "f" with hook)
  0x09F2,  // Bengali rupee mark
  0x09F3,  // Bengali rupee sign
  0x0AF1,  // Guajarati rupee sign
  0x0BF9,  // Tamil rupee sign
  0x0E3F,  // Thai baht
  0x17DB,  // Khmer riel
  0x20A0,  // alternative euro sign
  0x20A1,  // Costa Rica, El Salvador (colon sign)
  0x20A2,  // Brazilian cruzeiro
  0x20A3,  // French Franc
  0x20A4,  // alternative lira sign
  0x20A5,  // mill sign (USA 1/10 cent)
  0x20A6,  // Nigerian Naira
  0x20A7,  // Spanish peseta
  0x20A8,  // Indian rupee
  0x20A9,  // Korean won
  0x20AA,  // Israeli new sheqel
  0x20AB,  // Vietnam dong
  0x20AC,  // euro sign
  0x20AD,  // Laotian kip
  0x20AE,  // Mongolian tugrik
  0x20AF,  // Greek drachma
  0x20B0,  // German penny
  0x20B1,  // Philippine peso (Mexican peso uses "$")
  0x2133,  // Old German mark (script capital M)
  0xFDFC,  // rial sign
  0xFFE0,  // fullwidth cents
  0xFFE1,  // fullwidth pounds
  0xFFE5,  // fullwidth Japanese yen
  0xFFE6,  // fullwidth Korean won
)

// Chinese bookquotes.
// They look like "<<" and ">>" except that they are single UTF8 chars
// (U+300A, U+300B). These are used in chinese as special
// punctuation, refering to the title of a book, an article, a movie,
// etc.  For example: "cellphone" means cellphone, but <<cellphone>>
// means (exclusively) the movie.
DEFINE_CHAR_PROPERTY_AS_SET(open_bookquote,
 0x300A
)

DEFINE_CHAR_PROPERTY_AS_SET(close_bookquote,
 0x300B
)

//======================================================================
// Token-level punctuation
//

// Token-prefix symbols, excluding currency symbols -- glom on
// to following token (esp. if no space after)
DEFINE_CHAR_PROPERTY_AS_SET(noncurrency_token_prefix_symbol,
  '#',
  0x2116,  // numero sign ("No")
)

// Token-prefix symbols -- glom on to following token (esp. if no space after)
DEFINE_CHAR_PROPERTY(token_prefix_symbol, prop) {
  prop->AddCharProperty("currency_symbol");
  prop->AddCharProperty("noncurrency_token_prefix_symbol");
}

// Token-suffix symbols -- glom on to preceding token (esp. if no space before)
DEFINE_CHAR_PROPERTY_AS_SET(token_suffix_symbol,
  '%',
  0x066A,  // Arabic percent sign
  0x2030,  // per mille
  0x2031,  // per ten thousand
  0x00A2,  // cents sign
  0x2125,  // ounces sign
  0x00AA,  // feminine ordinal indicator (Spanish)
  0x00BA,  // masculine ordinal indicator (Spanish)
  0x00B0,  // degrees
  0x2109,  // degrees Fahrenheit
  0x2103,  // degrees Celsius
  0x2126,  // ohms
  0x212A,  // Kelvin
  0x212B,  // Angstroms ("A" with circle on top)
  0x00A9,  // copyright
  0x2117,  // sound recording copyright (circled "P")
  0x2122,  // trade mark
  0x00AE,  // registered trade mark
  0x2120,  // service mark
  0x2106,  // cada una ("c/a" == "each" in Spanish)
  0x2020,  // dagger (can be used for footnotes)
  0x2021,  // double dagger (can be used for footnotes)
)

// Subscripts
DEFINE_CHAR_PROPERTY_AS_SET(subscript_symbol,
  0x2080,  // subscript 0
  0x2081,  // subscript 1
  0x2082,  // subscript 2
  0x2083,  // subscript 3
  0x2084,  // subscript 4
  0x2085,  // subscript 5
  0x2086,  // subscript 6
  0x2087,  // subscript 7
  0x2088,  // subscript 8
  0x2089,  // subscript 9
  0x208A,  // subscript "+"
  0x208B,  // subscript "-"
  0x208C,  // subscript "="
  0x208D,  // subscript "("
  0x208E,  // subscript ")"
)

// Superscripts
DEFINE_CHAR_PROPERTY_AS_SET(superscript_symbol,
  0x2070,  // superscript 0
  0x00B9,  // superscript 1
  0x00B2,  // superscript 2
  0x00B3,  // superscript 3
  0x2074,  // superscript 4
  0x2075,  // superscript 5
  0x2076,  // superscript 6
  0x2077,  // superscript 7
  0x2078,  // superscript 8
  0x2079,  // superscript 9
  0x2071,  // superscript Latin small "i"
  0x207A,  // superscript "+"
  0x207B,  // superscript "-"
  0x207C,  // superscript "="
  0x207D,  // superscript "("
  0x207E,  // superscript ")"
  0x207F,  // superscript Latin small "n"
)

//======================================================================
// General punctuation
//

// Connector punctuation
// Code Pc from http://www.unicode.org/Public/UNIDATA/PropList.txt
// NB: This list is not necessarily exhaustive.
DEFINE_CHAR_PROPERTY_AS_SET(connector_punc,
  0x30fb,  // Katakana middle dot
  0xff65,  // halfwidth Katakana middle dot
  0x2040,  // character tie
)

// Dashes
// Code Pd from http://www.unicode.org/Public/UNIDATA/PropList.txt
// NB: This list is not necessarily exhaustive.
DEFINE_CHAR_PROPERTY_AS_SET(dash_punc,
  '-',
  '~',
  0x058a,  // Armenian hyphen
  0x1806,  // Mongolian todo soft hyphen
  RANGE(0x2010, 0x2015),  // hyphen..horizontal bar
  0x2053,  // swung dash -- from Table 6-3 of Unicode book
  0x207b,  // superscript minus
  0x208b,  // subscript minus
  0x2212,  // minus sign
  0x301c,  // wave dash
  0x3030,  // wavy dash
  RANGE(0xfe31, 0xfe32),  // presentation form for vertical em dash..en dash
  0xfe58,  // small em dash
  0xfe63,  // small hyphen-minus
  0xff0d,  // fullwidth hyphen-minus
)

// Other punctuation
// Code Po from http://www.unicode.org/Public/UNIDATA/UnicodeData.txt
// NB: This list is not exhaustive.
DEFINE_CHAR_PROPERTY_AS_SET(other_punc,
  ',',
  ':',
  ';',
  0x00b7,  // middle dot
  0x0387,  // Greek ano teleia
  0x05c3,  // Hebrew punctuation sof pasuq
  0x060c,  // Arabic comma
  0x061b,  // Arabic semicolon
  0x066b,  // Arabic decimal separator
  0x066c,  // Arabic thousands separator
  RANGE(0x0703, 0x70a),  // Syriac contraction and others
  0x070c,  // Syric harklean metobelus
  0x0e5a,  // Thai character angkhankhu
  0x0e5b,  // Thai character khomut
  0x0f08,  // Tibetan mark sbrul shad
  RANGE(0x0f0d, 0x0f12),  // Tibetan mark shad..Tibetan mark rgya gram shad
  0x1361,  // Ethiopic wordspace
  RANGE(0x1363, 0x1366),  // other Ethiopic chars
  0x166d,  // Canadian syllabics chi sign
  RANGE(0x16eb, 0x16ed),  // Runic single punctuation..Runic cross punctuation
  RANGE(0x17d5, 0x17d6),  // Khmer sign camnuc pii huuh and other
  0x17da,  // Khmer sign koomut
  0x1802,  // Mongolian comma
  RANGE(0x1804, 0x1805),  // Mongolian four dots and other
  0x1808,  // Mongolian manchu comma
  0x3001,  // ideographic comma
  RANGE(0xfe50, 0xfe51),  // small comma and others
  RANGE(0xfe54, 0xfe55),  // small semicolon and other
  0xff0c,  // fullwidth comma
  RANGE(0xff0e, 0xff0f),  // fullwidth stop..fullwidth solidus
  RANGE(0xff1a, 0xff1b),  // fullwidth colon..fullwidth semicolon
  0xff64,  // halfwidth ideographic comma
  0x2016,  // double vertical line
  RANGE(0x2032, 0x2034),  // prime..triple prime
  0xfe61,  // small asterisk
  0xfe68,  // small reverse solidus
  0xff3c,  // fullwidth reverse solidus
)

// All punctuation.
// Code P from http://www.unicode.org/Public/UNIDATA/PropList.txt
// NB: This list is not necessarily exhaustive.
DEFINE_CHAR_PROPERTY(punctuation, prop) {
  prop->AddCharProperty("open_punc");
  prop->AddCharProperty("close_punc");
  prop->AddCharProperty("leading_sentence_punc");
  prop->AddCharProperty("trailing_sentence_punc");
  prop->AddCharProperty("connector_punc");
  prop->AddCharProperty("dash_punc");
  prop->AddCharProperty("other_punc");
  prop->AddAsciiPredicate(&ispunct);
}

//======================================================================
// Separators
//

// Line separators
// Code Zl from http://www.unicode.org/Public/UNIDATA/PropList.txt
// NB: This list is not necessarily exhaustive.
DEFINE_CHAR_PROPERTY_AS_SET(line_separator,
  0x2028,                           // line separator
)

// Paragraph separators
// Code Zp from http://www.unicode.org/Public/UNIDATA/PropList.txt
// NB: This list is not necessarily exhaustive.
DEFINE_CHAR_PROPERTY_AS_SET(paragraph_separator,
  0x2029,                           // paragraph separator
)

// Space separators
// Code Zs from http://www.unicode.org/Public/UNIDATA/PropList.txt
// NB: This list is not necessarily exhaustive.
DEFINE_CHAR_PROPERTY_AS_SET(space_separator,
  0x0020,                           // space
  0x00a0,                           // no-break space
  0x1680,                           // Ogham space mark
  0x180e,                           // Mongolian vowel separator
  RANGE(0x2000, 0x200a),            // en quad..hair space
  0x202f,                           // narrow no-break space
  0x205f,                           // medium mathematical space
  0x3000,                           // ideographic space

  // Google additions
  0xe5e5,                           // "private" char used as space in Chinese
)

// Separators -- all line, paragraph, and space separators.
// Code Z from http://www.unicode.org/Public/UNIDATA/PropList.txt
// NB: This list is not necessarily exhaustive.
DEFINE_CHAR_PROPERTY(separator, prop) {
  prop->AddCharProperty("line_separator");
  prop->AddCharProperty("paragraph_separator");
  prop->AddCharProperty("space_separator");
  prop->AddAsciiPredicate(&isspace);
}

//======================================================================
// Alphanumeric Characters
//

// Digits
DEFINE_CHAR_PROPERTY_AS_SET(digit,
  RANGE('0', '9'),
  RANGE(0x0660, 0x0669),  // Arabic-Indic digits

  RANGE(0x06F0, 0x06F9),  // Eastern Arabic-Indic digits
)

//======================================================================
// Japanese Katakana
//

DEFINE_CHAR_PROPERTY_AS_SET(katakana,
  0x3099,  // COMBINING KATAKANA-HIRAGANA VOICED SOUND MARK
  0x309A,  // COMBINING KATAKANA-HIRAGANA SEMI-VOICED SOUND MARK
  0x309B,  // KATAKANA-HIRAGANA VOICED SOUND MARK
  0x309C,  // KATAKANA-HIRAGANA SEMI-VOICED SOUND MARK
  RANGE(0x30A0, 0x30FF),  // Fullwidth Katakana
  RANGE(0xFF65, 0xFF9F),  // Halfwidth Katakana
)

//======================================================================
// BiDi Directional Formatting Codes
//

// See http://www.unicode.org/reports/tr9/ for a description of Bidi
// and http://www.unicode.org/charts/PDF/U2000.pdf for the character codes.
DEFINE_CHAR_PROPERTY_AS_SET(directional_formatting_code,
  0x200E,  // LRM (Left-to-Right Mark)
  0x200F,  // RLM (Right-to-Left Mark)
  0x202A,  // LRE (Left-to-Right Embedding)
  0x202B,  // RLE (Right-to-Left Embedding)
  0x202C,  // PDF (Pop Directional Format)
  0x202D,  // LRO (Left-to-Right Override)
  0x202E,  // RLO (Right-to-Left Override)
)

//======================================================================
// Special collections
//

// NB: This does not check for all punctuation and symbols in the
// standard; just those listed in our code. See the definitions in
// char_properties.cc
DEFINE_CHAR_PROPERTY(punctuation_or_symbol, prop) {
  prop->AddCharProperty("punctuation");
  prop->AddCharProperty("subscript_symbol");
  prop->AddCharProperty("superscript_symbol");
  prop->AddCharProperty("token_prefix_symbol");
  prop->AddCharProperty("token_suffix_symbol");
}

}  // namespace syntaxnet
