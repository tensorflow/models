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

// char_properties.h - define is_X() tests for various character properties
//
// Character properties can be defined in two ways:
//
// (1) Set-based:
//
//     Enumerate the chars that have the property.  Example:
//
//       DEFINE_CHAR_PROPERTY_AS_SET(my_fave,
//         RANGE('0', '9'),
//         '\'',
//         0x00BF,   // Spanish inverted question mark
//       )
//
//     Characters are expressed as Unicode code points; note that ascii codes
//     are a subset.  RANGE() specifies an inclusive range of code points.
//
//     This defines two functions:
//
//       bool is_my_fave(const char *str, int len)
//       bool is_my_fave(int c)
//
//     Each returns true for precisely the 12 characters specified above.
//     Each takes a *single* UTf8 char as its argument -- the first expresses
//     it as a char * and a length, the second as a Unicode code point.
//     Please do not pass a string of multiple UTF8 chars to the first one.
//
//     To make is_my_fave() externally accessible, put in your .h file:
//
//       DECLARE_CHAR_PROPERTY(my_fave)
//
// (2) Function-based:
//
//     Specify a function that assigns the desired chars to a CharProperty
//     object.  Example:
//
//       DEFINE_CHAR_PROPERTY(my_other_fave, prop) {
//         for (int i = '0'; i <= '9'; i += 2) {
//           prop->AddChar(i);
//         }
//         prop->AddAsciiPredicate(&ispunct);
//         prop->AddCharProperty("currency_symbol");
//       }
//
//     This defines a function of one arg: CharProperty *prop.  The function
//     calls various CharProperty methods to populate the prop.  The last call
//     above, AddCharProperty(), adds the chars from another char property
//     ("currency_symbol").
//
//     As in the set-based case, put a DECLARE_CHAR_PROPERTY(my_other_fave)
//     in your .h if you want is_my_other_fave() to be externally accessible.
//

#ifndef SYNTAXNET_CHAR_PROPERTIES_H_
#define SYNTAXNET_CHAR_PROPERTIES_H_

#include <string>  // for string

#include "syntaxnet/registry.h"
#include "syntaxnet/utils.h"

// =====================================================================
// Registry for accessing CharProperties by name
//
// This is for internal use by the CharProperty class and macros; callers
// should not use it explicitly.
//

namespace syntaxnet {

class CharProperty;   // forward declaration

// Wrapper around a CharProperty, allowing it to be stored in a registry.
struct CharPropertyWrapper : RegisterableClass<CharPropertyWrapper> {
  virtual ~CharPropertyWrapper() { }
  virtual CharProperty *GetCharProperty() = 0;
};

#define REGISTER_CHAR_PROPERTY_WRAPPER(type, component) \
  REGISTER_SYNTAXNET_CLASS_COMPONENT(CharPropertyWrapper, type, component)

#define REGISTER_CHAR_PROPERTY(lsp, name)                         \
  struct name##CharPropertyWrapper : public CharPropertyWrapper { \
    CharProperty *GetCharProperty() { return lsp.get(); }         \
  };                                                              \
  REGISTER_CHAR_PROPERTY_WRAPPER(#name, name##CharPropertyWrapper)

// =====================================================================
// Macros for defining character properties
//

// Define is_X() functions to test whether a single UTF8 character has
// the 'X' char prop.
#define DEFINE_IS_X_CHAR_PROPERTY_FUNCTIONS(lsp, name) \
  bool is_##name(const char *str, int len) {                                 \
    return lsp->HoldsFor(str, len);                                          \
  }                                                                          \
  bool is_##name(int c) {                                                    \
    return lsp->HoldsFor(c);                                                 \
  }

// Define a char property by enumerating the unicode char points,
// or RANGE()s thereof, for which it holds.  Example:
//
//   DEFINE_CHAR_PROPERTY_AS_SET(my_fave,
//     'q',
//     RANGE('0', '9'),
//     0x20AB,
//   )
//
// "..." is a GNU extension.
#define DEFINE_CHAR_PROPERTY_AS_SET(name, unicodes...)                         \
  static const int k_##name##_unicodes[] = {unicodes};                         \
  static utils::LazyStaticPtr<CharProperty, const char *, const int *, size_t> \
      name##_char_property = {#name, k_##name##_unicodes,                      \
                              arraysize(k_##name##_unicodes)};                 \
  REGISTER_CHAR_PROPERTY(name##_char_property, name);                          \
  DEFINE_IS_X_CHAR_PROPERTY_FUNCTIONS(name##_char_property, name)

// Specify a range (inclusive) of Unicode character values.
// Example: RANGE('0', '9') specifies the 10 digits.
// For use as an element in a DEFINE_CHAR_PROPERTY_AS_SET() list.
static const int kPreUnicodeRange = -1;
static const int kPostUnicodeRange = -2;
#define RANGE(lower, upper) \
  kPreUnicodeRange, lower, upper, kPostUnicodeRange

// A function to initialize a CharProperty.
typedef void CharPropertyInitializer(CharProperty *prop);

// Define a char property by specifying a block of code that initializes it.
// Example:
//
//   DEFINE_CHAR_PROPERTY(my_other_fave, prop) {
//     for (int i = '0'; i <= '9'; i += 2) {
//       prop->AddChar(i);
//     }
//     prop->AddAsciiPredicate(&ispunct);
//     prop->AddCharProperty("currency_symbol");
//   }
//
#define DEFINE_CHAR_PROPERTY(name, charpropvar)                       \
  static void init_##name##_char_property(CharProperty *charpropvar); \
  static utils::LazyStaticPtr<CharProperty, const char *,             \
                              CharPropertyInitializer *>              \
      name##_char_property = {#name, &init_##name##_char_property};   \
  REGISTER_CHAR_PROPERTY(name##_char_property, name);                 \
  DEFINE_IS_X_CHAR_PROPERTY_FUNCTIONS(name##_char_property, name)     \
  static void init_##name##_char_property(CharProperty *charpropvar)

// =====================================================================
// Macro for declaring character properties
//

#define DECLARE_CHAR_PROPERTY(name) \
  extern bool is_##name(const char *str, int len);                           \
  extern bool is_##name(int c);                                              \

// ===========================================================
// CharProperty - a property that holds for selected Unicode chars
//
// A CharProperty is semantically equivalent to set<char32>.
//
// The characters for which a CharProperty holds are represented as a trie,
// i.e., a tree that is indexed by successive bytes of the UTF-8 encoding
// of the characters.  This permits fast lookup (HoldsFor).
//

// A function that defines a subset of [0..255], e.g., isspace.
typedef int AsciiPredicate(int c);

class CharProperty {
 public:
  // Constructor for set-based char properties.
  CharProperty(const char *name, const int *unicodes, int num_unicodes);

  // Constructor for function-based char properties.
  CharProperty(const char *name, CharPropertyInitializer *init_fn);

  virtual ~CharProperty();

  // Various ways of adding chars to a CharProperty; for use only in
  // CharPropertyInitializer functions.
  void AddChar(int c);
  void AddCharRange(int c1, int c2);
  void AddAsciiPredicate(AsciiPredicate *pred);
  void AddCharProperty(const char *name);
  void AddCharSpec(const int *unicodes, int num_unicodes);

  // Return true iff the CharProperty holds for a single given UTF8 char.
  bool HoldsFor(const char *str, int len) const;

  // Return true iff the CharProperty holds for a single given Unicode char.
  bool HoldsFor(int c) const;

  // You can use this to enumerate the set elements (it was easier
  // than defining a real iterator).  Returns -1 if there are no more.
  // Call with -1 to get the first element.  Expects c == -1 or HoldsFor(c).
  int NextElementAfter(int c) const;

  // Return NULL or the CharProperty with the given name.  Looks up the name
  // in a CharProperty registry.
  static const CharProperty *Lookup(const char *name);

 private:
  void CheckUnicodeVal(int c) const;
  static string UnicodeToString(int c);

  const char *name_;
  struct CharPropertyImplementation *impl_;

  TF_DISALLOW_COPY_AND_ASSIGN(CharProperty);
};

//======================================================================
// Expression-level punctuation
//

// Punctuation that starts a sentence.
DECLARE_CHAR_PROPERTY(start_sentence_punc);

// Punctuation that ends a sentence.
DECLARE_CHAR_PROPERTY(end_sentence_punc);

// Punctuation, such as parens, that opens a "nested expression" of text.
DECLARE_CHAR_PROPERTY(open_expr_punc);

// Punctuation, such as parens, that closes a "nested expression" of text.
DECLARE_CHAR_PROPERTY(close_expr_punc);

// Chars that open a quotation.
DECLARE_CHAR_PROPERTY(open_quote);

// Chars that close a quotation.
DECLARE_CHAR_PROPERTY(close_quote);

// Punctuation chars that open an expression or a quotation.
DECLARE_CHAR_PROPERTY(open_punc);

// Punctuation chars that close an expression or a quotation.
DECLARE_CHAR_PROPERTY(close_punc);

// Punctuation chars that can come at the beginning of a sentence.
DECLARE_CHAR_PROPERTY(leading_sentence_punc);

// Punctuation chars that can come at the end of a sentence.
DECLARE_CHAR_PROPERTY(trailing_sentence_punc);

//======================================================================
// Token-level punctuation
//

// Token-prefix symbols -- glom on to following token
// (esp. if no space after) -- except for currency symbols.
DECLARE_CHAR_PROPERTY(noncurrency_token_prefix_symbol);

// Token-prefix symbols -- glom on to following token (esp. if no space after).
DECLARE_CHAR_PROPERTY(token_prefix_symbol);

// Token-suffix symbols -- glom on to preceding token (esp. if no space
// before).
DECLARE_CHAR_PROPERTY(token_suffix_symbol);

// Subscripts.
DECLARE_CHAR_PROPERTY(subscript_symbol);

// Superscripts.
DECLARE_CHAR_PROPERTY(superscript_symbol);

//======================================================================
// General punctuation
//

// Connector punctuation.
DECLARE_CHAR_PROPERTY(connector_punc);

// Dashes.
DECLARE_CHAR_PROPERTY(dash_punc);

// Other punctuation.
DECLARE_CHAR_PROPERTY(other_punc);

// All punctuation.
DECLARE_CHAR_PROPERTY(punctuation);

//======================================================================
// Special symbols
//

// Currency symbols.
DECLARE_CHAR_PROPERTY(currency_symbol);

// Chinese bookquotes.
DECLARE_CHAR_PROPERTY(open_bookquote);
DECLARE_CHAR_PROPERTY(close_bookquote);

//======================================================================
// Separators
//

// Line separators.
DECLARE_CHAR_PROPERTY(line_separator);

// Paragraph separators.
DECLARE_CHAR_PROPERTY(paragraph_separator);

// Space separators.
DECLARE_CHAR_PROPERTY(space_separator);

// Separators -- all line, paragraph, and space separators.
DECLARE_CHAR_PROPERTY(separator);

//======================================================================
// Alphanumeric Characters
//

// Digits.
DECLARE_CHAR_PROPERTY(digit);

// Japanese Katakana.
DECLARE_CHAR_PROPERTY(katakana);

//======================================================================
// BiDi Directional Formatting Codes
//

// Explicit directional formatting codes (LRM, RLM, LRE, RLE, PDF, LRO, RLO)
// used by the bidirectional algorithm.
//
// Note: Use this only to classify characters. To actually determine
// directionality of BiDi text, look under i18n/bidi.
//
// See http://www.unicode.org/reports/tr9/ for a description of the algorithm
// and http://www.unicode.org/charts/PDF/U2000.pdf for the character codes.
DECLARE_CHAR_PROPERTY(directional_formatting_code);

//======================================================================
// Special collections
//

// NB: This does not check for all punctuation and symbols in the standard;
// just those listed in our code. See the definitions in char_properties.cc.
DECLARE_CHAR_PROPERTY(punctuation_or_symbol);

DECLARE_SYNTAXNET_CLASS_REGISTRY("char property wrapper", CharPropertyWrapper);

}  // namespace syntaxnet

#endif  // SYNTAXNET_CHAR_PROPERTIES_H_
