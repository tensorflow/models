/**
 * Copyright 2010 Google Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef UTIL_UTF8_PUBLIC_UNILIB_UTF8_UTILS_H_
#define UTIL_UTF8_PUBLIC_UNILIB_UTF8_UTILS_H_

// These definitions are self-contained and have no dependencies.
// They are also exported from unilib.h for legacy reasons.

#include "syntaxnet/base.h"
#include "third_party/utf/utf.h"

namespace UniLib {

// Returns true if 'c' is in the range [0, 0xD800) or [0xE000, 0x10FFFF]
// (i.e., is not a surrogate codepoint). See also
// IsValidCodepoint(const char* src) in util/utf8/public/unilib.h.
inline bool IsValidCodepoint(char32 c) {
  return (static_cast<uint32>(c) < 0xD800)
    || (c >= 0xE000 && c <= 0x10FFFF);
}

// Returns true if 'str' is the start of a structurally valid UTF-8
// sequence and is not a surrogate codepoint. Returns false if str.empty()
// or if str.length() < UniLib::OneCharLen(str[0]). Otherwise, this function
// will access 1-4 bytes of src, where n is UniLib::OneCharLen(src[0]).
inline bool IsUTF8ValidCodepoint(StringPiece str) {
  char32 c;
  int consumed;
  // It's OK if str.length() > consumed.
  return !str.empty()
      && isvalidcharntorune(str.data(), str.size(), &c, &consumed)
      && IsValidCodepoint(c);
}

// Returns the length (number of bytes) of the Unicode code point
// starting at src, based on inspecting just that one byte. This
// requires that src point to a well-formed UTF-8 string; the result
// is undefined otherwise.
inline int OneCharLen(const char* src) {
  return "\1\1\1\1\1\1\1\1\1\1\1\1\2\2\3\4"[(*src & 0xFF) >> 4];
}

// Returns true if this byte is a trailing UTF-8 byte (10xx xxxx)
inline bool IsTrailByte(char x) {
  // return (x & 0xC0) == 0x80;
  // Since trail bytes are always in [0x80, 0xBF], we can optimize:
  return static_cast<signed char>(x) < -0x40;
}

}  // namespace UniLib

#endif  // UTIL_UTF8_PUBLIC_UNILIB_UTF8_UTILS_H_
