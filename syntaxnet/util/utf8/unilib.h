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

// Routines to do manipulation of Unicode characters or text
//
// The StructurallyValid routines accept buffers of arbitrary bytes.
// For CoerceToStructurallyValid(), the input buffer and output buffers may
// point to exactly the same memory.
//
// In all other cases, the UTF-8 string must be structurally valid and
// have all codepoints in the range  U+0000 to U+D7FF or U+E000 to U+10FFFF.
// Debug builds take a fatal error for invalid UTF-8 input.
// The input and output buffers may not overlap at all.
//
// The char32 routines are here only for convenience; they convert to UTF-8
// internally and use the UTF-8 routines.

#ifndef UTIL_UTF8_UNILIB_H__
#define UTIL_UTF8_UNILIB_H__

#include <string>
#include "syntaxnet/base.h"

// We export OneCharLen, IsValidCodepoint, and IsTrailByte from here,
// but they are defined in unilib_utf8_utils.h.
//#include "util/utf8/public/unilib_utf8_utils.h"  // IWYU pragma: export

namespace UniLib {

// Returns the length in bytes of the prefix of src that is all
//  interchange valid UTF-8
int SpanInterchangeValid(const char* src, int byte_length);
inline int SpanInterchangeValid(const std::string& src) {
  return SpanInterchangeValid(src.data(), src.size());
}

// Returns true if the source is all interchange valid UTF-8
// "Interchange valid" is a stronger than structurally valid --
// no C0 or C1 control codes (other than CR LF HT FF) and no non-characters.
bool IsInterchangeValid(char32 codepoint);
inline bool IsInterchangeValid(const char* src, int byte_length) {
  return (byte_length == SpanInterchangeValid(src, byte_length));
}
inline bool IsInterchangeValid(const std::string& src) {
  return IsInterchangeValid(src.data(), src.size());
}

}  // namespace UniLib

#endif  // UTIL_UTF8_PUBLIC_UNILIB_H_
