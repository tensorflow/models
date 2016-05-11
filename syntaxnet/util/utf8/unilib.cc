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

// Author: sligocki@google.com (Shawn Ligocki)

#include "util/utf8/unilib.h"

#include "syntaxnet/base.h"
#include "third_party/utf/utf.h"

namespace UniLib {

// Codepoints not allowed for interchange are:
//   C0 (ASCII) controls: U+0000 to U+001F excluding Space (SP, U+0020),
//       Horizontal Tab (HT, U+0009), Line-Feed (LF, U+000A),
//       Form Feed (FF, U+000C) and Carriage-Return (CR, U+000D)
//   C1 controls: U+007F to U+009F
//   Surrogates: U+D800 to U+DFFF
//   Non-characters: U+FDD0 to U+FDEF and U+xxFFFE to U+xxFFFF for all xx
bool IsInterchangeValid(char32 c) {
  return !((c >= 0x00 && c <= 0x08) || c == 0x0B || (c >= 0x0E && c <= 0x1F) ||
           (c >= 0x7F && c <= 0x9F) ||
           (c >= 0xD800 && c <= 0xDFFF) ||
           (c >= 0xFDD0 && c <= 0xFDEF) || (c&0xFFFE) == 0xFFFE);
}

int SpanInterchangeValid(const char* begin, int byte_length) {
  char32 rune;
  const char* p = begin;
  const char* end = begin + byte_length;
  while (p < end) {
    int bytes_consumed = charntorune(&rune, p, end - p);
    // We want to accept Runeerror == U+FFFD as a valid char, but it is used
    // by chartorune to indicate error. Luckily, the real codepoint is size 3
    // while errors return bytes_consumed <= 1.
    if ((rune == Runeerror && bytes_consumed <= 1) ||
        !IsInterchangeValid(rune)) {
      break;  // Found
    }
    p += bytes_consumed;
  }
  return p - begin;
}

}  // namespace UniLib
