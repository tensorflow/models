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
//
// A basic main function to test that UnicodeText builds.

#include <stdio.h>
#include <stdlib.h>

#include <string>

#include "util/utf8/unicodetext.h"

int main(int argc, char** argv) {
  if (argc > 1) {
    printf("Bytes:\n");
    std::string bytes(argv[1]);
    for (std::string::const_iterator iter = bytes.begin();
         iter < bytes.end(); ++iter) {
      printf("  0x%02X\n", *iter);
    }

    printf("Unicode codepoints:\n");
    UnicodeText text(UTF8ToUnicodeText(bytes));
    for (UnicodeText::const_iterator iter = text.begin();
         iter < text.end(); ++iter) {
      printf("  U+%X\n", *iter);
    }
  }
  return EXIT_SUCCESS;
}
