/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tf_ops/text_distorter.h"  // seq_flow_lite

using tensorflow::uint32;

// Distorts the words in the text by inserting, deleting and swapping
// unicodes randomly with probability one third of distortion_probability.
std::string TextDistorter::DistortText(icu::UnicodeString* uword) {
  if (distortion_probability_ > 0.0 &&
      generator_.RandFloat() < distortion_probability_ && uword->length()) {
    // Distort text with non zero length with distortion_probability_.
    float distortion_type = generator_.RandFloat();
    uint32 rindex = generator_.Rand32() % uword->length();
    if (distortion_type < 0.33f) {
      // Remove character with one third probability.
      random_char_ = (*uword)[rindex];
      uword->remove(rindex, 1);
    } else if (distortion_type < 0.66f) {
      // Swap character with one third probability if there are more than 2
      // characters.
      if (uword->length() > 2) {
        random_char_ = (*uword)[rindex];
        uword->remove(rindex, 1);
        uword->insert(generator_.Rand32() % uword->length(), random_char_);
      }
    } else if (random_char_) {
      // Insert character with one third probability.
      uword->insert(rindex, random_char_);
    }
  }
  // Convert unicode sequence back to string.
  std::string word;
  icu::StringByteSink<std::string> sink(&word);
  uword->toUTF8(sink);
  return word;
}
