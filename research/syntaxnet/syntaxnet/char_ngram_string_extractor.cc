/* Copyright 2017 Google Inc. All Rights Reserved.

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

#include "syntaxnet/char_ngram_string_extractor.h"

#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {

string CharNgramStringExtractor::GetConfigId() const {
  return tensorflow::strings::StrCat(min_length_, ":", max_length_, ":",
                                     add_terminators_, ":", mark_boundaries_);
}

void CharNgramStringExtractor::Setup(const TaskContext &context) {
  min_length_ = context.Get("lexicon_min_char_ngram_length", min_length_);
  max_length_ = context.Get("lexicon_max_char_ngram_length", max_length_);
  add_terminators_ =
      context.Get("lexicon_char_ngram_include_terminators", add_terminators_);
  CHECK(!add_terminators_ || !mark_boundaries_)
      << "Can't use both terminators and boundaries";
  CHECK_GE(min_length_, 1);
  CHECK_LE(min_length_, max_length_);
}

}  // namespace syntaxnet
