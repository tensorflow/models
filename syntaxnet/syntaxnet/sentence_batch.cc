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

#include "syntaxnet/sentence_batch.h"

#include <memory>
#include <string>
#include <vector>

#include "syntaxnet/task_context.h"

namespace syntaxnet {

void SentenceBatch::Init(TaskContext *context) {
  reader_.reset(new TextReader(*context->GetInput(input_name_)));
  size_ = 0;
}

bool SentenceBatch::AdvanceSentence(int index) {
  if (sentences_[index] == nullptr) ++size_;
  sentences_[index].reset();
  std::unique_ptr<Sentence> sentence(reader_->Read());
  if (sentence == nullptr) {
    --size_;
    return false;
  }

  // Preprocess the new sentence for the parser state.
  sentences_[index] = std::move(sentence);
  return true;
}

}  // namespace syntaxnet
