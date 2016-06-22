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
  if (!use_sentence_feed_) {
    reader_.reset(new TextReader(*context->GetInput(input_name_)));
  }
  size_ = 0;
}

void SentenceBatch::FeedSentences(std::vector<std::unique_ptr<Sentence>> &sentences) {
  for (size_t i = 0; i < sentences.size(); i++) {
    feed_sentences_.push_back(std::move(sentences[i]));
  }
  sentences.clear();
}

bool SentenceBatch::AdvanceSentence(int index) {
  //LOG(INFO) << "SentenceBatch advancing to " << index;
  if (sentences_[index] == nullptr) ++size_;
  sentences_[index].reset();
  Sentence *sentenceptr = nullptr;
  //LOG(INFO) << "use_sentence_feed:" <<index<<": "<< use_sentence_feed_
  //  << " sentence_feed_index:" << sentence_feed_index_ << " size:"
  //  << (use_sentence_feed_ ? feed_sentences_.size() : -1);
  if (!use_sentence_feed_) {
    sentenceptr = reader_->Read();
  } else if (sentence_feed_index_ < feed_sentences_.size()) {
    sentenceptr = new Sentence();
    sentenceptr->CopyFrom(*feed_sentences_[sentence_feed_index_++]);
  }
  std::unique_ptr<Sentence> sentence(sentenceptr);
  if (sentence == nullptr) {
    --size_;
    return false;
  }

  // Preprocess the new sentence for the parser state.
  sentences_[index] = std::move(sentence);
  return true;
}

}  // namespace syntaxnet
