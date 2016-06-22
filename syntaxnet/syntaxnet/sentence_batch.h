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

#ifndef SYNTAXNET_SENTENCE_BATCH_H_
#define SYNTAXNET_SENTENCE_BATCH_H_

#include <memory>
#include <string>
#include <vector>

#include "syntaxnet/embedding_feature_extractor.h"
#include "syntaxnet/feature_extractor.h"
#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"
#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/sparse.pb.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/task_spec.pb.h"
#include "syntaxnet/term_frequency_map.h"

namespace syntaxnet {

// Helper class to manage generating batches of preprocessed ParserState objects
// by reading in multiple sentences in parallel.
class SentenceBatch {
 public:
   SentenceBatch(int batch_size, string input_name)
       : SentenceBatch(batch_size, input_name, false) {}

   SentenceBatch(int batch_size, string input_name, bool use_sentence_feed)
      : batch_size_(batch_size),
        input_name_(input_name),
        sentences_(batch_size),
        use_sentence_feed_(use_sentence_feed),
        sentence_feed_index_(0) {}

  // Initializes all resources and opens the corpus file.
  void Init(TaskContext *context);

  // Advances the index'th sentence in the batch to the next sentence. This will
  // create and preprocess a new ParserState for that element. Returns false if
  // EOF is reached (if EOF, also sets the state to be nullptr.)
  bool AdvanceSentence(int index);

  // Rewinds the corpus reader.
  void Rewind() {
    if (reader_ != nullptr) reader_->Reset();
    sentence_feed_index_ = 0;
  }

  int size() const { return size_; }

  Sentence *sentence(int index) { return sentences_[index].get(); }

  void FeedSentences(std::vector<std::unique_ptr<Sentence>> &sentences);

 private:
  // Running tally of non-nullptr states in the batch.
  int size_;

  // Maximum number of states in the batch.
  int batch_size_;

  // Input to read from the TaskContext.
  string input_name_;

  // Reader for the corpus.
  std::unique_ptr<TextReader> reader_;

  // Batch: Sentence objects.
  std::vector<std::unique_ptr<Sentence>> sentences_;

  // Sentence objects fed in, superceding the reader_ while non-empty
  std::vector<std::unique_ptr<Sentence>> feed_sentences_;

  bool use_sentence_feed_;

  int sentence_feed_index_;

};

}  // namespace syntaxnet

#endif  // SYNTAXNET_SENTENCE_BATCH_H_
