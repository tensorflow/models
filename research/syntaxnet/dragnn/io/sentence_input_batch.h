// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#ifndef DRAGNN_IO_SENTENCE_INPUT_BATCH_H_
#define DRAGNN_IO_SENTENCE_INPUT_BATCH_H_

#include <string>
#include <vector>

#include "dragnn/core/interfaces/input_batch.h"
#include "dragnn/io/syntaxnet_sentence.h"
#include "syntaxnet/base.h"

namespace syntaxnet {
namespace dragnn {

// Data accessor backed by a syntaxnet::Sentence object.
class SentenceInputBatch : public InputBatch {
 public:
  SentenceInputBatch() {}

  // Translates from a vector of stringified Sentence protos.
  void SetData(
      const std::vector<string> &stringified_sentence_protos) override;

  // Returns the size of the batch.
  int GetSize() const override { return data_.size(); }

  // Translates to a vector of stringified Sentence protos.
  const std::vector<string> GetSerializedData() const override;

  // Get the underlying Sentences.
  std::vector<SyntaxNetSentence> *data() { return &data_; }

 private:
  // The backing Sentence protos.
  std::vector<SyntaxNetSentence> data_;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_IO_SENTENCE_INPUT_BATCH_H_
