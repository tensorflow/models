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

#include "dragnn/io/sentence_input_batch.h"

#include "syntaxnet/sentence.pb.h"

namespace syntaxnet {
namespace dragnn {

void SentenceInputBatch::SetData(
    const std::vector<string> &stringified_sentence_protos) {
  for (const auto &stringified_proto : stringified_sentence_protos) {
    std::unique_ptr<Sentence> sentence(new Sentence);
    std::unique_ptr<WorkspaceSet> workspace_set(new WorkspaceSet);
    CHECK(sentence->ParseFromString(stringified_proto))
        << "Unable to parse string input as syntaxnet.Sentence.";
    SyntaxNetSentence aug_sentence(std::move(sentence),
                                   std::move(workspace_set));
    data_.push_back(std::move(aug_sentence));
  }
}

const std::vector<string> SentenceInputBatch::GetSerializedData() const {
  std::vector<string> output_data;
  output_data.resize(data_.size());
  for (int i = 0; i < data_.size(); ++i) {
    data_[i].sentence()->SerializeToString(&(output_data[i]));
  }
  return output_data;
}

}  // namespace dragnn
}  // namespace syntaxnet
