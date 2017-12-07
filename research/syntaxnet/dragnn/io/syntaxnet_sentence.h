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

#ifndef DRAGNN_IO_SYNTAXNET_SENTENCE_H_
#define DRAGNN_IO_SYNTAXNET_SENTENCE_H_

#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/workspace.h"

namespace syntaxnet {
namespace dragnn {

class SyntaxNetSentence {
 public:
  SyntaxNetSentence(std::unique_ptr<Sentence> sentence,
                    std::unique_ptr<WorkspaceSet> workspace)
      : sentence_(std::move(sentence)), workspace_(std::move(workspace)) {}

  Sentence *sentence() const { return sentence_.get(); }
  WorkspaceSet *workspace() const { return workspace_.get(); }

 private:
  std::unique_ptr<Sentence> sentence_;
  std::unique_ptr<WorkspaceSet> workspace_;
};

}  // namespace dragnn
}  // namespace syntaxnet

#endif  // DRAGNN_IO_SYNTAXNET_SENTENCE_H_
