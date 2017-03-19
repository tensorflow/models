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

#include "syntaxnet/whole_sentence_features.h"

#include <limits.h>
#include <algorithm>
#include <string>

#include "syntaxnet/base.h"
#include "syntaxnet/feature_types.h"
#include "syntaxnet/registry.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/workspace.h"

namespace syntaxnet {
namespace {

// Feature that extracts the sentence length in tokens.
class SentenceLength : public WholeSentenceFeatureFunction {
 public:
  void Init(TaskContext *context) override {
    max_length_ = GetIntParameter("max-length", INT_MAX - 1);
    CHECK_LT(max_length_, INT_MAX) << "max-length setting would overflow";
    set_feature_type(new NumericFeatureType(name(), max_length_ + 1));
  }

  FeatureValue Compute(const WorkspaceSet &workspaces, const Sentence &sentence,
                       const FeatureVector *result) const override {
    return std::min(sentence.token_size(), max_length_);
  }

 private:
  // Maximum allowed sentence length.
  int max_length_ = -1;
};

REGISTER_WHOLE_SENTENCE_FEATURE_FUNCTION("length", SentenceLength);

}  // namespace

REGISTER_SYNTAXNET_CLASS_REGISTRY("whole sentence feature function",
                                  WholeSentenceFeatureFunction);

}  // namespace syntaxnet
