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

#include <memory>

#include "syntaxnet/feature_extractor.h"
#include "syntaxnet/parser_features.h"
#include "syntaxnet/parser_state.h"
#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/term_frequency_map.h"
#include "syntaxnet/workspace.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace {

// Testing rig for exercising whole-sentence features, forwarded through parser
// features.
class WholeSentenceFeaturesTest : public ::testing::Test {
 protected:
  // Initializes the feature extractor from the |spec|.
  void Init(const string &spec) {
    extractor_.Parse(spec);
    extractor_.Setup(&context_);
    extractor_.Init(&context_);
    extractor_.RequestWorkspaces(&registry_);
    workspaces_.Reset(registry_);
    state_.reset(new ParserState(&sentence_, nullptr /* no transition state */,
                                 &label_map_));
    extractor_.Preprocess(&workspaces_, state_.get());
  }

  // Checks that the whole-sentence feature fired with the expected value.
  // Assumes Init() has been called.
  void ExpectValue(string value) {
    FeatureVector result;
    extractor_.ExtractFeatures(workspaces_, *state_, &result);
    ASSERT_EQ(result.size(), 1);
    EXPECT_EQ(value, result.type(0)->GetFeatureValueName(result.value(0)));
  }

  Token *AddToken() { return sentence_.add_token(); }

  TermFrequencyMap label_map_;
  TaskContext context_;
  WorkspaceRegistry registry_;
  ParserFeatureExtractor extractor_;
  Sentence sentence_;
  WorkspaceSet workspaces_;
  std::unique_ptr<ParserState> state_;
};

TEST_F(WholeSentenceFeaturesTest, SentenceLengthEmpty) {
  Init("sentence.length");

  ExpectValue("0");
}

TEST_F(WholeSentenceFeaturesTest, SentenceLengthPopulated) {
  AddToken()->set_word("test");
  AddToken()->set_word("test");
  AddToken()->set_word("test");

  Init("sentence.length");

  ExpectValue("3");
}

TEST_F(WholeSentenceFeaturesTest, SentenceLengthClipped) {
  AddToken()->set_word("test");
  AddToken()->set_word("test");
  AddToken()->set_word("test");

  Init("sentence.length(max-length=1)");

  ExpectValue("1");
}

}  // namespace
}  // namespace syntaxnet
