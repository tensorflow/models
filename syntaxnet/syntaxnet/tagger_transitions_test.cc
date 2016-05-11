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

#include <memory>
#include <string>

#include "syntaxnet/utils.h"
#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"
#include "syntaxnet/populate_test_inputs.h"
#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/task_spec.pb.h"
#include "syntaxnet/term_frequency_map.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {

class TaggerTransitionTest : public ::testing::Test {
 public:
  TaggerTransitionTest()
      : transition_system_(ParserTransitionSystem::Create("tagger")) {}

 protected:
  // Creates a label map and a tag map for testing based on the given
  // document and initializes the transition system appropriately.
  void SetUpForDocument(const Sentence &document) {
    input_label_map_ = context_.GetInput("label-map", "text", "");
    input_label_map_ = context_.GetInput("tag-map", "text", "");
    transition_system_->Setup(&context_);
    PopulateTestInputs::Defaults(document).Populate(&context_);
    label_map_.Load(TaskContext::InputFile(*input_label_map_),
                    0 /* minimum frequency */,
                    -1 /* maximum number of terms */);
    transition_system_->Init(&context_);
  }

  // Creates a cloned state from a sentence in order to test that cloning
  // works correctly for the new parser states.
  ParserState *NewClonedState(Sentence *sentence) {
    ParserState state(sentence, transition_system_->NewTransitionState(
                                    true /* training mode */),
                      &label_map_);
    return state.Clone();
  }

  // Performs gold transitions and check that the labels and heads recorded
  // in the parser state match gold heads and labels.
  void GoldParse(Sentence *sentence) {
    ParserState *state = NewClonedState(sentence);
    LOG(INFO) << "Initial parser state: " << state->ToString();
    while (!transition_system_->IsFinalState(*state)) {
      ParserAction action = transition_system_->GetNextGoldAction(*state);
      EXPECT_TRUE(transition_system_->IsAllowedAction(action, *state));
      LOG(INFO) << "Performing action: "
                << transition_system_->ActionAsString(action, *state);
      transition_system_->PerformActionWithoutHistory(action, state);
      LOG(INFO) << "Parser state: " << state->ToString();
    }
    delete state;
  }

  // Always takes the default action, and verifies that this leads to
  // a final state through a sequence of allowed actions.
  void DefaultParse(Sentence *sentence) {
    ParserState *state = NewClonedState(sentence);
    LOG(INFO) << "Initial parser state: " << state->ToString();
    while (!transition_system_->IsFinalState(*state)) {
      ParserAction action = transition_system_->GetDefaultAction(*state);
      EXPECT_TRUE(transition_system_->IsAllowedAction(action, *state));
      LOG(INFO) << "Performing action: "
                << transition_system_->ActionAsString(action, *state);
      transition_system_->PerformActionWithoutHistory(action, state);
      LOG(INFO) << "Parser state: " << state->ToString();
    }
    delete state;
  }

  TaskContext context_;
  TaskInput *input_label_map_ = nullptr;
  TermFrequencyMap label_map_;
  std::unique_ptr<ParserTransitionSystem> transition_system_;
};

TEST_F(TaggerTransitionTest, SingleSentenceDocumentTest) {
  string document_text;
  Sentence document;
  TF_CHECK_OK(ReadFileToString(
      tensorflow::Env::Default(),
      "syntaxnet/testdata/document",
      &document_text));
  LOG(INFO) << "see doc\n:" << document_text;
  CHECK(TextFormat::ParseFromString(document_text, &document));
  SetUpForDocument(document);
  GoldParse(&document);
  DefaultParse(&document);
}

}  // namespace syntaxnet
