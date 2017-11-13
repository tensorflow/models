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

#include <memory>

#include "syntaxnet/base.h"
#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"
#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/term_frequency_map.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {
namespace {

const char kSentence[] = R"(
  text: 'I saw a man with a telescope.'
  token { word: 'I' start: 0 end: 0 tag: 'PRP' category: 'PRON'
          head: 1 label: 'nsubj' break_level: NO_BREAK }
  token { word: 'saw' start: 2 end: 4 tag: 'VBD' category: 'VERB'
          label: 'ROOT' break_level: SPACE_BREAK }
  token { word: 'a' start: 6 end: 6 tag: 'DT' category: 'DET'
          head: 3 label: 'det' break_level: SPACE_BREAK }
  token { word: 'man' start: 8 end: 10 tag: 'NN' category: 'NOUN'
          head: 1 label: 'dobj' break_level: SPACE_BREAK }
  token { word: 'with' start: 12 end: 15 tag: 'IN' category: 'ADP'
          head: 1 label: 'prep' break_level: SPACE_BREAK }
  token { word: 'a' start: 17 end: 17 tag: 'DT' category: 'DET'
          head: 6 label: 'det' break_level: SPACE_BREAK }
  token { word: 'telescope' start: 19 end: 27 tag: 'NN' category: 'NOUN'
          head: 4 label: 'pobj'  break_level: SPACE_BREAK }
  token { word: '.' start: 28 end: 28 tag: '.' category: '.'
          head: 1 label: 'p' break_level: NO_BREAK }
)";

class HeadTransitionSystemTest : public ::testing::Test {
 public:
  HeadTransitionSystemTest() {
    transition_system_->Setup(&context_);
    transition_system_->Init(&context_);
    CHECK(TextFormat::ParseFromString(kSentence, &sentence_));
    for (auto &token : sentence_.token()) label_map_.Increment(token.label());
    state_.reset(new ParserState(
        &sentence_, transition_system_->NewTransitionState(true), &label_map_));
  }

 protected:
  TermFrequencyMap label_map_;
  TaskContext context_;
  std::unique_ptr<ParserTransitionSystem> transition_system_{
      ParserTransitionSystem::Create("heads")};
  Sentence sentence_;
  std::unique_ptr<ParserState> state_;
};

TEST_F(HeadTransitionSystemTest, Characteristics) {
  EXPECT_EQ(1, transition_system_->NumActionTypes());
  EXPECT_EQ(ParserTransitionSystem::kDynamicNumActions,
            transition_system_->NumActions(10));
}

TEST_F(HeadTransitionSystemTest, GoldParsesCorrectly) {
  LOG(INFO) << "Initial parser state: " << state_->ToString();
  while (!transition_system_->IsFinalState(*state_)) {
    ParserAction action = transition_system_->GetNextGoldAction(*state_);
    EXPECT_TRUE(transition_system_->IsAllowedAction(action, *state_));
    LOG(INFO) << "Performing action: "
              << transition_system_->ActionAsString(action, *state_);
    transition_system_->PerformActionWithoutHistory(action, state_.get());
    LOG(INFO) << "Parser state: " << state_->ToString();
  }
  for (int i = 0; i < state_->NumTokens(); ++i) {
    EXPECT_EQ(state_->GoldHead(i), state_->Head(i));
  }
}

}  // namespace
}  // namespace syntaxnet
