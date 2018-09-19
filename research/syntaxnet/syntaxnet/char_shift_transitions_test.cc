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

#include "syntaxnet/char_shift_transitions.h"

#include <memory>

#include "syntaxnet/parser_features.h"
#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/term_frequency_map.h"
#include "syntaxnet/workspace.h"
#include "tensorflow/core/platform/test.h"

namespace syntaxnet {

class CharShiftTransitionTest : public ::testing::Test {
 public:
  void SetUp() override {
    const char *str_sentence =
        "text: 'I saw a man with a กขค.' "
        "token { word: 'I' start: 0 end: 0 tag: 'PRP' category: 'PRON'"
        " head: 1 label: 'nsubj' break_level: NO_BREAK } "
        "token { word: 'saw' start: 2 end: 4 tag: 'VBD' category: 'VERB'"
        " label: 'ROOT' break_level: SPACE_BREAK } "
        "token { word: 'a' start: 6 end: 6 tag: 'DT' category: 'DET'"
        " head: 3 label: 'det' break_level: SPACE_BREAK } "
        "token { word: 'man' start: 8 end: 10 tag: 'NN' category: 'NOUN'"
        " head: 1 label: 'dobj' break_level: SPACE_BREAK } "
        "token { word: 'with' start: 12 end: 15 tag: 'IN' category: 'ADP'"
        " head: 1 label: 'prep' break_level: SPACE_BREAK } "
        "token { word: 'a' start: 17 end: 17 tag: 'DT' category: 'DET'"
        " head: 6 label: 'det' break_level: SPACE_BREAK } "
        "token { word: 'กขค' start: 19 end: 27 tag: 'NN' category: "
        "'NOUN'"
        " head: 4 label: 'pobj'  break_level: SPACE_BREAK } "
        "token { word: '.' start: 28 end: 28 tag: '.' category: '.'"
        " head: 1 label: 'p' break_level: NO_BREAK } ";
    TextFormat::ParseFromString(str_sentence, &sentence_);

    // Populates char-map manually.
    char_map_.Increment(" ");
    char_map_.Increment("I");
    char_map_.Increment("s");
    char_map_.Increment("a");
    char_map_.Increment("w");
    char_map_.Increment("a");
    char_map_.Increment("m");
    char_map_.Increment("a");
    char_map_.Increment("n");
    char_map_.Increment("w");
    char_map_.Increment("i");
    char_map_.Increment("t");
    char_map_.Increment("h");
    char_map_.Increment("a");
    char_map_.Increment("ก");
    char_map_.Increment("ข");
    char_map_.Increment("ค");
    string char_map_filename =
        tensorflow::strings::StrCat(tensorflow::testing::TmpDir(), "char-map");
    char_map_.Save(char_map_filename);
    AddInputToContext("char-map", char_map_filename, "text", "");
  }

  void PrepareCharTransition(bool left_to_right) {
    context_.SetParameter("left_to_right", left_to_right ? "true" : "false");
    transition_system_.reset(ParserTransitionSystem::Create("char-shift-only"));
    transition_system_->Setup(&context_);

    // Parser state.
    state_.reset(new ParserState(
        &sentence_, transition_system_->NewTransitionState(true), &label_map_));
    char_state_ = reinterpret_cast<const CharShiftTransitionState *>(
        state_->transition_state());
  }

  void PrepareShiftTransition(bool left_to_right) {
    context_.SetParameter("left_to_right", left_to_right ? "true" : "false");
    transition_system_.reset(ParserTransitionSystem::Create("shift-only"));
    transition_system_->Setup(&context_);
    state_.reset(new ParserState(
        &sentence_, transition_system_->NewTransitionState(true), &label_map_));
    char_state_ = nullptr;
  }

  void PrepareExtractor(const string &feature_name) {
    extractor_.Parse(feature_name);
    extractor_.Setup(&context_);
    extractor_.Init(&context_);
    extractor_.RequestWorkspaces(&registry_);
    workspace_.Reset(registry_);
    extractor_.Preprocess(&workspace_, state_.get());
  }

  void AddInputToContext(const string &name, const string &file_pattern,
                         const string &file_format,
                         const string &record_format) {
    TaskInput *input = context_.GetInput(name);
    TaskInput::Part *part = input->add_part();
    part->set_file_pattern(file_pattern);
    part->set_file_format(file_format);
    part->set_record_format(record_format);
  }

 protected:
  string MultiFeatureString(const FeatureVector &result) {
    std::vector<string> values;
    values.reserve(result.size());
    for (int i = 0; i < result.size(); ++i) {
      values.push_back(result.type(i)->GetFeatureValueName(result.value(i)));
    }
    return utils::Join(values, ",");
  }

  Sentence sentence_;

  TaskContext context_;
  TaskInput *input_label_map_ = nullptr;
  TaskInput *input_tag_map_ = nullptr;
  TermFrequencyMap char_map_;
  TermFrequencyMap label_map_;

  std::unique_ptr<ParserTransitionSystem> transition_system_;
  std::unique_ptr<ParserState> state_;
  const CharShiftTransitionState *char_state_;
  ParserFeatureExtractor extractor_;
  WorkspaceRegistry registry_;
  WorkspaceSet workspace_;
};

TEST_F(CharShiftTransitionTest, LRShift) {
  PrepareCharTransition(true);
  int expected_next = 0;
  const std::vector<string> expected_actions = {
      "I:I",    " :I",    "s:saw",  "a:saw",  "w:saw",  " :saw",
      "a:a",    " :a",    "m:man",  "a:man",  "n:man",  " :man",
      "w:with", "i:with", "t:with", "h:with", " :with", "a:a",
      " :a",    "ก:กขค",  "ข:กขค",  "ค:กขค",  ".:."};
  EXPECT_EQ(char_state_->Next(), expected_next);
  while (!transition_system_->IsFinalState(*state_)) {
    ParserAction action = transition_system_->GetNextGoldAction(*state_);
    EXPECT_TRUE(transition_system_->IsAllowedAction(action, *state_));
    EXPECT_EQ(transition_system_->ActionAsString(action, *state_),
              expected_actions[expected_next]);
    transition_system_->PerformActionWithoutHistory(action, state_.get());
    ++expected_next;
    EXPECT_EQ(char_state_->Next(), expected_next);
  }
}

TEST_F(CharShiftTransitionTest, RLShift) {
  PrepareCharTransition(false);
  int expected_next = 22;
  const std::vector<string> expected_actions = {
      "I:I",    " :saw",  "s:saw",  "a:saw",  "w:saw", " :a",
      "a:a",    " :man",  "m:man",  "a:man",  "n:man", " :with",
      "w:with", "i:with", "t:with", "h:with", " :a",   "a:a",
      " :กขค",  "ก:กขค",  "ข:กขค",  "ค:กขค",  ".:."};
  EXPECT_EQ(char_state_->Next(), expected_next);
  while (!transition_system_->IsFinalState(*state_)) {
    ParserAction action = transition_system_->GetNextGoldAction(*state_);
    EXPECT_TRUE(transition_system_->IsAllowedAction(action, *state_));
    EXPECT_EQ(transition_system_->ActionAsString(action, *state_),
              expected_actions[expected_next]);
    transition_system_->PerformActionWithoutHistory(action, state_.get());
    --expected_next;
    EXPECT_EQ(char_state_->Next(), expected_next);
  }
}

TEST_F(CharShiftTransitionTest, TextChar) {
  PrepareCharTransition(true);
  PrepareExtractor(
      "char-input(-2).text-char "
      "char-input(-1).text-char "
      "char-input.text-char "
      "char-input(1).text-char "
      "char-input(2).text-char ");
  FeatureVector features;
  ParserAction action;

  // "I s"
  features.clear();
  extractor_.ExtractFeatures(workspace_, *state_, &features);
  EXPECT_EQ(MultiFeatureString(features), "I,<SPACE>,s");

  // "I sa"
  action = transition_system_->GetNextGoldAction(*state_);
  transition_system_->PerformActionWithoutHistory(action, state_.get());

  // " saw"
  action = transition_system_->GetNextGoldAction(*state_);
  transition_system_->PerformActionWithoutHistory(action, state_.get());

  // "."
  while (!transition_system_->IsFinalState(*state_)) {
    action = transition_system_->GetNextGoldAction(*state_);
    transition_system_->PerformActionWithoutHistory(action, state_.get());
  }
  features.clear();
  extractor_.ExtractFeatures(workspace_, *state_, &features);
  EXPECT_EQ(MultiFeatureString(features), "ค,<UNKNOWN>");
}

TEST_F(CharShiftTransitionTest, LastCharFocus) {
  PrepareShiftTransition(true);
  PrepareExtractor(
      "input(-1).last-char-focus "
      "input.last-char-focus "
      "input(1).last-char-focus "
      "input(2).last-char-focus ");
  FeatureVector features;
  ParserAction action;

  // "I saw a"
  features.clear();
  extractor_.ExtractFeatures(workspace_, *state_, &features);
  EXPECT_EQ(MultiFeatureString(features), ",0,4,6");

  // "I saw a man"
  action = transition_system_->GetNextGoldAction(*state_);
  transition_system_->PerformActionWithoutHistory(action, state_.get());
  features.clear();
  extractor_.ExtractFeatures(workspace_, *state_, &features);
  EXPECT_EQ(MultiFeatureString(features), "0,4,6,10");

  // "saw a man with"
  action = transition_system_->GetNextGoldAction(*state_);
  transition_system_->PerformActionWithoutHistory(action, state_.get());

  // "."
  while (!transition_system_->IsFinalState(*state_)) {
    action = transition_system_->GetNextGoldAction(*state_);
    transition_system_->PerformActionWithoutHistory(action, state_.get());
  }
  features.clear();
  extractor_.ExtractFeatures(workspace_, *state_, &features);
  EXPECT_EQ(MultiFeatureString(features), "22,,,");
}

}  // namespace syntaxnet
