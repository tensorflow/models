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

#include "syntaxnet/binary_segment_state.h"
#include "syntaxnet/parser_features.h"
#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"
#include "syntaxnet/term_frequency_map.h"

namespace syntaxnet {

// Given an input of utf8 characters, the BinarySegmentTransitionSystem
// conducts word segmentation by performing one of the following two actions:
//  -START: starts a new word with the token at state.input, and also advances
//          the state.input.
//  -MERGE: adds the token at state.input to its prevous word, and also advances
//          state.input.
//
// Also see binary_segment_state.h for examples on handling spaces.
class BinarySegmentTransitionSystem : public ParserTransitionSystem {
 public:
  BinarySegmentTransitionSystem() {}
  ParserTransitionState *NewTransitionState(bool train_mode) const override {
    return new BinarySegmentState();
  }

  // Action types for the segmentation-transition system.
  enum ParserActionType {
    START = 0,
    MERGE = 1,
    CARDINAL = 2
  };

  static int StartAction() { return 0; }
  static int MergeAction() { return 1; }

  // The system always starts a new word by default.
  ParserAction GetDefaultAction(const ParserState &state) const override {
    return START;
  }

  // Returns the number of action types.
  int NumActionTypes() const override {
    return CARDINAL;
  }

  // Returns the number of possible actions.
  int NumActions(int num_labels) const override {
    return CARDINAL;
  }

  // Returns the next gold action for a given state according to the underlying
  // annotated sentence. The training data for the transition system is created
  // by the binary-segmenter-data task. If a token's break_level is NO_BREAK,
  // then it is a MERGE, START otherwise. The only exception is that the first
  // token in a sentence for the transition sysytem is always a START.
  ParserAction GetNextGoldAction(const ParserState &state) const override {
    if (state.Next() == 0) return StartAction();
    const Token &token = state.GetToken(state.Next());
    return (token.break_level() != Token::NO_BREAK ?
        StartAction() : MergeAction());
  }

  // Both START and MERGE can be applied to any tokens in the sentence.
  bool IsAllowedAction(
      ParserAction action, const ParserState &state) const override {
    return true;
  }

  // Performs the specified action on a given parser state, without adding the
  // action to the state's history.
  void PerformActionWithoutHistory(
      ParserAction action, ParserState *state) const override {
    // Note when the action is less than 0, it is treated as a START.
    if (action < 0 || action == StartAction()) {
      MutableTransitionState(state)->AddStart(state->Next(), state);
    }
    state->Advance();
  }

  // Allows backoff to best allowable transition.
  bool BackOffToBestAllowableTransition() const override { return true; }

  // A state is a deterministic state iff no tokens have been consumed.
  bool IsDeterministicState(const ParserState &state) const override {
    return state.Next() == 0;
  }

  // For binary segmentation, a state is a final state iff all tokens have been
  // consumed.
  bool IsFinalState(const ParserState &state) const override {
    return state.EndOfInput();
  }

  // Returns a string representation of a parser action.
  string ActionAsString(
      ParserAction action, const ParserState &state) const override {
    return action == StartAction() ? "START" : "MERGE";
  }

  // Downcasts the TransitionState in ParserState to an BinarySegmentState.
  static BinarySegmentState *MutableTransitionState(ParserState *state) {
    return static_cast<BinarySegmentState *>(state->mutable_transition_state());
  }
};

REGISTER_TRANSITION_SYSTEM("binary-segment-transitions",
                           BinarySegmentTransitionSystem);

// Parser feature locator that returns the token in the sentence that is
// argument() positions from the provided focus token.
class OffsetFeatureLocator : public ParserIndexLocator<OffsetFeatureLocator> {
 public:
  // Update the current focus to a new location.  If the initial focus or new
  // focus is outside the range of the sentence, returns -2.
  void UpdateArgs(const WorkspaceSet &workspaces, const ParserState &state,
                  int *focus) const {
    if (*focus < -1 || *focus >= state.sentence().token_size()) {
      *focus = -2;
      return;
    }
    int new_focus = *focus + argument();
    if (new_focus < -1 || new_focus >= state.sentence().token_size()) {
      *focus = -2;
      return;
    }
    *focus = new_focus;
  }
};

REGISTER_PARSER_IDX_FEATURE_FUNCTION("offset", OffsetFeatureLocator);

// Feature function that returns the id of bigram constructed by concatenating
// word fields of tokens at focus and focus + 1.
class CharBigramFunction : public ParserIndexFeatureFunction {
 public:
  void Setup(TaskContext *context) override {
    input_map_ = context->GetInput("char-ngram-map", "text", "");
  }

  void Init(TaskContext *context) override {
    min_freq_ = GetIntParameter("char-bigram-min-freq", 2);
    bigram_map_.Load(TaskContext::InputFile(*input_map_), min_freq_, -1);
    unknown_id_ = bigram_map_.Size();
    outside_id_ = unknown_id_ + 1;
    set_feature_type(
        new ResourceBasedFeatureType<CharBigramFunction>(name(), this, {}));
  }

  int64 NumValues() const {
    return outside_id_ + 1;
  }

  // Returns the string representation of the given feature value.
  string GetFeatureValueName(FeatureValue value) const {
    if (value == outside_id_) return "<OUTSIDE>";
    if (value == unknown_id_) return "<UNKNOWN>";
    return bigram_map_.GetTerm(value);
  }

  FeatureValue Compute(const WorkspaceSet &workspaces, const ParserState &state,
                       int focus, const FeatureVector *result) const override {
    if (focus < 0 || focus >= state.sentence().token_size() - 1) {
      return outside_id_;
    }
    const int start = state.GetToken(focus).start();
    const int length = state.GetToken(focus + 1).end() - start + 1;
    CHECK_GT(length, 0);
    CHECK_LE(start + length, state.sentence().text().size());
    const char *data = state.sentence().text().data() + start;
    return bigram_map_.LookupIndex(string(data, length), unknown_id_);
  }

 private:
  // Task input for the word to id map. Not owned.
  TaskInput *input_map_ = nullptr;
  TermFrequencyMap bigram_map_;

  // Special ids of unknown words and out-of-range.
  int unknown_id_ = 0;
  int outside_id_ = 0;

  // Minimum frequency for term map.
  int min_freq_ = 2;
};

REGISTER_PARSER_IDX_FEATURE_FUNCTION("char-bigram", CharBigramFunction);

// Feature function that returns the id of the n-th most recently constructed
// word. Note that the argument, n, should be larger than 0. When equals to 0,
// it points to the word which is not yet completed.
class LastWordFeatureFunction : public ParserFeatureFunction {
 public:
  void Setup(TaskContext *context) override {
    input_word_map_ = context->GetInput("word-map", "text", "");
  }

  void Init(TaskContext *context) override {
    min_freq_ = GetIntParameter("min-freq", 0);
    max_num_terms_ = GetIntParameter("max-num-terms", 0);
    word_map_.Load(
        TaskContext::InputFile(*input_word_map_), min_freq_, max_num_terms_);
    unknown_id_ = word_map_.Size();
    outside_id_ = unknown_id_ + 1;
    set_feature_type(
        new ResourceBasedFeatureType<LastWordFeatureFunction>(
        name(), this, {}));
  }

  int64 NumValues() const {
    return outside_id_ + 1;
  }

  // Returns the string representation of the given feature value.
  string GetFeatureValueName(FeatureValue value) const {
    if (value == outside_id_) return "<OUTSIDE>";
    if (value == unknown_id_) return "<UNKNOWN>";
    DCHECK_GE(value, 0);
    DCHECK_LT(value, word_map_.Size());
    return word_map_.GetTerm(value);
  }

  FeatureValue Compute(const WorkspaceSet &workspaces, const ParserState &state,
                       const FeatureVector *result) const override {
    // n should be larger than 0, since the current word is still under
    // construction.
    const int n = argument();
    CHECK_GT(n, 0);
    const auto *segment_state = static_cast<const BinarySegmentState *>(
        state.transition_state());
    if (n >= segment_state->NumStarts(state)) {
      return outside_id_;
    }

    const auto &sentence = state.sentence();
    const int start = segment_state->LastStart(n, state);
    const int end = segment_state->LastStart(n - 1, state) - 1;
    CHECK_GE(end, start);

    const int start_offset = state.GetToken(start).start();
    const int length = state.GetToken(end).end() - start_offset + 1;
    const auto *data = sentence.text().data() + start_offset;
    return word_map_.LookupIndex(string(data, length), unknown_id_);
  }

 private:
  // Task input for the word to id map. Not owned.
  TaskInput *input_word_map_ = nullptr;
  TermFrequencyMap word_map_;

  // Special ids of unknown words and out-of-range.
  int unknown_id_ = 0;
  int outside_id_ = 0;

  // Minimum frequency for term map.
  int min_freq_;

  // Maximum number of terms for term map.
  int max_num_terms_;
};

REGISTER_PARSER_FEATURE_FUNCTION("last-word", LastWordFeatureFunction);

}  // namespace syntaxnet
