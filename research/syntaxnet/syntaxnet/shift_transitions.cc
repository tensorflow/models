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

// Shift transition system.
//
// This transition system has one type of actions:
//  - The SHIFT action pushes the next input token to the stack and
//    advances to the next input token.
//
// For this very simple transition system, we don't need a specific
// TransitionState because we have no additional information to remember.
// We use it to compute look-ahead in DRAGNN by using its representations in
// downstream tasks.

#include <string>

#include "syntaxnet/parser_features.h"
#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"
#include "syntaxnet/sentence_features.h"
#include "syntaxnet/shared_store.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/term_frequency_map.h"
#include "syntaxnet/utils.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {

class ShiftTransitionState : public ParserTransitionState {
 public:
  explicit ShiftTransitionState(bool left_to_right)
      : left_to_right_(left_to_right) {}

  explicit ShiftTransitionState(const ShiftTransitionState *state)
      : left_to_right_(state->left_to_right_) {}

  ParserTransitionState *Clone() const override {
    return new ShiftTransitionState(this);
  }

  // Set the initial value of next in ParserState.
  void Init(ParserState *state) override {
    if (!left_to_right_) {
      // Start from the last word of the sentence if we transit from right to
      // left.
      state->Advance(state->sentence().token_size() - 1);
    }
  }

  bool IsTokenCorrect(const ParserState &state, int index) const override {
    return true;
  }

  string ToString(const ParserState &state) const override { return ""; }

 private:
  bool left_to_right_;
};

class ShiftTransitionSystem : public ParserTransitionSystem {
 public:
  static const ParserAction kShiftAction = 0;

  // Determines the direction of the system.
  void Setup(TaskContext *context) override {
    // TODO(googleuser): Use FetchDeprecated.
    if (context->Get("left-to-right", "<NOT-SET>") == "<NOT-SET>") {
      left_to_right_ = context->Get("left_to_right", true);
    } else {
      left_to_right_ = context->Get("left-to-right", true);
      LOG(WARNING) << "'left-to-right' parameter set: this is DEPRECATED. "
                   << "Use 'left_to_right' instead.";
    }
  }

  // The shift transition system doesn't actually look at the dependency tree,
  // so it does allow non-projective trees.
  bool AllowsNonProjective() const override { return true; }

  // Returns the number of action types.
  int NumActionTypes() const override { return 1; }

  // Returns the number of possible actions.
  int NumActions(int num_labels) const override { return 1; }

  ParserAction GetDefaultAction(const ParserState &state) const override {
    return kShiftAction;
  }

  // At anytime, the gold action is to shift.
  ParserAction GetNextGoldAction(const ParserState &state) const override {
    return kShiftAction;
  }

  // Checks if the action is allowed in a given parser state.
  bool IsAllowedAction(ParserAction action,
                       const ParserState &state) const override {
    return left_to_right_ ? (!state.EndOfInput()) : (state.Next() >= 0);
  }

  // Makes a shift by pushing the next input token on the stack and moving to
  // the next position.
  void PerformActionWithoutHistory(ParserAction action,
                                   ParserState *state) const override {
    DCHECK(!IsFinalState(*state));
    if (!IsFinalState(*state)) {
      int next = state->Next();
      state->Push(next);
      next = left_to_right_ ? (next + 1) : (next - 1);
      state->Advance(next);
    }
  }

  bool IsFinalState(const ParserState &state) const override {
    return left_to_right_ ? state.EndOfInput() : (state.Next() < 0);
  }

  // Returns a string representation of a parser action.
  string ActionAsString(ParserAction action,
                        const ParserState &state) const override {
    string current_word = state.GetToken(state.Next()).word();
    return current_word;
  }

  // All states are deterministic in this transition system.
  bool IsDeterministicState(const ParserState &state) const override {
    return true;
  }

  // Returns a new transition state.
  ParserTransitionState *NewTransitionState(bool training_mode) const override {
    return new ShiftTransitionState(get_direction());
  }

  bool get_direction() const { return left_to_right_; }

 private:
  bool left_to_right_ = true;
};

REGISTER_TRANSITION_SYSTEM("shift-only", ShiftTransitionSystem);

}  // namespace syntaxnet
