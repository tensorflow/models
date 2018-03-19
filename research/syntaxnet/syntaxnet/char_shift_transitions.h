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

// Character-level shift transition system.
//
// This transition system has one type of action:
//  - The SHIFT action advances the next input pointer to the next input
//    character.
//
// For this transition system, we need a simple TransitionState that keeps track
// of an input pointer into characters.

#ifndef SYNTAXNET_CHAR_SHIFT_TRANSITIONS_H_
#define SYNTAXNET_CHAR_SHIFT_TRANSITIONS_H_

#include "syntaxnet/base.h"
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

// CharShiftTransitionState is similar to ParserState, but operates on
// character-level instead of token-level. It contains of a pointer to the next
// input character.
class CharShiftTransitionState : public ParserTransitionState {
 public:
  explicit CharShiftTransitionState(bool left_to_right)
      : left_to_right_(left_to_right) {}

  ParserTransitionState *Clone() const override;

  // Set the initial value of next in ParserState.
  void Init(ParserState *state) override;

  // Returns the index of the next input character.
  int Next() const;

  // Returns the character index relative to the next input character. If no
  // such character exists, returns -2.
  int Input(int offset) const;

  // Returns the character at the given index i. Returns an empty string if the
  // index is out of range.
  string GetChar(const ParserState &state, int i) const;

  // Sets the next input character. Useful for transition systems that do not
  // necessarily process characters in order.
  void Advance(int next);

  // Returns true if all characters have been processed.
  bool EndOfInput() const;

  // Returns true if the character index i is at a token start.
  bool IsTokenStart(int i) const;

  // Returns true if the character index i is at a token end.
  bool IsTokenEnd(int i) const;

  int num_chars() const { return num_chars_; }

    // Whether a parsed token should be considered correct for evaluation.
  bool IsTokenCorrect(const ParserState &state, int index) const override {
    return true;
  }

  // Returns a human readable string representation of this state.
  string ToString(const ParserState &state) const override {
    return "";
  }

 private:
  // Number of characters in the sentence.
  int num_chars_;

  // Index of the next input character.
  int next_;

  // Whether the input characters are read from left to right.
  const bool left_to_right_;

  // Int vectors both of size num_chars_ for storing character positons and
  // lengths (in bytes).
  std::vector<int> char_pos_map_;
  std::vector<int> char_len_map_;

  // Boolean vectors both of size num_chars_. token_starts[i]/token_ends[i]
  // is true iff the character index i is a token start/end.
  std::vector<bool> token_starts_;
  std::vector<bool> token_ends_;
};

class CharShiftTransitionSystem : public ParserTransitionSystem {
 public:
  static const ParserAction kShiftAction = 0;

  CharShiftTransitionSystem() {}

  // Determines the direction of the system.
  void Setup(TaskContext *context) override;

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

  // At any time, the gold action is to shift.
  ParserAction GetNextGoldAction(const ParserState &state) const override {
    return kShiftAction;
  }

  // Checks if the action is allowed in a given parser state.
  bool IsAllowedAction(ParserAction action,
                       const ParserState &state) const override;

  // Performs a shift by pushing the next input token on the stack and moving to
  // the next position.
  void PerformActionWithoutHistory(ParserAction action,
                                   ParserState *state) const override;

  bool IsFinalState(const ParserState &state) const override;

  // Returns a string representation of a parser action.
  string ActionAsString(ParserAction action,
                        const ParserState &state) const override;

  // All states are deterministic in this transition system.
  bool IsDeterministicState(const ParserState &state) const override {
    return true;
  }

  // Returns a new transition state.
  ParserTransitionState *NewTransitionState(bool training_mode) const override;

  bool left_to_right() const { return left_to_right_; }

 private:
  bool left_to_right_ = true;
};

}  // namespace syntaxnet

#endif  // SYNTAXNET_CHAR_SHIFT_TRANSITIONS_H_
