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
#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"

namespace syntaxnet {

// Given an input of utf8 characters, the BinarySegmentTransitionSystem
// conducts word segmentation by performing one of the following two actions:
//  -START: starts a new word with the token at state.input, and also advances
//          the state.input.
//  -MERGE: adds the token at state.input to its prevous word, and also advances
//          state.input.
//
// Also see nlp/saft/components/segmentation/transition/binary-segment-state.h
// for examples on handling spaces.
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

}  // namespace syntaxnet
