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

#ifndef SYNTAXNET_HEAD_LABEL_TRANSITIONS_H_
#define SYNTAXNET_HEAD_LABEL_TRANSITIONS_H_

#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"

namespace syntaxnet {

// Heads and labels transition system. Predicts the syntactic heads and labels
// of a sentence directly.
//
// In this transition system actions encode heads and their labels, so the
// space of actions is num_labels*N (for a sentence with N tokens.) A token
// that points to itself is interpreted as a root. Unlike the heads transition
// system followed by labels, we allow root arcs to receive non-root
// dependency labels and vice versa since, unlike in the labels transition
// system, it is unclear whether the arc or label prediction should take
// precedence.
//
// Actions are interpreted as follows:
//
// For input pointer at position i:
//   head  = A / num_labels
//   label = A % num_labels
//   if head == i  : Add a root arc to token i (with given label)
//   if head != i  : Add an arc head -> i (with given label)
//
// Note that in syntaxnet.Sentence, root arcs are token.head() == -1, whereas
// here, we use a self-loop to represent roots.
class HeadLabelTransitionSystem : public ParserTransitionSystem {
 public:
  class State;  // defined in the .cc file

  int NumActionTypes() const override { return 1; }
  int NumActions(int num_labels) const override { return kDynamicNumActions; }

  // The default action is to assign itself as root.
  ParserAction GetDefaultAction(const ParserState &state) const override;

  // Returns the next gold action for a given state according to the
  // underlying annotated sentence.
  ParserAction GetNextGoldAction(const ParserState &state) const override;

  // Checks if the action is allowed in a given parser state.
  bool IsAllowedAction(ParserAction action,
                       const ParserState &state) const override;

  // Performs the specified action on a given parser state, without adding the
  // action to the state's history.
  void PerformActionWithoutHistory(ParserAction action,
                                   ParserState *state) const override;

  // Returns true if the state is at the end of the input.
  bool IsFinalState(const ParserState &state) const override;

  // Returns a string representation of a parser action.
  string ActionAsString(ParserAction action,
                        const ParserState &state) const override;

  // Returns a new transition state to be used to enhance the parser state.
  ParserTransitionState *NewTransitionState(bool training_mode) const override;

  // Returns false, since no states are deterministic.
  bool IsDeterministicState(const ParserState &state) const override {
    return false;
  }

 private:
  // Given a ParseState, decodes an action into a base action and a label.
  void DecodeActionWithState(ParserAction action, const ParserState &state,
                             ParserAction *base_action, int *label) const;

  // Given a ParseState, encodes a base action and a label into a single-valued
  // function.
  ParserAction EncodeActionWithState(ParserAction base_action, int label,
                                     const ParserState &state) const;
};

}  // namespace syntaxnet

#endif  // SYNTAXNET_HEAD_LABEL_TRANSITIONS_H_
