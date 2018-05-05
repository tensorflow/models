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

#ifndef SYNTAXNET_HEAD_TRANSITIONS_H_
#define SYNTAXNET_HEAD_TRANSITIONS_H_

#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"

namespace syntaxnet {

// Head transition system. Predicts the syntactic heads of a sentence directly.
//
// For a sentence with N tokens, actions are interpreted as follows:
//
// For input pointer at position i:
//
//   Action A == i  : Add a root arc to token i.
//   Action A != i  : Add an arc A -> i.
//
// Note that in the Sentence proto, root arcs are token.head() == -1, whereas
// here, we use a self-loop to represent roots.
class HeadTransitionSystem : public ParserTransitionSystem {
 public:
  class State;  // defined in the .cc file

  int NumActionTypes() const override { return 1; }
  int NumActions(int num_labels) const override { return kDynamicNumActions; }

  // Returns the default action, which is to assign itself as root.
  ParserAction GetDefaultAction(const ParserState &state) const override;

  // Returns the next gold head for a given state according to the underlying
  // annotated sentence.
  ParserAction GetNextGoldAction(const ParserState &state) const override;

  // Returns true if the action is allowed in a given parser state.
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
};

}  // namespace syntaxnet

#endif  // SYNTAXNET_HEAD_TRANSITIONS_H_
