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

#ifndef SYNTAXNET_LABEL_TRANSITIONS_H_
#define SYNTAXNET_LABEL_TRANSITIONS_H_

#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"

namespace syntaxnet {

// Label transition system.  For each token in a sentence, predicts the label of
// the dependency between that token and its head.
class LabelTransitionSystem : public ParserTransitionSystem {
 public:
  class State;  // defined in the .cc file

  // There is one type of action: predicting a label.
  int NumActionTypes() const override { return 1; }
  int NumActions(int num_labels) const override { return num_labels; }

  // Returns the default action, which is to assign the root label.
  ParserAction GetDefaultAction(const ParserState &state) const override;

  // Returns the gold label for the current token of the |state|.
  ParserAction GetNextGoldAction(const ParserState &state) const override;

  // Returns true if the |action| is allowed for the |state|.
  bool IsAllowedAction(ParserAction action,
                       const ParserState &state) const override;

  // Performs the |action| on the |state|, without adding the |action| to the
  // |state|'s history.
  void PerformActionWithoutHistory(ParserAction action,
                                   ParserState *state) const override;

  // Returns true if the |state| is at the end of the input.
  bool IsFinalState(const ParserState &state) const override;

  // Returns a string representation of performing the |action| on the |state|.
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

#endif  // SYNTAXNET_LABEL_TRANSITIONS_H_
