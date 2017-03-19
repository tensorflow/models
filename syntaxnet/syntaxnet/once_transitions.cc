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

#include "syntaxnet/base.h"
#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"

using tensorflow::strings::StrCat;

namespace syntaxnet {
namespace {

// A transition system that has exactly one action and performs exactly one
// action per state, regardless of sentence length.
class OnceTransitionSystem : public ParserTransitionSystem {
 public:
  // A transition state that allows exactly one transition.
  class State : public ParserTransitionState {
   public:
    // Implements TransitionState.
    State *Clone() const override { return new State(*this); }
    void Init(ParserState *state) override {}
    bool IsTokenCorrect(const ParserState &state, int index) const override {
      return true;
    }
    string ToString(const ParserState &state) const override {
      return StrCat("done=", done() ? "true" : "false");
    }

    // Records that a transition has been performed.
    void PerformAction() { done_ = true; }

    // Returns true if no more transitions are allowed.
    bool done() const { return done_; }

   private:
    bool done_ = false;  // true if no more transitions are allowed
  };

  // Implements ParserTransitionSystem.
  int NumActionTypes() const override { return 1; }
  int NumActions(int num_labels) const override { return 1; }
  ParserAction GetDefaultAction(const ParserState &state) const override {
    DCHECK(!IsFinalState(state));
    return 0;
  }
  ParserAction GetNextGoldAction(const ParserState &state) const override {
    return GetDefaultAction(state);
  }
  void PerformActionWithoutHistory(ParserAction action,
                                   ParserState *state) const override {
    DCHECK(!IsFinalState(*state));
    static_cast<State *>(state->mutable_transition_state())->PerformAction();
  }
  bool IsAllowedAction(ParserAction action,
                       const ParserState &state) const override {
    return action == 0 && !IsFinalState(state);
  }
  bool IsFinalState(const ParserState &state) const override {
    return static_cast<const State *>(state.transition_state())->done();
  }
  string ActionAsString(ParserAction action,
                        const ParserState &state) const override {
    return StrCat("action:", action);
  }
  ParserTransitionState *NewTransitionState(bool training_mode) const override {
    return new State();
  }
  bool IsDeterministicState(const ParserState &state) const override {
    return true;  // all states have only one action
  }
};

REGISTER_TRANSITION_SYSTEM("once", OnceTransitionSystem);

}  // namespace
}  // namespace syntaxnet
