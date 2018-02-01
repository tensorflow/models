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

#include "syntaxnet/head_transitions.h"

#include "syntaxnet/base.h"

using tensorflow::strings::StrAppend;
using tensorflow::strings::StrCat;

namespace syntaxnet {

// Parser transition state for head transitions.
class HeadTransitionSystem::State : public ParserTransitionState {
 public:
  // Returns a copy of this state.
  State *Clone() const override { return new State(*this); }

  // Does nothing; no need for additional initialization.
  void Init(ParserState *state) override {}

  // Copies the selected heads to the |sentence|.
  void AddParseToDocument(const ParserState &state, bool rewrite_root_labels,
                          Sentence *sentence) const override {
    for (int i = 0; i < state.NumTokens(); ++i) {
      Token *token = sentence->mutable_token(i);
      if (state.Head(i) != -1) {
        token->set_head(state.Head(i));
      } else {
        token->clear_head();
      }
    }
  }

  // Returns true if the head and gold head match.
  bool IsTokenCorrect(const ParserState &state, int index) const override {
    return state.GoldHead(index) == state.Head(index);
  }

  // Returns a string representation of the |state|.
  string ToString(const ParserState &state) const override {
    string str = "[";
    for (int i = 0; i < state.NumTokens(); ++i) {
      StrAppend(&str, i == 0 ? "" : " ", state.Head(i));
    }
    StrAppend(&str, "]");
    return str;
  }
};

ParserAction HeadTransitionSystem::GetDefaultAction(
    const ParserState &state) const {
  return state.Next();
}

ParserAction HeadTransitionSystem::GetNextGoldAction(
    const ParserState &state) const {
  if (state.EndOfInput()) {
    LOG(ERROR) << "Oracle called on invalid state: " << state.ToString();
    return 0;
  }
  const int current = state.Next();
  const int head = state.GoldHead(current);
  return head == -1 ? current : head;
}

void HeadTransitionSystem::PerformActionWithoutHistory(
    ParserAction action, ParserState *state) const {
  CHECK(IsAllowedAction(action, *state)) << "Illegal action " << action
                                         << " at state: " << state->ToString();

  const int current = state->Next();
  if (action == current) action = -1;  // self connect = root
  VLOG(2) << "Adding arc: " << current << " <- " << action;
  state->AddArc(current, action, 0);
  state->Advance();
}

bool HeadTransitionSystem::IsAllowedAction(ParserAction action,
                                           const ParserState &state) const {
  if (state.EndOfInput()) return false;
  return action >= 0 && action < state.sentence().token_size();
}

bool HeadTransitionSystem::IsFinalState(const ParserState &state) const {
  return state.EndOfInput();
}

string HeadTransitionSystem::ActionAsString(ParserAction action,
                                            const ParserState &state) const {
  if (!IsAllowedAction(action, state)) return StrCat("INVALID:", action);

  const auto &sentence = state.sentence();
  const int current = state.Next();
  return StrCat(action == current ? "ROOT" : sentence.token(action).word(),
                "->", sentence.token(current).word());
}

ParserTransitionState *HeadTransitionSystem::NewTransitionState(
    bool training_mode) const {
  return new State();
}

REGISTER_TRANSITION_SYSTEM("heads", HeadTransitionSystem);

}  // namespace syntaxnet
