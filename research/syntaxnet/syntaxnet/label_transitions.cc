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

#include "syntaxnet/label_transitions.h"

#include "syntaxnet/base.h"

using tensorflow::strings::StrAppend;
using tensorflow::strings::StrCat;

namespace syntaxnet {

// Parser transition state for label transitions.
class LabelTransitionSystem::State : public ParserTransitionState {
 public:
  // Returns a copy of this state.
  State *Clone() const override { return new State(*this); }

  // Does nothing; no need for additional initialization.
  void Init(ParserState *state) override {}

  // Copies the selected labels to the |sentence|.
  void AddParseToDocument(const ParserState &state, bool rewrite_root_labels,
                          Sentence *sentence) const override {
    for (int i = 0; i < state.NumTokens(); ++i) {
      Token *token = sentence->mutable_token(i);
      token->set_label(state.LabelAsString(state.Label(i)));
      if (rewrite_root_labels && state.Head(i) == -1) {
        token->set_label(state.LabelAsString(state.RootLabel()));
      }
    }
  }

  // Returns true if the label and gold label match.
  bool IsTokenCorrect(const ParserState &state, int index) const override {
    return state.GoldLabel(index) == state.Label(index);
  }

  // Returns a string representation of the |state|.
  string ToString(const ParserState &state) const override {
    string str = "[";
    for (int i = 0; i < state.NumTokens(); ++i) {
      StrAppend(&str, i == 0 ? "" : " ", state.LabelAsString(state.Label(i)));
    }
    StrAppend(&str, "]");
    return str;
  }
};

ParserAction LabelTransitionSystem::GetDefaultAction(
    const ParserState &state) const {
  return state.RootLabel();
}

ParserAction LabelTransitionSystem::GetNextGoldAction(
    const ParserState &state) const {
  if (state.EndOfInput()) {
    LOG(ERROR) << "Oracle called on invalid state: " << state.ToString();
    return 0;
  }
  const int current = state.Next();
  return state.GoldLabel(current);
}

void LabelTransitionSystem::PerformActionWithoutHistory(
    ParserAction action, ParserState *state) const {
  const int current = state->Next();
  const int head = state->GoldHead(current);
  CHECK(IsAllowedAction(action, *state))
      << "Illegal action " << action << " (root label " << state->RootLabel()
      << ") with current=" << current << " and head=" << head
      << " at state: " << state->ToString() << "\ndocument:\n"
      << state->sentence().DebugString();

  VLOG(2) << "Adding arc: " << action << " (" << current << " <- " << head
          << ")";
  state->AddArc(current, head, action);
  state->Advance();
}

bool LabelTransitionSystem::IsAllowedAction(ParserAction action,
                                            const ParserState &state) const {
  if (state.EndOfInput()) return false;
  if (action < 0 || action >= state.NumLabels()) return false;

  const int head = state.GoldHead(state.Next());
  const bool is_root_head = head < 0;
  const bool is_root_label = action == state.RootLabel();

  // The root label is allowed iff the head is the root.
  return is_root_head == is_root_label;
}

bool LabelTransitionSystem::IsFinalState(const ParserState &state) const {
  return state.EndOfInput();
}

string LabelTransitionSystem::ActionAsString(ParserAction action,
                                             const ParserState &state) const {
  if (!IsAllowedAction(action, state)) return StrCat("INVALID:", action);

  const auto &sentence = state.sentence();
  const int current = state.Next();
  const int head = state.GoldHead(current);
  return StrCat(state.LabelAsString(action), "(",
                sentence.token(current).word(), "<-",
                head == -1 ? "ROOT" : sentence.token(head).word(), ")");
}

ParserTransitionState *LabelTransitionSystem::NewTransitionState(
    bool training_mode) const {
  return new State();
}

REGISTER_TRANSITION_SYSTEM("labels", LabelTransitionSystem);

}  // namespace syntaxnet
