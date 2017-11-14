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

#include "syntaxnet/head_label_transitions.h"

#include "syntaxnet/base.h"

using tensorflow::strings::StrAppend;
using tensorflow::strings::StrCat;

namespace syntaxnet {

// Parser transition state for head & label transitions.
class HeadLabelTransitionSystem::State : public ParserTransitionState {
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
      token->set_head(state.Head(i));
      token->set_label(state.LabelAsString(state.Label(i)));
      if (rewrite_root_labels && state.Head(i) == -1) {
        token->set_label(state.LabelAsString(state.RootLabel()));
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

ParserAction HeadLabelTransitionSystem::GetDefaultAction(
    const ParserState &state) const {
  const int default_head = state.Next();
  const int default_label = state.RootLabel();
  return EncodeActionWithState(default_head, default_label, state);
}

ParserAction HeadLabelTransitionSystem::GetNextGoldAction(
    const ParserState &state) const {
  if (state.EndOfInput()) {
    LOG(ERROR) << "Oracle called on invalid state: " << state.ToString();
    return 0;
  }
  const int current = state.Next();
  int head = state.GoldHead(current);
  const int label = state.GoldLabel(current);

  // In syntaxnet.Sentence, root arcs are token.head() == -1, whereas
  // here, we use a self-loop to represent roots. So we need to convert here.
  head = head == -1 ? current : head;
  return EncodeActionWithState(head, label, state);
}

void HeadLabelTransitionSystem::PerformActionWithoutHistory(
    ParserAction action, ParserState *state) const {
  CHECK(IsAllowedAction(action, *state))
      << "Illegal action " << action << " at state: " << state->ToString();

  const int current = state->Next();
  int head, label;
  DecodeActionWithState(action, *state, &head, &label);

  VLOG(2) << "Adding arc: " << label << " (" << current << " <- " << head
          << ")";
  state->AddArc(current, head == current ? -1 : head, label);
  state->Advance();
}

bool HeadLabelTransitionSystem::IsAllowedAction(
    ParserAction action, const ParserState &state) const {
  if (state.EndOfInput()) return false;

  // Unlike the labels transition system, we allow root tokens to receive
  // non-root dependency labels and vice versa.
  return action >= 0 && action < state.NumTokens() * state.NumLabels();
}

bool HeadLabelTransitionSystem::IsFinalState(const ParserState &state) const {
  return state.EndOfInput();
}

string HeadLabelTransitionSystem::ActionAsString(
    ParserAction action, const ParserState &state) const {
  if (!IsAllowedAction(action, state)) return StrCat("INVALID:", action);

  const auto &sentence = state.sentence();
  const int current = state.Next();
  int head, label;
  DecodeActionWithState(action, state, &head, &label);
  return StrCat(state.LabelAsString(label), "(",
                sentence.token(current).word(), "<-",
                head == current ? "ROOT" : sentence.token(head).word(), ")");
}

ParserTransitionState *HeadLabelTransitionSystem::NewTransitionState(
    bool training_mode) const {
  return new State();
}

void HeadLabelTransitionSystem::DecodeActionWithState(ParserAction action,
                                                      const ParserState &state,
                                                      ParserAction *base_action,
                                                      int *label) const {
  const int num_labels = state.NumLabels();
  *base_action = action / num_labels;
  *label = action % num_labels;
}

ParserAction HeadLabelTransitionSystem::EncodeActionWithState(
    ParserAction base_action, int label, const ParserState &state) const {
  return base_action * state.NumLabels() + label;
}

REGISTER_TRANSITION_SYSTEM("heads_labels", HeadLabelTransitionSystem);

}  // namespace syntaxnet
