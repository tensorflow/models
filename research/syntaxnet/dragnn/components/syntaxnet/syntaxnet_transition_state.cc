// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "dragnn/components/syntaxnet/syntaxnet_transition_state.h"

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace syntaxnet {
namespace dragnn {

SyntaxNetTransitionState::SyntaxNetTransitionState(
    std::unique_ptr<ParserState> parser_state, SyntaxNetSentence *sentence)
    : parser_state_(std::move(parser_state)),
      sentence_(sentence),
      is_gold_(false) {
  score_ = 0;
  current_beam_index_ = -1;
  parent_beam_index_ = 0;
  step_for_token_.resize(sentence->sentence()->token_size(), -1);
  parent_for_token_.resize(sentence->sentence()->token_size(), -1);
  parent_step_for_token_.resize(sentence->sentence()->token_size(), -1);
}

void SyntaxNetTransitionState::Init(const TransitionState &parent) {
  score_ = parent.GetScore();
  parent_beam_index_ = parent.GetBeamIndex();
}

std::unique_ptr<SyntaxNetTransitionState> SyntaxNetTransitionState::Clone()
    const {
  // Create a new state from a clone of the underlying parser state.
  std::unique_ptr<ParserState> cloned_state(parser_state_->Clone());
  std::unique_ptr<SyntaxNetTransitionState> new_state(
      new SyntaxNetTransitionState(std::move(cloned_state), sentence_));

  // Copy relevant data members and set non-copied ones to flag values.
  new_state->score_ = score_;
  new_state->current_beam_index_ = current_beam_index_;
  new_state->parent_beam_index_ = parent_beam_index_;
  new_state->step_for_token_ = step_for_token_;
  new_state->parent_step_for_token_ = parent_step_for_token_;
  new_state->parent_for_token_ = parent_for_token_;

  // Copy trace if it exists.
  if (trace_) {
    new_state->trace_.reset(new ComponentTrace(*trace_));
  }

  return new_state;
}

int SyntaxNetTransitionState::ParentBeamIndex() const {
  return parent_beam_index_;
}

int SyntaxNetTransitionState::GetBeamIndex() const {
  return current_beam_index_;
}

bool SyntaxNetTransitionState::IsGold() const { return is_gold_; }

void SyntaxNetTransitionState::SetGold(bool is_gold) { is_gold_ = is_gold; }

void SyntaxNetTransitionState::SetBeamIndex(int index) {
  current_beam_index_ = index;
}

float SyntaxNetTransitionState::GetScore() const { return score_; }

void SyntaxNetTransitionState::SetScore(float score) { score_ = score; }

string SyntaxNetTransitionState::HTMLRepresentation() const {
  // Crude HTML string showing the stack and the word on the input.
  string html = "Stack: ";
  for (int i = parser_state_->StackSize() - 1; i >= 0; --i) {
    const int word_idx = parser_state_->Stack(i);
    if (word_idx >= 0) {
      tensorflow::strings::StrAppend(
          &html, parser_state_->GetToken(word_idx).word(), " ");
    }
  }
  tensorflow::strings::StrAppend(&html, "| Input: ");
  const int word_idx = parser_state_->Input(0);
  if (word_idx >= 0) {
    tensorflow::strings::StrAppend(
        &html, parser_state_->GetToken(word_idx).word(), " ");
  }

  return html;
}

}  // namespace dragnn
}  // namespace syntaxnet
