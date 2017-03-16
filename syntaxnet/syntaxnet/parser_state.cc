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

#include "syntaxnet/parser_state.h"

#include "syntaxnet/sentence.pb.h"
#include "syntaxnet/term_frequency_map.h"
#include "syntaxnet/utils.h"

namespace syntaxnet {

const char ParserState::kRootLabel[] = "ROOT";

ParserState::ParserState(Sentence *sentence,
                         ParserTransitionState *transition_state,
                         const TermFrequencyMap *label_map)
    : sentence_(sentence),
      num_tokens_(sentence->token_size()),
      transition_state_(transition_state),
      label_map_(label_map),
      root_label_(label_map->LookupIndex(kRootLabel,
                                         kDefaultRootLabel  /* unknown */)),
      next_(0) {
  // Initialize the stack. Some transition systems could also push the
  // artificial root on the stack, so we make room for that as well.
  stack_.reserve(num_tokens_ + 1);

  // Allocate space for head indices and labels. Initialize the head for all
  // tokens to be the artificial root node, i.e. token -1.
  head_.resize(num_tokens_, -1);
  label_.resize(num_tokens_, RootLabel());

  // Transition system-specific preprocessing.
  if (transition_state_ != nullptr) transition_state_->Init(this);
}

ParserState::~ParserState() { delete transition_state_; }

ParserState *ParserState::Clone() const {
  ParserState *new_state = new ParserState();
  new_state->sentence_ = sentence_;
  new_state->num_tokens_ = num_tokens_;
  new_state->alternative_ = alternative_;
  new_state->transition_state_ =
      (transition_state_ == nullptr ? nullptr : transition_state_->Clone());
  new_state->label_map_ = label_map_;
  new_state->root_label_ = root_label_;
  new_state->next_ = next_;
  new_state->stack_.assign(stack_.begin(), stack_.end());
  new_state->head_.assign(head_.begin(), head_.end());
  new_state->label_.assign(label_.begin(), label_.end());
  new_state->score_ = score_;
  new_state->is_gold_ = is_gold_;
  new_state->keep_history_ = keep_history_;
  new_state->history_.assign(history_.begin(), history_.end());
  return new_state;
}

int ParserState::RootLabel() const { return root_label_; }

int ParserState::NumLabels() const {
  return label_map_->Size() + (RootLabel() == kDefaultRootLabel ? 1 : 0);
}

int ParserState::Next() const {
  CHECK_GE(next_, -1);
  CHECK_LE(next_, num_tokens_);
  return next_;
}

int ParserState::Input(int offset) const {
  const int index = next_ + offset;
  return index >= -1 && index < num_tokens_ ? index : -2;
}

void ParserState::Advance() {
  CHECK_LT(next_, num_tokens_) << utils::Join(history_, ",");
  ++next_;
}

void ParserState::Advance(int next) {
  CHECK_LE(next, num_tokens_) << utils::Join(history_, ",");
  next_ = next;
}

bool ParserState::EndOfInput() const { return next_ == num_tokens_; }

void ParserState::Push(int index) {
  CHECK_LE(stack_.size(), num_tokens_) << utils::Join(history_, ",");
  stack_.push_back(index);
}

int ParserState::Pop() {
  CHECK(!StackEmpty()) << utils::Join(history_, ",");
  const int result = stack_.back();
  stack_.pop_back();
  return result;
}

int ParserState::Top() const {
  CHECK(!StackEmpty()) << utils::Join(history_, ",");
  return stack_.back();
}

int ParserState::Stack(int position) const {
  if (position < 0) return -2;
  const int index = stack_.size() - 1 - position;
  return (index < 0) ? -2 : stack_[index];
}

int ParserState::StackSize() const { return stack_.size(); }

bool ParserState::StackEmpty() const { return stack_.empty(); }

int ParserState::Head(int index) const {
  CHECK_GE(index, -1);
  CHECK_LT(index, num_tokens_);
  return index == -1 ? -1 : head_[index];
}

int ParserState::Label(int index) const {
  CHECK_GE(index, -1);
  CHECK_LT(index, num_tokens_);
  return index == -1 ? RootLabel() : label_[index];
}

int ParserState::Parent(int index, int n) const {
  // Find the n-th parent by applying the head function n times.
  CHECK_GE(index, -1);
  CHECK_LT(index, num_tokens_);
  while (n-- > 0) index = Head(index);
  return index;
}

int ParserState::LeftmostChild(int index, int n) const {
  CHECK_GE(index, -1);
  CHECK_LT(index, num_tokens_);
  while (n-- > 0) {
    // Find the leftmost child by scanning from start until a child is
    // encountered.
    int i;
    for (i = -1; i < index; ++i) {
      if (Head(i) == index) break;
    }
    if (i == index) return -2;
    index = i;
  }
  return index;
}

int ParserState::RightmostChild(int index, int n) const {
  CHECK_GE(index, -1);
  CHECK_LT(index, num_tokens_);
  while (n-- > 0) {
    // Find the rightmost child by scanning backward from end until a child
    // is encountered.
    int i;
    for (i = num_tokens_ - 1; i > index; --i) {
      if (Head(i) == index) break;
    }
    if (i == index) return -2;
    index = i;
  }
  return index;
}

int ParserState::LeftSibling(int index, int n) const {
  // Find the n-th left sibling by scanning left until the n-th child of the
  // parent is encountered.
  CHECK_GE(index, -1);
  CHECK_LT(index, num_tokens_);
  if (index == -1 && n > 0) return -2;
  int i = index;
  while (n > 0) {
    --i;
    if (i == -1) return -2;
    if (Head(i) == Head(index)) --n;
  }
  return i;
}

int ParserState::RightSibling(int index, int n) const {
  // Find the n-th right sibling by scanning right until the n-th child of the
  // parent is encountered.
  CHECK_GE(index, -1);
  CHECK_LT(index, num_tokens_);
  if (index == -1 && n > 0) return -2;
  int i = index;
  while (n > 0) {
    ++i;
    if (i == num_tokens_) return -2;
    if (Head(i) == Head(index)) --n;
  }
  return i;
}

void ParserState::AddArc(int index, int head, int label) {
  CHECK_GE(index, 0);
  CHECK_LT(index, num_tokens_);
  head_[index] = head;
  label_[index] = label;
}

int ParserState::GoldHead(int index) const {
  // A valid ParserState index is transformed to a valid Sentence index,
  // then the gold head is extracted.
  CHECK_GE(index, -1);
  CHECK_LT(index, num_tokens_);
  if (index == -1) return -1;
  const int offset = 0;
  const int gold_head = GetToken(index).head();
  return gold_head == -1 ? -1 : gold_head - offset;
}

int ParserState::GoldLabel(int index) const {
  // A valid ParserState index is transformed to a valid Sentence index,
  // then the gold label is extracted.
  CHECK_GE(index, -1);
  CHECK_LT(index, num_tokens_);
  if (index == -1) return RootLabel();
  string gold_label;
  gold_label = GetToken(index).label();
  return label_map_->LookupIndex(gold_label, RootLabel() /* unknown */);
}

void ParserState::AddParseToDocument(Sentence *sentence,
                                     bool rewrite_root_labels) const {
  transition_state_->AddParseToDocument(*this, rewrite_root_labels, sentence);
}

bool ParserState::IsTokenCorrect(int index) const {
  return transition_state_->IsTokenCorrect(*this, index);
}

string ParserState::LabelAsString(int label) const {
  if (label == root_label_) return kRootLabel;
  if (label >= 0 && label < label_map_->Size()) {
    return label_map_->GetTerm(label);
  }
  return "";
}

string ParserState::ToString() const {
  return transition_state_->ToString(*this);
}

}  // namespace syntaxnet
