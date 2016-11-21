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

// Tagger transition system.
//
// This transition system has one type of actions:
//  - The SHIFT action pushes the next input token to the stack and
//    advances to the next input token, assigning a part-of-speech tag to the
//    token that was shifted.
//
// The transition system operates with parser actions encoded as integers:
//  - A SHIFT action is encoded as number starting from 0.

#include <string>

#include "syntaxnet/parser_features.h"
#include "syntaxnet/parser_state.h"
#include "syntaxnet/parser_transitions.h"
#include "syntaxnet/sentence_features.h"
#include "syntaxnet/shared_store.h"
#include "syntaxnet/task_context.h"
#include "syntaxnet/term_frequency_map.h"
#include "syntaxnet/utils.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace syntaxnet {

class TaggerTransitionState : public ParserTransitionState {
 public:
  explicit TaggerTransitionState(const TermFrequencyMap *tag_map,
                                 const TagToCategoryMap *tag_to_category)
      : tag_map_(tag_map), tag_to_category_(tag_to_category) {}

  explicit TaggerTransitionState(const TaggerTransitionState *state)
      : TaggerTransitionState(state->tag_map_, state->tag_to_category_) {
    tag_ = state->tag_;
    gold_tag_ = state->gold_tag_;
  }

  // Clones the transition state by returning a new object.
  ParserTransitionState *Clone() const override {
    return new TaggerTransitionState(this);
  }

  // Reads gold tags for each token.
  void Init(ParserState *state) override {
    tag_.resize(state->sentence().token_size(), -1);
    gold_tag_.resize(state->sentence().token_size(), -1);
    for (int pos = 0; pos < state->sentence().token_size(); ++pos) {
      int tag = tag_map_->LookupIndex(state->GetToken(pos).tag(), -1);
      gold_tag_[pos] = tag;
    }
  }

  // Returns the tag assigned to a given token.
  int Tag(int index) const {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, tag_.size());
    return index == -1 ? -1 : tag_[index];
  }

  // Sets this tag on the token at index.
  void SetTag(int index, int tag) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, tag_.size());
    tag_[index] = tag;
  }

  // Returns the gold tag for a given token.
  int GoldTag(int index) const {
    DCHECK_GE(index, -1);
    DCHECK_LT(index, gold_tag_.size());
    return index == -1 ? -1 : gold_tag_[index];
  }

  // Returns the string representation of a POS tag, or an empty string
  // if the tag is invalid.
  string TagAsString(int tag) const {
    if (tag >= 0 && tag < tag_map_->Size()) {
      return tag_map_->GetTerm(tag);
    }
    return "";
  }

  // Adds transition state specific annotations to the document.
  void AddParseToDocument(const ParserState &state, bool rewrite_root_labels,
                          Sentence *sentence) const override {
    for (size_t i = 0; i < tag_.size(); ++i) {
      Token *token = sentence->mutable_token(i);
      token->set_tag(TagAsString(Tag(i)));
      if (tag_to_category_) {
        token->set_category(tag_to_category_->GetCategory(token->tag()));
      }
    }
  }

  // Whether a parsed token should be considered correct for evaluation.
  bool IsTokenCorrect(const ParserState &state, int index) const override {
    return GoldTag(index) == Tag(index);
  }

  // Returns a human readable string representation of this state.
  string ToString(const ParserState &state) const override {
    string str;
    for (int i = state.StackSize(); i > 0; --i) {
      const string &word = state.GetToken(state.Stack(i - 1)).word();
      if (i != state.StackSize() - 1) str.append(" ");
      tensorflow::strings::StrAppend(
          &str, word, "[", TagAsString(Tag(state.StackSize() - i)), "]");
    }
    for (int i = state.Next(); i < state.NumTokens(); ++i) {
      tensorflow::strings::StrAppend(&str, " ", state.GetToken(i).word());
    }
    return str;
  }

 private:
  // Currently assigned POS tags for each token in this sentence.
  vector<int> tag_;

  // Gold POS tags from the input document.
  vector<int> gold_tag_;

  // Tag map used for conversions between integer and string representations
  // part of speech tags. Not owned.
  const TermFrequencyMap *tag_map_ = nullptr;

  // Tag to category map. Not owned.
  const TagToCategoryMap *tag_to_category_ = nullptr;

  TF_DISALLOW_COPY_AND_ASSIGN(TaggerTransitionState);
};

class TaggerTransitionSystem : public ParserTransitionSystem {
 public:
  ~TaggerTransitionSystem() override { SharedStore::Release(tag_map_); }

  // Determines tag map location.
  void Setup(TaskContext *context) override {
    input_tag_map_ = context->GetInput("tag-map", "text", "");
    join_category_to_pos_ = context->GetBoolParameter("join_category_to_pos");
    input_tag_to_category_ = context->GetInput("tag-to-category", "text", "");
  }

  // Reads tag map and tag to category map.
  void Init(TaskContext *context) override {
    const string tag_map_path = TaskContext::InputFile(*input_tag_map_);
    tag_map_ = SharedStoreUtils::GetWithDefaultName<TermFrequencyMap>(
        tag_map_path, 0, 0);
    if (!join_category_to_pos_) {
      const string tag_to_category_path =
          TaskContext::InputFile(*input_tag_to_category_);
      tag_to_category_ = SharedStoreUtils::GetWithDefaultName<TagToCategoryMap>(
          tag_to_category_path);
    }
  }

  // The SHIFT action uses the same value as the corresponding action type.
  static ParserAction ShiftAction(int tag) { return tag; }

  // The tagger transition system doesn't look at the dependency tree, so it
  // allows non-projective trees.
  bool AllowsNonProjective() const override { return true; }

  // Returns the number of action types.
  int NumActionTypes() const override { return 1; }

  // Returns the number of possible actions.
  int NumActions(int num_labels) const override { return tag_map_->Size(); }

  // The default action for a given state is assigning the most frequent tag.
  ParserAction GetDefaultAction(const ParserState &state) const override {
    return ShiftAction(0);
  }

  // Returns the next gold action for a given state according to the
  // underlying annotated sentence.
  ParserAction GetNextGoldAction(const ParserState &state) const override {
    if (!state.EndOfInput()) {
      return ShiftAction(TransitionState(state).GoldTag(state.Next()));
    }
    return ShiftAction(0);
  }

  // Checks if the action is allowed in a given parser state.
  bool IsAllowedAction(ParserAction action,
                       const ParserState &state) const override {
    return !state.EndOfInput();
  }

  // Makes a shift by pushing the next input token on the stack and moving to
  // the next position.
  void PerformActionWithoutHistory(ParserAction action,
                                   ParserState *state) const override {
    DCHECK(!state->EndOfInput());
    if (!state->EndOfInput()) {
      MutableTransitionState(state)->SetTag(state->Next(), action);
      state->Push(state->Next());
      state->Advance();
    }
  }

  // We are in a final state when we reached the end of the input and the stack
  // is empty.
  bool IsFinalState(const ParserState &state) const override {
    return state.EndOfInput();
  }

  // Returns a string representation of a parser action.
  string ActionAsString(ParserAction action,
                        const ParserState &state) const override {
    return tensorflow::strings::StrCat("SHIFT(", tag_map_->GetTerm(action),
                                       ")");
  }

  // No state is deterministic in this transition system.
  bool IsDeterministicState(const ParserState &state) const override {
    return false;
  }

  // Returns a new transition state to be used to enhance the parser state.
  ParserTransitionState *NewTransitionState(bool training_mode) const override {
    return new TaggerTransitionState(tag_map_, tag_to_category_);
  }

  // Downcasts the const ParserTransitionState in ParserState to a const
  // TaggerTransitionState.
  static const TaggerTransitionState &TransitionState(
      const ParserState &state) {
    return *static_cast<const TaggerTransitionState *>(
        state.transition_state());
  }

  // Downcasts the ParserTransitionState in ParserState to an
  // TaggerTransitionState.
  static TaggerTransitionState *MutableTransitionState(ParserState *state) {
    return static_cast<TaggerTransitionState *>(
        state->mutable_transition_state());
  }

  // Input for the tag map. Not owned.
  TaskInput *input_tag_map_ = nullptr;

  // Tag map used for conversions between integer and string representations
  // part of speech tags. Owned through SharedStore.
  const TermFrequencyMap *tag_map_ = nullptr;

  // Input for the tag to category map. Not owned.
  TaskInput *input_tag_to_category_ = nullptr;

  // Tag to category map. Owned through SharedStore.
  const TagToCategoryMap *tag_to_category_ = nullptr;

  bool join_category_to_pos_ = false;
};

REGISTER_TRANSITION_SYSTEM("tagger", TaggerTransitionSystem);

// Feature function for retrieving the tag assigned to a token by the tagger
// transition system.
class PredictedTagFeatureFunction
    : public BasicParserSentenceFeatureFunction<Tag> {
 public:
  PredictedTagFeatureFunction() {}

  // Gets the TaggerTransitionState from the parser state and reads the assigned
  // tag at the focus index. Returns -1 if the focus is not within the sentence.
  FeatureValue Compute(const WorkspaceSet &workspaces, const ParserState &state,
                       int focus, const FeatureVector *result) const override {
    if (focus < 0 || focus >= state.sentence().token_size()) return -1;
    return static_cast<const TaggerTransitionState *>(state.transition_state())
        ->Tag(focus);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PredictedTagFeatureFunction);
};

REGISTER_PARSER_IDX_FEATURE_FUNCTION("pred-tag", PredictedTagFeatureFunction);

}  // namespace syntaxnet
