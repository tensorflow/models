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

// Morpher transition system.
//
// This transition system has one type of actions:
//  - The SHIFT action pushes the next input token to the stack and
//    advances to the next input token, assigning a part-of-speech tag to the
//    token that was shifted.
//
// The transition system operates with parser actions encoded as integers:
//  - A SHIFT action is encoded as number starting from 0.

#include <string>

#include "syntaxnet/morphology_label_set.h"
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

class MorphologyTransitionState : public ParserTransitionState {
 public:
  explicit MorphologyTransitionState(const MorphologyLabelSet *label_set)
      : label_set_(label_set) {}

  explicit MorphologyTransitionState(const MorphologyTransitionState *state)
      : MorphologyTransitionState(state->label_set_) {
    tag_ = state->tag_;
    gold_tag_ = state->gold_tag_;
  }

  // Clones the transition state by returning a new object.
  ParserTransitionState *Clone() const override {
    return new MorphologyTransitionState(this);
  }

  // Reads gold tags for each token.
  void Init(ParserState *state) override {
    tag_.resize(state->sentence().token_size(), -1);
    gold_tag_.resize(state->sentence().token_size(), -1);
    for (int pos = 0; pos < state->sentence().token_size(); ++pos) {
      const Token &token = state->GetToken(pos);

      // NOTE: we allow token to not have a TokenMorphology extension or for the
      // TokenMorphology to be absent from the label_set_ because this can
      // happen at test time.
      gold_tag_[pos] = label_set_->LookupExisting(
          token.GetExtension(TokenMorphology::morphology));
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

  // Returns the proto corresponding to the tag, or an empty proto if the tag is
  // not found.
  const TokenMorphology &TagAsProto(int tag) const {
    if (tag >= 0 && tag < label_set_->Size()) {
      return label_set_->Lookup(tag);
    }
    return TokenMorphology::default_instance();
  }

  // Adds transition state specific annotations to the document.
  void AddParseToDocument(const ParserState &state, bool rewrite_root_labels,
                          Sentence *sentence) const override {
    for (int i = 0; i < tag_.size(); ++i) {
      Token *token = sentence->mutable_token(i);
      *token->MutableExtension(TokenMorphology::morphology) =
          TagAsProto(Tag(i));
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
          &str, word, "[",
          TagAsProto(Tag(state.StackSize() - i)).ShortDebugString(), "]");
    }
    for (int i = state.Next(); i < state.NumTokens(); ++i) {
      tensorflow::strings::StrAppend(&str, " ", state.GetToken(i).word());
    }
    return str;
  }

 private:
  // Currently assigned morphological analysis for each token in this sentence.
  std::vector<int> tag_;

  // Gold morphological analysis from the input document.
  std::vector<int> gold_tag_;

  // Tag map used for conversions between integer and string representations
  // part of speech tags. Not owned.
  const MorphologyLabelSet *label_set_ = nullptr;

  TF_DISALLOW_COPY_AND_ASSIGN(MorphologyTransitionState);
};

class MorphologyTransitionSystem : public ParserTransitionSystem {
 public:
  ~MorphologyTransitionSystem() override { SharedStore::Release(label_set_); }

  // Determines tag map location.
  void Setup(TaskContext *context) override {
    context->GetInput("morph-label-set");
  }

  // Reads tag map and tag to category map.
  void Init(TaskContext *context) override {
    const string fname =
        TaskContext::InputFile(*context->GetInput("morph-label-set"));
    label_set_ =
        SharedStoreUtils::GetWithDefaultName<MorphologyLabelSet>(fname);
  }

  // The SHIFT action uses the same value as the corresponding action type.
  static ParserAction ShiftAction(int tag) { return tag; }

  // The morpher transition system doesn't look at the dependency tree, so it
  // allows non-projective trees.
  bool AllowsNonProjective() const override { return true; }

  // Returns the number of action types.
  int NumActionTypes() const override { return 1; }

  // Returns the number of possible actions.
  int NumActions(int num_labels) const override { return label_set_->Size(); }

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
    return tensorflow::strings::StrCat(
        "SHIFT(", label_set_->Lookup(action).ShortDebugString(), ")");
  }

  // No state is deterministic in this transition system.
  bool IsDeterministicState(const ParserState &state) const override {
    return false;
  }

  // Returns a new transition state to be used to enhance the parser state.
  ParserTransitionState *NewTransitionState(bool training_mode) const override {
    return new MorphologyTransitionState(label_set_);
  }

  // Downcasts the const ParserTransitionState in ParserState to a const
  // MorphologyTransitionState.
  static const MorphologyTransitionState &TransitionState(
      const ParserState &state) {
    return *static_cast<const MorphologyTransitionState *>(
        state.transition_state());
  }

  // Downcasts the ParserTransitionState in ParserState to an
  // MorphologyTransitionState.
  static MorphologyTransitionState *MutableTransitionState(ParserState *state) {
    return static_cast<MorphologyTransitionState *>(
        state->mutable_transition_state());
  }

  // Input for the tag map. Not owned.
  TaskInput *input_label_set_ = nullptr;

  // Tag map used for conversions between integer and string representations
  // morphology labels. Owned through SharedStore.
  const MorphologyLabelSet *label_set_;
};

REGISTER_TRANSITION_SYSTEM("morpher", MorphologyTransitionSystem);

// Feature function for retrieving the tag assigned to a token by the tagger
// transition system.
class PredictedMorphTagFeatureFunction : public ParserIndexFeatureFunction {
 public:
  PredictedMorphTagFeatureFunction() {}

  // Determines tag map location.
  void Setup(TaskContext *context) override {
    context->GetInput("morph-label-set", "recordio", "token-morphology");
  }

  // Reads tag map.
  void Init(TaskContext *context) override {
    const string fname =
        TaskContext::InputFile(*context->GetInput("morph-label-set"));
    label_set_ = SharedStore::Get<MorphologyLabelSet>(fname, fname);
    set_feature_type(new FullLabelFeatureType(name(), label_set_));
  }

  // Gets the MorphologyTransitionState from the parser state and reads the
  // assigned
  // tag at the focus index. Returns -1 if the focus is not within the sentence.
  FeatureValue Compute(const WorkspaceSet &workspaces, const ParserState &state,
                       int focus, const FeatureVector *result) const override {
    if (focus < 0 || focus >= state.sentence().token_size()) return -1;
    return static_cast<const MorphologyTransitionState *>(
               state.transition_state())
        ->Tag(focus);
  }

 private:
  // Tag map used for conversions between integer and string representations
  // part of speech tags. Owned through SharedStore.
  const MorphologyLabelSet *label_set_;

  TF_DISALLOW_COPY_AND_ASSIGN(PredictedMorphTagFeatureFunction);
};

REGISTER_PARSER_IDX_FEATURE_FUNCTION("pred-morph-tag",
                                     PredictedMorphTagFeatureFunction);

}  // namespace syntaxnet
